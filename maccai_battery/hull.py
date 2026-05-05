# =============================================================================
# maccai_battery.hull — Materials Project convex hull analysis
# =============================================================================
# Queries the Materials Project database to evaluate the thermodynamic
# stability of DFT-relaxed structures by computing their energy above the
# convex hull.
#
# Workflow:
#   1. Load DFT-relaxed energies from candidates.ndjson
#   2. Query the Materials Project for all stable/metastable phases in the
#      target chemical space (e.g. Li-Fe-P-O)
#   3. Build a pymatgen PhaseDiagram from MP reference data
#   4. Compute ΔH_hull for each candidate structure
#   5. Write results back to candidates.ndjson
#
# Prerequisites:
#   pip install mp-api pymatgen
#   Set MP_API_KEY environment variable (get your key at materialsproject.org)
#
# Usage:
#   from maccai_battery.hull import HullAnalyzer
#
#   analyzer = HullAnalyzer(cfg)
#   results  = analyzer.run(candidates)
#   analyzer.print_ranking(results)
# =============================================================================

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# MP POTCAR titles — must match what MaterialsProject2020Compatibility expects.
# Generic "PAW_PBE Fe" is rejected; the actual MP titles use suffixes like
# _pv, _sv etc.  Extend this dict if you add new elements to the chemical system.
# ---------------------------------------------------------------------------
_MP_POTCAR_MAP: Dict[str, str] = {
    "Li": "PAW_PBE Li_sv",
    "Fe": "PAW_PBE Fe_pv",
    "Mn": "PAW_PBE Mn_pv",
    "Co": "PAW_PBE Co",
    "Ni": "PAW_PBE Ni_pv",
    "V":  "PAW_PBE V_sv",
    "Ti": "PAW_PBE Ti_pv",
    "Cr": "PAW_PBE Cr_pv",
    "Cu": "PAW_PBE Cu_pv",
    "Na": "PAW_PBE Na_pv",
    "K":  "PAW_PBE K_sv",
    "Ca": "PAW_PBE Ca_sv",
    "Mg": "PAW_PBE Mg_pv",
    "Al": "PAW_PBE Al",
    "Si": "PAW_PBE Si",
    "P":  "PAW_PBE P",
    "S":  "PAW_PBE S",
    "O":  "PAW_PBE O",
    "F":  "PAW_PBE F",
    "Cl": "PAW_PBE Cl",
}

# ---------------------------------------------------------------------------
# Standard MP GGA+U Hubbard U values — same values used in the MP database.
# These MUST match MP's values exactly so that MP2020Compatibility applies
# the correct energy correction to your candidates.
# Only elements in this dict get GGA+U treatment; everything else is plain GGA.
# ---------------------------------------------------------------------------
_MP_HUBBARD_U: Dict[str, float] = {
    "Fe": 5.3,
    "Mn": 3.9,
    "Co": 3.32,
    "Ni": 6.2,
    "V":  3.25,
    "Cr": 3.7,
    "Mo": 4.38,
    "W":  6.2,
}


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class HullResult:
    """Thermodynamic stability result for a single candidate.

    Attributes
    ----------
    candidate_id : str
        The ``"id"`` field from candidates.ndjson.
    formula : str
        Reduced chemical formula.
    e_above_hull_eV_per_atom : Optional[float]
        Energy above the convex hull in eV/atom.
        - 0.0          → on the hull (thermodynamically stable)
        - 0.0 – 0.05   → likely synthesisable (low metastability)
        - 0.05 – 0.1   → possibly synthesisable under special conditions
        - > 0.1        → probably metastable or unstable
        ``None`` if the calculation could not be completed.
    dft_energy_eV_per_atom : Optional[float]
        DFT total energy per atom used for the hull calculation (eV/atom).
    mp_stable_phases : List[str]
        Competing MP stable phases with decomposition fractions.
    is_stable : Optional[bool]
        True if e_above_hull ≤ stability_threshold from config.
    stability_threshold_eV : float
        The threshold used for ``is_stable`` (from config).
    used_ml_structure : bool
        True if the ML-relaxed geometry was used as fallback (DFT relax failed).
    error : Optional[str]
        Exception message if the computation failed, else None.
    """

    candidate_id: str
    formula: str
    e_above_hull_eV_per_atom: Optional[float]   = None
    dft_energy_eV_per_atom: Optional[float]     = None
    mp_stable_phases: List[str]                 = field(default_factory=list)
    is_stable: Optional[bool]                   = None
    stability_threshold_eV: float               = 0.1
    used_ml_structure: bool                     = False
    error: Optional[str]                        = None

    @property
    def success(self) -> bool:
        return self.error is None and self.e_above_hull_eV_per_atom is not None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidate_id":             self.candidate_id,
            "formula":                  self.formula,
            "e_above_hull_eV_per_atom": self.e_above_hull_eV_per_atom,
            "dft_energy_eV_per_atom":   self.dft_energy_eV_per_atom,
            "mp_stable_phases":         self.mp_stable_phases,
            "is_stable":                self.is_stable,
            "stability_threshold_eV":   self.stability_threshold_eV,
            "used_ml_structure":        self.used_ml_structure,
            "error":                    self.error,
        }

    def summary_line(self) -> str:
        if not self.success:
            return f"[ERROR] {self.formula} ({self.candidate_id}): {self.error}"
        status = "STABLE" if self.is_stable else "METASTABLE"
        suffix = " [ML geometry]" if self.used_ml_structure else ""
        return (
            f"[{status}] {self.formula} ({self.candidate_id})  "
            f"ΔH_hull = {self.e_above_hull_eV_per_atom:.4f} eV/atom{suffix}"
        )


# ---------------------------------------------------------------------------
# Main hull analyzer class
# ---------------------------------------------------------------------------

class HullAnalyzer:
    """Compute energy above the convex hull for a set of DFT-relaxed candidates.

    Uses the Materials Project REST API (v2) to fetch reference phase energies
    and constructs a pymatgen ``PhaseDiagram`` locally.

    Parameters
    ----------
    cfg : PipelineConfig
        Fully loaded pipeline configuration.
    api_key : str, optional
        Materials Project API key.  If omitted, reads ``MP_API_KEY``
        from the environment.  Get your key at https://materialsproject.org.

    Raises
    ------
    EnvironmentError
        If no API key is available.
    ImportError
        If ``mp-api`` or ``pymatgen`` is not installed.
    """

    def __init__(self, cfg, api_key: Optional[str] = None) -> None:
        self._cfg      = cfg
        self._api_key  = api_key or os.environ.get("MP_API_KEY")
        self._hull_cfg = getattr(cfg, "hull", None)
        self._threshold = (
            self._hull_cfg.stability_threshold_eV
            if self._hull_cfg is not None
            else 0.1
        )
        # Cache the phase diagram by chemical-system key so that calling
        # run() multiple times doesn't re-hit the MP API each time.
        self._pd_cache: Dict[str, Tuple] = {}

        if not self._api_key:
            raise EnvironmentError(
                "No Materials Project API key found.\n"
                "Either pass api_key= to HullAnalyzer() or set the "
                "MP_API_KEY environment variable.\n"
                "Get your free key at: https://materialsproject.org/api"
            )

        _check_imports()
        logger.debug("HullAnalyzer initialised (threshold=%.3f eV/atom).", self._threshold)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        candidates: List[Dict[str, Any]],
        energy_source: str = "relax",
    ) -> List[HullResult]:
        """Compute hull distances for a list of candidate records.

        Parameters
        ----------
        candidates : list of dict
            Candidate records from ``CandidateDatabase.load_all()``.
            Each must have ``dft_jobs.relax.energy_eV_per_atom`` (or
            ``dft_jobs.scf.energy_eV_per_atom`` when *energy_source*
            is ``"scf"``).
        energy_source : {"relax", "scf"}
            Which DFT stage to take energies from.
            - ``"relax"`` (default): use the fully relaxed DFT energy.
            - ``"scf"``           : use the SCF single-point energy (less accurate).

        Returns
        -------
        list of HullResult
            One result per candidate with a non-null DFT energy.
            Candidates without a completed DFT energy are skipped.
        """
        # Collect candidates that have a DFT energy
        with_energy = []
        for rec in candidates:
            e = _get_dft_energy(rec, energy_source)
            if e is not None:
                with_energy.append((rec, e))

        if not with_energy:
            logger.warning(
                "No candidates have a completed %s energy — nothing to do.\n"
                "Run the DFT pipeline and merge results with "
                "scripts/05_merge_dft_results.py first.",
                energy_source,
            )
            return []

        logger.info(
            "Running hull analysis on %d candidates with %s energies ...",
            len(with_energy), energy_source,
        )

        # Determine the full set of elements we need MP reference data for
        elements = _parse_elements(self._cfg.generation.chemical_system)
        logger.info("Fetching MP reference data for elements: %s", elements)

        # Build (or retrieve from cache) the phase diagram
        phase_diagram, compat = self._build_phase_diagram(elements)
        logger.info(
            "Phase diagram ready with %d entries.",
            len(phase_diagram.all_entries),
        )

        # Evaluate every candidate
        results: List[HullResult] = []
        for rec, e_pa in with_energy:
            result = self._evaluate_candidate(rec, e_pa, phase_diagram, compat)
            results.append(result)
            if result.success:
                logger.info("  %s", result.summary_line())
            else:
                logger.warning("  %s", result.summary_line())

        # Summary counts
        stable     = [r for r in results if r.is_stable]
        metastable = [r for r in results if r.success and not r.is_stable]
        failed     = [r for r in results if not r.success]
        logger.info(
            "Hull analysis complete: %d stable, %d metastable, %d failed.",
            len(stable), len(metastable), len(failed),
        )

        return results

    def print_ranking(self, results: List[HullResult]) -> None:
        """Print a formatted ranking table sorted by ΔH_hull."""
        successful = sorted(
            [r for r in results if r.success],
            key=lambda r: r.e_above_hull_eV_per_atom,  # type: ignore[arg-type]
        )
        failed = [r for r in results if not r.success]

        width = 80
        print(f"\n{'=' * width}")
        print(f"  Hull Analysis — Stability Ranking")
        print(f"  Threshold: ΔH ≤ {self._threshold:.3f} eV/atom = stable")
        print(f"{'=' * width}")
        print(f"  {'Rank':>4}  {'ID':<14}  {'Formula':<16}  "
              f"{'ΔH_hull (eV/a)':>16}  {'Status':<20}")
        print(f"  {'-' * 74}")

        for rank, r in enumerate(successful, start=1):
            status = "✓ STABLE" if r.is_stable else "  metastable"
            if r.used_ml_structure:
                status += " [ML geom]"
            print(
                f"  {rank:>4}  {r.candidate_id:<14}  {r.formula:<16}  "
                f"{r.e_above_hull_eV_per_atom:>16.4f}  {status:<20}"
            )

        if failed:
            print(f"\n  Failed ({len(failed)}):")
            for r in failed:
                print(f"    {r.candidate_id}: {r.error}")

        print(f"{'=' * width}\n")

    def update_database(
        self,
        results: List[HullResult],
        db,             # CandidateDatabase
    ) -> int:
        """Write hull analysis results back into candidates.ndjson.

        Stores results under a ``"hull_analysis"`` key in each record.

        Returns
        -------
        int
            Number of records updated.
        """
        n_updated = 0
        for result in results:
            if not result.success:
                continue
            ok = db.update_field(
                result.candidate_id,
                "hull_analysis",
                result.to_dict(),
            )
            if ok:
                n_updated += 1

        logger.info("Updated %d records with hull analysis results.", n_updated)
        return n_updated

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_phase_diagram(self, elements: List[str]) -> Tuple:
        """Fetch MP reference energies and build a pymatgen PhaseDiagram.

        Results are cached by chemical system so the MP API is only hit once
        per unique element set per HullAnalyzer instance.

        Parameters
        ----------
        elements : list of str
            Element symbols to include in the chemical space.

        Returns
        -------
        (PhaseDiagram, MaterialsProject2020Compatibility)
        """
        # --- Cache check: return immediately if already built ---
        cache_key = "-".join(sorted(elements))
        if cache_key in self._pd_cache:
            logger.debug("Phase diagram cache hit for '%s'.", cache_key)
            return self._pd_cache[cache_key]

        from mp_api.client import MPRester                              # type: ignore[import]
        from pymatgen.analysis.phase_diagram import PhaseDiagram
        from pymatgen.entries.compatibility import MaterialsProject2020Compatibility

        with MPRester(self._api_key) as mpr:
            # Fetch ALL entries in the chemsys — no energy_above_hull filter.
            # Filtering to e.g. (0, 0.05) can silently drop phases that anchor
            # the hull at intermediate compositions, making hull distances wrong.
            entries = mpr.get_entries_in_chemsys(elements=elements)

        if not entries:
            raise RuntimeError(
                f"No MP entries found for chemical space {elements}. "
                f"Check your API key and chemical system spelling."
            )

        logger.debug("Fetched %d raw MP entries.", len(entries))

        # Apply the standard MP2020 compatibility scheme.
        # This corrects for oxide/peroxide errors and GGA+U energy offsets
        # so that your candidate energies are on the same scale as MP's data.
        compat = MaterialsProject2020Compatibility()
        entries_corrected = compat.process_entries(entries, clean=True)

        if not entries_corrected:
            raise RuntimeError(
                "All MP entries were rejected by MaterialsProject2020Compatibility. "
                "This usually means entries are missing required parameters. "
                "Try fetching with compatible_only=True in MPRester."
            )

        logger.debug(
            "%d / %d MP entries survived compatibility processing.",
            len(entries_corrected), len(entries),
        )

        phase_diagram = PhaseDiagram(entries_corrected)

        # Store in cache before returning
        self._pd_cache[cache_key] = (phase_diagram, compat)
        return phase_diagram, compat

    def _evaluate_candidate(
        self,
        record: Dict[str, Any],
        e_pa: float,
        phase_diagram,
        compat,
    ) -> HullResult:
        """Compute ΔH_hull for a single candidate record.

        Steps
        -----
        1. Resolve the structure (DFT-relaxed → ML-relaxed fallback).
        2. Determine Hubbard U values from _MP_HUBBARD_U for elements present.
        3. Build a ComputedStructureEntry with correct MP-compatible parameters.
        4. Apply MP2020Compatibility corrections.
        5. Query the phase diagram for e_above_hull and decomposition products.
        """
        from pymatgen.core import Composition, Structure
        from pymatgen.entries.computed_entries import ComputedStructureEntry

        cid     = record.get("id", "unknown")
        formula = record.get("formula", "unknown")

        result = HullResult(
            candidate_id           = cid,
            formula                = formula,
            dft_energy_eV_per_atom = e_pa,
            stability_threshold_eV = self._threshold,
        )

        try:
            # ----------------------------------------------------------
            # 1. Resolve structure
            #    Priority: DFT-relaxed → ML-relaxed fallback → error
            # ----------------------------------------------------------
            structure_dict = (
                record.get("dft_jobs", {}).get("relax", {}).get("structure")
                or record.get("structure")          # legacy / top-level fallback
            )

            if structure_dict:
                structure = Structure.from_dict(structure_dict)
            else:
                # Fallback to ML-relaxed geometry (less accurate but better than nothing)
                ml_path = record.get("structure_files", {}).get("ml_relaxed")
                if ml_path and Path(ml_path).exists():
                    try:
                        from ase.io import read
                        from pymatgen.io.ase import AseAtomsAdaptor
                        atoms = read(ml_path)
                        structure = AseAtomsAdaptor.get_structure(atoms)
                        result.used_ml_structure = True
                        logger.warning(
                            "%s (%s): no DFT-relaxed structure found — "
                            "falling back to ML-relaxed geometry. "
                            "Hull distance will be less accurate.",
                            cid, formula,
                        )
                    except Exception as ase_err:
                        raise ValueError(
                            f"ML-relaxed structure at '{ml_path}' could not be read: {ase_err}"
                        )
                else:
                    raise ValueError(
                        f"No structure available for {cid} ({formula}). "
                        f"Run 05_merge_dft_results.py to store the relaxed structure, "
                        f"or ensure structure_files.ml_relaxed is set in the record."
                    )

            # ----------------------------------------------------------
            # 2. Derive atom count and total energy from the actual cell.
            #    Using structure.num_sites (not the reduced-formula count)
            #    is essential for non-primitive supercells.
            # ----------------------------------------------------------
            n_atoms = structure.num_sites
            e_total = e_pa * n_atoms

            # ----------------------------------------------------------
            # 3. Hubbard U values
            #    Use the standard MP values from _MP_HUBBARD_U so that
            #    MP2020Compatibility applies exactly the same GGA+U
            #    correction as it does to the reference entries.
            #    Only elements actually present in the composition get U.
            # ----------------------------------------------------------
            comp       = Composition(formula)
            elements_in_comp = set(comp.as_dict().keys())

            hubbards   = {
                el: _MP_HUBBARD_U[el]
                for el in elements_in_comp
                if el in _MP_HUBBARD_U
            }
            is_hubbard = bool(hubbards)
            run_type   = "GGA+U" if is_hubbard else "GGA"

            logger.debug(
                "%s: run_type=%s, hubbards=%s",
                cid, run_type, hubbards,
            )

            # ----------------------------------------------------------
            # 4. Build the ComputedStructureEntry.
            #    ComputedStructureEntry (not ComputedEntry) is required
            #    here because MP2020Compatibility reads the structure to
            #    detect oxide/peroxide oxygen species for corrections.
            # ----------------------------------------------------------
            raw_entry = ComputedStructureEntry(
                structure = structure,
                energy    = e_total,
                entry_id  = cid,
                parameters = {
                    "run_type":   run_type,
                    "is_hubbard": is_hubbard,
                    "hubbards":   hubbards,
                    "potcar_spec": [
                        {
                            "titel": _MP_POTCAR_MAP.get(el, f"PAW_PBE {el}"),
                            "hash":  None,
                        }
                        for el in sorted(elements_in_comp)
                    ],
                },
            )

            # ----------------------------------------------------------
            # 5. Apply MP2020Compatibility corrections
            # ----------------------------------------------------------
            corrected = compat.process_entries([raw_entry], clean=True)

            if not corrected:
                # Uncomment for debugging:
                # logger.debug(compat.explain(raw_entry))
                raise ValueError(
                    f"Entry rejected by MaterialsProject2020Compatibility for "
                    f"{formula} ({cid}). "
                    f"Check POTCAR titles and Hubbard U values match MP conventions."
                )

            entry = corrected[0]

            # ----------------------------------------------------------
            # 6. Compute hull distance and decomposition products
            # ----------------------------------------------------------
            e_hull = phase_diagram.get_e_above_hull(entry)

            decomp, _ = phase_diagram.get_decomp_and_e_above_hull(entry)

            # Store decomposition products with their fractional amounts
            # e.g. "LiFePO4 (0.50)" — useful for understanding what the
            # candidate decomposes into if it's metastable.
            competing = sorted(
                f"{e.composition.reduced_formula} ({amt:.3f})"
                for e, amt in decomp.items()
            )

            result.e_above_hull_eV_per_atom = float(e_hull)
            result.mp_stable_phases         = competing
            result.is_stable                = float(e_hull) <= self._threshold

        except Exception as exc:
            result.error = f"{type(exc).__name__}: {exc}"
            logger.debug(
                "Hull evaluation failed for %s (%s): %s",
                cid, formula, exc, exc_info=True,
            )

        return result


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def compute_hull_distances(
    cfg,
    candidates: List[Dict[str, Any]],
    energy_source: str = "relax",
    api_key: Optional[str] = None,
) -> List[HullResult]:
    """One-shot convenience wrapper around :class:`HullAnalyzer`.

    Parameters
    ----------
    cfg : PipelineConfig
    candidates : list of dict
    energy_source : {"relax", "scf"}
    api_key : str, optional

    Returns
    -------
    list of HullResult
    """
    analyzer = HullAnalyzer(cfg, api_key=api_key)
    return analyzer.run(candidates, energy_source=energy_source)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_elements(chemical_system: str) -> List[str]:
    """Split ``"Li-Fe-P-O"`` into ``["Li", "Fe", "P", "O"]``."""
    return [el.strip() for el in chemical_system.split("-") if el.strip()]


def _get_dft_energy(
    record: Dict[str, Any],
    source: str,
) -> Optional[float]:
    """Extract the DFT energy per atom from a candidate record.

    Parameters
    ----------
    record : dict
        Candidate record from the NDJSON database.
    source : {"relax", "scf"}
        Which DFT stage to use.

    Returns
    -------
    float or None
    """
    dft   = record.get("dft_jobs", {})
    stage = dft.get(source, {})
    return stage.get("energy_eV_per_atom")


def _check_imports() -> None:
    """Raise ImportError with a helpful message if dependencies are missing."""
    missing = []
    try:
        import mp_api  # noqa: F401
    except ImportError:
        missing.append("mp-api")

    try:
        from pymatgen.analysis.phase_diagram import PhaseDiagram  # noqa: F401
    except ImportError:
        missing.append("pymatgen")

    if missing:
        raise ImportError(
            f"Missing required packages for hull analysis: {missing}\n"
            f"Install with: pip install {' '.join(missing)}"
        )


# ---------------------------------------------------------------------------
# Quick-check: validate MP API key without running the full pipeline
# ---------------------------------------------------------------------------

def check_mp_api_key(api_key: Optional[str] = None) -> bool:
    """Verify that the Materials Project API key is valid.

    Parameters
    ----------
    api_key : str, optional
        API key to test.  Falls back to ``MP_API_KEY`` environment variable.

    Returns
    -------
    bool
        ``True`` if the key is valid and the API is reachable.

    Raises
    ------
    EnvironmentError
        If no key is provided and ``MP_API_KEY`` is not set.
    """
    key = api_key or os.environ.get("MP_API_KEY")
    if not key:
        raise EnvironmentError(
            "MP_API_KEY environment variable is not set.\n"
            "Get your free key at https://materialsproject.org/api"
        )

    _check_imports()

    try:
        from mp_api.client import MPRester  # type: ignore[import]
        with MPRester(key) as mpr:
            entries = mpr.get_entries_in_chemsys(["Li", "O"])
        logger.info("MP API key is valid. Retrieved %d Li-O entries.", len(entries))
        return True
    except Exception as exc:
        logger.error("MP API key validation failed: %s", exc)
        return False