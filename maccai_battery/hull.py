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
        Competing MP stable phases used to build the hull for this composition.
    is_stable : Optional[bool]
        True if e_above_hull ≤ stability_threshold from config.
    stability_threshold_eV : float
        The threshold used for ``is_stable`` (from config).
    error : Optional[str]
        Exception message if the computation failed, else None.
    """

    candidate_id: str
    formula: str
    e_above_hull_eV_per_atom: Optional[float]       = None
    dft_energy_eV_per_atom: Optional[float]         = None
    mp_stable_phases: List[str]                     = field(default_factory=list)
    is_stable: Optional[bool]                       = None
    stability_threshold_eV: float                   = 0.1
    error: Optional[str]                            = None

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
            "error":                    self.error,
        }

    def summary_line(self) -> str:
        if not self.success:
            return f"[ERROR] {self.formula} ({self.candidate_id}): {self.error}"
        status = "STABLE" if self.is_stable else "METASTABLE"
        return (
            f"[{status}] {self.formula} ({self.candidate_id})  "
            f"ΔH_hull = {self.e_above_hull_eV_per_atom:.4f} eV/atom"
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
        self._cfg     = cfg
        self._api_key = api_key or os.environ.get("MP_API_KEY")
        self._hull_cfg = getattr(cfg, "hull", None)
        self._threshold = (
            self._hull_cfg.stability_threshold_eV
            if self._hull_cfg is not None
            else 0.1
        )

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
        # Collect those candidates that have a DFT energy
        with_energy = []
        for rec in candidates:
            e = _get_dft_energy(rec, energy_source)
            if e is not None:
                with_energy.append((rec, e))

        if not with_energy:
            logger.warning(
                "No candidates have a completed %s energy — nothing to do.\n"
                "Run the DFT pipeline (Colab notebook) and merge results "
                "back with scripts/04_merge_dft_results.py first.",
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

        # Build the phase diagram from MP data
        phase_diagram, compat = self._build_phase_diagram(elements)
        logger.info(
            "Phase diagram built with %d entries.",
            len(phase_diagram.all_entries),
        )

        # Extract MP elemental reference energies (corrected VASP scale).
        # Used in _evaluate_candidate to convert QE formation energies onto
        # the same energy scale as the MP phase diagram.
        mp_el_refs: Dict[str, float] = {
            str(el): entry.energy_per_atom
            for el, entry in phase_diagram.el_refs.items()
        }
        logger.debug(
            "MP elemental refs (eV/atom): %s",
            {el: f"{e:.4f}" for el, e in mp_el_refs.items()},
        )

        # Evaluate every candidate
        results: List[HullResult] = []
        for rec, e_pa in with_energy:
            result = self._evaluate_candidate(rec, e_pa, phase_diagram, compat, mp_el_refs)
            results.append(result)
            if result.success:
                logger.info("  %s", result.summary_line())
            else:
                logger.warning("  %s", result.summary_line())

        # Summary
        stable     = [r for r in results if r.is_stable]
        metastable = [r for r in results if r.success and not r.is_stable]
        failed     = [r for r in results if not r.success]
        logger.info(
            "Hull analysis complete: %d stable, %d metastable, %d failed.",
            len(stable), len(metastable), len(failed),
        )

        return results

    def print_ranking(self, results: List[HullResult]) -> None:
        """Print a formatted ranking table sorted by ΔH_hull.

        Parameters
        ----------
        results : list of HullResult
        """
        successful = sorted(
            [r for r in results if r.success],
            key=lambda r: r.e_above_hull_eV_per_atom,  # type: ignore[arg-type]
        )
        failed = [r for r in results if not r.success]

        width = 72
        print(f"\n{'=' * width}")
        print(f"  Hull Analysis — Stability Ranking")
        print(f"  Threshold: ΔH ≤ {self._threshold:.3f} eV/atom = stable")
        print(f"{'=' * width}")
        print(f"  {'Rank':>4}  {'ID':<12}  {'Formula':<14}  "
              f"{'ΔH_hull (eV/a)':>16}  {'Status':<12}")
        print(f"  {'-' * 64}")

        for rank, r in enumerate(successful, start=1):
            status = "✓ STABLE" if r.is_stable else "  metastable"
            print(
                f"  {rank:>4}  {r.candidate_id:<12}  {r.formula:<14}  "
                f"{r.e_above_hull_eV_per_atom:>16.4f}  {status}"
            )

        if failed:
            print(f"\n  Failed ({len(failed)}):")
            for r in failed:
                print(f"    {r.candidate_id}: {r.error}")

        print(f"{'=' * width}\n")

    def update_database(
        self,
        results: List[HullResult],
        db,              # CandidateDatabase
    ) -> int:
        """Write hull analysis results back into candidates.ndjson.

        Stores results under a ``"hull_analysis"`` key in each record.

        Parameters
        ----------
        results : list of HullResult
        db : CandidateDatabase

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
    # Internal
    # ------------------------------------------------------------------

    def _build_phase_diagram(self, elements: List[str]):
        """Fetch MP reference energies and build a pymatgen PhaseDiagram.

        Parameters
        ----------
        elements : list of str
            Element symbols to include in the chemical space.

        Returns
        -------
        pymatgen.analysis.phase_diagram.PhaseDiagram
        """
        from mp_api.client import MPRester  # type: ignore[import]
        from pymatgen.analysis.phase_diagram import PhaseDiagram
        from pymatgen.entries.compatibility import MaterialsProject2020Compatibility

        with MPRester(self._api_key) as mpr:
            # Fetch ALL entries in the chemsys — no energy_above_hull filter.
            # Filtering to (0, 0.05) can silently drop phases that anchor the
            # hull at intermediate compositions, making hull distances wrong.
            entries = mpr.get_entries_in_chemsys(elements=elements)

        if not entries:
            raise RuntimeError(
                f"No MP entries found for chemical space {elements}. "
                f"Check your API key and chemical system spelling."
            )

        logger.debug("Fetched %d raw MP entries.", len(entries))

        # Apply the standard MP2020 compatibility scheme to the reference data.
        # This ensures the MP entries are on a consistent energy scale
        # (oxide/peroxide corrections, GGA+U corrections for d-electron systems).
        compat = MaterialsProject2020Compatibility()
        entries_corrected = compat.process_entries(entries, clean=True)

        if not entries_corrected:
            raise RuntimeError(
                "All MP entries were rejected by MaterialsProject2020Compatibility. "
                "This usually means the entries are missing required parameters. "
                "Try fetching with compatible_only=True in MPRester."
            )

        logger.debug(
            "%d / %d MP entries survived compatibility processing.",
            len(entries_corrected), len(entries),
        )
        return PhaseDiagram(entries_corrected), compat

    def _evaluate_candidate(
        self,
        record: Dict[str, Any],
        e_pa: float,
        phase_diagram,
        compat,
        mp_el_refs: Dict[str, float],
    ) -> HullResult:
        """Compute ΔH_hull for a single candidate using formation energies.

        QE and VASP (MP) have different absolute energy scales because they
        use different pseudopotential implementations.  Directly comparing
        QE total energies to MP entries gives nonsensical hull distances.

        This method corrects for that by:
          1. Computing the QE formation energy using QE elemental references
             from config (cancels the code-specific offset per element).
          2. Converting to the VASP scale by adding back the MP elemental
             reference energies extracted from the phase diagram.
          3. Placing the result on the phase diagram as a PDEntry (no further
             compatibility corrections — the energy is already on the right scale).

        Parameters
        ----------
        record : dict
        e_pa : float — QE DFT energy per atom (eV/atom)
        phase_diagram : PhaseDiagram — built from corrected MP entries
        compat : MaterialsProject2020Compatibility — unused, kept for API compat
        mp_el_refs : dict — {element: corrected_eV_per_atom} from phase_diagram.el_refs
        """
        from pymatgen.core import Composition
        from pymatgen.analysis.phase_diagram import PDEntry

        cid     = record.get("id", "unknown")
        formula = record.get("formula", "unknown")

        result = HullResult(
            candidate_id            = cid,
            formula                 = formula,
            dft_energy_eV_per_atom  = e_pa,
            stability_threshold_eV  = self._threshold,
        )

        try:
            comp    = Composition(formula)
            amounts = comp.as_dict()          # {element: count} for one formula unit
            n_atoms = int(sum(amounts.values()))
            e_total = e_pa * n_atoms          # total QE energy for this formula unit (eV)

            # ----------------------------------------------------------
            # 1. Read QE elemental reference energies from config.
            # ----------------------------------------------------------
            qe_refs = getattr(
                getattr(self._cfg, "elemental_references", None),
                "energies_eV_per_atom", {}
            ) or {}

            missing_qe = [el for el in amounts if qe_refs.get(el) is None]
            if missing_qe:
                raise ValueError(
                    f"Missing QE elemental reference energies for {missing_qe}. "
                    f"Run QE SCF on those elemental phases and add to "
                    f"elemental_references.energies_eV_per_atom in config.yaml."
                )

            # ----------------------------------------------------------
            # 2. Compute QE formation energy (eV for the formula unit).
            #    ΔH_f(QE) = E_QE(candidate) − Σ n_i × E_QE(element_i)
            # ----------------------------------------------------------
            e_ref_qe = sum(n * float(qe_refs[el]) for el, n in amounts.items())
            dh_f_qe  = e_total - e_ref_qe

            # ----------------------------------------------------------
            # 3. Convert to VASP-equivalent energy.
            #    E_vasp = ΔH_f(QE) + Σ n_i × E_MP_ref(element_i)
            #    mp_el_refs are already on the corrected MP2020 scale.
            # ----------------------------------------------------------
            missing_mp = [el for el in amounts if el not in mp_el_refs]
            if missing_mp:
                raise ValueError(
                    f"MP elemental references not found for {missing_mp}. "
                    f"Check that get_entries_in_chemsys fetched all required elements."
                )

            e_ref_mp     = sum(n * mp_el_refs[el] for el, n in amounts.items())
            e_vasp_equiv = dh_f_qe + e_ref_mp

            logger.debug(
                "%s: ΔH_f(QE)=%.4f eV/atom  E_vasp_equiv=%.4f eV/atom",
                cid, dh_f_qe / n_atoms, e_vasp_equiv / n_atoms,
            )

            # ----------------------------------------------------------
            # 4. Place on the phase diagram (no further corrections).
            # ----------------------------------------------------------
            entry = PDEntry(comp, e_vasp_equiv, entry_id=cid)

            decomp, e_hull = phase_diagram.get_decomp_and_e_above_hull(
                entry, allow_negative=True
            )
            e_hull = max(0.0, float(e_hull))   # below-hull → treat as on the hull

            competing = sorted(
                {str(e.composition.reduced_formula) for e in decomp}
            )

            result.e_above_hull_eV_per_atom = e_hull
            result.mp_stable_phases         = competing
            result.is_stable                = e_hull <= self._threshold

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
    dft = record.get("dft_jobs", {})
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
            # Minimal query to test authentication
            entries = mpr.get_entries_in_chemsys(["Li", "O"])
        logger.info("MP API key is valid. Retrieved %d Li-O entries.", len(entries))
        return True
    except Exception as exc:
        logger.error("MP API key validation failed: %s", exc)
        return False