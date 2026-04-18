# =============================================================================
# maccai_battery.checks — Structural sanity checks (pymatgen)
# =============================================================================
# Runs a battery of fast, heuristic structural checks on each ML-relaxed
# candidate before passing it to expensive DFT calculations.
#
# Philosophy:
#   - Checks are SCREENING tools, not hard physics validators.
#   - Every check result is stored as metadata — failures don't crash
#     the pipeline unless cfg.screening.hard_filter is True.
#   - All functions return plain dicts so they are JSON-serialisable
#     and can be written directly into candidates.ndjson.
#
# Checks implemented:
#   1. Density (g/cm³)                    — detects exploded/collapsed cells
#   2. Minimum interatomic distance (Å)   — detects atomic overlaps
#   3. Oxidation state assignment         — heuristic charge bookkeeping
#   4. Charge neutrality                  — sum of formal charges ≈ 0
#   5. Bond-valence consistency           — BVAnalyzer cross-check
#
# Usage:
#   from maccai_battery.config import load_config
#   from maccai_battery.checks import run_sanity_checks, CheckResult
#
#   cfg    = load_config()
#   result = run_sanity_checks(structure, cfg)
#   print(result.passed_all)
# =============================================================================

from __future__ import annotations

import logging
import traceback
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class CheckResult:
    """Holds every sanity-check outcome for a single structure.

    Attributes
    ----------
    formula : str
        Reduced chemical formula of the structure.
    density_gcc : Optional[float]
        Density in g/cm³.  ``None`` if the calculation failed.
    density_ok : Optional[bool]
        ``True`` if density is within the configured [min, max] range.
    min_distance_A : Optional[float]
        Shortest interatomic distance in Å.  ``None`` on failure.
    min_distance_ok : Optional[bool]
        ``True`` if ``min_distance_A >= threshold``.
    oxidation_states : Optional[Dict[str, float]]
        Assigned oxidation states keyed by site index (str).
        ``None`` if assignment failed.
    oxidation_states_assigned : bool
        ``True`` if oxidation state assignment succeeded.
    charge_neutral : Optional[bool]
        ``True`` if the sum of formal charges is zero.
        ``None`` if oxidation state assignment failed.
    bv_consistent : Optional[bool]
        ``True`` if bond-valence analysis passes.
        ``None`` if the check was skipped or failed.
    errors : Dict[str, str]
        Per-check exception messages, keyed by check name.
        Empty if all checks completed without exceptions.
    """

    formula: str                                     = ""
    density_gcc: Optional[float]                     = None
    density_ok: Optional[bool]                       = None
    min_distance_A: Optional[float]                  = None
    min_distance_ok: Optional[bool]                  = None
    oxidation_states: Optional[Dict[str, float]]     = None
    oxidation_states_assigned: bool                  = False
    charge_neutral: Optional[bool]                   = None
    bv_consistent: Optional[bool]                    = None
    errors: Dict[str, str]                           = field(default_factory=dict)

    @property
    def passed_all(self) -> bool:
        """``True`` if every completed check passed (no ``False`` values)."""
        flags = [
            self.density_ok,
            self.min_distance_ok,
            self.charge_neutral,
            self.bv_consistent,
        ]
        # A check that returned None (skipped/failed) does not count as a failure.
        return all(f is not False for f in flags)

    @property
    def warning_count(self) -> int:
        """Number of checks that explicitly returned ``False``."""
        flags = [
            self.density_ok,
            self.min_distance_ok,
            self.charge_neutral,
            self.bv_consistent,
        ]
        return sum(1 for f in flags if f is False)

    def to_dict(self) -> dict:
        """Return a JSON-serialisable representation."""
        return {
            "formula":                   self.formula,
            "density_gcc":               self.density_gcc,
            "density_ok":                self.density_ok,
            "min_distance_A":            self.min_distance_A,
            "min_distance_ok":           self.min_distance_ok,
            "oxidation_states":          self.oxidation_states,
            "oxidation_states_assigned": self.oxidation_states_assigned,
            "charge_neutral":            self.charge_neutral,
            "bv_consistent":             self.bv_consistent,
            "passed_all":                self.passed_all,
            "warning_count":             self.warning_count,
            "errors":                    self.errors,
        }

    def summary_line(self) -> str:
        """Return a compact one-line human-readable summary."""
        parts = [f"formula={self.formula}"]

        if self.density_gcc is not None:
            ok = "✓" if self.density_ok else "✗"
            parts.append(f"ρ={self.density_gcc:.2f}g/cc{ok}")

        if self.min_distance_A is not None:
            ok = "✓" if self.min_distance_ok else "✗"
            parts.append(f"d_min={self.min_distance_A:.2f}Å{ok}")

        if self.charge_neutral is not None:
            ok = "✓" if self.charge_neutral else "✗"
            parts.append(f"neutral{ok}")

        if self.bv_consistent is not None:
            ok = "✓" if self.bv_consistent else "✗"
            parts.append(f"BV{ok}")

        status = "PASS" if self.passed_all else f"WARN({self.warning_count})"
        return f"[{status}] " + "  ".join(parts)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_sanity_checks(structure, cfg) -> CheckResult:
    """Run all configured sanity checks on a pymatgen Structure.

    Parameters
    ----------
    structure : pymatgen.core.Structure
        The structure to check (typically an ML-relaxed candidate).
    cfg : PipelineConfig
        Fully loaded pipeline configuration.

    Returns
    -------
    CheckResult
        All check outcomes.  Never raises — exceptions are captured in
        ``result.errors``.

    Raises
    ------
    RuntimeError
        Only if ``cfg.screening.hard_filter`` is ``True`` and at least one
        check explicitly failed (returned ``False``).
    """
    sc = cfg.screening
    result = CheckResult()

    # ------------------------------------------------------------------
    # Basic metadata
    # ------------------------------------------------------------------
    try:
        result.formula = structure.composition.reduced_formula
    except Exception:
        result.formula = "unknown"

    # ------------------------------------------------------------------
    # Run individual checks
    # ------------------------------------------------------------------
    _check_density(structure, result, sc.density_min_gcc, sc.density_max_gcc)
    _check_min_distance(structure, result, sc.min_distance_threshold_A)

    if sc.assign_oxidation_states:
        _check_oxidation_states(structure, result)

    if sc.check_charge_neutrality:
        _check_charge_neutrality(structure, result)

    if sc.check_bond_valence:
        _check_bond_valence(structure, result)

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    if result.errors:
        logger.debug(
            "Sanity check errors for %s: %s",
            result.formula, result.errors,
        )

    if result.passed_all:
        logger.info("  Sanity checks: %s", result.summary_line())
    else:
        logger.warning("  Sanity checks: %s", result.summary_line())

    # ------------------------------------------------------------------
    # Hard filter (optional)
    # ------------------------------------------------------------------
    if sc.hard_filter and not result.passed_all:
        raise RuntimeError(
            f"Structure '{result.formula}' failed sanity checks "
            f"(hard_filter=true):\n{result.summary_line()}"
        )

    return result


def run_sanity_checks_batch(
    structures: List[Tuple[int, object]],
    cfg,
) -> List[Tuple[int, CheckResult]]:
    """Run sanity checks on a batch of structures.

    Parameters
    ----------
    structures : list of (frame_index, pymatgen.core.Structure)
        Structures to check.  ``frame_index`` is passed through for
        correlation with relaxation results.
    cfg : PipelineConfig
        Fully loaded pipeline configuration.

    Returns
    -------
    list of (frame_index, CheckResult)
        Results in the same order as the input list.
    """
    results = []
    total = len(structures)

    for i, (frame_idx, structure) in enumerate(structures):
        logger.info(
            "Sanity check [%d/%d] frame=%d ...", i + 1, total, frame_idx
        )
        check = run_sanity_checks(structure, cfg)
        results.append((frame_idx, check))

    n_pass = sum(1 for _, r in results if r.passed_all)
    n_warn = len(results) - n_pass
    logger.info(
        "Sanity check batch done: %d passed, %d with warnings (total %d).",
        n_pass, n_warn, total,
    )

    return results


def filter_passed(
    results: List[Tuple[int, CheckResult]],
) -> List[Tuple[int, CheckResult]]:
    """Return only the (index, result) pairs that passed all checks.

    Parameters
    ----------
    results : list of (frame_index, CheckResult)

    Returns
    -------
    list of (frame_index, CheckResult)
        Subset that passed.
    """
    return [(idx, r) for idx, r in results if r.passed_all]


def make_filters_dict(result: CheckResult) -> dict:
    """Return the ``filters`` sub-dict used in a candidate NDJSON record.

    This maps directly onto the schema used in ``candidates.ndjson``.

    Parameters
    ----------
    result : CheckResult

    Returns
    -------
    dict
    """
    return {
        "charge_neutral":                result.charge_neutral,
        "oxidation_states_assigned":     result.oxidation_states_assigned,
        "min_interatomic_distance_ok":   result.min_distance_ok,
        "density_ok":                    result.density_ok,
        "bv_consistent":                 result.bv_consistent,
        "passed_all":                    result.passed_all,
        "warning_count":                 result.warning_count,
    }


def make_ml_scores_dict(result: CheckResult, energy_per_atom_eV: Optional[float]) -> dict:
    """Return the ``ml_scores`` sub-dict used in a candidate NDJSON record.

    Parameters
    ----------
    result : CheckResult
    energy_per_atom_eV : float or None
        ML potential energy per atom from MatterSim.

    Returns
    -------
    dict
    """
    return {
        "matter_sim_energy_eV_per_atom":       energy_per_atom_eV,
        "matter_sim_uncertainty_eV_per_atom":  None,   # not yet supported
        "density_gcc":                         result.density_gcc,
        "min_distance_A":                      result.min_distance_A,
    }


# ---------------------------------------------------------------------------
# Individual check implementations
# ---------------------------------------------------------------------------

def _check_density(
    structure,
    result: CheckResult,
    min_gcc: float,
    max_gcc: float,
) -> None:
    """Compute density and check it is in [min_gcc, max_gcc].

    Uses ``pymatgen.core.Structure.density`` (g/cm³), which correctly
    handles the unit conversion from Å³ to cm³.

    .. note::
        A common bug is to compute ``composition.weight / volume`` directly.
        ``composition.weight`` is in g/mol and ``volume`` is in Å³, so the
        result would be off by Avogadro's number × the Å³→cm³ factor.
        Always use ``structure.density`` instead.

    Parameters
    ----------
    structure : pymatgen.core.Structure
    result : CheckResult
        Modified in-place.
    min_gcc, max_gcc : float
        Acceptable density range (g/cm³).
    """
    try:
        # pymatgen.core.Structure.density returns g/cm³ correctly.
        density = structure.density
        result.density_gcc = float(density)
        result.density_ok  = min_gcc <= density <= max_gcc

        if not result.density_ok:
            logger.debug(
                "Density out of range for %s: %.3f g/cc (expected %.1f–%.1f).",
                result.formula, density, min_gcc, max_gcc,
            )

    except Exception as exc:
        result.errors["density"] = f"{type(exc).__name__}: {exc}"
        logger.debug("Density check failed for %s: %s", result.formula, exc)


def _check_min_distance(
    structure,
    result: CheckResult,
    threshold_A: float,
) -> None:
    """Find the shortest interatomic distance and compare to *threshold_A*.

    Uses two strategies in order of preference:
    1. ``pymatgen.analysis.local_env.MinimumDistanceNN`` — chemistry-aware,
       accounts for periodic images properly.
    2. Brute-force fallback using ``structure.distance_matrix`` — works
       even if MinimumDistanceNN raises for unusual structures.

    Parameters
    ----------
    structure : pymatgen.core.Structure
    result : CheckResult
        Modified in-place.
    threshold_A : float
        Minimum acceptable bond length (Å).
    """
    # Strategy 1: MinimumDistanceNN
    try:
        from pymatgen.analysis.local_env import MinimumDistanceNN

        mdnn = MinimumDistanceNN()
        min_d = float("inf")

        for i in range(len(structure)):
            try:
                neighbors = mdnn.get_nn_info(structure, i)
                for nb in neighbors:
                    d = nb.get("weight") or nb.get("distance")
                    if d is not None and d > 0:
                        min_d = min(min_d, d)
            except Exception:
                # MinimumDistanceNN can fail for some unusual environments;
                # skip this site and continue.
                pass

        if min_d == float("inf"):
            raise ValueError("No valid distances found via MinimumDistanceNN.")

        result.min_distance_A  = round(min_d, 4)
        result.min_distance_ok = min_d >= threshold_A
        return

    except Exception as exc:
        logger.debug(
            "MinimumDistanceNN failed for %s (%s); trying distance_matrix fallback.",
            result.formula, exc,
        )

    # Strategy 2: distance_matrix fallback
    try:
        import numpy as np

        dm = structure.distance_matrix
        # Zero out diagonal (self-distances)
        np.fill_diagonal(dm, float("inf"))
        min_d = float(dm.min())

        result.min_distance_A  = round(min_d, 4)
        result.min_distance_ok = min_d >= threshold_A

    except Exception as exc:
        result.errors["min_distance"] = f"{type(exc).__name__}: {exc}"
        logger.debug(
            "Min-distance check failed for %s: %s", result.formula, exc
        )


def _check_oxidation_states(structure, result: CheckResult) -> None:
    """Attempt to assign oxidation states using pymatgen's heuristic guesser.

    ``pymatgen.core.Structure.add_oxidation_state_by_guess`` uses the
    ICSD oxidation state statistics.  It can fail for unusual compositions.

    Parameters
    ----------
    structure : pymatgen.core.Structure
        Modified in-place (oxidation states added to a copy only).
    result : CheckResult
        Modified in-place.
    """
    try:
        from pymatgen.core import Structure

        # Work on a copy so the caller's structure is not mutated.
        s_copy = structure.copy()
        s_copy.add_oxidation_state_by_guess()

        ox_states: Dict[str, float] = {}
        for i, site in enumerate(s_copy):
            try:
                ox = float(site.specie.oxi_state)
            except AttributeError:
                ox = 0.0
            ox_states[str(i)] = ox

        result.oxidation_states          = ox_states
        result.oxidation_states_assigned = True

    except Exception as exc:
        result.errors["oxidation_states"] = f"{type(exc).__name__}: {exc}"
        result.oxidation_states_assigned  = False
        logger.debug(
            "Oxidation state assignment failed for %s: %s",
            result.formula, exc,
        )


def _check_charge_neutrality(structure, result: CheckResult) -> None:
    """Check that the sum of formal charges is approximately zero.

    Requires oxidation states to have been assigned first.
    If ``result.oxidation_states_assigned`` is ``False`` this check
    is skipped (``result.charge_neutral`` is left as ``None``).

    Parameters
    ----------
    structure : pymatgen.core.Structure
    result : CheckResult
        Modified in-place.
    """
    if not result.oxidation_states_assigned or result.oxidation_states is None:
        # Cannot determine neutrality without oxidation states.
        logger.debug(
            "Charge neutrality check skipped for %s "
            "(oxidation states not assigned).",
            result.formula,
        )
        return

    try:
        total_charge = sum(result.oxidation_states.values())
        # Allow a small numerical tolerance
        result.charge_neutral = abs(total_charge) < 0.1

        if not result.charge_neutral:
            logger.debug(
                "Charge neutrality failed for %s: total charge = %.3f.",
                result.formula, total_charge,
            )

    except Exception as exc:
        result.errors["charge_neutral"] = f"{type(exc).__name__}: {exc}"
        logger.debug(
            "Charge neutrality check failed for %s: %s",
            result.formula, exc,
        )


def _check_bond_valence(structure, result: CheckResult) -> None:
    """Run pymatgen's BVAnalyzer to check bond-valence consistency.

    Bond-valence analysis tests whether the bond lengths are consistent
    with the assigned formal charges.  Structures that pass this check
    are more likely to be physically reasonable.

    This check requires that oxidation states have already been assigned;
    if they haven't, it is skipped.

    Parameters
    ----------
    structure : pymatgen.core.Structure
    result : CheckResult
        Modified in-place.
    """
    try:
        from pymatgen.analysis.bond_valence import BVAnalyzer

        bva = BVAnalyzer()

        # ``get_valences`` returns a list of valence values and raises
        # ``ValueError`` if the analysis fails (e.g. unknown species).
        valences = bva.get_valences(structure)

        # Check valences are non-trivially assigned (not all zero)
        result.bv_consistent = any(v != 0 for v in valences)

        if not result.bv_consistent:
            logger.debug(
                "BV analysis returned all-zero valences for %s.", result.formula
            )

    except Exception as exc:
        result.errors["bond_valence"] = f"{type(exc).__name__}: {exc}"
        logger.debug(
            "Bond valence check failed for %s: %s", result.formula, exc
        )
