# =============================================================================
# maccai_battery.relaxation — MatterSim ML-relaxation interface
# =============================================================================
# Performs fast geometry relaxation on MatterGen-generated structures using
# the MatterSim ML interatomic potential via the ASE interface.
#
# Features:
#   - Typed config integration
#   - Structured logging
#   - Per-structure error isolation (one failure never kills the full run)
#   - Graceful EMT fallback if MatterSim is unavailable
#   - FrechetCellFilter for full cell + atom relaxation
#   - Per-structure EXTXYZ output with energy metadata in comment line
#
# MatterSim must be installed in its own conda environment.
# See README.md → "Environment Setup" for instructions.
#
# Usage:
#   from maccai_battery.config import load_config
#   from maccai_battery.relaxation import relax_structures
#
#   cfg = load_config()
#   records = relax_structures(cfg, atoms_list)
# =============================================================================

from __future__ import annotations

import logging
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class RelaxationResult:
    """Holds the outcome of a single structure relaxation.

    Attributes
    ----------
    frame_index : int
        Zero-based index of the structure in the source EXTXYZ file.
    formula : str
        Chemical formula of the structure (e.g. ``"LiFePO4"``).
    ml_relaxed_path : Optional[Path]
        Path to the saved ML-relaxed EXTXYZ file, or ``None`` on failure.
    energy_eV : Optional[float]
        Total potential energy reported by the calculator (eV).
    energy_per_atom_eV : Optional[float]
        Energy per atom (eV/atom).
    n_atoms : int
        Number of atoms in the structure.
    converged : bool
        Whether the BFGS optimizer reached ``fmax`` within ``max_steps``.
    used_fallback : bool
        ``True`` if the EMT fallback potential was used instead of MatterSim.
    wall_time_s : float
        Wall-clock time for this relaxation (seconds).
    error : Optional[str]
        Exception message if relaxation failed, ``None`` otherwise.
    """

    frame_index: int
    formula: str
    ml_relaxed_path: Optional[Path]        = None
    energy_eV: Optional[float]             = None
    energy_per_atom_eV: Optional[float]    = None
    n_atoms: int                            = 0
    converged: bool                         = False
    used_fallback: bool                     = False
    wall_time_s: float                      = 0.0
    error: Optional[str]                    = None

    @property
    def success(self) -> bool:
        """``True`` if the relaxation completed without a fatal error."""
        return self.error is None and self.ml_relaxed_path is not None

    def to_dict(self) -> dict:
        """Return a JSON-serialisable dictionary representation."""
        return {
            "frame_index":          self.frame_index,
            "formula":              self.formula,
            "ml_relaxed_path":      str(self.ml_relaxed_path) if self.ml_relaxed_path else None,
            "energy_eV":            self.energy_eV,
            "energy_per_atom_eV":   self.energy_per_atom_eV,
            "n_atoms":              self.n_atoms,
            "converged":            self.converged,
            "used_fallback":        self.used_fallback,
            "wall_time_s":          round(self.wall_time_s, 3),
            "error":                self.error,
        }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def relax_structures(
    cfg,                      # PipelineConfig — avoid circular import at module level
    atoms_list: List,
    source_extxyz: Optional[Path] = None,
) -> List[RelaxationResult]:
    """ML-relax a list of ASE Atoms objects using MatterSim (or EMT fallback).

    Each structure is relaxed independently.  Failures are caught per
    structure and stored in the ``error`` field of the result; they do not
    abort the remaining structures.

    Parameters
    ----------
    cfg : PipelineConfig
        Fully loaded and validated pipeline configuration.
    atoms_list : list of ase.Atoms
        Structures to relax (typically from :func:`generation.load_generated_structures`).
    source_extxyz : Path, optional
        Path of the source EXTXYZ file, used only for log messages.

    Returns
    -------
    list of RelaxationResult
        One result per input structure, in the same order.
        Check ``result.success`` before using ``result.ml_relaxed_path``.
    """
    rel_cfg  = cfg.relaxation
    proj_cfg = cfg.project

    # ------------------------------------------------------------------
    # Output directory
    # ------------------------------------------------------------------
    out_dir = proj_cfg.dir_path("ml_relaxed")
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("ML-relaxed structures → %s", out_dir)

    # ------------------------------------------------------------------
    # Determine how many structures to relax
    # ------------------------------------------------------------------
    total = len(atoms_list)
    limit = rel_cfg.max_structures if rel_cfg.max_structures is not None else total
    limit = min(limit, total)

    if limit < total:
        logger.info(
            "Relaxing %d of %d structures (max_structures=%d).",
            limit, total, limit,
        )
    else:
        logger.info("Relaxing all %d structures.", total)

    # ------------------------------------------------------------------
    # Prepare calculator
    # ------------------------------------------------------------------
    calculator, used_fallback = _get_calculator(rel_cfg.device, rel_cfg.emt_fallback)

    if used_fallback:
        logger.warning(
            "MatterSim not available — using ASE EMT fallback potential. "
            "Energies and geometries will be significantly less accurate."
        )

    # ------------------------------------------------------------------
    # Main relaxation loop
    # ------------------------------------------------------------------
    results: List[RelaxationResult] = []

    for idx, atoms in enumerate(atoms_list[:limit]):
        formula = atoms.get_chemical_formula()
        logger.info(
            "  [%d/%d] Relaxing %s (%d atoms) ...",
            idx + 1, limit, formula, len(atoms),
        )

        result = _relax_single(
            atoms       = atoms,
            frame_index = idx,
            formula     = formula,
            calculator  = calculator,
            used_fallback = used_fallback,
            out_dir     = out_dir,
            fmax        = rel_cfg.fmax,
            max_steps   = rel_cfg.max_steps,
            relax_cell  = rel_cfg.relax_cell,
        )

        results.append(result)

        if result.success:
            logger.info(
                "    → %.4f eV/atom | converged=%s | %.1f s",
                result.energy_per_atom_eV,
                result.converged,
                result.wall_time_s,
            )
        else:
            logger.warning(
                "    → FAILED for %s (frame %d): %s",
                formula, idx, result.error,
            )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    n_ok   = sum(1 for r in results if r.success)
    n_fail = len(results) - n_ok
    logger.info(
        "Relaxation complete: %d succeeded, %d failed (out of %d).",
        n_ok, n_fail, len(results),
    )

    return results


def load_ml_relaxed_structure(path: Path):
    """Load a single ML-relaxed structure from an EXTXYZ file.

    Parameters
    ----------
    path : Path
        Path to the ``*_ml_relaxed.extxyz`` file.

    Returns
    -------
    ase.Atoms
        The relaxed structure.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    RuntimeError
        If the file cannot be parsed by ASE.
    """
    from ase import io as ase_io

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"ML-relaxed file not found: {path}")

    try:
        atoms = ase_io.read(str(path))
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load ML-relaxed structure from {path}: {exc}"
        ) from exc

    return atoms


def rank_by_ml_energy(results: List[RelaxationResult]) -> List[RelaxationResult]:
    """Return *results* sorted by ML energy per atom (ascending).

    Failed structures (``result.success == False``) are placed at the end.

    Parameters
    ----------
    results : list of RelaxationResult

    Returns
    -------
    list of RelaxationResult
        Sorted copy (original list is not modified).
    """
    successful  = [r for r in results if r.success and r.energy_per_atom_eV is not None]
    failed      = [r for r in results if not r.success or r.energy_per_atom_eV is None]

    successful.sort(key=lambda r: r.energy_per_atom_eV)  # type: ignore[arg-type]
    return successful + failed


def top_n_candidates(
    results: List[RelaxationResult],
    n: int,
) -> List[RelaxationResult]:
    """Return the *n* most stable (lowest ML energy/atom) successful results.

    Parameters
    ----------
    results : list of RelaxationResult
    n : int
        Number of candidates to return.

    Returns
    -------
    list of RelaxationResult
        At most *n* results, sorted by ascending energy per atom.
    """
    ranked = rank_by_ml_energy(results)
    return ranked[:n]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_calculator(device: str, emt_fallback: bool) -> Tuple[object, bool]:
    """Obtain a calculator for geometry relaxation.

    Tries MatterSim first.  Falls back to ASE EMT if:
    - MatterSim is not installed, AND ``emt_fallback`` is ``True``.

    Parameters
    ----------
    device : str
        Compute device string (``"cpu"``, ``"cuda"``, ``"mps"``).
    emt_fallback : bool
        Allow falling back to the EMT potential.

    Returns
    -------
    (calculator, used_fallback)
        The ASE-compatible calculator object and a boolean flag indicating
        whether the EMT fallback was used.

    Raises
    ------
    ImportError
        If MatterSim is unavailable and ``emt_fallback`` is ``False``.
    """
    # Attempt MatterSim
    try:
        import importlib.util
        if importlib.util.find_spec("mattersim") is None:
            raise ImportError("mattersim module not found")

        from mattersim.forcefield import MatterSimCalculator  # type: ignore[import]

        calc = MatterSimCalculator(device=device)
        logger.debug("Using MatterSimCalculator (device=%s).", device)
        return calc, False

    except Exception as exc:
        if not emt_fallback:
            raise ImportError(
                "MatterSim is not available and emt_fallback is disabled.\n"
                "Either install MatterSim or set relaxation.emt_fallback: true "
                "in config.yaml."
            ) from exc

        logger.warning(
            "MatterSim unavailable (%s). Falling back to ASE EMT.", exc
        )

    # EMT fallback
    try:
        from ase.calculators.emt import EMT  # type: ignore[import]
        return EMT(), True
    except ImportError as exc:
        raise ImportError(
            "Neither MatterSim nor ASE EMT could be imported. "
            "Install ASE with: pip install ase"
        ) from exc


def _relax_single(
    atoms,
    frame_index: int,
    formula: str,
    calculator,
    used_fallback: bool,
    out_dir: Path,
    fmax: float,
    max_steps: int,
    relax_cell: bool,
) -> RelaxationResult:
    """Relax a single structure and save the result.

    Parameters
    ----------
    atoms : ase.Atoms
        Structure to relax (modified in-place by ASE).
    frame_index : int
        Index of this frame in the source EXTXYZ file.
    formula : str
        Chemical formula string (for naming output files).
    calculator : ASE-compatible calculator
        The ML potential or EMT fallback.
    used_fallback : bool
        Whether *calculator* is the EMT fallback.
    out_dir : Path
        Directory to write the relaxed EXTXYZ file.
    fmax : float
        Force convergence threshold (eV/Å).
    max_steps : int
        Maximum number of BFGS steps.
    relax_cell : bool
        If ``True``, use :class:`ase.filters.FrechetCellFilter` to relax
        both atomic positions and the cell; otherwise relax atoms only.

    Returns
    -------
    RelaxationResult
    """
    from ase import io as ase_io
    from ase.optimize import BFGS

    # Build result skeleton
    result = RelaxationResult(
        frame_index   = frame_index,
        formula       = formula,
        n_atoms       = len(atoms),
        used_fallback = used_fallback,
    )

    t0 = time.monotonic()

    try:
        # ------------------------------------------------------------------
        # Prepare atoms
        # ------------------------------------------------------------------
        # Ensure periodic boundary conditions and wrap atoms into the cell.
        atoms.set_pbc(True)
        atoms.wrap()
        atoms.calc = calculator

        # ------------------------------------------------------------------
        # Set up optimiser target
        # ------------------------------------------------------------------
        if relax_cell:
            try:
                from ase.filters import FrechetCellFilter  # type: ignore[import]
                target = FrechetCellFilter(atoms)
                logger.debug("Using FrechetCellFilter (cell + atoms relaxation).")
            except ImportError:
                # Older ASE versions don't have FrechetCellFilter
                from ase.constraints import ExpCellFilter  # type: ignore[import]
                target = ExpCellFilter(atoms)
                logger.debug("Using ExpCellFilter (FrechetCellFilter not available).")
        else:
            target = atoms
            logger.debug("Relaxing atoms only (cell fixed).")

        # ------------------------------------------------------------------
        # Log file per structure
        # ------------------------------------------------------------------
        log_path = out_dir / f"frame{frame_index:04d}_bfgs.log"

        # ------------------------------------------------------------------
        # Run BFGS
        # ------------------------------------------------------------------
        opt = BFGS(target, logfile=str(log_path))
        converged = opt.run(fmax=fmax, steps=max_steps)

        # ------------------------------------------------------------------
        # Extract energies
        # ------------------------------------------------------------------
        e_tot = atoms.get_potential_energy()
        e_pa  = e_tot / len(atoms)

        result.energy_eV          = float(e_tot)
        result.energy_per_atom_eV = float(e_pa)
        result.converged          = bool(converged)

        # ------------------------------------------------------------------
        # Save relaxed structure
        # ------------------------------------------------------------------
        out_path = out_dir / f"generated_crystals_frame{frame_index}_ml_relaxed.extxyz"
        _write_extxyz_with_energy(atoms, e_tot, out_path)
        result.ml_relaxed_path = out_path

    except Exception as exc:
        result.error = f"{type(exc).__name__}: {exc}"
        logger.debug(
            "Relaxation failed for frame %d (%s):\n%s",
            frame_index, formula, traceback.format_exc(),
        )

    finally:
        result.wall_time_s = time.monotonic() - t0

    return result


def _write_extxyz_with_energy(atoms, energy_eV: float, path: Path) -> None:
    """Write an ASE Atoms object to an EXTXYZ file with energy in the comment line.

    The energy is stored in a format that downstream tools (including
    :func:`generation.summarise_generated`) can parse:

    ``Lattice="..." Properties=... energy=-123.456 ...``

    Parameters
    ----------
    atoms : ase.Atoms
        The relaxed structure to save.
    energy_eV : float
        Total potential energy (eV) to embed in the comment line.
    path : Path
        Destination file path.
    """
    from ase import io as ase_io

    # MatterSim may have already written results into atoms.info;
    # remove them to avoid conflict with ASE's extxyz writer.
    for key in ["energy", "free_energy", "stress"]:
        atoms.info.pop(key, None)
    for key in ["forces"]:
        atoms.arrays.pop(key, None)

    # ASE stores extra info in atoms.info dict; it writes these to the
    # EXTXYZ comment line automatically.
    atoms.info["energy"] = energy_eV
    atoms.info["ml_energy_eV_per_atom"] = energy_eV / len(atoms)

    ase_io.write(str(path), atoms, format="extxyz")
    logger.debug("Saved relaxed structure: %s", path.name)
