# =============================================================================
# maccai_battery.dft.workflow — Full DFT screening + relaxation orchestrator
# =============================================================================
# This module implements the two-stage DFT workflow used in the MACCAI Battery
# pipeline:
#
#   Stage 1 — SCF Screening (run_scf_screening)
#   ─────────────────────────────────────────────
#   • Select the top dft_screening.n_candidates ML-ranked structures from the
#     candidate database (by ML energy per atom).
#   • For each candidate: load the ml_relaxed extxyz, convert to pymatgen
#     Structure, generate a QE SCF input, run pw.x, parse the output, and
#     update the candidate record in the database.
#   • Returns a List[SCFResult] sorted in the original selection order.
#
#   Stage 2 — Ionic Relaxation (run_dft_relax)
#   ─────────────────────────────────────────────
#   • Select the top dft_relax.n_candidates structures by DFT SCF energy
#     (from Stage 1 results or directly from the database).
#   • For each candidate: generate a QE relax input, run pw.x, parse the
#     output, and update the candidate record.
#   • Returns a List[RelaxResult].
#
#   Full pipeline (run)
#   ─────────────────────────────────────────────
#   • Runs Stage 1 then Stage 2 and returns (scf_results, relax_results).
#
# Error handling
# ──────────────
# Per-structure failures (bad structure file, QE crash, missing pseudopotential,
# etc.) are caught and stored in the corresponding result's ``error`` field.
# They do NOT abort the loop — all other candidates are processed normally.
# This is intentional: a single failed structure should not stop the pipeline
# from evaluating the remaining candidates.
#
# Usage
# ─────
#   from maccai_battery.config import load_config
#   from maccai_battery.dft import DFTWorkflow
#
#   cfg = load_config()
#   wf  = DFTWorkflow(cfg)
#
#   # Full pipeline
#   scf_results, relax_results = wf.run()
#
#   # Step-by-step (e.g. if you want to inspect SCF results first)
#   scf_results   = wf.run_scf_screening()
#   relax_results = wf.run_dft_relax(scf_results)
# =============================================================================

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from maccai_battery.config import PipelineConfig
    from maccai_battery.database import CandidateDatabase

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class SCFResult:
    """Outcome of a single QE SCF screening calculation.

    Attributes
    ----------
    candidate_id : str
        Database ID of the candidate, e.g. ``"MG-abc12345"``.
    formula : str
        Reduced chemical formula, e.g. ``"LiFePO4"``.
    run_dir : Path
        Absolute path to the directory where ``qe.in`` and ``qe.out`` live.
    energy_eV : float or None
        Total DFT SCF energy in eV.  ``None`` if parsing failed or QE crashed.
    energy_eV_per_atom : float or None
        Total DFT SCF energy normalised by atom count (eV/atom).
    n_atoms : int
        Number of atoms in the cell (0 if unknown).
    converged : bool
        ``True`` if QE reported ``"convergence has been achieved"``.
    wall_time_s : float
        Elapsed wall time of the pw.x process in seconds.
    error : str or None
        Human-readable description of what went wrong, or ``None`` on success.
    """

    candidate_id: str
    formula: str
    run_dir: Path
    energy_eV: Optional[float]
    energy_eV_per_atom: Optional[float]
    n_atoms: int
    converged: bool
    wall_time_s: float
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        """``True`` when the calculation produced a usable energy.

        A result is considered successful when:
        * No error was recorded during input generation, QE execution, or
          output parsing.
        * A total energy was successfully parsed from the output.

        Note that SCF non-convergence does **not** set ``success=False`` on
        its own — a non-converged energy may still be useful for screening.
        Check :attr:`converged` separately if strict convergence is required.
        """
        return self.error is None and self.energy_eV is not None

    def to_dict(self) -> dict:
        """Serialise to a JSON-compatible dictionary.

        Returns
        -------
        dict
            All attributes, with ``run_dir`` converted to string and
            ``success`` included as a derived key.
        """
        return {
            "candidate_id":       self.candidate_id,
            "formula":            self.formula,
            "run_dir":            str(self.run_dir),
            "energy_eV":          self.energy_eV,
            "energy_eV_per_atom": self.energy_eV_per_atom,
            "n_atoms":            self.n_atoms,
            "converged":          self.converged,
            "wall_time_s":        self.wall_time_s,
            "error":              self.error,
            "success":            self.success,
        }


@dataclass
class RelaxResult:
    """Outcome of a single QE ionic-relaxation calculation.

    Attributes
    ----------
    candidate_id : str
        Database ID of the candidate, e.g. ``"MG-abc12345"``.
    formula : str
        Reduced chemical formula, e.g. ``"LiFePO4"``.
    run_dir : Path
        Absolute path to the directory where ``qe.in`` and ``qe.out`` live.
    energy_eV : float or None
        Final DFT total energy after ionic relaxation, in eV.  This is the
        energy of the last converged ionic step.
    energy_eV_per_atom : float or None
        Final DFT energy normalised by atom count (eV/atom).
    n_atoms : int
        Number of atoms in the cell (0 if unknown).
    wall_time_s : float
        Elapsed wall time of the pw.x process in seconds.
    error : str or None
        Human-readable description of what went wrong, or ``None`` on success.
    """

    candidate_id: str
    formula: str
    run_dir: Path
    energy_eV: Optional[float]
    energy_eV_per_atom: Optional[float]
    n_atoms: int
    wall_time_s: float
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        """``True`` when the calculation produced a usable final energy.

        A result is considered successful when no error was recorded and a
        total energy was parsed from the output.  The ionic relaxation need
        not have fully converged (e.g. it may have hit ``nstep``) for the
        result to be considered successful.
        """
        return self.error is None and self.energy_eV is not None

    def to_dict(self) -> dict:
        """Serialise to a JSON-compatible dictionary.

        Returns
        -------
        dict
        """
        return {
            "candidate_id":       self.candidate_id,
            "formula":            self.formula,
            "run_dir":            str(self.run_dir),
            "energy_eV":          self.energy_eV,
            "energy_eV_per_atom": self.energy_eV_per_atom,
            "n_atoms":            self.n_atoms,
            "wall_time_s":        self.wall_time_s,
            "error":              self.error,
            "success":            self.success,
        }


# ---------------------------------------------------------------------------
# Workflow class
# ---------------------------------------------------------------------------


class DFTWorkflow:
    """Orchestrates the full DFT screening and relaxation workflow.

    This class ties together input generation (:mod:`~maccai_battery.dft.input_generator`),
    job execution (:mod:`~maccai_battery.dft.runner`), output parsing
    (:mod:`~maccai_battery.dft.parser`), and database updates
    (:class:`~maccai_battery.database.CandidateDatabase`) into two high-level
    methods.

    The workflow is designed to be *fault-tolerant at the per-structure level*:
    any exception during structure loading, input generation, QE execution, or
    output parsing is caught and stored in the corresponding result's ``error``
    field.  Processing continues for all remaining candidates.

    Parameters
    ----------
    cfg : PipelineConfig
        Fully loaded and validated pipeline configuration.  Controls all DFT
        parameters (k-points, cutoffs, pseudopotentials, nproc, etc.) and
        directory paths.

    Examples
    --------
    Run the complete pipeline in one call:

    >>> from maccai_battery.config import load_config
    >>> from maccai_battery.dft import DFTWorkflow
    >>> cfg = load_config()
    >>> wf  = DFTWorkflow(cfg)
    >>> scf_results, relax_results = wf.run()

    Run stages independently (e.g. to inspect SCF results before relaxing):

    >>> scf_results   = wf.run_scf_screening()
    >>> relax_results = wf.run_dft_relax(scf_results)
    """

    def __init__(self, cfg: "PipelineConfig") -> None:
        self.cfg = cfg
        self._db: Optional["CandidateDatabase"] = None

    # ------------------------------------------------------------------
    # Lazy database property
    # ------------------------------------------------------------------

    @property
    def db(self) -> "CandidateDatabase":
        """Lazily initialised :class:`~maccai_battery.database.CandidateDatabase`.

        The database is opened on first access and cached for the lifetime
        of this :class:`DFTWorkflow` instance.
        """
        if self._db is None:
            from maccai_battery.database import CandidateDatabase

            self._db = CandidateDatabase(self.cfg)
        return self._db

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_structure(extxyz_path: Path):
        """Load an extxyz file and return a pymatgen Structure.

        Parameters
        ----------
        extxyz_path : Path
            Path to the ML-relaxed extxyz file.

        Returns
        -------
        pymatgen.core.Structure

        Raises
        ------
        ImportError
            If ase or pymatgen is not installed.
        FileNotFoundError
            If the extxyz file does not exist.
        RuntimeError
            If the structure cannot be parsed.
        """
        from ase import io as ase_io  # type: ignore[import]
        from pymatgen.io.ase import AseAtomsAdaptor  # type: ignore[import]

        extxyz_path = Path(extxyz_path)
        if not extxyz_path.exists():
            raise FileNotFoundError(
                f"ML-relaxed extxyz file not found: {extxyz_path}\n"
                "Run the ML relaxation step (step 2) before DFT."
            )

        atoms = ase_io.read(str(extxyz_path))
        structure = AseAtomsAdaptor.get_structure(atoms)
        return structure

    def _resolve_ml_relaxed_path(self, record: Dict) -> Path:
        """Resolve the ml_relaxed extxyz path for a database record.

        The ``structure_files.ml_relaxed`` field stores a path that is
        *relative to the project output directory*.  This method converts
        it to an absolute Path.

        Parameters
        ----------
        record : dict
            A candidate database record.

        Returns
        -------
        Path
            Absolute path to the ml_relaxed extxyz file.

        Raises
        ------
        KeyError
            If the record does not have a ``structure_files.ml_relaxed`` entry.
        """
        rel = record.get("structure_files", {}).get("ml_relaxed")
        if rel is None:
            raise KeyError(
                f"Candidate {record.get('id', '?')} has no 'ml_relaxed' "
                "structure file in the database record. "
                "Run the ML relaxation step before DFT."
            )
        return self.cfg.project.output_path / rel

    @staticmethod
    def _make_run_dir_str(run_dir: Path, output_path: Path) -> str:
        """Return a portable (relative-if-possible) string for the run_dir.

        Storing a path relative to the project output directory keeps the
        database portable if the project is moved to a different machine.

        Parameters
        ----------
        run_dir : Path
            Absolute path to the QE run directory.
        output_path : Path
            Project output root (used as the anchor for relative conversion).

        Returns
        -------
        str
        """
        from maccai_battery.utils import relative_path

        return str(relative_path(run_dir, output_path))

    # ------------------------------------------------------------------
    # Stage 1 — SCF screening
    # ------------------------------------------------------------------

    def run_scf_screening(self) -> List[SCFResult]:
        """Run QE SCF on the top ML-ranked candidates and update the database.

        Selects ``cfg.dft_screening.n_candidates`` candidates from the
        database ordered by ascending ML energy per atom, generates a QE SCF
        input for each, runs ``pw.x``, parses the total energy, and writes
        the result back to the candidate database.

        Per-structure failures are caught and recorded in the corresponding
        :class:`SCFResult` without aborting the remaining calculations.

        Returns
        -------
        list of SCFResult
            One entry per selected candidate, in selection order (lowest ML
            energy first).  Inspect ``result.success`` and ``result.error``
            to identify failed calculations.

        Notes
        -----
        * Run directories are created at
          ``<output>/dft/scf/scf_0/``, ``scf_1/``, … in the same order as
          the ML-energy ranking.
        * The ``qe.in`` input file and ``qe.out`` output file are written
          inside each run directory.
        * The database is updated after **each** individual calculation so
          that partial results are preserved even if the pipeline is
          interrupted.
        """
        from maccai_battery.dft.input_generator import make_scf_input, write_qe_input
        from maccai_battery.dft.parser import parse_scf_stdout
        from maccai_battery.dft.runner import run_qe_pw
        from maccai_battery.utils import ensure_dir

        scr = self.cfg.dft_screening
        n = scr.n_candidates

        logger.info("=" * 60)
        logger.info("DFT SCF SCREENING — selecting top %d ML candidates", n)
        logger.info("=" * 60)

        candidates = self.db.top_n_by_ml_energy(n, passed_checks_only=False)
        if not candidates:
            logger.warning(
                "No candidates found in the database. "
                "Run step 3 (sanity checks) before DFT."
            )
            return []

        total = len(candidates)
        if total < n:
            logger.warning(
                "Requested %d SCF candidates but only %d found in database.",
                n,
                total,
            )

        scf_base_dir = self.cfg.project.dir_path("dft_scf")
        ensure_dir(scf_base_dir)

        results: List[SCFResult] = []

        for idx, record in enumerate(candidates):
            cid = record.get("id", f"unknown_{idx}")
            formula = record.get("formula", "?")
            step_label = f"[SCF {idx + 1}/{total}]"

            logger.info(
                "%s  id=%-20s  formula=%s",
                step_label,
                cid,
                formula,
            )

            run_dir = scf_base_dir / f"scf_{idx}"
            ensure_dir(run_dir)

            # -------- 1. Load structure -----------------------------------
            try:
                ml_path = self._resolve_ml_relaxed_path(record)
                structure = self._load_structure(ml_path)
                logger.debug(
                    "%s  Loaded structure from %s (%d atoms)",
                    step_label,
                    ml_path.name,
                    len(structure),
                )
            except Exception as exc:
                msg = f"Structure load failed: {exc}"
                logger.error("%s  %s", step_label, msg)
                result = SCFResult(
                    candidate_id=cid,
                    formula=formula,
                    run_dir=run_dir,
                    energy_eV=None,
                    energy_eV_per_atom=None,
                    n_atoms=0,
                    converged=False,
                    wall_time_s=0.0,
                    error=msg,
                )
                results.append(result)
                self._update_scf_db(cid, result, "failed")
                continue

            # -------- 2. Generate QE SCF input ----------------------------
            try:
                pw_in = make_scf_input(structure, self.cfg)
                write_qe_input(pw_in, run_dir / "qe.in")
                logger.debug("%s  Wrote QE SCF input → %s/qe.in", step_label, run_dir)
            except Exception as exc:
                msg = f"Input generation failed: {exc}"
                logger.error("%s  %s", step_label, msg)
                result = SCFResult(
                    candidate_id=cid,
                    formula=formula,
                    run_dir=run_dir,
                    energy_eV=None,
                    energy_eV_per_atom=None,
                    n_atoms=0,
                    converged=False,
                    wall_time_s=0.0,
                    error=msg,
                )
                results.append(result)
                self._update_scf_db(cid, result, "failed")
                continue

            # -------- 3. Run QE -------------------------------------------
            try:
                qe_result = run_qe_pw(
                    run_dir=run_dir,
                    nproc=scr.nproc,
                )
            except Exception as exc:
                msg = f"QE launch failed: {exc}"
                logger.error("%s  %s", step_label, msg)
                result = SCFResult(
                    candidate_id=cid,
                    formula=formula,
                    run_dir=run_dir,
                    energy_eV=None,
                    energy_eV_per_atom=None,
                    n_atoms=0,
                    converged=False,
                    wall_time_s=0.0,
                    error=msg,
                )
                results.append(result)
                self._update_scf_db(cid, result, "failed")
                continue

            # -------- 4. Parse output ------------------------------------
            try:
                summary = parse_scf_stdout(qe_result.stdout)
            except Exception as exc:
                # Parsing failure is non-fatal — we still have the raw output
                logger.warning(
                    "%s  Output parsing raised an exception: %s — "
                    "energies will be None.",
                    step_label,
                    exc,
                )
                summary = {
                    "total_energy_Ry":    None,
                    "total_energy_eV":    None,
                    "n_atoms":            None,
                    "energy_per_atom_eV": None,
                    "scf_converged":      False,
                    "fermi_energy_Ry":    None,
                }

            energy_eV = summary.get("total_energy_eV")
            n_atoms = summary.get("n_atoms") or 0
            converged = summary.get("scf_converged", False)
            energy_per_atom = summary.get("energy_per_atom_eV")

            # Combine QE process error with convergence info for error field
            error_field: Optional[str] = None
            if not qe_result.success:
                error_field = qe_result.error
            elif energy_eV is None:
                error_field = (
                    "Energy not found in QE output — "
                    "SCF may not have run or output is truncated."
                )

            result = SCFResult(
                candidate_id=cid,
                formula=formula,
                run_dir=run_dir,
                energy_eV=energy_eV,
                energy_eV_per_atom=energy_per_atom,
                n_atoms=n_atoms,
                converged=converged,
                wall_time_s=qe_result.wall_time_s,
                error=error_field,
            )
            results.append(result)

            db_status = "done" if qe_result.success else "failed"
            self._update_scf_db(cid, result, db_status)

            # -------- 5. Log summary -------------------------------------
            if energy_per_atom is not None:
                logger.info(
                    "%s  E = %+.6f eV/atom  converged=%s  t=%.1fs",
                    step_label,
                    energy_per_atom,
                    converged,
                    qe_result.wall_time_s,
                )
            else:
                logger.warning(
                    "%s  Energy not parsed (converged=%s  t=%.1fs  error=%s)",
                    step_label,
                    converged,
                    qe_result.wall_time_s,
                    error_field,
                )

        # Summary
        n_ok = sum(1 for r in results if r.success)
        n_conv = sum(1 for r in results if r.converged)
        logger.info(
            "SCF screening complete: %d/%d succeeded, %d/%d converged.",
            n_ok,
            total,
            n_conv,
            total,
        )
        if n_ok > 0:
            best = min(
                (r for r in results if r.energy_eV_per_atom is not None),
                key=lambda r: r.energy_eV_per_atom,
                default=None,
            )
            if best:
                logger.info(
                    "Best SCF candidate: %s  E = %+.6f eV/atom  (%s)",
                    best.formula,
                    best.energy_eV_per_atom,
                    best.candidate_id,
                )

        return results

    # ------------------------------------------------------------------
    # Stage 2 — Ionic relaxation
    # ------------------------------------------------------------------

    def run_dft_relax(
        self,
        scf_results: Optional[List[SCFResult]] = None,
    ) -> List[RelaxResult]:
        """Run QE ionic relaxation on the top SCF-ranked candidates.

        Parameters
        ----------
        scf_results : list of SCFResult or None, optional
            If provided, candidates are selected from this list by sorting on
            ``energy_eV_per_atom`` (ascending) and taking the top
            ``cfg.dft_relax.n_candidates`` successful entries.  If ``None``,
            the top candidates are queried directly from the database using
            :meth:`~maccai_battery.database.CandidateDatabase.top_n_by_dft_scf_energy`.

        Returns
        -------
        list of RelaxResult
            One entry per selected candidate, in selection order (lowest SCF
            energy first).

        Notes
        -----
        * Run directories are created at
          ``<output>/dft/relax/relax_0/``, ``relax_1/``, …
        * The database is updated after each individual calculation.
        * Even if a structure's SCF calculation did not fully converge, it
          may still be selected for relaxation if it has the lowest available
          SCF energy.
        """
        from maccai_battery.dft.input_generator import make_relax_input, write_qe_input
        from maccai_battery.dft.parser import parse_relax_stdout
        from maccai_battery.dft.runner import run_qe_pw
        from maccai_battery.utils import ensure_dir

        relax_cfg = self.cfg.dft_relax
        n = relax_cfg.n_candidates

        logger.info("=" * 60)
        logger.info("DFT IONIC RELAXATION — selecting top %d SCF candidates", n)
        logger.info("=" * 60)

        # -------- Candidate selection ------------------------------------
        if scf_results is not None:
            # Select from the provided SCF results
            completed = [
                r for r in scf_results
                if r.success and r.energy_eV_per_atom is not None
            ]
            completed.sort(key=lambda r: r.energy_eV_per_atom)  # type: ignore[arg-type]
            top_ids = [r.candidate_id for r in completed[:n]]
            candidates = []
            for cid in top_ids:
                rec = self.db.get_by_id(cid)
                if rec is not None:
                    candidates.append(rec)
                else:
                    logger.warning(
                        "Candidate %s from SCF results not found in database "
                        "— skipping relax.",
                        cid,
                    )
            logger.info(
                "Selected %d candidates from SCF results (requested %d).",
                len(candidates),
                n,
            )
        else:
            candidates = self.db.top_n_by_dft_scf_energy(n)
            logger.info(
                "Selected %d candidates from database SCF energy ranking "
                "(requested %d).",
                len(candidates),
                n,
            )

        if not candidates:
            logger.warning(
                "No candidates available for relaxation. "
                "Ensure SCF screening has been run first."
            )
            return []

        total = len(candidates)
        relax_base_dir = self.cfg.project.dir_path("dft_relax")
        ensure_dir(relax_base_dir)

        results: List[RelaxResult] = []

        for idx, record in enumerate(candidates):
            cid = record.get("id", f"unknown_{idx}")
            formula = record.get("formula", "?")
            step_label = f"[RELAX {idx + 1}/{total}]"

            # Log SCF energy if available for context
            scf_e = (
                record.get("dft_jobs", {})
                .get("scf", {})
                .get("energy_eV_per_atom")
            )
            logger.info(
                "%s  id=%-20s  formula=%s  SCF=%.6f eV/atom",
                step_label,
                cid,
                formula,
                scf_e if scf_e is not None else float("nan"),
            )

            run_dir = relax_base_dir / f"relax_{idx}"
            ensure_dir(run_dir)

            # -------- 1. Load structure -----------------------------------
            try:
                ml_path = self._resolve_ml_relaxed_path(record)
                structure = self._load_structure(ml_path)
                logger.debug(
                    "%s  Loaded structure from %s (%d atoms)",
                    step_label,
                    ml_path.name,
                    len(structure),
                )
            except Exception as exc:
                msg = f"Structure load failed: {exc}"
                logger.error("%s  %s", step_label, msg)
                result = RelaxResult(
                    candidate_id=cid,
                    formula=formula,
                    run_dir=run_dir,
                    energy_eV=None,
                    energy_eV_per_atom=None,
                    n_atoms=0,
                    wall_time_s=0.0,
                    error=msg,
                )
                results.append(result)
                self._update_relax_db(cid, result, "failed")
                continue

            # -------- 2. Generate QE relax input -------------------------
            try:
                pw_in = make_relax_input(structure, self.cfg, run_dir)
                write_qe_input(pw_in, run_dir / "qe.in")
                logger.debug(
                    "%s  Wrote QE relax input → %s/qe.in", step_label, run_dir
                )
            except Exception as exc:
                msg = f"Input generation failed: {exc}"
                logger.error("%s  %s", step_label, msg)
                result = RelaxResult(
                    candidate_id=cid,
                    formula=formula,
                    run_dir=run_dir,
                    energy_eV=None,
                    energy_eV_per_atom=None,
                    n_atoms=0,
                    wall_time_s=0.0,
                    error=msg,
                )
                results.append(result)
                self._update_relax_db(cid, result, "failed")
                continue

            # -------- 3. Run QE -------------------------------------------
            try:
                qe_result = run_qe_pw(
                    run_dir=run_dir,
                    nproc=relax_cfg.nproc,
                )
            except Exception as exc:
                msg = f"QE launch failed: {exc}"
                logger.error("%s  %s", step_label, msg)
                result = RelaxResult(
                    candidate_id=cid,
                    formula=formula,
                    run_dir=run_dir,
                    energy_eV=None,
                    energy_eV_per_atom=None,
                    n_atoms=0,
                    wall_time_s=0.0,
                    error=msg,
                )
                results.append(result)
                self._update_relax_db(cid, result, "failed")
                continue

            # -------- 4. Parse output ------------------------------------
            try:
                summary = parse_relax_stdout(qe_result.stdout)
            except Exception as exc:
                logger.warning(
                    "%s  Output parsing raised an exception: %s — "
                    "energies will be None.",
                    step_label,
                    exc,
                )
                summary = {
                    "total_energy_Ry":    None,
                    "total_energy_eV":    None,
                    "n_atoms":            None,
                    "energy_per_atom_eV": None,
                    "scf_converged":      False,
                    "fermi_energy_Ry":    None,
                    "n_ionic_steps":      0,
                }

            energy_eV = summary.get("total_energy_eV")
            n_atoms = summary.get("n_atoms") or 0
            energy_per_atom = summary.get("energy_per_atom_eV")
            n_ionic_steps = summary.get("n_ionic_steps", 0)

            error_field: Optional[str] = None
            if not qe_result.success:
                error_field = qe_result.error
            elif energy_eV is None:
                error_field = (
                    "Energy not found in QE relax output — "
                    "relaxation may not have run or output is truncated."
                )

            result = RelaxResult(
                candidate_id=cid,
                formula=formula,
                run_dir=run_dir,
                energy_eV=energy_eV,
                energy_eV_per_atom=energy_per_atom,
                n_atoms=n_atoms,
                wall_time_s=qe_result.wall_time_s,
                error=error_field,
            )
            results.append(result)

            db_status = "done" if qe_result.success else "failed"
            self._update_relax_db(cid, result, db_status)

            # -------- 5. Log summary -------------------------------------
            if energy_per_atom is not None:
                logger.info(
                    "%s  E = %+.6f eV/atom  ionic_steps=%d  t=%.1fs",
                    step_label,
                    energy_per_atom,
                    n_ionic_steps,
                    qe_result.wall_time_s,
                )
            else:
                logger.warning(
                    "%s  Energy not parsed  (ionic_steps=%d  t=%.1fs  error=%s)",
                    step_label,
                    n_ionic_steps,
                    qe_result.wall_time_s,
                    error_field,
                )

        # Summary
        n_ok = sum(1 for r in results if r.success)
        logger.info(
            "Ionic relaxation complete: %d/%d succeeded.", n_ok, total
        )
        if n_ok > 0:
            best = min(
                (r for r in results if r.energy_eV_per_atom is not None),
                key=lambda r: r.energy_eV_per_atom,  # type: ignore[arg-type]
                default=None,
            )
            if best:
                logger.info(
                    "Best relaxed candidate: %s  E = %+.6f eV/atom  (%s)",
                    best.formula,
                    best.energy_eV_per_atom,
                    best.candidate_id,
                )

        return results

    # ------------------------------------------------------------------
    # Full pipeline entry-point
    # ------------------------------------------------------------------

    def run(self) -> Tuple[List[SCFResult], List[RelaxResult]]:
        """Run the complete DFT workflow: SCF screening followed by relaxation.

        This is a convenience wrapper that calls :meth:`run_scf_screening`
        and passes its results to :meth:`run_dft_relax`.

        Returns
        -------
        tuple of (list[SCFResult], list[RelaxResult])
            Both lists in candidate-selection order.

        Notes
        -----
        If SCF screening produced no successful results (e.g. all QE jobs
        failed), :meth:`run_dft_relax` falls back to the database to look
        for any pre-existing SCF energies before giving up.
        """
        logger.info("Starting full DFT workflow (SCF screening + ionic relaxation)")
        scf_results = self.run_scf_screening()
        relax_results = self.run_dft_relax(scf_results)
        logger.info(
            "DFT workflow complete: %d SCF + %d relax calculations.",
            len(scf_results),
            len(relax_results),
        )
        return scf_results, relax_results

    # ------------------------------------------------------------------
    # Private database update helpers
    # ------------------------------------------------------------------

    def _update_scf_db(
        self,
        candidate_id: str,
        result: SCFResult,
        status: str,
    ) -> None:
        """Write SCF result to the candidate database (best-effort).

        Failures are logged as warnings but do not raise so that the main
        loop is not interrupted.

        Parameters
        ----------
        candidate_id : str
        result : SCFResult
        status : str
            ``"done"`` or ``"failed"``.
        """
        try:
            run_dir_str = self._make_run_dir_str(
                result.run_dir, self.cfg.project.output_path
            )
            self.db.update_dft_scf(
                candidate_id,
                energy_eV=result.energy_eV,
                n_atoms=result.n_atoms if result.n_atoms > 0 else None,
                run_dir=run_dir_str,
                status=status,
            )
        except Exception as exc:  # pragma: no cover
            logger.warning(
                "Failed to update SCF result in database for %s: %s",
                candidate_id,
                exc,
            )

    def _update_relax_db(
        self,
        candidate_id: str,
        result: RelaxResult,
        status: str,
    ) -> None:
        """Write relax result to the candidate database (best-effort).

        Parameters
        ----------
        candidate_id : str
        result : RelaxResult
        status : str
            ``"done"`` or ``"failed"``.
        """
        try:
            run_dir_str = self._make_run_dir_str(
                result.run_dir, self.cfg.project.output_path
            )
            self.db.update_dft_relax(
                candidate_id,
                energy_eV=result.energy_eV,
                n_atoms=result.n_atoms if result.n_atoms > 0 else None,
                run_dir=run_dir_str,
                status=status,
            )
        except Exception as exc:  # pragma: no cover
            logger.warning(
                "Failed to update relax result in database for %s: %s",
                candidate_id,
                exc,
            )
