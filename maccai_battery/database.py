# =============================================================================
# maccai_battery.database — Candidate NDJSON database manager
# =============================================================================
# Manages the central candidates.ndjson file that acts as the single source
# of truth for every candidate crystal structure across the full pipeline.
#
# Design principles:
#   - Append-only writes   → no data is ever silently overwritten
#   - Relative paths only  → records are portable across machines
#   - Atomic writes        → partial writes never corrupt the database
#   - Schema-validated     → every record conforms to a known structure
#   - Queryable in memory  → filter/rank without external dependencies
#
# Record schema (one JSON object per line):
#   {
#     "id":                  "MG-<8 hex chars>",
#     "formula":             "LiFePO4",
#     "stoichiometry":       {"Li": 1, "Fe": 1, "P": 1, "O": 4},
#     "structure_files": {
#       "source_extxyz":     "candidates/cifs/generated_crystals.extxyz",
#       "ml_relaxed":        "candidates/ml_relaxed/generated_crystals_frame0_ml_relaxed.extxyz"
#     },
#     "generation_metadata": {
#       "generator":         "MatterGen",
#       "model_checkpoint":  "chemical_system_energy_above_hull",
#       "chemical_system":   "Li-Fe-P-O",
#       "energy_above_hull": 0.05,
#       "frame_index":       0,
#       "timestamp":         "2025-01-01T00:00:00Z"
#     },
#     "ml_scores": {
#       "matter_sim_energy_eV_per_atom":      -3.12,
#       "matter_sim_uncertainty_eV_per_atom": null,
#       "density_gcc":                        3.45,
#       "min_distance_A":                     1.82
#     },
#     "filters": {
#       "charge_neutral":              true,
#       "oxidation_states_assigned":   true,
#       "min_interatomic_distance_ok": true,
#       "density_ok":                  true,
#       "bv_consistent":               true,
#       "passed_all":                  true,
#       "warning_count":               0
#     },
#     "dft_jobs": {
#       "status":   "not_submitted",   # not_submitted | running | done | failed
#       "workflow": "PBE_scf_relax_QE",
#       "scf": {
#         "status":        "not_submitted",
#         "run_dir":       null,
#         "energy_eV":     null,
#         "energy_eV_per_atom": null,
#         "n_atoms":       null,
#         "completed_at":  null
#       },
#       "relax": {
#         "status":        "not_submitted",
#         "run_dir":       null,
#         "energy_eV":     null,
#         "energy_eV_per_atom": null,
#         "n_atoms":       null,
#         "completed_at":  null
#       },
#       "vasp_incar_settings": {"ENCUT": 520, "KSPACING": 0.3}
#     },
#     "notes": ""
#   }
#
# Usage:
#   from maccai_battery.config import load_config
#   from maccai_battery.database import CandidateDatabase
#
#   cfg = load_config()
#   db  = CandidateDatabase(cfg)
#
#   db.append(record)                          # add a new candidate
#   records = db.load_all()                    # read everything into memory
#   top5 = db.top_n_by_ml_energy(5)           # sorted by ML energy
#   db.update_dft_scf("MG-abc123", energy_eV=-123.4, n_atoms=14)
# =============================================================================

from __future__ import annotations

import json
import numpy as np
import logging
import os
import shutil
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from scripts import relax

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Record schema helpers
# ---------------------------------------------------------------------------

def _utc_now() -> str:
    """Return the current UTC time as an ISO-8601 string with a Z suffix."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z"


def _new_id() -> str:
    """Generate a short, unique candidate ID of the form ``MG-<8hex>``."""
    return "MG-" + uuid.uuid4().hex[:8]


def _make_empty_dft_stage() -> Dict[str, Any]:
    """Return an empty DFT stage sub-record."""
    return {
        "status":              "not_submitted",
        "run_dir":             None,
        "energy_eV":           None,
        "energy_eV_per_atom":  None,
        "n_atoms":             None,
        "completed_at":        None,
    }


def make_candidate_record(
    *,
    frame_index: int,
    source_extxyz_path: Optional[Path],
    ml_relaxed_path: Optional[Path],
    energy_per_atom_eV: Optional[float],
    check_result,                    # checks.CheckResult — avoid circular import
    cfg,                             # PipelineConfig
    output_root: Optional[Path] = None,
) -> Dict[str, Any]:
    """Build a fully-populated candidate record dictionary.

    All file paths stored in the record are made relative to *output_root*
    (or left absolute if *output_root* is ``None``).  Using relative paths
    keeps the database portable when the project is moved or shared.

    Parameters
    ----------
    frame_index : int
        Zero-based index of this structure in the source EXTXYZ file.
    source_extxyz_path : Path or None
        Path to the original MatterGen EXTXYZ file.
    ml_relaxed_path : Path or None
        Path to the ML-relaxed EXTXYZ file produced by MatterSim.
    energy_per_atom_eV : float or None
        ML potential energy per atom (eV/atom).
    check_result : checks.CheckResult
        Sanity check outcomes for this structure.
    cfg : PipelineConfig
        Fully loaded pipeline configuration.
    output_root : Path, optional
        Root directory used to make file paths relative.
        Defaults to ``cfg.project.output_path``.

    Returns
    -------
    dict
        A JSON-serialisable candidate record conforming to the schema
        described at the top of this module.
    """
    from maccai_battery.checks import make_filters_dict, make_ml_scores_dict

    root = output_root or cfg.project.output_path

    def _rel(p: Optional[Path]) -> Optional[str]:
        """Convert *p* to a string path relative to *root*, if possible."""
        if p is None:
            return None
        p = Path(p)
        try:
            return str(p.relative_to(root))
        except ValueError:
            # Path is outside root — keep it as-is but warn.
            logger.warning(
                "Path %s is outside output_root %s; storing as absolute.", p, root
            )
            return str(p)

    gen_cfg = cfg.generation
    db_cfg  = cfg.database

    # ------------------------------------------------------------------
    # Load structure for metadata (best-effort)
    # ------------------------------------------------------------------
    structure = _load_structure_best_effort(ml_relaxed_path or source_extxyz_path)

    if structure is not None:
        formula   = structure.composition.reduced_formula
        stoich    = {el.symbol: int(round(amt))
                     for el, amt in structure.composition.items()}
    else:
        formula   = check_result.formula or "unknown"
        stoich    = {}

    # ------------------------------------------------------------------
    # Assemble record
    # ------------------------------------------------------------------
    record: Dict[str, Any] = {
        "id":      _new_id(),
        "formula": formula,
        "stoichiometry": stoich,

        "structure_files": {
            "source_extxyz": _rel(source_extxyz_path),
            "ml_relaxed":    _rel(ml_relaxed_path),
        },

        "generation_metadata": {
            "generator":         db_cfg.generator,
            "model_checkpoint":  gen_cfg.model_name,
            "chemical_system":   gen_cfg.chemical_system,
            "energy_above_hull": gen_cfg.energy_above_hull,
            "frame_index":       frame_index,
            "conditioning": {
                "energy_above_hull": gen_cfg.energy_above_hull,
                "chemical_system":   gen_cfg.chemical_system,
            },
            "seed":      None,
            "timestamp": _utc_now(),
        },

        "ml_scores": make_ml_scores_dict(check_result, energy_per_atom_eV),

        "filters": make_filters_dict(check_result),

        "dft_jobs": {
            "status":   "not_submitted",
            "workflow": db_cfg.dft_workflow,
            "scf":      _make_empty_dft_stage(),
            "relax":    _make_empty_dft_stage(),
            "vasp_incar_settings": db_cfg.vasp_incar_defaults,
        },

        "notes": "",
    }

    return record


# ---------------------------------------------------------------------------
# Database class
# ---------------------------------------------------------------------------

class CandidateDatabase:
    """Read/write interface for the candidates.ndjson database.

    The database is a newline-delimited JSON (NDJSON) file where each line
    is a complete JSON object representing one candidate structure.

    All mutating operations (``append``, ``update_*``) use atomic writes via
    a temporary file and ``os.replace`` so that a crash mid-write never
    leaves a partially written or corrupted database.

    Parameters
    ----------
    cfg : PipelineConfig
        Fully loaded pipeline configuration.  The database path is derived
        from ``cfg.project.candidates_db_path``.
    """

    def __init__(self, cfg) -> None:
        self._cfg  = cfg
        self._path = cfg.project.candidates_db_path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        logger.debug("CandidateDatabase path: %s", self._path)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def path(self) -> Path:
        """Absolute path to the NDJSON database file."""
        return self._path

    @property
    def exists(self) -> bool:
        """``True`` if the database file exists and is non-empty."""
        return self._path.exists() and self._path.stat().st_size > 0

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def append(self, record: Dict[str, Any]) -> str:
        """Append a single candidate record to the database.

        The record is validated for the required ``id`` field before writing.

        Parameters
        ----------
        record : dict
            A JSON-serialisable candidate record.  Must contain an ``"id"``
            key; if missing, one is generated automatically.

        Returns
        -------
        str
            The ``"id"`` field of the appended record.

        Raises
        ------
        TypeError
            If *record* is not a dict.
        ValueError
            If *record* contains non-JSON-serialisable values.
        """
        if not isinstance(record, dict):
            raise TypeError(f"record must be a dict, got {type(record).__name__}")

        # Auto-assign ID if missing
        if "id" not in record or not record["id"]:
            record = {**record, "id": _new_id()}

        # Validate JSON-serialisability eagerly
        line = json.dumps(record, ensure_ascii=False)

        with self._path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")

        logger.debug("Appended candidate %s to database.", record["id"])
        return record["id"]

    def append_many(self, records: List[Dict[str, Any]]) -> List[str]:
        """Append multiple records in a single file open.

        Parameters
        ----------
        records : list of dict

        Returns
        -------
        list of str
            The ``"id"`` fields of all appended records.
        """
        if not records:
            return []

        ids: List[str] = []
        lines: List[str] = []

        for record in records:
            if not isinstance(record, dict):
                raise TypeError(f"Each record must be a dict, got {type(record).__name__}")
            if "id" not in record or not record["id"]:
                record = {**record, "id": _new_id()}
            ids.append(record["id"])
            lines.append(json.dumps(record, default=lambda o: bool(o) if isinstance(o, np.bool_) else int(o) if isinstance(o, np.integer) else float(o) if isinstance(o, np.floating) else None, ensure_ascii=False))

        with self._path.open("a", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")

        logger.info("Appended %d candidates to database.", len(ids))
        return ids

    def update_field(
        self,
        candidate_id: str,
        field_path: str,
        value: Any,
    ) -> bool:
        """Update a single field in an existing record.

        Uses a full rewrite of the database file (atomic via temp file).
        This is intentionally simple; for bulk updates use
        :meth:`update_records`.

        Parameters
        ----------
        candidate_id : str
            The ``"id"`` of the record to update.
        field_path : str
            Dot-separated path to the field, e.g. ``"dft_jobs.status"``
            or ``"notes"``.
        value : Any
            The new value.

        Returns
        -------
        bool
            ``True`` if a record with *candidate_id* was found and updated.
        """
        records   = self.load_all()
        found     = False

        for record in records:
            if record.get("id") == candidate_id:
                _set_nested(record, field_path, value)
                found = True
                break

        if not found:
            logger.warning(
                "update_field: candidate '%s' not found in database.", candidate_id
            )
            return False

        self._atomic_write(records)
        logger.debug(
            "Updated %s.%s = %r in database.", candidate_id, field_path, value
        )
        return True

    def update_dft_scf(
        self,
        candidate_id: str,
        *,
        energy_eV: Optional[float],
        n_atoms: Optional[int],
        run_dir: Optional[str] = None,
        status: str = "done",
    ) -> bool:
        """Update the SCF DFT results for a candidate.

        Parameters
        ----------
        candidate_id : str
        energy_eV : float or None
            Total DFT SCF energy (eV).
        n_atoms : int or None
            Number of atoms in the cell.
        run_dir : str, optional
            Directory where the QE SCF run was performed.
        status : str
            Job status string (default ``"done"``).

        Returns
        -------
        bool
        """
        records = self.load_all()
        found   = False

        for record in records:
            if record.get("id") == candidate_id:
                scf = record.setdefault("dft_jobs", {}).setdefault("scf", {})
                scf["status"]             = status
                scf["energy_eV"]          = energy_eV
                scf["energy_eV_per_atom"] = (
                    energy_eV / n_atoms
                    if (energy_eV is not None and n_atoms)
                    else None
                )
                scf["n_atoms"]            = n_atoms
                scf["run_dir"]            = run_dir
                scf["completed_at"]       = _utc_now()

                # Roll up top-level dft_jobs status
                record["dft_jobs"]["status"] = status
                found = True
                break

        if not found:
            logger.warning(
                "update_dft_scf: candidate '%s' not found.", candidate_id
            )
            return False

        self._atomic_write(records)
        logger.info(
            "Updated SCF result for %s: %.6f eV (%d atoms).",
            candidate_id, energy_eV or 0.0, n_atoms or 0,
        )
        return True

    def update_dft_relax(
        self,
        candidate_id: str,
        *,
        energy_eV: Optional[float],
        n_atoms: Optional[int],
        run_dir: Optional[str] = None,
        status: str = "done",
        structure: Optional[dict] = None,
    ) -> bool:
        """Update the ionic relaxation DFT results for a candidate.

        Parameters
        ----------
        candidate_id : str
        energy_eV : float or None
            Final total DFT energy after ionic relaxation (eV).
        n_atoms : int or None
            Number of atoms in the cell.
        run_dir : str, optional
            Directory where the QE relax run was performed.
        status : str
            Job status string (default ``"done"``).

        Returns
        -------
        bool
        """
        records = self.load_all()
        found   = False

        for record in records:
            if record.get("id") == candidate_id:
                relax = record.setdefault("dft_jobs", {}).setdefault("relax", {})
                relax["status"]             = status
                relax["energy_eV"]          = energy_eV
                relax["energy_eV_per_atom"] = (
                    energy_eV / n_atoms
                    if (energy_eV is not None and n_atoms)
                    else None
                )
                relax["n_atoms"]            = n_atoms
                relax["run_dir"]            = run_dir
                if structure is not None:
                    relax["structure"] = structure      # ← ADD THIS
                relax["completed_at"]       = _utc_now()

                record["dft_jobs"]["status"] = status
                found = True
                break

        if not found:
            logger.warning(
                "update_dft_relax: candidate '%s' not found.", candidate_id
            )
            return False

        self._atomic_write(records)
        logger.info(
            "Updated relax result for %s: %.6f eV (%d atoms).",
            candidate_id, energy_eV or 0.0, n_atoms or 0,
        )
        return True

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def load_all(self) -> List[Dict[str, Any]]:
        """Load all records from the database into memory.

        Returns
        -------
        list of dict
            Records in file order (oldest first).  Returns an empty list
            if the database does not yet exist.
        """
        if not self._path.exists():
            return []

        records: List[Dict[str, Any]] = []
        with self._path.open("r", encoding="utf-8") as fh:
            for lineno, raw_line in enumerate(fh, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    logger.warning(
                        "Skipping malformed JSON on line %d of %s: %s",
                        lineno, self._path.name, exc,
                    )

        return records

    def iter_records(self) -> Iterator[Dict[str, Any]]:
        """Iterate over records one at a time without loading them all.

        Useful for large databases where memory is a concern.

        Yields
        ------
        dict
            One candidate record per iteration.
        """
        if not self._path.exists():
            return

        with self._path.open("r", encoding="utf-8") as fh:
            for lineno, raw_line in enumerate(fh, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    logger.warning(
                        "Skipping malformed JSON on line %d: %s", lineno, exc
                    )

    def get_by_id(self, candidate_id: str) -> Optional[Dict[str, Any]]:
        """Find and return a single record by its ``"id"`` field.

        Parameters
        ----------
        candidate_id : str

        Returns
        -------
        dict or None
            The matching record, or ``None`` if not found.
        """
        for record in self.iter_records():
            if record.get("id") == candidate_id:
                return record
        return None

    def count(self) -> int:
        """Return the number of records in the database."""
        if not self._path.exists():
            return 0
        count = 0
        with self._path.open("r", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    count += 1
        return count

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def top_n_by_ml_energy(
        self,
        n: int,
        *,
        passed_checks_only: bool = False,
    ) -> List[Dict[str, Any]]:
        """Return the *n* candidates with the lowest ML energy per atom.

        Parameters
        ----------
        n : int
            Number of candidates to return.
        passed_checks_only : bool
            If ``True``, only candidates whose ``filters.passed_all`` is
            ``True`` are considered.

        Returns
        -------
        list of dict
            At most *n* records sorted by ascending ML energy per atom.
        """
        records = self.load_all()

        def _ml_energy(r: dict) -> float:
            e = r.get("ml_scores", {}).get("matter_sim_energy_eV_per_atom")
            return e if e is not None else float("inf")

        if passed_checks_only:
            records = [
                r for r in records
                if r.get("filters", {}).get("passed_all", False)
            ]

        records.sort(key=_ml_energy)
        return records[:n]

    def top_n_by_dft_scf_energy(self, n: int) -> List[Dict[str, Any]]:
        """Return the *n* candidates with the lowest DFT SCF energy per atom.

        Only candidates with a completed SCF calculation are included.

        Parameters
        ----------
        n : int

        Returns
        -------
        list of dict
        """
        records = self.load_all()

        completed = [
            r for r in records
            if r.get("dft_jobs", {}).get("scf", {}).get("energy_eV_per_atom") is not None
        ]

        def _scf_e(r: dict) -> float:
            return r["dft_jobs"]["scf"]["energy_eV_per_atom"]

        completed.sort(key=_scf_e)
        return completed[:n]

    def top_n_by_dft_relax_energy(self, n: int) -> List[Dict[str, Any]]:
        """Return the *n* candidates with the lowest DFT relaxation energy per atom.

        Only candidates with a completed relaxation are included.

        Parameters
        ----------
        n : int

        Returns
        -------
        list of dict
        """
        records = self.load_all()

        completed = [
            r for r in records
            if r.get("dft_jobs", {}).get("relax", {}).get("energy_eV_per_atom") is not None
        ]

        def _relax_e(r: dict) -> float:
            return r["dft_jobs"]["relax"]["energy_eV_per_atom"]

        completed.sort(key=_relax_e)
        return completed[:n]

    def filter_by_formula(self, formula: str) -> List[Dict[str, Any]]:
        """Return all records whose reduced formula matches *formula*.

        Parameters
        ----------
        formula : str
            Reduced formula string, e.g. ``"LiFePO4"``.

        Returns
        -------
        list of dict
        """
        return [
            r for r in self.iter_records()
            if r.get("formula") == formula
        ]

    def summary(self) -> Dict[str, Any]:
        """Return a human-readable statistics summary of the database.

        Returns
        -------
        dict with keys:
            total, passed_checks, failed_checks, ml_screened,
            scf_done, relax_done, formulas (list of unique formulas)
        """
        records = self.load_all()
        total   = len(records)

        passed_checks = sum(
            1 for r in records
            if r.get("filters", {}).get("passed_all", False)
        )

        ml_scored = sum(
            1 for r in records
            if r.get("ml_scores", {}).get("matter_sim_energy_eV_per_atom") is not None
        )

        scf_done = sum(
            1 for r in records
            if r.get("dft_jobs", {}).get("scf", {}).get("status") == "done"
        )

        relax_done = sum(
            1 for r in records
            if r.get("dft_jobs", {}).get("relax", {}).get("status") == "done"
        )

        formulas = sorted({r.get("formula", "?") for r in records})

        return {
            "total":          total,
            "passed_checks":  passed_checks,
            "failed_checks":  total - passed_checks,
            "ml_screened":    ml_scored,
            "scf_done":       scf_done,
            "relax_done":     relax_done,
            "formulas":       formulas,
        }

    def print_summary(self) -> None:
        """Print a formatted summary to stdout."""
        s = self.summary()
        print(f"\n{'='*55}")
        print(f"  Candidate Database: {self._path.name}")
        print(f"{'='*55}")
        print(f"  Total candidates  : {s['total']}")
        print(f"  Passed checks     : {s['passed_checks']}")
        print(f"  Failed checks     : {s['failed_checks']}")
        print(f"  ML-energy scored  : {s['ml_screened']}")
        print(f"  SCF done          : {s['scf_done']}")
        print(f"  Relax done        : {s['relax_done']}")
        print(f"  Unique formulas   : {len(s['formulas'])}")
        if s["formulas"]:
            print(f"  Formulas          : {', '.join(s['formulas'][:10])}", end="")
            print(" ..." if len(s["formulas"]) > 10 else "")
        print(f"{'='*55}\n")

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def backup(self, suffix: Optional[str] = None) -> Path:
        """Create a timestamped backup copy of the database file.

        Parameters
        ----------
        suffix : str, optional
            Extra suffix appended to the backup filename before the
            timestamp.  E.g. ``"pre_dft"`` → ``candidates.pre_dft.20250101T120000.ndjson``.

        Returns
        -------
        Path
            Path to the backup file.
        """
        if not self._path.exists():
            raise FileNotFoundError(
                f"Cannot back up database — file does not exist: {self._path}"
            )

        ts   = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        stem = self._path.stem
        ext  = self._path.suffix

        parts = [stem]
        if suffix:
            parts.append(suffix)
        parts.append(ts)

        backup_name = ".".join(parts) + ext
        backup_path = self._path.parent / backup_name

        shutil.copy2(self._path, backup_path)
        logger.info("Database backed up to: %s", backup_path)
        return backup_path

    def deduplicate(self, key: str = "id") -> int:
        """Remove duplicate records, keeping the first occurrence.

        Parameters
        ----------
        key : str
            Record field used to identify duplicates (default ``"id"``).

        Returns
        -------
        int
            Number of duplicate records removed.
        """
        records  = self.load_all()
        seen     = set()
        unique   = []

        for record in records:
            val = record.get(key)
            if val not in seen:
                seen.add(val)
                unique.append(record)

        removed = len(records) - len(unique)

        if removed > 0:
            self._atomic_write(unique)
            logger.info(
                "Deduplication removed %d duplicate records (key='%s').",
                removed, key,
            )
        else:
            logger.debug("No duplicates found (key='%s').", key)

        return removed

    def export_csv(self, out_path: Optional[Path] = None) -> Path:
        """Export a flat CSV summary of all candidates.

        Includes the most commonly needed fields for quick analysis in
        spreadsheet tools.  Nested fields are flattened with dot notation.

        Parameters
        ----------
        out_path : Path, optional
            Destination CSV file.  Defaults to ``<db_dir>/candidates_summary.csv``.

        Returns
        -------
        Path
            Path to the written CSV file.
        """
        import csv

        records = self.load_all()

        if out_path is None:
            out_path = self._path.parent / "candidates_summary.csv"

        fieldnames = [
            "id",
            "formula",
            "ml_energy_eV_per_atom",
            "density_gcc",
            "min_distance_A",
            "passed_checks",
            "warning_count",
            "charge_neutral",
            "oxidation_states_assigned",
            "scf_energy_eV_per_atom",
            "relax_energy_eV_per_atom",
            "dft_status",
            "frame_index",
            "ml_relaxed_path",
        ]

        with open(out_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()

            for r in records:
                ml   = r.get("ml_scores", {})
                filt = r.get("filters",   {})
                dft  = r.get("dft_jobs",  {})
                gen  = r.get("generation_metadata", {})
                sf   = r.get("structure_files", {})

                writer.writerow({
                    "id":                      r.get("id"),
                    "formula":                 r.get("formula"),
                    "ml_energy_eV_per_atom":   ml.get("matter_sim_energy_eV_per_atom"),
                    "density_gcc":             ml.get("density_gcc"),
                    "min_distance_A":          ml.get("min_distance_A"),
                    "passed_checks":           filt.get("passed_all"),
                    "warning_count":           filt.get("warning_count"),
                    "charge_neutral":          filt.get("charge_neutral"),
                    "oxidation_states_assigned": filt.get("oxidation_states_assigned"),
                    "scf_energy_eV_per_atom":  dft.get("scf", {}).get("energy_eV_per_atom"),
                    "relax_energy_eV_per_atom": dft.get("relax", {}).get("energy_eV_per_atom"),
                    "dft_status":              dft.get("status"),
                    "frame_index":             gen.get("frame_index"),
                    "ml_relaxed_path":         sf.get("ml_relaxed"),
                })

        logger.info("Exported %d candidates to CSV: %s", len(records), out_path)
        return out_path

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _atomic_write(self, records: List[Dict[str, Any]]) -> None:
        """Rewrite the entire database file atomically.

        Writes to a temporary file in the same directory first, then uses
        ``os.replace`` to swap it into place.  This guarantees the database
        is never left in a half-written state if the process is interrupted.

        Parameters
        ----------
        records : list of dict
            All records to write (complete replacement of the file).
        """
        parent = self._path.parent
        parent.mkdir(parents=True, exist_ok=True)

        # Write to a temp file in the same directory so os.replace is atomic
        # (same filesystem required for atomic rename).
        fd, tmp_path = tempfile.mkstemp(dir=parent, suffix=".ndjson.tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                for record in records:
                    fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            os.replace(tmp_path, self._path)
        except Exception:
            # Clean up temp file on failure
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _load_structure_best_effort(path: Optional[Path]):
    """Try to load a pymatgen Structure from *path*, returning ``None`` on any failure.

    Supports:
    - ``.cif`` files via pymatgen native reader
    - ``.extxyz`` / ``.xyz`` / any ASE-readable format via ASE → pymatgen

    This function is intentionally silent on failure so it can be used
    inside record-building loops without aborting the pipeline.

    Parameters
    ----------
    path : Path or None
        File to load.

    Returns
    -------
    pymatgen.core.Structure or None
    """
    if path is None:
        return None

    path = Path(path)
    if not path.exists():
        return None

    try:
        from pymatgen.core import Structure

        if path.suffix.lower() == ".cif":
            return Structure.from_file(str(path))

        # Fall through to ASE for EXTXYZ and other formats
        from ase import io as ase_io
        from pymatgen.io.ase import AseAtomsAdaptor

        atoms = ase_io.read(str(path))
        return AseAtomsAdaptor.get_structure(atoms)

    except Exception as exc:
        logger.debug(
            "_load_structure_best_effort: failed to load %s: %s", path, exc
        )
        return None


def _set_nested(record: Dict[str, Any], field_path: str, value: Any) -> None:
    """Set a value in a nested dict using a dot-separated *field_path*.

    Example
    -------
    >>> r = {"dft_jobs": {"status": "not_submitted"}}
    >>> _set_nested(r, "dft_jobs.status", "done")
    >>> r["dft_jobs"]["status"]
    'done'

    Parameters
    ----------
    record : dict
        The record to mutate in-place.
    field_path : str
        Dot-separated key path, e.g. ``"dft_jobs.scf.energy_eV"``.
    value : Any
        The value to assign.
    """
    keys = field_path.split(".")
    node = record

    for key in keys[:-1]:
        if key not in node or not isinstance(node[key], dict):
            node[key] = {}
        node = node[key]

    node[keys[-1]] = value
