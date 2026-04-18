#!/usr/bin/env python3
# =============================================================================
# 03_sanity_check.py — Structural sanity checks & candidate database builder
# =============================================================================
# Step 3 of the MACCAI battery materials discovery pipeline.
#
# What this script does:
#   1. Loads config and all ML-relaxed EXTXYZ files from step 2
#   2. Converts each structure to pymatgen format
#   3. Runs a battery of fast heuristic sanity checks:
#        - Density (g/cm³) in expected range
#        - Minimum interatomic distance (detects overlaps)
#        - Oxidation state assignment
#        - Charge neutrality
#        - Bond-valence consistency
#   4. Builds a structured candidate record for every structure
#   5. Appends all records to candidates.ndjson (the pipeline database)
#   6. Exports a CSV summary for quick inspection
#   7. Prints a ranked table of candidates by ML energy
#
# Prerequisites:
#   - Step 2 (02_relax.py) must have completed successfully
#   - pymatgen must be installed (pip install pymatgen)
#   - config.yaml must exist at the project root
#
# Usage:
#   python scripts/03_sanity_check.py
#   python scripts/03_sanity_check.py --config /path/to/config.yaml
#   python scripts/03_sanity_check.py --ml-relaxed-dir output/candidates/ml_relaxed
#   python scripts/03_sanity_check.py --no-backup   # skip pre-existing DB backup
#
# Outputs:
#   output/candidates.ndjson                      (appended — NOT overwritten)
#   output/candidates_summary.csv                 (flat CSV for spreadsheet tools)
#   output/logs/03_sanity_check.log
#
# Notes:
#   - Failed checks are stored as metadata and do NOT crash the pipeline
#     (unless screening.hard_filter: true in config.yaml).
#   - Re-running this script appends new records; use `db.deduplicate()` to
#     clean up if you re-run on the same structures.
# =============================================================================

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Tuple

# ---------------------------------------------------------------------------
# Make sure the package root is on sys.path when running as a script
# ---------------------------------------------------------------------------
_SCRIPT_DIR  = Path(__file__).resolve().parent
_PROJECT_DIR = _SCRIPT_DIR.parent
if str(_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(_PROJECT_DIR))

from maccai_battery.config import load_config
from maccai_battery.checks import run_sanity_checks, CheckResult
from maccai_battery.database import CandidateDatabase, make_candidate_record
from maccai_battery.utils import (
    setup_logging,
    find_ml_relaxed_files,
    parse_extxyz_energy,
    deduplicate_by_fingerprint,
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Step 3: Structural sanity checks and candidate database builder.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help=(
            "Path to config.yaml. "
            "Defaults to $MACCAI_CONFIG env var or <project_root>/config.yaml."
        ),
    )
    parser.add_argument(
        "--ml-relaxed-dir",
        type=Path,
        default=None,
        help=(
            "Directory containing *_ml_relaxed.extxyz files from step 2. "
            "Defaults to output/candidates/ml_relaxed/ (from config.yaml)."
        ),
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help=(
            "Skip backing up the existing candidates.ndjson before appending. "
            "A backup is created by default if the database already exists."
        ),
    )
    parser.add_argument(
        "--hard-filter",
        action="store_true",
        default=None,
        help=(
            "Override config: treat failed sanity checks as fatal errors "
            "and skip those structures. "
            "Default: false (store failures as metadata, continue pipeline)."
        ),
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Console log verbosity level.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Structure loading helpers
# ---------------------------------------------------------------------------

def _load_pymatgen_structure(path: Path):
    """Load a pymatgen Structure from an EXTXYZ file via ASE.

    Parameters
    ----------
    path : Path
        Path to the ``*_ml_relaxed.extxyz`` file.

    Returns
    -------
    pymatgen.core.Structure or None
        ``None`` if loading fails for any reason.
    """
    try:
        from ase import io as ase_io
        from pymatgen.io.ase import AseAtomsAdaptor

        atoms = ase_io.read(str(path))
        return AseAtomsAdaptor.get_structure(atoms)
    except Exception:
        return None


def _extract_frame_index(path: Path) -> int:
    """Extract the integer frame index from a filename like
    ``generated_crystals_frame42_ml_relaxed.extxyz``.

    Falls back to 0 if the pattern is not found.
    """
    import re
    m = re.search(r"frame(\d+)_ml_relaxed", path.name)
    return int(m.group(1)) if m else 0


# ---------------------------------------------------------------------------
# Structure deduplication
# ---------------------------------------------------------------------------

def _make_fingerprint_fast(structure, volume_tol: float, density_tol: float) -> str:
    """Build a bucketed fingerprint from formula + cell volume + density.

    Two structures are considered duplicates when they share the same reduced
    formula AND their volume and density fall within the configured tolerances.

    Bucketing works by rounding each continuous value to the nearest
    ``tolerance``-sized bin, so structures that differ by less than the
    tolerance map to the same bin key.

    Parameters
    ----------
    structure : pymatgen.core.Structure
    volume_tol : float
        Bin width for cell volume (Å³).
    density_tol : float
        Bin width for density (g/cm³).

    Returns
    -------
    str
        A compact fingerprint key, e.g. ``"LiFePO4|v=150|d=3.4"``.
    """
    import math

    formula = structure.composition.reduced_formula
    volume  = structure.volume
    density = structure.density

    # Round to nearest bucket
    v_bin = round(volume  / volume_tol)  * volume_tol   if volume_tol  > 0 else volume
    d_bin = round(density / density_tol) * density_tol  if density_tol > 0 else density

    return f"{formula}|v={v_bin:.1f}|d={d_bin:.3f}"


def _make_fingerprint_rdf(structure) -> str:
    """Build a fingerprint from pymatgen's radial distribution function.

    More accurate than the fast method but slower (~ 0.5–2 s per structure).
    Falls back to the fast method if pymatgen's fingerprint module is
    unavailable.

    Parameters
    ----------
    structure : pymatgen.core.Structure

    Returns
    -------
    str
        A hex fingerprint string.
    """
    import hashlib

    try:
        from pymatgen.analysis.structure_matcher import StructureMatcher

        # Use StructureMatcher to detect true equivalence.
        # We encode via a canonical string instead of pairwise matching.
        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

        sga = SpacegroupAnalyzer(structure, symprec=0.1)
        sym_struct = sga.get_conventional_standard_structure()
        key = (
            sym_struct.composition.reduced_formula
            + str(round(sym_struct.volume, 1))
            + sym_struct.formula
        )
        return hashlib.md5(key.encode()).hexdigest()

    except Exception:
        # Fallback to fast method
        return _make_fingerprint_fast(structure, volume_tol=5.0, density_tol=0.05)


def _deduplicate_structures(
    paths_and_structures: List[Tuple[Path, object]],
    cfg,
    logger,
) -> Tuple[List[Tuple[Path, object]], int]:
    """Remove duplicate structures from a list before processing.

    Keeps the first occurrence of each unique fingerprint.

    Parameters
    ----------
    paths_and_structures : list of (Path, pymatgen.core.Structure)
        Each tuple is (ml_relaxed_path, pymatgen_structure).
    cfg : PipelineConfig
    logger

    Returns
    -------
    (deduplicated_list, n_removed)
        ``deduplicated_list`` — unique structures only (order preserved).
        ``n_removed``         — number of duplicates that were dropped.
    """
    dedup_cfg = cfg.deduplication

    if not dedup_cfg.enabled:
        return paths_and_structures, 0

    method = dedup_cfg.fingerprint_method

    def _fp(item: Tuple[Path, object]) -> str:
        _path, structure = item
        if method == "pymatgen_rdf":
            return _make_fingerprint_rdf(structure)
        # default: formula_volume_density
        return _make_fingerprint_fast(
            structure,
            volume_tol  = dedup_cfg.volume_tolerance_A3,
            density_tol = dedup_cfg.density_tolerance_gcc,
        )

    unique, dup_indices = deduplicate_by_fingerprint(paths_and_structures, _fp)
    n_removed = len(dup_indices)

    if n_removed > 0 and dedup_cfg.log_removed:
        for idx in dup_indices:
            path, _ = paths_and_structures[idx]
            logger.warning(
                "  [dedup] Removed duplicate: %s (index %d, method=%s)",
                path.name, idx, method,
            )

    return unique, n_removed


def _extract_source_extxyz(ml_relaxed_dir: Path, cfg) -> Optional[Path]:
    """Try to find the canonical source EXTXYZ (generated_crystals.extxyz).

    Checks the canonical cifs/ location first, then the parent directory
    of *ml_relaxed_dir*.

    Returns
    -------
    Path or None
        Path to the source file, or ``None`` if not found.
    """
    # Check the canonical location set up by step 1
    canonical = cfg.project.dir_path("cifs") / "generated_crystals.extxyz"
    if canonical.exists():
        return canonical

    # Fallback: parent directory of ml_relaxed
    parent_candidate = ml_relaxed_dir.parent / "generated_crystals.extxyz"
    if parent_candidate.exists():
        return parent_candidate

    return None


# ---------------------------------------------------------------------------
# Processing loop
# ---------------------------------------------------------------------------

def _process_structures(
    ml_relaxed_files: List[Path],
    source_extxyz: Optional[Path],
    cfg,
    logger,
) -> Tuple[List[dict], int, int]:
    """Run sanity checks on all ML-relaxed files and build candidate records.

    Parameters
    ----------
    ml_relaxed_files : list of Path
        Sorted list of ``*_ml_relaxed.extxyz`` files.
    source_extxyz : Path or None
        The original MatterGen EXTXYZ (for provenance metadata).
    cfg : PipelineConfig
        Fully loaded pipeline configuration.
    logger : logging.Logger

    Returns
    -------
    (records, n_pass, n_fail)
        - ``records``  : list of candidate record dicts ready for the DB
        - ``n_pass``   : number of structures that passed all checks
        - ``n_fail``   : number of structures with at least one failed check
    """
    records: List[dict] = []
    n_pass = 0
    n_fail = 0

    # ------------------------------------------------------------------
    # Step 0: Load all structures first so we can deduplicate
    # ------------------------------------------------------------------
    logger.info("  Loading %d structures for deduplication check ...", len(ml_relaxed_files))

    paths_and_structures: List[Tuple[Path, object]] = []
    load_failed: List[Path] = []

    for path in ml_relaxed_files:
        structure = _load_pymatgen_structure(path)
        if structure is None:
            logger.warning("    Could not load structure from %s — will skip.", path.name)
            load_failed.append(path)
            n_fail += 1
        else:
            paths_and_structures.append((path, structure))

    # ------------------------------------------------------------------
    # Step 1: Deduplicate
    # ------------------------------------------------------------------
    if cfg.deduplication.enabled and paths_and_structures:
        logger.info(
            "  Running deduplication (method=%s) on %d structures ...",
            cfg.deduplication.fingerprint_method,
            len(paths_and_structures),
        )
        paths_and_structures, n_removed = _deduplicate_structures(
            paths_and_structures, cfg, logger
        )
        if n_removed > 0:
            logger.info(
                "  Deduplication removed %d duplicate structure(s). "
                "%d unique structures remain.",
                n_removed, len(paths_and_structures),
            )
            n_fail += n_removed
        else:
            logger.info("  No duplicates found.")
    else:
        logger.info("  Deduplication disabled — skipping.")

    total = len(paths_and_structures)
    logger.info("  Processing %d unique structures ...", total)
    logger.info("")

    for i, (path, structure) in enumerate(paths_and_structures):
        frame_index = _extract_frame_index(path)

        logger.info(
            "  [%d/%d] Processing frame%d  (%s) ...",
            i + 1, total, frame_index, path.name,
        )

        # structure is already loaded (reuse from dedup pass above)
        if structure is None:
            logger.warning(
                "    Could not load structure from %s — skipping.", path.name
            )
            n_fail += 1
            continue

        # ------------------------------------------------------------------
        # Extract ML energy from EXTXYZ comment line
        # ------------------------------------------------------------------
        try:
            ml_energy_ev = parse_extxyz_energy(path)
            ml_energy_per_atom = ml_energy_ev / len(structure)
        except (ValueError, ZeroDivisionError) as exc:
            logger.debug(
                "    Could not parse ML energy for frame%d: %s", frame_index, exc
            )
            ml_energy_ev        = None
            ml_energy_per_atom  = None

        # ------------------------------------------------------------------
        # Run sanity checks
        # ------------------------------------------------------------------
        try:
            check: CheckResult = run_sanity_checks(structure, cfg)
        except RuntimeError as exc:
            # Only raised when hard_filter=True and a check failed
            logger.warning("    Hard-filter rejection for frame%d: %s", frame_index, exc)
            n_fail += 1
            continue

        if check.passed_all:
            n_pass += 1
            logger.info("    %s", check.summary_line())
        else:
            n_fail += 1
            logger.warning("    %s", check.summary_line())

        # ------------------------------------------------------------------
        # Build candidate record
        # ------------------------------------------------------------------
        record = make_candidate_record(
            frame_index        = frame_index,
            source_extxyz_path = source_extxyz,
            ml_relaxed_path    = path,
            energy_per_atom_eV = ml_energy_per_atom,
            check_result       = check,
            cfg                = cfg,
        )

        records.append(record)

    return records, n_pass, n_fail


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    # ------------------------------------------------------------------
    # Load config
    # ------------------------------------------------------------------
    cfg = load_config(args.config)

    # Apply CLI overrides
    if args.hard_filter is not None:
        cfg.screening.hard_filter = args.hard_filter

    # ------------------------------------------------------------------
    # Set up logging
    # ------------------------------------------------------------------
    log_dir  = cfg.project.dir_path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "03_sanity_check.log"

    logger = setup_logging(
        level    = args.log_level,
        log_file = log_file,
    )

    logger.info("=" * 60)
    logger.info("MACCAI Battery Pipeline — Step 3: Sanity Checks & Database")
    logger.info("=" * 60)
    logger.info("Config file  : %s", args.config or "(default)")
    logger.info("Log file     : %s", log_file)

    # ------------------------------------------------------------------
    # Print screening settings
    # ------------------------------------------------------------------
    sc = cfg.screening
    logger.info("")
    logger.info("Screening settings:")
    logger.info("  Min distance threshold  : %.2f Å", sc.min_distance_threshold_A)
    logger.info("  Density range           : %.1f – %.1f g/cm³",
                sc.density_min_gcc, sc.density_max_gcc)
    logger.info("  Assign oxidation states : %s", sc.assign_oxidation_states)
    logger.info("  Check charge neutrality : %s", sc.check_charge_neutrality)
    logger.info("  Check bond valence      : %s", sc.check_bond_valence)
    logger.info("  Hard filter             : %s", sc.hard_filter)
    logger.info("")

    # ------------------------------------------------------------------
    # Locate ML-relaxed files
    # ------------------------------------------------------------------
    ml_relaxed_dir = args.ml_relaxed_dir
    if ml_relaxed_dir is None:
        ml_relaxed_dir = cfg.project.dir_path("ml_relaxed")

    ml_relaxed_dir = Path(ml_relaxed_dir)

    if not ml_relaxed_dir.exists():
        logger.error(
            "ML-relaxed directory not found: %s\n"
            "Did step 2 (02_relax.py) complete successfully?",
            ml_relaxed_dir,
        )
        sys.exit(1)

    ml_relaxed_files = find_ml_relaxed_files(ml_relaxed_dir)

    if not ml_relaxed_files:
        logger.error(
            "No *_ml_relaxed.extxyz files found in %s.\n"
            "Did step 2 (02_relax.py) complete successfully?",
            ml_relaxed_dir,
        )
        sys.exit(1)

    logger.info("Found %d ML-relaxed structures in: %s", len(ml_relaxed_files), ml_relaxed_dir)

    # ------------------------------------------------------------------
    # Locate source EXTXYZ (for provenance)
    # ------------------------------------------------------------------
    source_extxyz = _extract_source_extxyz(ml_relaxed_dir, cfg)
    if source_extxyz:
        logger.info("Source EXTXYZ (provenance): %s", source_extxyz)
    else:
        logger.warning(
            "Source EXTXYZ not found — provenance metadata will be incomplete."
        )

    # ------------------------------------------------------------------
    # Open / backup the candidate database
    # ------------------------------------------------------------------
    db = CandidateDatabase(cfg)

    if db.exists and not args.no_backup:
        backup_path = db.backup(suffix="pre_step3")
        logger.info("Database backed up to: %s", backup_path)
    elif db.exists:
        logger.info("Existing database found (%d records) — no backup requested.", db.count())
    else:
        logger.info("No existing database — a new one will be created.")

    # ------------------------------------------------------------------
    # Run sanity checks and build records
    # ------------------------------------------------------------------
    logger.info("")
    logger.info("Running sanity checks on %d structures ...", len(ml_relaxed_files))
    logger.info("")

    records, n_pass, n_fail = _process_structures(
        ml_relaxed_files = ml_relaxed_files,
        source_extxyz    = source_extxyz,
        cfg              = cfg,
        logger           = logger,
    )

    # ------------------------------------------------------------------
    # Write all records to the database
    # ------------------------------------------------------------------
    if not records:
        logger.error("No valid records to write — all structures failed. Exiting.")
        sys.exit(1)

    logger.info("")
    logger.info("Writing %d records to database ...", len(records))

    ids = db.append_many(records)

    logger.info("Wrote %d records to: %s", len(ids), db.path)

    # ------------------------------------------------------------------
    # Deduplicate (in case the script is re-run on the same data)
    # ------------------------------------------------------------------
    n_removed = db.deduplicate(key="id")
    if n_removed > 0:
        logger.info("Removed %d duplicate records (re-run detected).", n_removed)

    # ------------------------------------------------------------------
    # Export CSV summary
    # ------------------------------------------------------------------
    csv_path = db.export_csv()
    logger.info("CSV summary exported to: %s", csv_path)

    # ------------------------------------------------------------------
    # Print ranked table (by ML energy per atom)
    # ------------------------------------------------------------------
    top_candidates = db.top_n_by_ml_energy(
        n                  = min(20, len(records)),
        passed_checks_only = False,
    )

    if top_candidates:
        logger.info("")
        logger.info("Top candidates by ML energy per atom:")
        logger.info(
            "  %-10s  %-12s  %-8s  %-16s  %-10s  %s",
            "ID", "Formula", "Nat", "E (eV/atom)", "Density", "Checks",
        )
        logger.info("  " + "-" * 75)

        for r in top_candidates:
            ml      = r.get("ml_scores", {})
            filt    = r.get("filters", {})
            e_pa    = ml.get("matter_sim_energy_eV_per_atom")
            density = ml.get("density_gcc")
            nat     = sum(r.get("stoichiometry", {}).values()) or "-"

            e_str       = f"{e_pa:.4f}" if e_pa is not None else "N/A"
            density_str = f"{density:.2f}" if density is not None else "N/A"
            check_str   = "PASS" if filt.get("passed_all") else f"WARN({filt.get('warning_count', '?')})"

            logger.info(
                "  %-10s  %-12s  %-8s  %-16s  %-10s  %s",
                r.get("id", "?"),
                r.get("formula", "?"),
                nat,
                e_str,
                density_str,
                check_str,
            )

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    logger.info("")
    db.print_summary()

    logger.info("=" * 60)
    logger.info("Sanity check & database build complete")
    logger.info("  Structures processed : %d", len(ml_relaxed_files))
    logger.info("  Records written      : %d", len(ids))
    logger.info("  Passed all checks    : %d", n_pass)
    logger.info("  Failed ≥1 check      : %d", n_fail)
    logger.info("")
    logger.info("Database  : %s", db.path)
    logger.info("CSV       : %s", csv_path)
    logger.info("Log       : %s", log_file)
    logger.info("")
    logger.info("Next step:")
    logger.info(
        "  Upload the following to Google Drive, then run the Colab notebook:"
    )
    logger.info("    %s", cfg.project.dir_path("ml_relaxed"))
    logger.info("    %s", db.path)
    logger.info("")


if __name__ == "__main__":
    main()
