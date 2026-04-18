#!/usr/bin/env python3
# =============================================================================
# 05_merge_dft_results.py — Merge DFT results into candidates.ndjson
# =============================================================================
# Step 5 of the MACCAI battery materials discovery pipeline.
#
# What this script does:
#   1. Scans the DFT output directories (scf/ and relax/) for completed QE runs
#   2. Parses pw.x output files for total energies, n_atoms, convergence status
#   3. Matches each QE run back to its candidate record in candidates.ndjson
#      using the EXTXYZ filename embedded in the run directory name
#   4. Updates candidates.ndjson with DFT results (atomic write — safe)
#   5. Prints a final ranked table of all DFT-validated candidates
#
# Prerequisites:
#   - Step 3 (03_sanity_check.py) must have completed (candidates.ndjson exists)
#   - Step 4 (04_dft.py) must have completed (QE output files in output/dft/)
#   - QE output files must be present on this machine
#   - config.yaml must exist at the project root
#
# NOTE: If you ran the full pipeline with 04_dft.py, this script is run
# automatically at the end of that step. Use this script to re-merge results
# from external QE runs or to re-process partial outputs.
#
# Usage:
#   python scripts/05_merge_dft_results.py
#   python scripts/05_merge_dft_results.py --config /path/to/config.yaml
#   python scripts/05_merge_dft_results.py --dft-root output/dft
#   python scripts/05_merge_dft_results.py --dry-run       # preview, no writes
#   python scripts/05_merge_dft_results.py --stage scf     # only merge SCF
#   python scripts/05_merge_dft_results.py --stage relax   # only merge relax
#
# Expected DFT directory layout (created by 04_dft.py):
#
#   output/dft/
#   ├── scf/
#   │   ├── scf_0/          ← QE SCF run for the 1st ML-selected structure
#   │   │   ├── qe.in
#   │   │   └── qe.out      ← parsed for energy + convergence
#   │   ├── scf_1/
#   │   │   └── qe.out
#   │   └── ...
#   └── relax/
#       ├── relax_0/
#       │   ├── qe.in
#       │   └── qe.out
#       └── ...
#
# The script infers which candidate each run belongs to by:
#   1. For SCF: correlating the sequential run index with the ML-energy ranking
#      stored in candidates.ndjson (same selection order used in step 4).
#   2. For relax: matching the run directory name to the ml_relaxed filename.
#
# Outputs:
#   output/candidates.ndjson      (updated in-place, atomically)
#   output/dft_merge_report.txt   (human-readable merge summary)
#   output/logs/05_merge_dft_results.log
# =============================================================================

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Ensure package is importable when run as a script
# ---------------------------------------------------------------------------
_SCRIPT_DIR  = Path(__file__).resolve().parent
_PROJECT_DIR = _SCRIPT_DIR.parent
if str(_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(_PROJECT_DIR))

from maccai_battery.config import load_config
from maccai_battery.database import CandidateDatabase
from maccai_battery.utils import (
    setup_logging,
    parse_qe_total_energy_ry,
    parse_qe_n_atoms,
    parse_qe_scf_converged,
    extract_scf_summary,
    ry_to_ev,
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Step 5: Merge QE DFT results into candidates.ndjson.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to config.yaml (default: project root).",
    )
    parser.add_argument(
        "--dft-root",
        type=Path,
        default=None,
        help=(
            "Root directory of QE outputs. "
            "Expected sub-dirs: scf/ and relax/. "
            "Defaults to output/dft/ (from config.yaml)."
        ),
    )
    parser.add_argument(
        "--stage",
        choices=["scf", "relax", "both"],
        default="both",
        help="Which DFT stage(s) to merge (default: both).",
    )
    parser.add_argument(
        "--qe-output-name",
        default="qe.out",
        help="Filename of the QE pw.x output inside each run directory.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse outputs and show what would be written, but do not modify the database.",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Skip backing up candidates.ndjson before writing.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Console log verbosity.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# QE output parsing
# ---------------------------------------------------------------------------

def _read_qe_output(out_path: Path) -> str:
    """Read a QE pw.x output file and return its contents as a string."""
    if not out_path.exists():
        raise FileNotFoundError(f"QE output not found: {out_path}")
    return out_path.read_text(encoding="utf-8", errors="replace")


def _parse_run_dir(run_dir: Path, qe_output_name: str) -> Optional[Dict]:
    """Parse a single QE run directory and return a result dict.

    Returns ``None`` if the output file is missing or unparseable.

    Returns
    -------
    dict with keys:
        run_dir, converged, energy_ry, energy_ev, n_atoms, energy_ev_per_atom
    """
    out_path = run_dir / qe_output_name
    if not out_path.exists():
        return None

    try:
        text    = _read_qe_output(out_path)
        summary = extract_scf_summary(text)
        return {
            "run_dir":            str(run_dir),
            "converged":          summary["converged"],
            "energy_ry":          summary["energy_ry"],
            "energy_ev":          summary["energy_ev"],
            "n_atoms":            summary["n_atoms"],
            "energy_ev_per_atom": summary["energy_ev_per_atom"],
            "fermi_ev":           summary["fermi_ev"],
        }
    except Exception as exc:
        return {
            "run_dir":  str(run_dir),
            "error":    str(exc),
        }


# ---------------------------------------------------------------------------
# SCF directory scanning
# ---------------------------------------------------------------------------

def _find_scf_dirs(scf_root: Path) -> List[Tuple[int, Path]]:
    """Find all SCF run directories and return them sorted by index.

    Expects directories named ``scf_0``, ``scf_1``, ... or any prefix
    followed by an integer.

    Parameters
    ----------
    scf_root : Path
        Directory containing individual SCF run sub-directories.

    Returns
    -------
    list of (index, path) sorted by ascending index.
    """
    import re

    pairs: List[Tuple[int, Path]] = []
    for d in scf_root.iterdir():
        if not d.is_dir():
            continue
        m = re.search(r"(\d+)$", d.name)
        if m:
            pairs.append((int(m.group(1)), d))

    pairs.sort(key=lambda x: x[0])
    return pairs


def _find_relax_dirs(relax_root: Path) -> List[Path]:
    """Find all relaxation run directories.

    Returns all sub-directories of *relax_root*, sorted by name.
    """
    dirs = sorted(
        [d for d in relax_root.iterdir() if d.is_dir()],
        key=lambda d: d.name,
    )
    return dirs


# ---------------------------------------------------------------------------
# Candidate matching
# ---------------------------------------------------------------------------

def _match_scf_to_candidates(
    scf_results: List[Tuple[int, Dict]],
    db_records: List[Dict],
    n_ml_select: int,
    logger,
) -> List[Tuple[str, Dict]]:
    """Match SCF results to candidate IDs using ML-energy ranking.

    The Colab notebook selects the top-N ML candidates by energy and
    runs SCF on them in that order (scf_0 = best ML, scf_1 = 2nd best, ...).

    Parameters
    ----------
    scf_results : list of (index, parsed_result_dict)
        Parsed QE results, sorted by run index.
    db_records : list of dict
        All records from candidates.ndjson.
    n_ml_select : int
        Number of candidates selected for SCF (from dft_screening.n_candidates).
    logger

    Returns
    -------
    list of (candidate_id, result_dict)
    """
    # Replicate the ML-energy ranking used in the Colab notebook
    def _ml_energy(r: dict) -> float:
        e = r.get("ml_scores", {}).get("matter_sim_energy_eV_per_atom")
        return e if e is not None else float("inf")

    ranked = sorted(db_records, key=_ml_energy)
    top_n  = ranked[:n_ml_select]

    matched: List[Tuple[str, Dict]] = []

    for run_idx, result in scf_results:
        if run_idx >= len(top_n):
            logger.warning(
                "SCF run index %d is out of range for top-%d ML candidates — skipping.",
                run_idx, n_ml_select,
            )
            continue
        candidate_id = top_n[run_idx]["id"]
        matched.append((candidate_id, result))
        logger.debug(
            "Matched scf_%d → candidate %s (%s)",
            run_idx, candidate_id, top_n[run_idx].get("formula", "?"),
        )

    return matched


def _match_relax_to_candidates(
    relax_dirs: List[Path],
    db_records: List[Dict],
    logger,
) -> List[Tuple[str, Dict, Path]]:
    """Match relaxation run directories to candidate IDs by filename.

    The Colab notebook names relax directories after the ML-relaxed EXTXYZ
    file, e.g.::

        generated_crystals_frame48_ml_relaxed_relax/

    We search candidates.ndjson for a record whose ``ml_relaxed`` path
    contains ``frame48``.

    Parameters
    ----------
    relax_dirs : list of Path
    db_records : list of dict
    logger

    Returns
    -------
    list of (candidate_id, parsed_result_dict, run_dir_path)
    """
    import re

    # Build a frame-index → candidate_id lookup
    frame_to_record: Dict[int, Dict] = {}
    for record in db_records:
        ml_path = record.get("structure_files", {}).get("ml_relaxed", "") or ""
        m = re.search(r"frame(\d+)_ml_relaxed", ml_path)
        if m:
            frame_to_record[int(m.group(1))] = record

    matched: List[Tuple[str, Dict, Path]] = []

    for run_dir in relax_dirs:
        # Extract frame index from the run directory name
        m = re.search(r"frame(\d+)_ml_relaxed", run_dir.name)
        if not m:
            logger.debug(
                "Relax dir '%s' does not contain a frame index — skipping.", run_dir.name
            )
            continue

        frame_idx = int(m.group(1))
        record    = frame_to_record.get(frame_idx)

        if record is None:
            logger.warning(
                "No candidate found for frame%d (relax dir: %s).", frame_idx, run_dir.name
            )
            continue

        result = _parse_run_dir(run_dir, "qe.out")
        if result is None:
            logger.warning("No QE output found in %s — skipping.", run_dir)
            continue

        matched.append((record["id"], result, run_dir))
        logger.debug(
            "Matched relax frame%d → candidate %s (%s)",
            frame_idx, record["id"], record.get("formula", "?"),
        )

    return matched


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------

def _write_report(
    report_path: Path,
    scf_matches: List[Tuple[str, Dict]],
    relax_matches: List[Tuple[str, Dict, Path]],
    db_records: List[Dict],
) -> None:
    """Write a plain-text merge report to *report_path*."""
    lines = [
        "=" * 72,
        "  MACCAI Battery Pipeline — DFT Merge Report",
        "=" * 72,
        "",
        f"  SCF results merged  : {len(scf_matches)}",
        f"  Relax results merged: {len(relax_matches)}",
        "",
    ]

    if scf_matches:
        lines += [
            "  SCF Results:",
            f"  {'Candidate':12s}  {'Formula':12s}  {'Converged':9s}  {'nat':>5}  {'E (eV/atom)':>14}",
            "  " + "-" * 58,
        ]
        # Build formula lookup
        id_to_formula = {r["id"]: r.get("formula", "?") for r in db_records}

        for cid, res in scf_matches:
            formula   = id_to_formula.get(cid, "?")
            converged = str(res.get("converged", "?"))
            nat       = res.get("n_atoms") or "?"
            e_pa      = res.get("energy_ev_per_atom")
            e_str     = f"{e_pa:.6f}" if e_pa is not None else "N/A"
            err       = res.get("error")

            if err:
                lines.append(f"  {cid:12s}  {formula:12s}  ERROR: {err}")
            else:
                lines.append(
                    f"  {cid:12s}  {formula:12s}  {converged:9s}  {str(nat):>5}  {e_str:>14}"
                )

    lines.append("")

    if relax_matches:
        lines += [
            "  Relaxation Results:",
            f"  {'Candidate':12s}  {'Formula':12s}  {'Converged':9s}  {'nat':>5}  {'E (eV/atom)':>14}",
            "  " + "-" * 58,
        ]
        id_to_formula = {r["id"]: r.get("formula", "?") for r in db_records}

        for cid, res, run_dir in relax_matches:
            formula   = id_to_formula.get(cid, "?")
            converged = str(res.get("converged", "?"))
            nat       = res.get("n_atoms") or "?"
            e_pa      = res.get("energy_ev_per_atom")
            e_str     = f"{e_pa:.6f}" if e_pa is not None else "N/A"
            err       = res.get("error")

            if err:
                lines.append(f"  {cid:12s}  {formula:12s}  ERROR: {err}")
            else:
                lines.append(
                    f"  {cid:12s}  {formula:12s}  {converged:9s}  {str(nat):>5}  {e_str:>14}"
                )

    lines += ["", "=" * 72, ""]
    report_path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    # ------------------------------------------------------------------
    # Load config
    # ------------------------------------------------------------------
    cfg = load_config(args.config)

    # ------------------------------------------------------------------
    # Set up logging
    # ------------------------------------------------------------------
    log_dir  = cfg.project.dir_path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "05_merge_dft_results.log"

    logger = setup_logging(level=args.log_level, log_file=log_file)

    logger.info("=" * 60)
    logger.info("MACCAI Battery Pipeline — Step 5: Merge DFT Results")
    logger.info("=" * 60)
    logger.info("Config     : %s", args.config or "(default)")
    logger.info("Stage      : %s", args.stage)
    logger.info("Dry run    : %s", args.dry_run)
    logger.info("Log file   : %s", log_file)

    # ------------------------------------------------------------------
    # Locate DFT root directory
    # ------------------------------------------------------------------
    dft_root = args.dft_root
    if dft_root is None:
        # Try output/dft/ first, then the individual stage dirs
        dft_root = cfg.project.output_path / "dft"

    dft_root = Path(dft_root)
    scf_root   = dft_root / "scf"
    relax_root = dft_root / "relax"

    logger.info("DFT root   : %s", dft_root)

    if not dft_root.exists():
        logger.error(
            "DFT root directory not found: %s\n"
            "Download your Colab outputs to this path and re-run, or use --dft-root.",
            dft_root,
        )
        sys.exit(1)

    # ------------------------------------------------------------------
    # Load candidate database
    # ------------------------------------------------------------------
    db = CandidateDatabase(cfg)

    if not db.exists:
        logger.error(
            "Candidate database not found: %s\n"
            "Run step 3 (03_sanity_check.py) first.",
            db.path,
        )
        sys.exit(1)

    db_records = db.load_all()
    logger.info("Loaded %d candidate records from database.", len(db_records))

    # Back up database before modifying
    if not args.dry_run and not args.no_backup:
        backup = db.backup(suffix="pre_dft_merge")
        logger.info("Database backed up to: %s", backup)

    # ------------------------------------------------------------------
    # Merge SCF results
    # ------------------------------------------------------------------
    scf_matches: List[Tuple[str, Dict]] = []

    if args.stage in ("scf", "both"):
        if not scf_root.exists():
            logger.warning("SCF directory not found: %s — skipping SCF merge.", scf_root)
        else:
            logger.info("")
            logger.info("Scanning SCF results in: %s", scf_root)

            scf_dirs = _find_scf_dirs(scf_root)
            logger.info("Found %d SCF run directories.", len(scf_dirs))

            # Parse each run directory
            scf_parsed: List[Tuple[int, Dict]] = []
            for run_idx, run_dir in scf_dirs:
                result = _parse_run_dir(run_dir, args.qe_output_name)
                if result is None:
                    logger.warning(
                        "No %s found in %s — skipping.", args.qe_output_name, run_dir
                    )
                    continue
                if "error" in result:
                    logger.warning("Parse error for scf_%d: %s", run_idx, result["error"])
                else:
                    conv_str = "✓" if result.get("converged") else "✗"
                    e_pa     = result.get("energy_ev_per_atom")
                    e_str    = f"{e_pa:.4f}" if e_pa is not None else "N/A"
                    logger.info(
                        "  scf_%d: converged=%s  E=%s eV/atom  nat=%s",
                        run_idx, conv_str, e_str, result.get("n_atoms", "?"),
                    )
                scf_parsed.append((run_idx, result))

            # Match to candidates
            scf_matches = _match_scf_to_candidates(
                scf_parsed,
                db_records,
                cfg.dft_screening.n_candidates,
                logger,
            )

            if not args.dry_run:
                for cid, result in scf_matches:
                    if "error" in result:
                        db.update_dft_scf(
                            cid,
                            energy_eV = None,
                            n_atoms   = None,
                            run_dir   = result.get("run_dir"),
                            status    = "failed",
                        )
                    else:
                        db.update_dft_scf(
                            cid,
                            energy_eV = result.get("energy_ev"),
                            n_atoms   = result.get("n_atoms"),
                            run_dir   = result.get("run_dir"),
                            status    = "done" if result.get("converged") else "unconverged",
                        )

            logger.info("SCF merge: %d candidates updated.", len(scf_matches))

    # ------------------------------------------------------------------
    # Merge relaxation results
    # ------------------------------------------------------------------
    relax_matches: List[Tuple[str, Dict, Path]] = []

    if args.stage in ("relax", "both"):
        if not relax_root.exists():
            logger.warning(
                "Relax directory not found: %s — skipping relax merge.", relax_root
            )
        else:
            logger.info("")
            logger.info("Scanning relaxation results in: %s", relax_root)

            relax_dirs = _find_relax_dirs(relax_root)
            logger.info("Found %d relaxation run directories.", len(relax_dirs))

            relax_matches = _match_relax_to_candidates(relax_dirs, db_records, logger)

            for cid, result, run_dir in relax_matches:
                if "error" in result:
                    logger.warning(
                        "  %-12s — parse error: %s", cid, result["error"]
                    )
                else:
                    conv_str = "✓" if result.get("converged") else "✗"
                    e_pa     = result.get("energy_ev_per_atom")
                    e_str    = f"{e_pa:.4f}" if e_pa is not None else "N/A"
                    logger.info(
                        "  %-12s — converged=%s  E=%s eV/atom  nat=%s",
                        cid, conv_str, e_str, result.get("n_atoms", "?"),
                    )

            if not args.dry_run:
                for cid, result, run_dir in relax_matches:
                    if "error" in result:
                        db.update_dft_relax(
                            cid,
                            energy_eV = None,
                            n_atoms   = None,
                            run_dir   = str(run_dir),
                            status    = "failed",
                        )
                    else:
                        db.update_dft_relax(
                            cid,
                            energy_eV = result.get("energy_ev"),
                            n_atoms   = result.get("n_atoms"),
                            run_dir   = str(run_dir),
                            status    = "done" if result.get("converged") else "unconverged",
                        )

            logger.info("Relax merge: %d candidates updated.", len(relax_matches))

    # ------------------------------------------------------------------
    # Write text report
    # ------------------------------------------------------------------
    report_path = cfg.project.output_path / "dft_merge_report.txt"

    if not args.dry_run:
        _write_report(report_path, scf_matches, relax_matches, db_records)
        logger.info("")
        logger.info("Merge report written to: %s", report_path)

    # ------------------------------------------------------------------
    # Print final DFT ranking
    # ------------------------------------------------------------------
    if not args.dry_run:
        # Reload to reflect updates
        db_records = db.load_all()

    logger.info("")

    relax_done = db.top_n_by_dft_relax_energy(20)
    if relax_done:
        logger.info("Final ranking by DFT relaxation energy:")
        logger.info(
            "  %-10s  %-12s  %-8s  %-16s",
            "ID", "Formula", "Nat", "E relax (eV/atom)",
        )
        logger.info("  " + "-" * 52)
        for rec in relax_done:
            relax = rec.get("dft_jobs", {}).get("relax", {})
            e_pa  = relax.get("energy_ev_per_atom")
            nat   = relax.get("n_atoms")
            logger.info(
                "  %-10s  %-12s  %-8s  %s",
                rec.get("id", "?"),
                rec.get("formula", "?"),
                str(nat or "?"),
                f"{e_pa:.6f}" if e_pa is not None else "N/A",
            )

    scf_done = db.top_n_by_dft_scf_energy(20)
    if scf_done and not relax_done:
        logger.info("Final ranking by DFT SCF energy (no relaxation data yet):")
        for rec in scf_done:
            scf  = rec.get("dft_jobs", {}).get("scf", {})
            e_pa = scf.get("energy_ev_per_atom")
            nat  = scf.get("n_atoms")
            logger.info(
                "  %-10s  %-12s  %-8s  %s eV/atom",
                rec.get("id", "?"),
                rec.get("formula", "?"),
                str(nat or "?"),
                f"{e_pa:.6f}" if e_pa is not None else "N/A",
            )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    db.print_summary()

    logger.info("=" * 60)
    logger.info("DFT merge complete")
    logger.info("  SCF merged          : %d", len(scf_matches))
    logger.info("  Relax merged        : %d", len(relax_matches))

    if args.dry_run:
        logger.info("")
        logger.info("DRY RUN — no changes written to database.")
    else:
        logger.info("  Database updated    : %s", db.path)
        logger.info("  Merge report        : %s", report_path)

    logger.info("  Log file            : %s", log_file)
    logger.info("")
    logger.info("Next step:")
    logger.info("  python scripts/06_hull_analysis.py")
    logger.info("")


if __name__ == "__main__":
    main()
