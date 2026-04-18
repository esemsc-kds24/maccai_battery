#!/usr/bin/env python3
# =============================================================================
# 02_relax.py — MatterSim ML geometry relaxation
# =============================================================================
# Step 2 of the MACCAI battery materials discovery pipeline.
#
# What this script does:
#   1. Loads config and the generated EXTXYZ from step 1
#   2. Relaxes each structure using MatterSim (or EMT fallback)
#   3. Saves one *_ml_relaxed.extxyz per structure
#   4. Writes relaxation results to a CSV summary
#   5. Prints energy ranking
#
# Prerequisites:
#   - Step 1 (01_generate.py) must have completed successfully
#   - MatterSim installed in its own conda environment (mattersim_env)
#   - This script must be run with the mattersim_env kernel/interpreter
#   - config.yaml must exist at the project root
#
# Usage:
#   conda activate mattersim_env
#   python scripts/02_relax.py
#   python scripts/02_relax.py --config /path/to/config.yaml
#   python scripts/02_relax.py --max-structures 10   # override max from config
#   python scripts/02_relax.py --extxyz path/to/generated_crystals.extxyz
#
# Outputs:
#   output/candidates/ml_relaxed/generated_crystals_frame<N>_ml_relaxed.extxyz
#   output/candidates/ml_relaxed_summary.csv
#   output/logs/02_relax.log
#
# Notes:
#   - Each structure is relaxed independently; one failure does not stop the run.
#   - If MatterSim is unavailable and emt_fallback=true in config.yaml, the
#     ASE EMT potential is used as a fallback (much less accurate).
#   - Use relaxation.device: "cpu" in config.yaml on Apple Silicon Macs to
#     avoid MPS instability.
# =============================================================================

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Make sure the package root is on sys.path when running as a script
# ---------------------------------------------------------------------------
_SCRIPT_DIR  = Path(__file__).resolve().parent
_PROJECT_DIR = _SCRIPT_DIR.parent
if str(_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(_PROJECT_DIR))

from maccai_battery.config import load_config
from maccai_battery.generation import load_generated_structures
from maccai_battery.relaxation import relax_structures, rank_by_ml_energy
from maccai_battery.utils import setup_logging, format_energy_table


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Step 2: MatterSim ML geometry relaxation.",
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
        "--extxyz",
        type=Path,
        default=None,
        help=(
            "Path to the source EXTXYZ file from step 1. "
            "Defaults to output/candidates/cifs/generated_crystals.extxyz "
            "(as set in config.yaml)."
        ),
    )
    parser.add_argument(
        "--max-structures",
        type=int,
        default=None,
        help=(
            "Maximum number of structures to relax. "
            "Overrides relaxation.max_structures in config.yaml."
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cpu", "cuda", "mps"],
        help=(
            "Compute device for MatterSim. "
            "Overrides relaxation.device in config.yaml."
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
# Helpers
# ---------------------------------------------------------------------------

def _find_source_extxyz(cfg) -> Path:
    """Locate the generated_crystals.extxyz produced by step 1.

    Search order:
      1. Canonical copy in output/candidates/cifs/
      2. Any .extxyz in that directory

    Raises
    ------
    FileNotFoundError
        If no EXTXYZ file can be found.
    """
    cif_dir  = cfg.project.dir_path("cifs")
    canonical = cif_dir / "generated_crystals.extxyz"

    if canonical.exists():
        return canonical

    # Fallback: search the cifs directory for any extxyz
    candidates = sorted(cif_dir.glob("*.extxyz"))
    if candidates:
        return candidates[0]

    raise FileNotFoundError(
        f"No EXTXYZ file found in {cif_dir}.\n"
        f"Did step 1 (01_generate.py) complete successfully?\n"
        f"Expected: {canonical}"
    )


def _write_csv_summary(results, out_path: Path) -> None:
    """Write a CSV summary of relaxation results to *out_path*."""
    fieldnames = [
        "frame_index",
        "formula",
        "n_atoms",
        "energy_eV",
        "energy_per_atom_eV",
        "converged",
        "used_fallback",
        "wall_time_s",
        "ml_relaxed_path",
        "error",
    ]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            row = r.to_dict()
            # Make path a string for CSV
            if row.get("ml_relaxed_path"):
                row["ml_relaxed_path"] = str(row["ml_relaxed_path"])
            writer.writerow(row)


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
    if args.max_structures is not None:
        cfg.relaxation.max_structures = args.max_structures
    if args.device is not None:
        cfg.relaxation.device = args.device

    # ------------------------------------------------------------------
    # Set up logging
    # ------------------------------------------------------------------
    log_dir  = cfg.project.dir_path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "02_relax.log"

    logger = setup_logging(
        level    = args.log_level,
        log_file = log_file,
    )

    logger.info("=" * 60)
    logger.info("MACCAI Battery Pipeline — Step 2: ML Relaxation")
    logger.info("=" * 60)
    logger.info("Config file  : %s", args.config or "(default)")
    logger.info("Log file     : %s", log_file)

    # ------------------------------------------------------------------
    # Print relaxation plan
    # ------------------------------------------------------------------
    rel = cfg.relaxation
    logger.info("")
    logger.info("Relaxation settings:")
    logger.info("  Device          : %s", rel.device)
    logger.info("  Max structures  : %s", rel.max_structures or "all")
    logger.info("  fmax            : %.3f eV/Å", rel.fmax)
    logger.info("  Max steps       : %d", rel.max_steps)
    logger.info("  Relax cell      : %s", rel.relax_cell)
    logger.info("  EMT fallback    : %s", rel.emt_fallback)
    logger.info("")

    # ------------------------------------------------------------------
    # Locate source EXTXYZ
    # ------------------------------------------------------------------
    extxyz_path = args.extxyz
    if extxyz_path is None:
        try:
            extxyz_path = _find_source_extxyz(cfg)
        except FileNotFoundError as exc:
            logger.error("%s", exc)
            sys.exit(1)

    extxyz_path = Path(extxyz_path)
    if not extxyz_path.exists():
        logger.error("Source EXTXYZ not found: %s", extxyz_path)
        sys.exit(1)

    logger.info("Source EXTXYZ: %s", extxyz_path)

    # ------------------------------------------------------------------
    # Load structures
    # ------------------------------------------------------------------
    logger.info("Loading generated structures ...")
    try:
        atoms_list = load_generated_structures(extxyz_path)
    except Exception as exc:
        logger.exception("Failed to load structures from %s: %s", extxyz_path, exc)
        sys.exit(1)

    logger.info("Loaded %d structures.", len(atoms_list))

    # ------------------------------------------------------------------
    # Run relaxation
    # ------------------------------------------------------------------
    logger.info("")
    logger.info("Starting ML relaxation ...")
    try:
        results = relax_structures(
            cfg        = cfg,
            atoms_list = atoms_list,
            source_extxyz = extxyz_path,
        )
    except Exception as exc:
        logger.exception("Relaxation run failed unexpectedly: %s", exc)
        sys.exit(1)

    # ------------------------------------------------------------------
    # Write CSV summary
    # ------------------------------------------------------------------
    out_dir     = cfg.project.dir_path("ml_relaxed")
    csv_path    = out_dir / "ml_relaxed_summary.csv"
    _write_csv_summary(results, csv_path)
    logger.info("CSV summary written to: %s", csv_path)

    # ------------------------------------------------------------------
    # Print energy ranking
    # ------------------------------------------------------------------
    ranked = rank_by_ml_energy(results)
    successful = [r for r in ranked if r.success]
    failed     = [r for r in ranked if not r.success]

    if successful:
        rows = [
            (f"frame{r.frame_index} ({r.formula})", r.energy_eV or 0.0, r.n_atoms)
            for r in successful
        ]
        print(format_energy_table(rows, title="ML Relaxation — Energy Ranking"))

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Relaxation complete")
    logger.info("  Total processed  : %d", len(results))
    logger.info("  Succeeded        : %d", len(successful))
    logger.info("  Failed           : %d", len(failed))
    if successful:
        best = successful[0]
        logger.info(
            "  Best candidate   : frame%d (%s) @ %.4f eV/atom",
            best.frame_index, best.formula, best.energy_per_atom_eV or float("nan"),
        )
    if failed:
        logger.warning("  Failed structures:")
        for r in failed:
            logger.warning("    frame%d (%s): %s", r.frame_index, r.formula, r.error)

    logger.info("")
    logger.info("Output directory : %s", out_dir)
    logger.info("CSV summary      : %s", csv_path)
    logger.info("Log file         : %s", log_file)
    logger.info("")
    logger.info("Next step:")
    logger.info("  python scripts/03_sanity_check.py")
    logger.info("")


if __name__ == "__main__":
    main()
