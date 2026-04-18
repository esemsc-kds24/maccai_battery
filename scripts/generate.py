#!/usr/bin/env python3
# =============================================================================
# 01_generate.py — MatterGen crystal structure generation
# =============================================================================
# Step 1 of the MACCAI battery materials discovery pipeline.
#
# What this script does:
#   1. Loads the pipeline configuration from config.yaml
#   2. Runs MatterGen conditional generation (CLI-based)
#   3. Validates the output EXTXYZ file
#   4. Prints a lightweight summary of generated structures
#
# Prerequisites:
#   - MatterGen installed in a separate conda environment (mattergen_env)
#   - This script must be run with the mattergen_env kernel/interpreter
#   - config.yaml must exist at the project root
#
# Usage:
#   conda activate mattergen_env
#   python scripts/01_generate.py
#   python scripts/01_generate.py --config /path/to/config.yaml
#   python scripts/01_generate.py --dry-run        # print command without running
#
# Outputs:
#   output/candidates/cifs/mattergen_results/<model>_<system>/generated_crystals.extxyz
#   output/candidates/cifs/generated_crystals.extxyz   (canonical copy)
#   output/logs/01_generate.log
#
# Environment:
#   On Apple Silicon Macs, set PYTORCH_ENABLE_MPS_FALLBACK=1 automatically
#   (controlled by generation.pytorch_mps_fallback in config.yaml).
# =============================================================================

from __future__ import annotations

import argparse
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
from maccai_battery.generation import (
    run_mattergen,
    summarise_generated,
    count_structures,
)
from maccai_battery.utils import setup_logging


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Step 1: MatterGen crystal structure generation.",
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
        "--dry-run",
        action="store_true",
        help="Print the MatterGen command that would be run, then exit.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Console log verbosity level.",
    )
    return parser.parse_args()


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
    # Set up logging (console + file)
    # ------------------------------------------------------------------
    log_dir  = cfg.project.dir_path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "01_generate.log"

    logger = setup_logging(
        level    = args.log_level,
        log_file = log_file,
    )

    logger.info("=" * 60)
    logger.info("MACCAI Battery Pipeline — Step 1: Generation")
    logger.info("=" * 60)
    logger.info("Config file : %s", args.config or "(default)")
    logger.info("Project dir : %s", cfg.project.output_path)
    logger.info("Log file    : %s", log_file)

    # ------------------------------------------------------------------
    # Print generation plan
    # ------------------------------------------------------------------
    gen = cfg.generation
    logger.info("")
    logger.info("Generation plan:")
    logger.info("  Chemical system     : %s", gen.chemical_system)
    logger.info("  E above hull (max)  : %.3f eV/atom", gen.energy_above_hull)
    logger.info("  Model checkpoint    : %s", gen.model_name)
    logger.info("  Batch size          : %d", gen.batch_size)
    logger.info("  Number of batches   : %d", gen.num_batches)
    logger.info("  Total structures    : %d", gen.total_structures)
    logger.info("  Guidance factor     : %.1f", gen.diffusion_guidance_factor)
    logger.info("  MPS fallback (Mac)  : %s", gen.pytorch_mps_fallback)
    logger.info("")

    # ------------------------------------------------------------------
    # Dry-run mode: just print the command
    # ------------------------------------------------------------------
    if args.dry_run:
        from maccai_battery.generation import _build_results_dir, _build_command
        results_dir = _build_results_dir(cfg)
        cmd         = _build_command(cfg, results_dir)
        print("\n[DRY RUN] Command that would be executed:\n")
        print("  " + " ".join(cmd))
        print(f"\n[DRY RUN] Output directory:\n  {results_dir}\n")
        sys.exit(0)

    # ------------------------------------------------------------------
    # Run generation
    # ------------------------------------------------------------------
    logger.info("Starting MatterGen generation ...")
    try:
        extxyz_path = run_mattergen(cfg)
    except FileNotFoundError as exc:
        logger.error(
            "MatterGen executable not found.\n"
            "  Make sure you have activated the mattergen_env environment:\n"
            "    conda activate mattergen_env\n"
            "  Original error: %s",
            exc,
        )
        sys.exit(1)
    except Exception as exc:
        logger.exception("MatterGen generation failed: %s", exc)
        sys.exit(1)

    # ------------------------------------------------------------------
    # Summarise output
    # ------------------------------------------------------------------
    logger.info("")
    logger.info("Generation complete!")
    logger.info("Output file: %s", extxyz_path)

    n_total = count_structures(extxyz_path)
    logger.info("Total structures generated: %d", n_total)

    logger.info("")
    logger.info("Top 10 structures by ML energy:")
    logger.info("  %-6s  %-20s  %s", "Frame", "Formula", "Energy (eV)")
    logger.info("  %s", "-" * 46)

    summary = summarise_generated(extxyz_path)
    summary_sorted = sorted(summary, key=lambda x: x[2])

    for frame_idx, formula, energy in summary_sorted[:10]:
        energy_str = f"{energy:.6f}" if energy == energy else "N/A"  # NaN check
        logger.info("  %-6d  %-20s  %s", frame_idx, formula, energy_str)

    if n_total > 10:
        logger.info("  ... and %d more structures.", n_total - 10)

    logger.info("")
    logger.info("Next step:")
    logger.info(
        "  Activate mattersim_env and run:  python scripts/02_relax.py"
    )
    logger.info("")
    logger.info("Log written to: %s", log_file)


if __name__ == "__main__":
    main()
