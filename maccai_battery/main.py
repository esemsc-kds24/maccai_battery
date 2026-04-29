#!/usr/bin/env python3
# =============================================================================
# main.py — MACCAI Battery Materials Discovery Pipeline (End-to-End Entry Point)
# =============================================================================
#
# This script provides a single command to run the full pipeline or any
# contiguous subset of steps.
#
# Pipeline steps:
#   1. generate      — MatterGen crystal structure generation
#   2. relax         — MatterSim ML geometry relaxation
#   3. sanity_check  — Structural checks & candidate database builder
#   4. dft           — Quantum ESPRESSO SCF screening + ionic relaxation
#   5. merge         — Merge DFT results into candidates.ndjson
#   6. hull          — Materials Project convex hull stability analysis
#
# Usage:
#   python main.py                        # run all steps 1–6
#   python main.py --steps 1 2 3          # run only steps 1, 2, 3
#   python main.py --from-step 3          # run steps 3 through 6
#   python main.py --to-step 3            # run steps 1 through 3
#   python main.py --step 4               # run only step 4
#   python main.py --config my_config.yaml
#   python main.py --dry-run              # print plan, do not run anything
#
# Step-specific pass-through options:
#   python main.py --step 4 --nproc 8 --n-scf 10 --n-relax 4
#   python main.py --step 4 --dft-stage scf
#   python main.py --step 2 --device cuda
#   python main.py --step 6 --energy-source scf
#
# Environment notes:
#   Step 1 requires the MatterGen environment:
#     conda activate mattergen
#   Steps 2–6 require the main pipeline environment:
#     conda activate maccai
#
#   See README.md for full environment setup instructions.
#
# Quick start (GPU machine):
#   conda activate maccai
#   export MP_API_KEY="your_key_here"
#   python main.py --from-step 2          # assumes step 1 ran in mattergen env
# =============================================================================

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from typing import List, Optional

# ---------------------------------------------------------------------------
# Make sure the package is importable even when run from the repo root
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))


# ---------------------------------------------------------------------------
# Step registry
# ---------------------------------------------------------------------------

STEPS = {
    1: {
        "name": "generate",
        "label": "Crystal generation (MatterGen)",
        "script": "scripts/0generate.py",
        "env_note": "Requires: conda activate mattergen",
    },
    2: {
        "name": "relax",
        "label": "ML relaxation (MatterSim)",
        "script": "scripts/relax.py",
        "env_note": "Requires: conda activate maccai",
    },
    3: {
        "name": "sanity_check",
        "label": "Sanity checks & candidate database",
        "script": "scripts/sanity_check.py",
        "env_note": "Requires: conda activate maccai",
    },
    4: {
        "name": "dft",
        "label": "DFT screening & relaxation (Quantum ESPRESSO)",
        "script": "scripts/dft.py",
        "env_note": "Requires: conda activate maccai + pw.x on PATH",
    },
    5: {
        "name": "merge",
        "label": "Merge DFT results",
        "script": "scripts/merge_dft_results.py",
        "env_note": "Requires: conda activate maccai",
    },
    6: {
        "name": "hull",
        "label": "Hull stability analysis (Materials Project)",
        "script": "scripts/hull_analysis.py",
        "env_note": "Requires: conda activate maccai + MP_API_KEY set",
    },
}

ALL_STEPS = sorted(STEPS.keys())


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python main.py",
        description=(
            "MACCAI Battery Materials Discovery Pipeline\n"
            "End-to-end ML-to-DFT pipeline: MatterGen → MatterSim → QE → Hull\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              # Run the full pipeline (all 6 steps):
              python main.py

              # Run only steps 1, 2, 3:
              python main.py --steps 1 2 3

              # Resume from step 3 (after step 1-2 completed):
              python main.py --from-step 3

              # Run steps 1 through 3 only:
              python main.py --to-step 3

              # Run step 4 with 16 MPI processes and SCF only:
              python main.py --step 4 --nproc 16 --dft-stage scf

              # Run step 2 on GPU:
              python main.py --step 2 --device cuda

              # Dry-run (print plan without executing):
              python main.py --dry-run

              # Run hull analysis using SCF energies:
              python main.py --step 6 --energy-source scf
        """),
    )

    # ---- Step selection (mutually exclusive group) ----
    step_group = parser.add_mutually_exclusive_group()
    step_group.add_argument(
        "--step",
        type=int,
        choices=ALL_STEPS,
        metavar="{1..6}",
        help="Run exactly one step by number.",
    )
    step_group.add_argument(
        "--steps",
        type=int,
        nargs="+",
        choices=ALL_STEPS,
        metavar="N",
        help="Run specific steps, e.g. --steps 1 2 3.",
    )
    step_group.add_argument(
        "--from-step",
        type=int,
        choices=ALL_STEPS,
        metavar="{1..6}",
        help="Run from this step through step 6.",
    )
    step_group.add_argument(
        "--to-step",
        type=int,
        choices=ALL_STEPS,
        metavar="{1..6}",
        help="Run from step 1 through this step.",
    )

    # ---- Common options ----
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
        help="Print what would be run without executing any scripts.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity for this orchestrator (not for sub-scripts).",
    )

    # ---- Step 2 (relax) options ----
    relax_group = parser.add_argument_group("Step 2 — ML relaxation options")
    relax_group.add_argument(
        "--device",
        choices=["cpu", "cuda", "mps"],
        default=None,
        help="Compute device for MatterSim (overrides config.yaml).",
    )
    relax_group.add_argument(
        "--max-structures",
        type=int,
        default=None,
        metavar="N",
        help="Max structures to relax (overrides relaxation.max_structures).",
    )

    # ---- Step 4 (dft) options ----
    dft_group = parser.add_argument_group("Step 4 — DFT options")
    dft_group.add_argument(
        "--dft-stage",
        choices=["scf", "relax", "all"],
        default=None,
        metavar="STAGE",
        help="DFT stage to run: 'scf', 'relax', or 'all' (default: all).",
    )
    dft_group.add_argument(
        "--n-scf",
        type=int,
        default=None,
        metavar="N",
        help="Number of candidates for QE SCF screening.",
    )
    dft_group.add_argument(
        "--n-relax",
        type=int,
        default=None,
        metavar="M",
        help="Number of candidates for QE ionic relaxation.",
    )
    dft_group.add_argument(
        "--nproc",
        type=int,
        default=None,
        metavar="N",
        help="MPI processes for pw.x (step 4).",
    )

    # ---- Step 6 (hull) options ----
    hull_group = parser.add_argument_group("Step 6 — Hull analysis options")
    hull_group.add_argument(
        "--energy-source",
        choices=["scf", "relax"],
        default=None,
        help="DFT energy source for hull analysis.",
    )
    hull_group.add_argument(
        "--threshold",
        type=float,
        default=None,
        metavar="EV",
        help="Energy-above-hull stability threshold (eV/atom).",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Step selection logic
# ---------------------------------------------------------------------------

def _resolve_steps(args: argparse.Namespace) -> List[int]:
    """Determine which steps to run based on CLI arguments."""
    if args.step is not None:
        return [args.step]
    if args.steps is not None:
        return sorted(set(args.steps))
    if args.from_step is not None:
        return [s for s in ALL_STEPS if s >= args.from_step]
    if args.to_step is not None:
        return [s for s in ALL_STEPS if s <= args.to_step]
    return ALL_STEPS  # default: run everything


# ---------------------------------------------------------------------------
# Script runner
# ---------------------------------------------------------------------------

def _build_cmd(step_num: int, args: argparse.Namespace) -> List[str]:
    """Build the subprocess command for a given step."""
    step = STEPS[step_num]
    script = str(_HERE / step["script"])
    cmd = [sys.executable, script]

    # Common args
    if args.config:
        cmd += ["--config", str(args.config)]
    if args.log_level != "INFO":
        cmd += ["--log-level", args.log_level]

    # Step-specific args
    if step_num == 1:
        if args.dry_run:
            cmd.append("--dry-run")

    elif step_num == 2:
        if args.device:
            cmd += ["--device", args.device]
        if args.max_structures is not None:
            cmd += ["--max-structures", str(args.max_structures)]

    elif step_num == 4:
        if args.dft_stage:
            cmd += ["--stage", args.dft_stage]
        if args.n_scf is not None:
            cmd += ["--n-scf", str(args.n_scf)]
        if args.n_relax is not None:
            cmd += ["--n-relax", str(args.n_relax)]
        if args.nproc is not None:
            cmd += ["--nproc", str(args.nproc)]
        if args.dry_run:
            cmd.append("--dry-run")

    elif step_num == 6:
        if args.energy_source:
            cmd += ["--energy-source", args.energy_source]
        if args.threshold is not None:
            cmd += ["--threshold", str(args.threshold)]
        if args.dry_run:
            cmd.append("--dry-run")

    return cmd


def _run_step(step_num: int, args: argparse.Namespace, logger: logging.Logger) -> bool:
    """Run a single pipeline step as a subprocess.

    Returns True on success, False on failure.
    """
    step = STEPS[step_num]
    cmd = _build_cmd(step_num, args)

    logger.info("")
    logger.info("┌─────────────────────────────────────────────────────────┐")
    logger.info("│  Step %d: %-50s│", step_num, step["label"])
    logger.info("└─────────────────────────────────────────────────────────┘")
    logger.info("  Command: %s", " ".join(cmd))
    logger.info("")

    t0 = time.monotonic()
    try:
        result = subprocess.run(cmd, check=False)
        elapsed = time.monotonic() - t0

        if result.returncode == 0:
            logger.info("  ✓ Step %d completed in %.1f s.", step_num, elapsed)
            return True
        else:
            logger.error(
                "  ✗ Step %d FAILED (exit code %d) after %.1f s.",
                step_num, result.returncode, elapsed,
            )
            return False

    except FileNotFoundError as exc:
        elapsed = time.monotonic() - t0
        logger.error("  ✗ Step %d FAILED — script not found: %s", step_num, exc)
        return False
    except KeyboardInterrupt:
        elapsed = time.monotonic() - t0
        logger.warning("  ⚠ Step %d interrupted by user after %.1f s.", step_num, elapsed)
        raise


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    # ── Logging ──────────────────────────────────────────────────────────────
    log_fmt = "%(asctime)s  %(levelname)-8s  %(message)s"
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format=log_fmt,
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger("maccai.pipeline")

    # ── Resolve steps ────────────────────────────────────────────────────────
    steps_to_run = _resolve_steps(args)

    # ── Header ───────────────────────────────────────────────────────────────
    logger.info("")
    logger.info("╔══════════════════════════════════════════════════════════╗")
    logger.info("║      MACCAI Battery Materials Discovery Pipeline         ║")
    logger.info("╚══════════════════════════════════════════════════════════╝")
    logger.info("")
    logger.info("Config       : %s", args.config or "(default: config.yaml)")
    logger.info("Steps to run : %s", steps_to_run)
    if args.dry_run:
        logger.info("Mode         : DRY RUN — no scripts will be executed")
    logger.info("")

    # ── Print step plan ───────────────────────────────────────────────────────
    logger.info("Pipeline plan:")
    for snum in steps_to_run:
        step = STEPS[snum]
        logger.info("  [%d] %-42s  %s", snum, step["label"], step["env_note"])
    logger.info("")

    # ── Environment warning for step 1 ───────────────────────────────────────
    if 1 in steps_to_run and len(steps_to_run) > 1:
        logger.warning(
            "WARNING: Step 1 (MatterGen) requires a separate conda environment.\n"
            "         If you are not in the 'mattergen' environment, step 1 will fail.\n"
            "         Consider running: python main.py --step 1   (in mattergen env)\n"
            "         Then:             python main.py --from-step 2  (in maccai env)"
        )
        logger.info("")

    # ── Dry-run: just print and exit ─────────────────────────────────────────
    if args.dry_run:
        logger.info("[DRY RUN] Commands that would be executed:")
        for snum in steps_to_run:
            cmd = _build_cmd(snum, args)
            logger.info("  Step %d: %s", snum, " ".join(cmd))
        logger.info("")
        logger.info("[DRY RUN] No scripts were executed.")
        sys.exit(0)

    # ── Execute steps ─────────────────────────────────────────────────────────
    t_pipeline_start = time.monotonic()
    completed: List[int] = []
    failed_at: Optional[int] = None

    try:
        for snum in steps_to_run:
            success = _run_step(snum, args, logger)
            if success:
                completed.append(snum)
            else:
                failed_at = snum
                break
    except KeyboardInterrupt:
        logger.warning("")
        logger.warning("Pipeline interrupted by user.")
        if completed:
            logger.warning("Completed steps: %s", completed)
        sys.exit(130)

    # ── Final summary ─────────────────────────────────────────────────────────
    total_elapsed = time.monotonic() - t_pipeline_start

    logger.info("")
    logger.info("╔══════════════════════════════════════════════════════════╗")
    logger.info("║                    Pipeline Summary                      ║")
    logger.info("╚══════════════════════════════════════════════════════════╝")
    logger.info("")

    for snum in steps_to_run:
        step = STEPS[snum]
        if snum in completed:
            status = "✓ DONE"
        elif snum == failed_at:
            status = "✗ FAILED"
        else:
            status = "  SKIPPED"
        logger.info("  [%d] %s  %s", snum, status, step["label"])

    logger.info("")
    logger.info("  Total time : %.1f s  (%.1f min)", total_elapsed, total_elapsed / 60)
    logger.info("")

    if failed_at is not None:
        remaining = [s for s in steps_to_run if s > failed_at]
        logger.error(
            "Pipeline stopped at step %d. Check the log above for errors.",
            failed_at,
        )
        if remaining:
            logger.info(
                "To resume from the next step, run:\n"
                "  python main.py --from-step %d",
                failed_at + 1 if failed_at + 1 in STEPS else failed_at,
            )
        sys.exit(1)
    else:
        logger.info("All steps completed successfully.")
        logger.info("")
        if 6 in completed:
            logger.info(
                "Results are in:  %s",
                Path("output").resolve() if Path("output").exists() else "output/",
            )
            logger.info("  candidates.ndjson   — full candidate database with DFT energies")
            logger.info("  hull_analysis_report.txt — stability ranking")
        elif 3 in completed:
            logger.info("Next: run step 4 (DFT) when QE is available:")
            logger.info("  python main.py --from-step 4")
        sys.exit(0)


if __name__ == "__main__":
    main()
