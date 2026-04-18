#!/usr/bin/env python3
# =============================================================================
# 04_dft.py — Quantum ESPRESSO DFT screening and ionic relaxation
# =============================================================================
# Step 4 of the MACCAI battery materials discovery pipeline.
#
# What this script does:
#   1. Loads the candidate database built by step 3
#   2. Selects the top N ML-relaxed structures (by ML energy / atom)
#   3. Generates Quantum ESPRESSO SCF input files and runs pw.x
#   4. Ranks structures by DFT SCF energy, selects the top M
#   5. Generates QE ionic-relaxation inputs and runs pw.x
#   6. Updates candidates.ndjson with all DFT energies
#   7. Prints a ranked summary of DFT results
#
# Prerequisites:
#   - Step 3 (03_sanity_check.py) must have completed (candidates.ndjson exists)
#   - Quantum ESPRESSO must be installed and pw.x must be on PATH
#       Ubuntu/Debian: sudo apt-get install quantum-espresso
#       From source:   https://www.quantum-espresso.org/
#   - Pseudopotential UPF files must be present at the path in config.yaml:
#       pseudopotentials.pseudo_dir (default: ./pseudo/)
#       Run: python scripts/download_pseudopotentials.py
#   - pymatgen-io-espresso must be installed:
#       pip install git+https://github.com/Griffin-Group/pymatgen-io-espresso
#   - config.yaml must exist at the project root
#
# Usage:
#   python scripts/04_dft.py                     # full workflow: SCF + relax
#   python scripts/04_dft.py --stage scf         # SCF screening only
#   python scripts/04_dft.py --stage relax       # relax only (needs prior SCF)
#   python scripts/04_dft.py --n-scf 8           # override n_candidates for SCF
#   python scripts/04_dft.py --n-relax 3         # override n_candidates for relax
#   python scripts/04_dft.py --nproc 4           # MPI processes for pw.x
#   python scripts/04_dft.py --config /path/to/config.yaml
#   python scripts/04_dft.py --dry-run           # print plan, do not run QE
#
# Outputs:
#   output/dft/scf/scf_0/qe.in       (QE SCF input for candidate 0)
#   output/dft/scf/scf_0/qe.out      (QE SCF output)
#   output/dft/relax/relax_0/qe.in   (QE relax input for candidate 0)
#   output/dft/relax/relax_0/qe.out  (QE relax output)
#   output/candidates.ndjson          (DFT energies merged in)
#   output/dft_scf_summary.txt        (SCF ranking table)
#   output/dft_relax_summary.txt      (relax ranking table)
#   output/logs/04_dft.log
#
# K-point convergence notes (copied from config.yaml comments):
#   - SCF [2,2,2] is for RANKING ONLY — not suitable for hull analysis
#   - Relax [3,3,3] is a compromise for ionic relaxation
#   - For production-quality hull energies, use ≥ [4,4,4] and re-run SCF
#
# Device / parallelism:
#   - On a GPU machine, set --nproc to the number of available CPU cores
#   - QE pw.x is MPI-parallel (CPU), not GPU-accelerated in the default build
#   - Example for a 16-core machine: python scripts/04_dft.py --nproc 16
# =============================================================================

from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path
from typing import List, Optional

# ---------------------------------------------------------------------------
# Ensure the package root is on sys.path when run as a standalone script
# ---------------------------------------------------------------------------
_SCRIPT_DIR  = Path(__file__).resolve().parent
_PROJECT_DIR = _SCRIPT_DIR.parent
if str(_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(_PROJECT_DIR))

from maccai_battery.config import load_config, PipelineConfig
from maccai_battery.database import CandidateDatabase
from maccai_battery.utils import setup_logging


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Step 4: Quantum ESPRESSO DFT screening and ionic relaxation. "
            "Runs SCF on top ML candidates, then relaxes the lowest-energy subset."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              # Full pipeline (recommended):
              python scripts/04_dft.py

              # SCF only (useful for quick screening):
              python scripts/04_dft.py --stage scf

              # Relax only (after SCF has been run):
              python scripts/04_dft.py --stage relax

              # Override candidate counts and use 8 MPI processes:
              python scripts/04_dft.py --n-scf 12 --n-relax 4 --nproc 8

              # Dry-run: print what would be done without running QE:
              python scripts/04_dft.py --dry-run
        """),
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
        "--stage",
        choices=["scf", "relax", "all"],
        default="all",
        help=(
            "Which DFT stage to run. "
            "'scf'   → SCF screening only. "
            "'relax' → ionic relaxation only (requires prior SCF data in the DB). "
            "'all'   → SCF then relax (default)."
        ),
    )
    parser.add_argument(
        "--n-scf",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Number of ML-relaxed candidates to pass to QE SCF. "
            "Overrides dft_screening.n_candidates in config.yaml."
        ),
    )
    parser.add_argument(
        "--n-relax",
        type=int,
        default=None,
        metavar="M",
        help=(
            "Number of SCF-ranked candidates to pass to QE ionic relaxation. "
            "Overrides dft_relax.n_candidates in config.yaml."
        ),
    )
    parser.add_argument(
        "--nproc",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Number of MPI processes for pw.x. "
            "Overrides both dft_screening.nproc and dft_relax.nproc in config.yaml. "
            "Use the number of available CPU cores on a GPU machine."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Print the DFT plan (candidate count, k-points, nproc, directories) "
            "without actually running pw.x. Useful for verifying the setup."
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

def _print_plan(cfg: PipelineConfig, stage: str, logger) -> None:
    """Log a human-readable summary of the DFT plan."""
    scf = cfg.dft_screening
    rel = cfg.dft_relax
    pseudo = cfg.pseudopotentials

    logger.info("")
    logger.info("DFT Plan")
    logger.info("  Chemical system        : %s", cfg.generation.chemical_system)
    logger.info("  Pseudopotential dir    : %s", pseudo.pseudo_dir)
    for el, fname in pseudo.files.items():
        logger.info("    %s → %s", el, fname)
    logger.info("")

    if stage in ("scf", "all"):
        logger.info("  SCF Screening")
        logger.info("    Candidates           : %d", scf.n_candidates)
        logger.info("    MPI processes        : %d", scf.nproc)
        logger.info("    k-points             : %s", scf.kpoints)
        logger.info("    ecutwfc / ecutrho    : %.0f / %.0f Ry", scf.ecutwfc, scf.ecutrho)
        logger.info("    smearing             : %s @ %.3f Ry", scf.smearing, scf.degauss)
        logger.info("    conv_thr             : %.0e Ry", scf.conv_thr)
        logger.info("    spin_polarised       : %s", scf.spin_polarised)
        logger.info("    starting_magnetization:")
        for el, mag in scf.starting_magnetization.items():
            logger.info("      %s: %.2f", el, mag)
        logger.info("")

    if stage in ("relax", "all"):
        logger.info("  Ionic Relaxation")
        logger.info("    Candidates           : %d", rel.n_candidates)
        logger.info("    MPI processes        : %d", rel.nproc)
        logger.info("    k-points             : %s", rel.kpoints)
        logger.info("    ecutwfc / ecutrho    : %.0f / %.0f Ry", rel.ecutwfc, rel.ecutrho)
        logger.info("    ion_dynamics         : %s", rel.ion_dynamics)
        logger.info("    nstep                : %d", rel.nstep)
        logger.info("    conv_thr             : %.0e Ry", rel.conv_thr)
        logger.info("    spin_polarised       : %s", rel.spin_polarised)
        logger.info("")

    scf_dir   = cfg.project.dir_path("dft_scf")
    relax_dir = cfg.project.dir_path("dft_relax")
    logger.info("  Output directories")
    logger.info("    SCF   → %s", scf_dir)
    logger.info("    Relax → %s", relax_dir)
    logger.info("")


def _print_scf_results(results: List, logger) -> None:
    """Log a ranked SCF results table."""
    successful = [r for r in results if r.success]
    failed     = [r for r in results if not r.success]

    logger.info("")
    logger.info("SCF Results — ranked by energy")
    logger.info("  %-5s  %-20s  %12s  %12s  %s",
                "Rank", "Formula", "E (eV)", "E/atom (eV)", "Converged")
    logger.info("  %s", "-" * 72)

    sorted_ok = sorted(
        successful,
        key=lambda r: r.energy_eV_per_atom if r.energy_eV_per_atom is not None else float("inf"),
    )
    for rank, r in enumerate(sorted_ok, 1):
        e_str  = f"{r.energy_eV:.4f}"  if r.energy_eV is not None         else "N/A"
        ea_str = f"{r.energy_eV_per_atom:.4f}" if r.energy_eV_per_atom is not None else "N/A"
        logger.info("  %-5d  %-20s  %12s  %12s  %s",
                    rank, r.formula, e_str, ea_str, "✓" if r.converged else "✗")

    if failed:
        logger.warning("")
        logger.warning("  Failed SCF runs (%d):", len(failed))
        for r in failed:
            logger.warning("    %s: %s", r.formula, r.error)
    logger.info("")


def _print_relax_results(results: List, logger) -> None:
    """Log a ranked relax results table."""
    successful = [r for r in results if r.success]
    failed     = [r for r in results if not r.success]

    logger.info("")
    logger.info("Relax Results — ranked by final DFT energy")
    logger.info("  %-5s  %-20s  %12s  %12s",
                "Rank", "Formula", "E (eV)", "E/atom (eV)")
    logger.info("  %s", "-" * 60)

    sorted_ok = sorted(
        successful,
        key=lambda r: r.energy_eV_per_atom if r.energy_eV_per_atom is not None else float("inf"),
    )
    for rank, r in enumerate(sorted_ok, 1):
        e_str  = f"{r.energy_eV:.4f}"  if r.energy_eV is not None         else "N/A"
        ea_str = f"{r.energy_eV_per_atom:.4f}" if r.energy_eV_per_atom is not None else "N/A"
        logger.info("  %-5d  %-20s  %12s  %12s", rank, r.formula, e_str, ea_str)

    if failed:
        logger.warning("")
        logger.warning("  Failed relax runs (%d):", len(failed))
        for r in failed:
            logger.warning("    %s: %s", r.formula, r.error)
    logger.info("")


def _write_summary_file(results: List, out_path: Path, title: str) -> None:
    """Write a plain-text summary table to *out_path*."""
    lines = [
        title,
        "=" * len(title),
        "",
        f"{'Rank':<5}  {'ID':<14}  {'Formula':<20}  {'E (eV)':>14}  {'E/atom (eV)':>14}  {'OK':>5}",
        "-" * 80,
    ]

    successful = sorted(
        [r for r in results if r.success],
        key=lambda r: r.energy_eV_per_atom if r.energy_eV_per_atom is not None else float("inf"),
    )
    failed = [r for r in results if not r.success]

    for rank, r in enumerate(successful, 1):
        e_str  = f"{r.energy_eV:.6f}"         if r.energy_eV is not None         else "N/A"
        ea_str = f"{r.energy_eV_per_atom:.6f}" if r.energy_eV_per_atom is not None else "N/A"
        ok     = "yes" if getattr(r, "converged", True) else "no"
        lines.append(
            f"{rank:<5}  {r.candidate_id:<14}  {r.formula:<20}  {e_str:>14}  {ea_str:>14}  {ok:>5}"
        )

    if failed:
        lines += ["", f"Failed ({len(failed)}):", "-" * 40]
        for r in failed:
            lines.append(f"  {r.candidate_id}  {r.formula}:  {r.error}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    # ------------------------------------------------------------------
    # Load config (apply CLI overrides before DFT workflow reads it)
    # ------------------------------------------------------------------
    cfg = load_config(args.config)

    if args.n_scf is not None:
        cfg.dft_screening.n_candidates = args.n_scf
    if args.n_relax is not None:
        cfg.dft_relax.n_candidates = args.n_relax
    if args.nproc is not None:
        cfg.dft_screening.nproc = args.nproc
        cfg.dft_relax.nproc     = args.nproc

    # ------------------------------------------------------------------
    # Set up logging
    # ------------------------------------------------------------------
    log_dir  = cfg.project.dir_path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "04_dft.log"

    logger = setup_logging(level=args.log_level, log_file=log_file)

    logger.info("=" * 60)
    logger.info("MACCAI Battery Pipeline — Step 4: DFT (QE)")
    logger.info("=" * 60)
    logger.info("Config file  : %s", args.config or "(default)")
    logger.info("Stage        : %s", args.stage)
    logger.info("Dry run      : %s", args.dry_run)
    logger.info("Log file     : %s", log_file)

    # ------------------------------------------------------------------
    # Verify the candidate database exists
    # ------------------------------------------------------------------
    db = CandidateDatabase(cfg)
    if not db.exists:
        logger.error(
            "Candidate database not found: %s\n"
            "Run step 3 (03_sanity_check.py) first.",
            db.path,
        )
        sys.exit(1)

    n_candidates = db.count()
    logger.info("Candidate database : %s (%d records)", db.path, n_candidates)

    if n_candidates == 0:
        logger.error("No candidates in database. Did step 3 complete successfully?")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Validate pseudopotential files exist before committing to a long run
    # ------------------------------------------------------------------
    pseudo_dir_str = cfg.pseudopotentials.pseudo_dir
    pseudo_dir = Path(pseudo_dir_str)
    if not pseudo_dir.is_absolute():
        pseudo_dir = (cfg.project._root / pseudo_dir_str).resolve()

    if not pseudo_dir.exists():
        logger.error(
            "Pseudopotential directory not found: %s\n"
            "Run: python scripts/download_pseudopotentials.py\n"
            "Or update pseudopotentials.pseudo_dir in config.yaml.",
            pseudo_dir,
        )
        sys.exit(1)

    missing_pseudos = []
    for el, fname in cfg.pseudopotentials.files.items():
        upf = pseudo_dir / fname
        if not upf.exists():
            missing_pseudos.append(f"  {el}: {upf}")
    if missing_pseudos:
        logger.error(
            "Missing pseudopotential files:\n%s\n"
            "Run: python scripts/download_pseudopotentials.py",
            "\n".join(missing_pseudos),
        )
        sys.exit(1)

    logger.info("Pseudopotential dir: %s (all files present)", pseudo_dir)

    # ------------------------------------------------------------------
    # Print DFT plan
    # ------------------------------------------------------------------
    _print_plan(cfg, args.stage, logger)

    if args.dry_run:
        logger.info("[DRY RUN] Exiting without running QE.")
        logger.info("")
        logger.info("To run for real:")
        logger.info("  python scripts/04_dft.py")
        sys.exit(0)

    # ------------------------------------------------------------------
    # Check QE is available
    # ------------------------------------------------------------------
    import shutil
    if shutil.which("pw.x") is None:
        logger.error(
            "pw.x not found on PATH. Install Quantum ESPRESSO:\n"
            "  Ubuntu/Debian: sudo apt-get install quantum-espresso\n"
            "  From source:   https://www.quantum-espresso.org/\n"
            "Or activate the conda environment that has QE installed."
        )
        sys.exit(1)

    # ------------------------------------------------------------------
    # Check pymatgen-io-espresso is importable
    # ------------------------------------------------------------------
    try:
        import importlib
        importlib.import_module("pymatgen.io.espresso")
    except ImportError:
        logger.error(
            "pymatgen-io-espresso is not installed.\n"
            "Install it with:\n"
            "  pip install git+https://github.com/Griffin-Group/pymatgen-io-espresso"
        )
        sys.exit(1)

    # ------------------------------------------------------------------
    # Import DFT workflow (deferred so errors above are clean)
    # ------------------------------------------------------------------
    from maccai_battery.dft import DFTWorkflow

    workflow = DFTWorkflow(cfg)

    # ------------------------------------------------------------------
    # Run SCF screening
    # ------------------------------------------------------------------
    scf_results = []
    if args.stage in ("scf", "all"):
        logger.info("=" * 60)
        logger.info("SCF Screening — running %d QE SCF calculations ...",
                    cfg.dft_screening.n_candidates)
        logger.info("=" * 60)

        try:
            scf_results = workflow.run_scf_screening()
        except Exception as exc:
            logger.exception("SCF screening run failed unexpectedly: %s", exc)
            sys.exit(1)

        _print_scf_results(scf_results, logger)

        n_ok   = sum(1 for r in scf_results if r.success)
        n_fail = len(scf_results) - n_ok
        logger.info(
            "SCF complete: %d succeeded, %d failed.", n_ok, n_fail
        )

        # Write SCF summary file
        scf_summary_path = cfg.project.output_path / "dft_scf_summary.txt"
        _write_summary_file(scf_results, scf_summary_path, "DFT SCF Screening Summary")
        logger.info("SCF summary written to: %s", scf_summary_path)

        if n_ok == 0 and args.stage == "all":
            logger.error(
                "No SCF calculations succeeded. Cannot proceed to relaxation.\n"
                "Check QE output files in: %s",
                cfg.project.dir_path("dft_scf"),
            )
            sys.exit(1)

    # ------------------------------------------------------------------
    # Run ionic relaxation
    # ------------------------------------------------------------------
    relax_results = []
    if args.stage in ("relax", "all"):
        n_for_relax = cfg.dft_relax.n_candidates

        logger.info("=" * 60)
        logger.info("Ionic Relaxation — running %d QE relax calculations ...",
                    n_for_relax)
        logger.info("=" * 60)

        # If we just ran SCF, pass results directly; otherwise load from DB
        scf_input = scf_results if scf_results else None

        try:
            relax_results = workflow.run_dft_relax(scf_results=scf_input)
        except Exception as exc:
            logger.exception("DFT relaxation run failed unexpectedly: %s", exc)
            sys.exit(1)

        _print_relax_results(relax_results, logger)

        n_ok   = sum(1 for r in relax_results if r.success)
        n_fail = len(relax_results) - n_ok
        logger.info(
            "Relaxation complete: %d succeeded, %d failed.", n_ok, n_fail
        )

        # Write relax summary file
        relax_summary_path = cfg.project.output_path / "dft_relax_summary.txt"
        _write_summary_file(relax_results, relax_summary_path, "DFT Ionic Relaxation Summary")
        logger.info("Relax summary written to: %s", relax_summary_path)

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    logger.info("")
    logger.info("=" * 60)
    logger.info("DFT Step Complete")
    logger.info("=" * 60)

    if scf_results:
        n_scf_ok = sum(1 for r in scf_results if r.success)
        logger.info("  SCF     : %d / %d succeeded", n_scf_ok, len(scf_results))
        if n_scf_ok > 0:
            best_scf = min(
                (r for r in scf_results if r.success and r.energy_eV_per_atom is not None),
                key=lambda r: r.energy_eV_per_atom,  # type: ignore[arg-type]
                default=None,
            )
            if best_scf:
                logger.info(
                    "  Best SCF  : %s  %.4f eV/atom",
                    best_scf.formula, best_scf.energy_eV_per_atom,
                )

    if relax_results:
        n_rel_ok = sum(1 for r in relax_results if r.success)
        logger.info("  Relax   : %d / %d succeeded", n_rel_ok, len(relax_results))
        if n_rel_ok > 0:
            best_rel = min(
                (r for r in relax_results if r.success and r.energy_eV_per_atom is not None),
                key=lambda r: r.energy_eV_per_atom,  # type: ignore[arg-type]
                default=None,
            )
            if best_rel:
                logger.info(
                    "  Best relax: %s  %.4f eV/atom",
                    best_rel.formula, best_rel.energy_eV_per_atom,
                )

    logger.info("")
    logger.info("  Candidate DB : %s", db.path)
    logger.info("  SCF outputs  : %s", cfg.project.dir_path("dft_scf"))
    logger.info("  Relax outputs: %s", cfg.project.dir_path("dft_relax"))
    logger.info("  Log file     : %s", log_file)
    logger.info("")
    logger.info("Next step:")
    logger.info("  python scripts/06_hull_analysis.py")
    logger.info(
        "  (or run 05_merge_dft_results.py first if you need to re-merge outputs)"
    )
    logger.info("")


if __name__ == "__main__":
    main()
