#!/usr/bin/env python3
# =============================================================================
# 06_hull_analysis.py — Materials Project convex hull stability analysis
# =============================================================================
# Step 6 of the MACCAI battery materials discovery pipeline.
#
# What this script does:
#   1. Loads DFT-relaxed energies from candidates.ndjson (written by step 5)
#   2. Queries the Materials Project API for all reference phases in the
#      target chemical space (e.g. Li-Fe-P-O)
#   3. Constructs a pymatgen PhaseDiagram from MP reference data
#   4. Computes ΔH_hull (energy above convex hull) for every candidate
#      with a completed DFT relaxation energy
#   5. Writes hull results back into candidates.ndjson under "hull_analysis"
#   6. Prints a ranked stability table
#
# Prerequisites:
#   - Step 5 (05_merge_dft_results.py) must have completed so that
#     candidates.ndjson contains dft_jobs.relax.energy_eV_per_atom values
#   - pip install mp-api pymatgen
#   - Set MP_API_KEY environment variable:
#       export MP_API_KEY="your_key_here"
#     Or pass --api-key on the command line.
#     Get your free key at: https://materialsproject.org/api
#
# Usage:
#   export MP_API_KEY="your_key_here"
#   python scripts/06_hull_analysis.py
#
#   python scripts/06_hull_analysis.py --config /path/to/config.yaml
#   python scripts/06_hull_analysis.py --energy-source scf   # use SCF energies
#   python scripts/06_hull_analysis.py --threshold 0.05      # tighter stability cutoff
#   python scripts/06_hull_analysis.py --dry-run             # print, do not write
#   python scripts/06_hull_analysis.py --check-key           # just test API key
#
# Outputs:
#   output/candidates.ndjson          (hull_analysis field updated per record)
#   output/hull_analysis_report.txt   (human-readable ranked table)
#   output/logs/06_hull_analysis.log
#
# Interpreting ΔH_hull:
#   0.000 eV/atom  → on the convex hull  (thermodynamically stable)
#   0.000–0.050    → likely synthesisable (low-lying metastable)
#   0.050–0.100    → marginal stability (sometimes synthesisable)
#   > 0.100        → probably too metastable for practical use
#
# Why hull analysis matters for battery materials:
#   - LiFePO4 (olivine) itself sits ON the hull → excellent indicator
#   - Candidates with ΔH_hull < 0.05 eV are worth further high-accuracy DFT
#   - Candidates with ΔH_hull > 0.1 eV are unlikely to form under synthesis
#     conditions and should be deprioritised
# =============================================================================

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Ensure package is importable when run as a script
# ---------------------------------------------------------------------------
_SCRIPT_DIR  = Path(__file__).resolve().parent
_PROJECT_DIR = _SCRIPT_DIR.parent
if str(_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(_PROJECT_DIR))

from maccai_battery.config import load_config
from maccai_battery.database import CandidateDatabase
from maccai_battery.hull import HullAnalyzer, check_mp_api_key, HullResult
from maccai_battery.utils import setup_logging


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Step 6: Materials Project convex hull stability analysis.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to config.yaml (default: project root).",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help=(
            "Materials Project API key. "
            "Defaults to the MP_API_KEY environment variable. "
            "Get your free key at https://materialsproject.org/api"
        ),
    )
    parser.add_argument(
        "--energy-source",
        choices=["relax", "scf"],
        default="relax",
        help=(
            "Which DFT stage to use for hull calculation. "
            "'relax' uses fully-relaxed energies (recommended). "
            "'scf' uses single-point SCF energies (less accurate)."
        ),
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help=(
            "Energy-above-hull threshold in eV/atom for labelling a "
            "candidate as 'stable'. Overrides hull.stability_threshold_eV "
            "in config.yaml. Default: 0.1 eV/atom."
        ),
    )
    parser.add_argument(
        "--chemical-system",
        type=str,
        default=None,
        help=(
            "Dash-separated chemical system for MP reference data, "
            "e.g. 'Li-Fe-P-O'. Defaults to generation.chemical_system "
            "in config.yaml."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute hull distances and print results, but do NOT update the database.",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Skip backing up candidates.ndjson before writing hull results.",
    )
    parser.add_argument(
        "--check-key",
        action="store_true",
        help="Test the MP API key and exit without running the full analysis.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Console log verbosity.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------

def _write_report(
    report_path: Path,
    results: List[HullResult],
    threshold: float,
    energy_source: str,
    chemical_system: str,
) -> None:
    """Write a plain-text hull analysis report.

    Parameters
    ----------
    report_path : Path
    results : list of HullResult
    threshold : float
        Stability threshold (eV/atom).
    energy_source : str
    chemical_system : str
    """
    successful = sorted(
        [r for r in results if r.success],
        key=lambda r: r.e_above_hull_eV_per_atom,  # type: ignore[arg-type]
    )
    failed = [r for r in results if not r.success]

    lines = [
        "=" * 78,
        "  MACCAI Battery Pipeline — Hull Analysis Report",
        "=" * 78,
        "",
        f"  Chemical system      : {chemical_system}",
        f"  DFT energy source    : {energy_source}",
        f"  Stability threshold  : ΔH ≤ {threshold:.3f} eV/atom",
        f"  Candidates evaluated : {len(results)}",
        f"  Stable (ΔH ≤ {threshold:.3f} eV/a) : {sum(1 for r in successful if r.is_stable)}",
        f"  Metastable           : {sum(1 for r in successful if not r.is_stable)}",
        f"  Failed               : {len(failed)}",
        "",
        "  Ranking (by ΔH_hull):",
        "",
        f"  {'Rank':>4}  {'ID':<12}  {'Formula':<16}  "
        f"{'E_DFT (eV/a)':>14}  {'ΔH_hull (eV/a)':>16}  {'Status':<14}  Competing phases",
        "  " + "-" * 110,
    ]

    for rank, r in enumerate(successful, start=1):
        status = "STABLE" if r.is_stable else "metastable"
        e_dft  = f"{r.dft_energy_eV_per_atom:.6f}" if r.dft_energy_eV_per_atom is not None else "N/A"
        e_hull = f"{r.e_above_hull_eV_per_atom:.6f}" if r.e_above_hull_eV_per_atom is not None else "N/A"
        phases = ", ".join(r.mp_stable_phases[:4])
        if len(r.mp_stable_phases) > 4:
            phases += " ..."

        lines.append(
            f"  {rank:>4}  {r.candidate_id:<12}  {r.formula:<16}  "
            f"{e_dft:>14}  {e_hull:>16}  {status:<14}  {phases}"
        )

    if failed:
        lines += [
            "",
            f"  Failed ({len(failed)}):",
        ]
        for r in failed:
            lines.append(f"    {r.candidate_id}: {r.error}")

    lines += [
        "",
        "=" * 78,
        "",
        "  NOTES:",
        "  ------",
        "  ΔH_hull = 0.000 eV/atom  → on the convex hull (thermodynamically stable)",
        "  ΔH_hull ≤ 0.050 eV/atom  → likely synthesisable",
        "  ΔH_hull ≤ 0.100 eV/atom  → marginal metastability",
        "  ΔH_hull > 0.100 eV/atom  → probably not synthesisable under standard conditions",
        "",
        "  Reference: Materials Project convex hull (PBE, VASP PAW)",
        "  Candidate energies: Quantum ESPRESSO PBE (this work)",
        "",
        "  IMPORTANT: Direct comparison of QE and MP energies assumes the same",
        "  pseudopotential family and DFT settings. Systematic offsets may exist.",
        "  For publication-quality results, re-compute with the same code/settings.",
        "",
        "=" * 78,
    ]

    report_path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Config monkey-patch for threshold override
# ---------------------------------------------------------------------------

class _ThresholdPatch:
    """Minimal stand-in so HullAnalyzer can read stability_threshold_eV."""

    def __init__(self, threshold: float) -> None:
        self.stability_threshold_eV = threshold


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
    log_file = log_dir / "06_hull_analysis.log"

    logger = setup_logging(level=args.log_level, log_file=log_file)

    logger.info("=" * 60)
    logger.info("MACCAI Battery Pipeline — Step 6: Hull Analysis")
    logger.info("=" * 60)
    logger.info("Config         : %s", args.config or "(default)")
    logger.info("Energy source  : %s", args.energy_source)
    logger.info("Dry run        : %s", args.dry_run)
    logger.info("Log file       : %s", log_file)

    # ------------------------------------------------------------------
    # Resolve API key
    # ------------------------------------------------------------------
    api_key = args.api_key or os.environ.get("MP_API_KEY")

    if not api_key:
        logger.error(
            "No Materials Project API key provided.\n"
            "  Set it with:  export MP_API_KEY='your_key_here'\n"
            "  Or pass it:   --api-key your_key_here\n"
            "  Get a free key at: https://materialsproject.org/api"
        )
        sys.exit(1)

    # ------------------------------------------------------------------
    # Check-key mode: just test and exit
    # ------------------------------------------------------------------
    if args.check_key:
        logger.info("Testing MP API key ...")
        try:
            ok = check_mp_api_key(api_key)
            if ok:
                logger.info("API key is valid!")
            else:
                logger.error("API key validation failed.")
                sys.exit(1)
        except Exception as exc:
            logger.error("API key check failed: %s", exc)
            sys.exit(1)
        sys.exit(0)

    # ------------------------------------------------------------------
    # Check for mp-api import
    # ------------------------------------------------------------------
    try:
        import mp_api  # noqa: F401
    except ImportError:
        logger.error(
            "mp-api is not installed.\n"
            "Install it with:  pip install mp-api"
        )
        sys.exit(1)

    # ------------------------------------------------------------------
    # Determine chemical system and threshold
    # ------------------------------------------------------------------
    chemical_system = args.chemical_system or cfg.generation.chemical_system
    threshold = (
        args.threshold
        if args.threshold is not None
        else getattr(getattr(cfg, "hull", None), "stability_threshold_eV", 0.1)
    )

    logger.info("Chemical system : %s", chemical_system)
    logger.info("Threshold       : %.3f eV/atom", threshold)

    # Inject threshold into a minimal config stub so HullAnalyzer reads it
    cfg.hull = _ThresholdPatch(threshold)  # type: ignore[attr-defined]
    # Also override chemical system if specified on CLI
    if args.chemical_system:
        cfg.generation.chemical_system = args.chemical_system

    # ------------------------------------------------------------------
    # Load candidate database
    # ------------------------------------------------------------------
    db = CandidateDatabase(cfg)

    if not db.exists:
        logger.error(
            "Candidate database not found: %s\n"
            "Run step 3 (03_sanity_check.py) and step 5 "
            "(05_merge_dft_results.py) first.",
            db.path,
        )
        sys.exit(1)

    candidates = db.load_all()
    logger.info("Loaded %d candidate records.", len(candidates))

    # Count how many have a DFT energy
    n_with_energy = sum(
        1 for r in candidates
        if r.get("dft_jobs", {})
           .get(args.energy_source, {})
           .get("energy_eV_per_atom") is not None
    )
    logger.info(
        "Candidates with %s energy: %d / %d",
        args.energy_source, n_with_energy, len(candidates),
    )

    if n_with_energy == 0:
        logger.error(
            "No candidates have a completed '%s' DFT energy.\n"
            "Run step 4 (04_dft.py) or step 5 (05_merge_dft_results.py) "
            "to populate DFT energies first.",
            args.energy_source,
        )
        sys.exit(1)

    # ------------------------------------------------------------------
    # Backup database before writing
    # ------------------------------------------------------------------
    if not args.dry_run and not args.no_backup:
        backup = db.backup(suffix="pre_hull")
        logger.info("Database backed up to: %s", backup)

    # ------------------------------------------------------------------
    # Run hull analysis
    # ------------------------------------------------------------------
    logger.info("")
    logger.info("Initialising HullAnalyzer ...")

    try:
        analyzer = HullAnalyzer(cfg, api_key=api_key)
    except EnvironmentError as exc:
        logger.error("%s", exc)
        sys.exit(1)
    except ImportError as exc:
        logger.error("%s", exc)
        sys.exit(1)

    logger.info("Querying Materials Project for '%s' reference phases ...", chemical_system)
    logger.info("(This may take 10–30 seconds depending on network speed.)")

    try:
        results = analyzer.run(candidates, energy_source=args.energy_source)
    except Exception as exc:
        logger.exception("Hull analysis failed: %s", exc)
        sys.exit(1)

    if not results:
        logger.warning("No hull results produced — check that DFT energies exist.")
        sys.exit(0)

    # ------------------------------------------------------------------
    # Print ranked table
    # ------------------------------------------------------------------
    analyzer.print_ranking(results)

    # ------------------------------------------------------------------
    # Write report
    # ------------------------------------------------------------------
    report_path = cfg.project.output_path / "hull_analysis_report.txt"

    if not args.dry_run:
        _write_report(
            report_path     = report_path,
            results         = results,
            threshold       = threshold,
            energy_source   = args.energy_source,
            chemical_system = chemical_system,
        )
        logger.info("Report written to: %s", report_path)

    # ------------------------------------------------------------------
    # Update database
    # ------------------------------------------------------------------
    if args.dry_run:
        logger.info("DRY RUN — hull results not written to database.")
    else:
        n_updated = analyzer.update_database(results, db)
        logger.info("Updated %d records in database.", n_updated)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    successful = [r for r in results if r.success]
    stable     = [r for r in successful if r.is_stable]
    meta       = [r for r in successful if not r.is_stable]
    failed     = [r for r in results if not r.success]

    logger.info("")
    logger.info("=" * 60)
    logger.info("Hull analysis complete")
    logger.info("  Evaluated    : %d candidates", len(results))
    logger.info("  Stable       : %d  (ΔH ≤ %.3f eV/atom)", len(stable), threshold)
    logger.info("  Metastable   : %d  (ΔH > %.3f eV/atom)", len(meta), threshold)
    logger.info("  Failed       : %d", len(failed))

    if stable:
        best = stable[0]
        logger.info("")
        logger.info(
            "  Most stable candidate: %s (%s)  ΔH_hull = %.4f eV/atom",
            best.candidate_id, best.formula, best.e_above_hull_eV_per_atom,
        )

    logger.info("")

    if not args.dry_run:
        logger.info("  Database : %s", db.path)
        logger.info("  Report   : %s", report_path)

    logger.info("  Log      : %s", log_file)
    logger.info("")

    # Advice on next steps
    if stable:
        logger.info("Next steps for stable candidates:")
        logger.info("  1. Re-run DFT with production-quality settings (denser k-grid,")
        logger.info("     higher Ecut, tighter convergence) for publication accuracy.")
        logger.info("  2. Run phonon calculations to confirm dynamic stability.")
        logger.info("  3. Compute electronic structure (band structure, DOS).")
        logger.info("  4. If on HPC: use VASP/Quantum ESPRESSO with VESTA for viz.")
    else:
        logger.info("No candidates met the ΔH ≤ %.3f eV/atom stability criterion.", threshold)
        logger.info("Consider:")
        logger.info("  - Loosening --threshold (e.g. 0.15 eV/atom)")
        logger.info("  - Generating more structures (increase num_batches in config.yaml)")
        logger.info("  - Trying a different chemical system")

    logger.info("")


if __name__ == "__main__":
    main()
