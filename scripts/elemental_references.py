#!/usr/bin/env python3
# =============================================================================
# scripts/elemental_references.py — Step 6: Elemental QE Reference Energies
# =============================================================================
#
# Computes QE SCF energies for the elemental reference phases required by the
# hull analysis (step 7).  Runs pw.x on:
#
#   Li  — BCC metal  (1-atom cell, ibrav=3)
#   Fe  — BCC metal  (1-atom cell, spin-pol, DFT+U)
#   P   — P4 molecule (4 atoms in 10 Å box)
#   O   — O2 molecule (2 atoms in 15 Å box, triplet)
#
# After a successful run the energies are written back into config.yaml under
# elemental_references.energies_eV_per_atom so that step 7 (hull) can use them.
#
# Usage:
#   python scripts/elemental_references.py                 # all 4 elements
#   python scripts/elemental_references.py --elements Li Fe
#   python scripts/elemental_references.py --dry-run       # show inputs only
#   python scripts/elemental_references.py --nproc 4       # MPI processes
#
# Output directories:
#   output/dft/references/Li/  output/dft/references/Fe/
#   output/dft/references/P/   output/dft/references/O/
# =============================================================================

from __future__ import annotations

import argparse
import logging
import re
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Make the package importable when run from the repo root
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from maccai_battery.config import load_config
from maccai_battery.dft.elemental_inputs import make_reference_inputs, REFERENCE_N_ATOMS

logger = logging.getLogger("maccai_battery")

# Rydberg → eV conversion (exact CODATA 2018 value used by QE)
_RY_TO_EV = 13.605693122994


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python scripts/elemental_references.py",
        description=(
            "Step 6: Run QE SCF on elemental reference phases (Li, Fe, P, O)\n"
            "and store the energies in config.yaml for hull analysis."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              # Run all 4 references:
              python scripts/elemental_references.py

              # Run only Li and O:
              python scripts/elemental_references.py --elements Li O

              # Preview inputs without running QE:
              python scripts/elemental_references.py --dry-run

              # Use 4 MPI processes:
              python scripts/elemental_references.py --nproc 4
        """),
    )
    parser.add_argument(
        "--elements",
        nargs="+",
        choices=list(REFERENCE_N_ATOMS.keys()),
        default=None,
        metavar="EL",
        help="Elements to compute (default: all four Li Fe P O).",
    )
    parser.add_argument(
        "--nproc",
        type=int,
        default=1,
        metavar="N",
        help="Number of MPI processes for pw.x (default: 1).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to config.yaml (default: repo-root/config.yaml).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write input files and print commands, but do not run QE.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# QE runner
# ---------------------------------------------------------------------------

def _find_pw() -> str:
    """Return the path to pw.x, or raise RuntimeError if not found."""
    import shutil
    pw = shutil.which("pw.x")
    if pw is None:
        raise RuntimeError(
            "pw.x not found on PATH.\n"
            "Make sure Quantum ESPRESSO is installed and pw.x is on your PATH."
        )
    return pw


def _run_qe(
    run_dir: Path,
    nproc: int,
    dry_run: bool,
) -> bool:
    """Run pw.x in run_dir.  Returns True on success."""
    pw = _find_pw()
    if nproc > 1:
        cmd = ["mpirun", "-np", str(nproc), pw, "-input", "qe.in"]
    else:
        cmd = [pw, "-input", "qe.in"]

    logger.info("  Command: %s", " ".join(cmd))
    logger.info("  Working dir: %s", run_dir)

    if dry_run:
        logger.info("  [DRY RUN] skipping pw.x execution.")
        return True

    t0 = time.monotonic()
    try:
        with open(run_dir / "qe.out", "w") as fout, \
             open(run_dir / "qe.err", "w") as ferr:
            result = subprocess.run(
                cmd,
                cwd=run_dir,
                stdout=fout,
                stderr=ferr,
                check=False,
            )
        elapsed = time.monotonic() - t0

        if result.returncode != 0:
            logger.error(
                "  pw.x exited with code %d after %.1f s.",
                result.returncode, elapsed,
            )
            logger.error("  See %s for details.", run_dir / "qe.err")
            return False

        logger.info("  pw.x finished in %.1f s.", elapsed)
        return True

    except FileNotFoundError as exc:
        logger.error("  Could not launch pw.x: %s", exc)
        return False


# ---------------------------------------------------------------------------
# Output parser
# ---------------------------------------------------------------------------

_ENERGY_RE = re.compile(
    r"!\s+total energy\s*=\s*([-\d.]+)\s*Ry", re.IGNORECASE
)


def _parse_total_energy_ry(out_file: Path) -> Optional[float]:
    """Extract the final total energy (Ry) from a QE output file."""
    if not out_file.exists():
        return None
    energy = None
    for line in out_file.read_text().splitlines():
        m = _ENERGY_RE.match(line.strip())
        if m:
            energy = float(m.group(1))
    return energy  # last match = final SCF iteration


def _parse_energy_ev_per_atom(run_dir: Path, element: str) -> Optional[float]:
    out_file = run_dir / "qe.out"
    e_ry = _parse_total_energy_ry(out_file)
    if e_ry is None:
        logger.error(
            "  Could not find 'total energy' in %s — SCF may not have converged.",
            out_file,
        )
        return None
    n_atoms = REFERENCE_N_ATOMS[element]
    e_ev = e_ry * _RY_TO_EV
    e_pa = e_ev / n_atoms
    logger.info(
        "  %s: E_total = %.6f Ry = %.6f eV, n_atoms = %d → %.6f eV/atom",
        element, e_ry, e_ev, n_atoms, e_pa,
    )
    return e_pa


# ---------------------------------------------------------------------------
# config.yaml updater
# ---------------------------------------------------------------------------

def _update_config_yaml(config_path: Path, energies: Dict[str, float]) -> None:
    """Patch config.yaml in-place: update elemental_references.energies_eV_per_atom."""
    text = config_path.read_text()
    for element, e_pa in energies.items():
        # Match both "null" and existing float values on the Li/Fe/P/O lines
        pattern = re.compile(
            r"^(\s+" + re.escape(element) + r"\s*:\s*)(?:null|[-\d.eE+]+)(.*)",
            re.MULTILINE,
        )
        replacement = rf"\g<1>{e_pa:.8f}\g<2>"
        new_text, n = pattern.subn(replacement, text)
        if n == 0:
            logger.warning(
                "Could not find '%s:' entry in elemental_references section of %s. "
                "Add it manually: %s: %.8f",
                element, config_path, element, e_pa,
            )
        else:
            text = new_text
            logger.info("  Updated config.yaml: %s = %.8f eV/atom", element, e_pa)

    config_path.write_text(text)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    # Logging
    log_fmt = "%(asctime)s  %(levelname)-8s  %(name)s — %(message)s"
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format=log_fmt,
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logger.info("=" * 60)
    logger.info("MACCAI Battery Pipeline — Step 6: Elemental References")
    logger.info("=" * 60)

    # Load config
    cfg = load_config(args.config)
    config_path = cfg.project._root / "config.yaml"
    elements: List[str] = args.elements or list(REFERENCE_N_ATOMS.keys())
    logger.info("Elements    : %s", elements)
    logger.info("nproc       : %d", args.nproc)
    logger.info("Dry run     : %s", args.dry_run)
    logger.info("")

    # Generate inputs
    all_inputs = make_reference_inputs(cfg)

    # Run dir: output/dft/references/<element>/
    base_dir = cfg.project.output_path / "dft" / "references"

    energies: Dict[str, float] = {}
    failed: List[str] = []

    for element in elements:
        logger.info("")
        logger.info("─" * 50)
        logger.info("Element: %s", element)
        logger.info("─" * 50)

        run_dir = base_dir / element
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "tmp").mkdir(exist_ok=True)

        input_text, n_atoms = all_inputs[element]
        input_file = run_dir / "qe.in"
        input_file.write_text(input_text)
        logger.info("  Input written : %s", input_file)

        if args.dry_run:
            logger.info("  --- Input file preview ---")
            for line in input_text.splitlines()[:30]:
                logger.info("  %s", line)
            logger.info("  [DRY RUN] pw.x would run in: %s", run_dir)
            continue

        success = _run_qe(run_dir, args.nproc, dry_run=False)
        if not success:
            failed.append(element)
            continue

        e_pa = _parse_energy_ev_per_atom(run_dir, element)
        if e_pa is None:
            failed.append(element)
            continue

        energies[element] = e_pa

    # Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("Results")
    logger.info("=" * 60)
    for el in elements:
        if el in energies:
            logger.info("  %-2s  %.8f eV/atom  ✓", el, energies[el])
        elif args.dry_run:
            logger.info("  %-2s  (dry run — not computed)", el)
        else:
            logger.info("  %-2s  FAILED  ✗", el)

    if args.dry_run:
        logger.info("")
        logger.info("[DRY RUN] No energies computed.  Re-run without --dry-run.")
        sys.exit(0)

    if failed:
        logger.error("")
        logger.error("Failed elements: %s", failed)
        logger.error(
            "Check QE output in: %s/<element>/qe.out", base_dir
        )

    # Write successful energies back to config.yaml
    if energies:
        logger.info("")
        logger.info("Updating config.yaml with computed energies ...")
        _update_config_yaml(config_path, energies)
        logger.info(
            "config.yaml updated.  Run step 7 (hull analysis) next:\n"
            "  python main.py --step 7"
        )

    if failed:
        sys.exit(1)
    else:
        logger.info("All elemental references computed successfully.")
        sys.exit(0)


if __name__ == "__main__":
    main()
