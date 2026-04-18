#!/usr/bin/env python3
# =============================================================================
# download_pseudopotentials.py — Automated pseudopotential downloader
# =============================================================================
# Downloads PBE pseudopotential files (.UPF) required for Quantum ESPRESSO
# DFT calculations in the MACCAI battery pipeline.
#
# Sources supported:
#   - PseudoDojo  (ONCV, PAW — recommended)
#   - PSLibrary   (USPP, PAW — widely used)
#   - GBRV        (USPP — fast, good for screening)
#   - SG15        (ONCV — norm-conserving, no ultrasoft)
#
# Default target: Li-Fe-P-O system (LFP battery cathode)
# Default library: PSLibrary PAW PBE (matches the config.yaml defaults)
#
# Usage:
#   python scripts/download_pseudopotentials.py
#   python scripts/download_pseudopotentials.py --elements Li Fe P O
#   python scripts/download_pseudopotentials.py --library gbrv
#   python scripts/download_pseudopotentials.py --library sg15
#   python scripts/download_pseudopotentials.py --out-dir /my/pseudo/dir
#   python scripts/download_pseudopotentials.py --dry-run     # list URLs only
#   python scripts/download_pseudopotentials.py --check       # verify existing files
#
# Outputs:
#   <out_dir>/<element>.UPF   for each requested element
#
# After downloading:
#   Update pseudopotentials.pseudo_dir in config.yaml to point to <out_dir>.
#   Update pseudopotentials.files in config.yaml to match the downloaded filenames.
#
# Notes:
#   - Always verify pseudopotentials against known test cases before production use.
#   - PAW pseudopotentials are generally more accurate but slower than USPP.
#   - For spin-polarized Fe calculations, use the spin-polarized (spn) variant.
# =============================================================================

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Pseudopotential catalogue
# ---------------------------------------------------------------------------
# Each entry: (filename, url, sha256_hex_or_None, description)
# sha256 is optional — if provided the download is verified.
# ---------------------------------------------------------------------------

@dataclass
class PseudoEntry:
    """One pseudopotential file entry."""
    element:     str
    filename:    str
    url:         str
    library:     str          # e.g. "pslibrary", "gbrv", "sg15"
    pp_type:     str          # "PAW", "USPP", "ONCV"
    functional:  str          = "PBE"
    sha256:      Optional[str] = None
    notes:       str           = ""


# ---------------------------------------------------------------------------
# PSLibrary (PAW PBE) — recommended for spin-polarised Fe calculations
# ---------------------------------------------------------------------------
_PSLIBRARY_BASE = (
    "https://pseudopotentials.quantum-espresso.org/upf_files"
)

PSLIBRARY_PAW: List[PseudoEntry] = [
    PseudoEntry(
        element  = "Li",
        filename = "Li.pbe-s-kjpaw_psl.1.0.0.UPF",
        url      = f"{_PSLIBRARY_BASE}/Li.pbe-s-kjpaw_psl.1.0.0.UPF",
        library  = "pslibrary",
        pp_type  = "PAW",
        notes    = "Li PAW PBE (PSLibrary 1.0). Consistent with Fe, P, O.",
    ),
    PseudoEntry(
        element  = "Fe",
        filename = "Fe.pbe-spn-kjpaw_psl.0.2.1.UPF",
        url      = f"{_PSLIBRARY_BASE}/Fe.pbe-spn-kjpaw_psl.0.2.1.UPF",
        library  = "pslibrary",
        pp_type  = "PAW",
        notes    = (
            "Fe PAW PBE spin-polarised (PSLibrary 0.2.1). "
            "Essential for Fe magnetism in LFP-type systems."
        ),
    ),
    PseudoEntry(
        element  = "P",
        filename = "P.pbe-n-kjpaw_psl.0.1.UPF",
        url      = f"{_PSLIBRARY_BASE}/P.pbe-n-kjpaw_psl.0.1.UPF",
        library  = "pslibrary",
        pp_type  = "PAW",
        notes    = "P PAW PBE (PSLibrary 0.1). Standard phosphorus pseudopotential.",
    ),
    PseudoEntry(
        element  = "O",
        filename = "O.pbe-n-kjpaw_psl.0.1.UPF",
        url      = f"{_PSLIBRARY_BASE}/O.pbe-n-kjpaw_psl.0.1.UPF",
        library  = "pslibrary",
        pp_type  = "PAW",
        notes    = "O PAW PBE (PSLibrary 0.1). Standard oxygen pseudopotential.",
    ),
]

# ---------------------------------------------------------------------------
# GBRV (USPP PBE) — faster, good for large-scale screening
# ---------------------------------------------------------------------------
_GBRV_BASE = (
    "https://www.physics.rutgers.edu/gbrv/all_pbe.tgz"
)

# GBRV ships as a tarball; individual file URLs via QE pseudopotential mirror
_GBRV_MIRROR = "https://pseudopotentials.quantum-espresso.org/upf_files"

GBRV_USPP: List[PseudoEntry] = [
    PseudoEntry(
        element  = "Li",
        filename = "li_pbe_v1.uspp.F.UPF",
        url      = f"{_GBRV_MIRROR}/li_pbe_v1.uspp.F.UPF",
        library  = "gbrv",
        pp_type  = "USPP",
        notes    = "Li USPP PBE (GBRV 1.2). Faster than PAW for Li.",
    ),
    PseudoEntry(
        element  = "Fe",
        filename = "fe_pbe_v1.5.uspp.F.UPF",
        url      = f"{_GBRV_MIRROR}/fe_pbe_v1.5.uspp.F.UPF",
        library  = "gbrv",
        pp_type  = "USPP",
        notes    = (
            "Fe USPP PBE (GBRV 1.5). Note: not spin-polarised by default; "
            "spin must be initialised via starting_magnetization in QE input."
        ),
    ),
    PseudoEntry(
        element  = "P",
        filename = "p_pbe_v1.5.uspp.F.UPF",
        url      = f"{_GBRV_MIRROR}/p_pbe_v1.5.uspp.F.UPF",
        library  = "gbrv",
        pp_type  = "USPP",
        notes    = "P USPP PBE (GBRV 1.5).",
    ),
    PseudoEntry(
        element  = "O",
        filename = "o_pbe_v1.2.uspp.F.UPF",
        url      = f"{_GBRV_MIRROR}/o_pbe_v1.2.uspp.F.UPF",
        library  = "gbrv",
        pp_type  = "USPP",
        notes    = "O USPP PBE (GBRV 1.2).",
    ),
]

# ---------------------------------------------------------------------------
# SG15 (ONCV PBE) — norm-conserving, good transferability
# ---------------------------------------------------------------------------
_SG15_BASE = (
    "http://www.quantum-simulation.org/potentials/sg15_oncv/upf"
)

SG15_ONCV: List[PseudoEntry] = [
    PseudoEntry(
        element  = "Li",
        filename = "Li_ONCV_PBE-1.2.upf",
        url      = f"{_SG15_BASE}/Li_ONCV_PBE-1.2.upf",
        library  = "sg15",
        pp_type  = "ONCV",
        notes    = "Li ONCV PBE (SG15). Norm-conserving, high transferability.",
    ),
    PseudoEntry(
        element  = "Fe",
        filename = "Fe_ONCV_PBE-1.2.upf",
        url      = f"{_SG15_BASE}/Fe_ONCV_PBE-1.2.upf",
        library  = "sg15",
        pp_type  = "ONCV",
        notes    = "Fe ONCV PBE (SG15). Norm-conserving.",
    ),
    PseudoEntry(
        element  = "P",
        filename = "P_ONCV_PBE-1.2.upf",
        url      = f"{_SG15_BASE}/P_ONCV_PBE-1.2.upf",
        library  = "sg15",
        pp_type  = "ONCV",
        notes    = "P ONCV PBE (SG15).",
    ),
    PseudoEntry(
        element  = "O",
        filename = "O_ONCV_PBE-1.2.upf",
        url      = f"{_SG15_BASE}/O_ONCV_PBE-1.2.upf",
        library  = "sg15",
        pp_type  = "ONCV",
        notes    = "O ONCV PBE (SG15).",
    ),
]

# ---------------------------------------------------------------------------
# Library index
# ---------------------------------------------------------------------------
LIBRARY_CATALOGUE: Dict[str, List[PseudoEntry]] = {
    "pslibrary": PSLIBRARY_PAW,
    "gbrv":      GBRV_USPP,
    "sg15":      SG15_ONCV,
}

LIBRARY_DESCRIPTIONS: Dict[str, str] = {
    "pslibrary": (
        "PSLibrary PAW/USPP PBE — recommended for LFP (spin-polarised Fe, "
        "matches config.yaml defaults)"
    ),
    "gbrv":      "GBRV USPP PBE — fast screening, lower accuracy than PAW",
    "sg15":      "SG15 ONCV PBE — norm-conserving, good transferability",
}


# ---------------------------------------------------------------------------
# Config-file snippet generator
# ---------------------------------------------------------------------------

def _config_snippet(entries: List[PseudoEntry], out_dir: Path) -> str:
    """Generate the config.yaml pseudopotentials section for the downloaded files."""
    lines = [
        "# Add this to your config.yaml pseudopotentials section:",
        "pseudopotentials:",
        f"  pseudo_dir: \"{out_dir}\"",
        "  files:",
    ]
    for e in entries:
        lines.append(f"    {e.element}: \"{e.filename}\"")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def _sha256_file(path: Path) -> str:
    """Return the SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _human_size(n_bytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n_bytes < 1024:
            return f"{n_bytes:.0f} {unit}" if unit == "B" else f"{n_bytes:.1f} {unit}"
        n_bytes //= 1024
    return f"{n_bytes:.1f} TB"


def _download_file(
    url: str,
    dest: Path,
    logger,
    timeout: int = 60,
    retries: int = 3,
) -> bool:
    """Download *url* to *dest* with progress reporting and retry logic.

    Parameters
    ----------
    url : str
    dest : Path
    logger
    timeout : int
        HTTP timeout in seconds.
    retries : int
        Number of retry attempts on failure.

    Returns
    -------
    bool
        True on success, False on failure.
    """
    for attempt in range(1, retries + 1):
        try:
            logger.info("  Downloading (attempt %d/%d): %s", attempt, retries, url)

            req = urllib.request.Request(
                url,
                headers={"User-Agent": "maccai-battery-pipeline/0.1 (pseudopotential downloader)"},
            )

            with urllib.request.urlopen(req, timeout=timeout) as response:
                total = int(response.headers.get("Content-Length", 0))
                downloaded = 0
                chunk_size = 32768  # 32 KB

                with open(dest, "wb") as out_fh:
                    while True:
                        chunk = response.read(chunk_size)
                        if not chunk:
                            break
                        out_fh.write(chunk)
                        downloaded += len(chunk)

                        if total:
                            pct = 100 * downloaded / total
                            bar = "#" * int(pct / 5)
                            print(
                                f"\r    [{bar:<20}] {pct:5.1f}%  "
                                f"{_human_size(downloaded)} / {_human_size(total)}",
                                end="",
                                flush=True,
                            )

            if total:
                print()  # newline after progress bar

            logger.info("  Saved: %s (%s)", dest.name, _human_size(dest.stat().st_size))
            return True

        except urllib.error.HTTPError as exc:
            logger.error(
                "  HTTP %d for %s", exc.code, url
            )
            if exc.code in (403, 404):
                # Permanent failure — no point retrying
                if dest.exists():
                    dest.unlink()
                return False

        except (urllib.error.URLError, OSError) as exc:
            logger.warning("  Attempt %d failed: %s", attempt, exc)
            if dest.exists():
                dest.unlink()

        if attempt < retries:
            wait = 2 ** attempt
            logger.info("  Retrying in %d s ...", wait)
            time.sleep(wait)

    logger.error("  All %d attempts failed for: %s", retries, url)
    return False


def _verify_upf(path: Path, logger) -> bool:
    """Check that a downloaded file looks like a valid UPF file.

    A UPF file should:
    - Be non-empty
    - Start with ``<UPF`` or ``<PP_INFO`` XML tags (v1 or v2 format)

    Parameters
    ----------
    path : Path
    logger

    Returns
    -------
    bool
    """
    if not path.exists() or path.stat().st_size == 0:
        logger.error("  File is missing or empty: %s", path)
        return False

    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            head = fh.read(512)
    except OSError as exc:
        logger.error("  Cannot read file %s: %s", path, exc)
        return False

    # UPF v1 starts with <PP_INFO, UPF v2 starts with <UPF
    if not (
        "<UPF" in head
        or "<PP_INFO" in head
        or "PP_HEADER" in head
        or "Generated" in head  # some older formats
    ):
        logger.warning(
            "  %s does not look like a valid UPF file (unexpected header). "
            "Check the URL or source.",
            path.name,
        )
        return False

    logger.info("  Verified UPF format: %s", path.name)
    return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download PBE pseudopotentials for the MACCAI battery pipeline.\n\n"
            "Available libraries:\n"
            + "\n".join(
                f"  {name:12s} — {desc}"
                for name, desc in LIBRARY_DESCRIPTIONS.items()
            )
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--elements",
        nargs="+",
        default=["Li", "Fe", "P", "O"],
        metavar="ELEMENT",
        help=(
            "Elements to download pseudopotentials for. "
            "Default: Li Fe P O  (the Li-Fe-P-O battery system)"
        ),
    )
    parser.add_argument(
        "--library",
        default="pslibrary",
        choices=list(LIBRARY_CATALOGUE.keys()),
        help=(
            "Pseudopotential library to use. "
            "Default: pslibrary (PAW PBE, recommended for LFP)."
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help=(
            "Directory to save pseudopotential files. "
            "Default: output/pseudo/ relative to this script's project root."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Print the URLs that would be downloaded without actually "
            "downloading anything."
        ),
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help=(
            "Check whether pseudopotential files already exist in --out-dir "
            "and verify their UPF format. Does not download missing files."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download files even if they already exist in --out-dir.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="HTTP request timeout in seconds (default: 60).",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Number of retry attempts per file on network failure (default: 3).",
    )
    parser.add_argument(
        "--list-libraries",
        action="store_true",
        help="List all available libraries and their entries, then exit.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    import logging

    # Basic logging setup (no file handler for this utility script)
    logging.basicConfig(
        level   = logging.INFO,
        format  = "%(levelname)-8s  %(message)s",
        stream  = sys.stdout,
    )
    logger = logging.getLogger("download_pseudopotentials")

    args = _parse_args()

    # ------------------------------------------------------------------
    # --list-libraries
    # ------------------------------------------------------------------
    if args.list_libraries:
        print("\nAvailable pseudopotential libraries:\n")
        for lib_name, entries in LIBRARY_CATALOGUE.items():
            print(f"  {lib_name}  ({LIBRARY_DESCRIPTIONS[lib_name]})")
            for e in entries:
                print(f"    {e.element:4s}  {e.filename:<45s}  [{e.pp_type}]")
                if e.notes:
                    # Wrap note at 70 chars
                    note_words = e.notes.split()
                    line, note_lines = [], []
                    for word in note_words:
                        line.append(word)
                        if len(" ".join(line)) > 60:
                            note_lines.append("           " + " ".join(line[:-1]))
                            line = [word]
                    note_lines.append("           " + " ".join(line))
                    print("\n".join(note_lines))
            print()
        sys.exit(0)

    # ------------------------------------------------------------------
    # Resolve output directory
    # ------------------------------------------------------------------
    _SCRIPT_DIR  = Path(__file__).resolve().parent
    _PROJECT_DIR = _SCRIPT_DIR.parent

    out_dir = args.out_dir
    if out_dir is None:
        out_dir = _PROJECT_DIR / "output" / "pseudo"

    out_dir = Path(out_dir).expanduser().resolve()

    if not args.check and not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("MACCAI Pseudopotential Downloader")
    logger.info("=" * 60)
    logger.info("Library    : %s", args.library)
    logger.info("Elements   : %s", " ".join(args.elements))
    logger.info("Output dir : %s", out_dir)
    logger.info("Dry run    : %s", args.dry_run)
    logger.info("Check only : %s", args.check)
    logger.info("Force      : %s", args.force)
    logger.info("")

    # ------------------------------------------------------------------
    # Select entries for the requested elements and library
    # ------------------------------------------------------------------
    all_entries = LIBRARY_CATALOGUE[args.library]
    wanted_elements = {el.strip().capitalize() for el in args.elements}

    selected: List[PseudoEntry] = []
    missing_elements: List[str] = []

    for element in args.elements:
        el = element.strip().capitalize()
        match = next((e for e in all_entries if e.element == el), None)
        if match:
            selected.append(match)
        else:
            missing_elements.append(el)

    if missing_elements:
        logger.error(
            "No pseudopotential found in '%s' library for element(s): %s\n"
            "Available elements: %s\n"
            "Try --library pslibrary or --list-libraries.",
            args.library,
            ", ".join(missing_elements),
            ", ".join(e.element for e in all_entries),
        )
        sys.exit(1)

    # ------------------------------------------------------------------
    # Print plan
    # ------------------------------------------------------------------
    logger.info("Pseudopotentials to download:")
    logger.info("  %-4s  %-45s  %-6s  %s", "El", "Filename", "Type", "Notes")
    logger.info("  " + "-" * 72)
    for e in selected:
        note_short = e.notes[:50] + ("..." if len(e.notes) > 50 else "")
        logger.info("  %-4s  %-45s  %-6s  %s", e.element, e.filename, e.pp_type, note_short)
    logger.info("")

    # ------------------------------------------------------------------
    # Dry run — just print URLs
    # ------------------------------------------------------------------
    if args.dry_run:
        print("\n[DRY RUN] Would download:\n")
        for e in selected:
            dest = out_dir / e.filename
            status = "EXISTS" if dest.exists() else "MISSING"
            print(f"  [{status}]  {e.url}")
            print(f"          → {dest}")
        print(
            f"\n[DRY RUN] Config snippet:\n\n"
            + _config_snippet(selected, out_dir)
            + "\n"
        )
        sys.exit(0)

    # ------------------------------------------------------------------
    # Check-only mode
    # ------------------------------------------------------------------
    if args.check:
        logger.info("Checking existing files in: %s\n", out_dir)
        all_ok = True
        for e in selected:
            dest = out_dir / e.filename
            if dest.exists():
                ok = _verify_upf(dest, logger)
                size_str = _human_size(dest.stat().st_size)
                status = "✓ OK" if ok else "✗ INVALID"
                logger.info("  %s  %-45s  %s", status, e.filename, size_str)
                if not ok:
                    all_ok = False
            else:
                logger.warning("  ✗ MISSING  %s", e.filename)
                all_ok = False

        if all_ok:
            logger.info("\nAll pseudopotentials present and valid.")
        else:
            logger.warning(
                "\nSome pseudopotentials are missing or invalid. "
                "Run without --check to download them."
            )
            sys.exit(1)
        sys.exit(0)

    # ------------------------------------------------------------------
    # Download
    # ------------------------------------------------------------------
    succeeded: List[PseudoEntry] = []
    failed: List[PseudoEntry]    = []

    for e in selected:
        dest = out_dir / e.filename

        if dest.exists() and not args.force:
            logger.info("  %-4s — already exists: %s (use --force to re-download)", e.element, e.filename)
            _verify_upf(dest, logger)
            succeeded.append(e)
            continue

        logger.info("Downloading %s (%s) ...", e.element, e.pp_type)

        ok = _download_file(
            url     = e.url,
            dest    = dest,
            logger  = logger,
            timeout = args.timeout,
            retries = args.retries,
        )

        if not ok:
            failed.append(e)
            logger.error(
                "  FAILED: %s\n"
                "  Possible fixes:\n"
                "    1. Check your internet connection.\n"
                "    2. Try a different library (--library gbrv or --library sg15).\n"
                "    3. Download manually from the URL above and place in:\n"
                "         %s",
                e.filename, out_dir,
            )
            continue

        # Verify the downloaded file
        valid = _verify_upf(dest, logger)
        if not valid:
            logger.warning(
                "  Downloaded file may not be a valid UPF — proceed with caution."
            )

        # Verify SHA256 if provided
        if e.sha256:
            actual = _sha256_file(dest)
            if actual == e.sha256:
                logger.info("  SHA256: ✓ verified")
            else:
                logger.error(
                    "  SHA256 mismatch!\n"
                    "    expected: %s\n"
                    "    actual  : %s\n"
                    "  The file may be corrupted or have been updated. "
                    "Delete it and try again.",
                    e.sha256, actual,
                )
                failed.append(e)
                dest.unlink(missing_ok=True)
                continue

        succeeded.append(e)
        logger.info("")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Download summary")
    logger.info("  Succeeded : %d / %d", len(succeeded), len(selected))
    logger.info("  Failed    : %d / %d", len(failed), len(selected))
    logger.info("  Output dir: %s", out_dir)
    logger.info("")

    if failed:
        logger.error("Failed elements: %s", ", ".join(e.element for e in failed))
        logger.error(
            "Manual download instructions:\n"
            "  1. Visit https://pseudopotentials.quantum-espresso.org\n"
            "  2. Search for each element with PBE functional\n"
            "  3. Download the .UPF file and place it in: %s",
            out_dir,
        )

    if succeeded:
        # Print config.yaml snippet
        print("\n" + "=" * 60)
        print("  Add this to your config.yaml:\n")
        print(_config_snippet(succeeded, out_dir))
        print("=" * 60 + "\n")

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
