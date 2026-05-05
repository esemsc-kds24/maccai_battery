# =============================================================================
# maccai_battery.utils — Shared utilities and logging setup
# =============================================================================
# Provides:
#   - Structured logging configuration (console + optional file handler)
#   - Unit conversion helpers (Ry → eV, Bohr → Å, etc.)
#   - EXTXYZ energy parsing
#   - Structure deduplication via fingerprinting
#   - Portable path helpers
#   - Progress reporting for long loops
#
# Usage:
#   from maccai_battery.utils import setup_logging, ry_to_ev, parse_extxyz_energy
# =============================================================================

from __future__ import annotations

import logging
import math
import re
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple, TypeVar

# ---------------------------------------------------------------------------
# Physical constants and unit conversions
# ---------------------------------------------------------------------------

#: Rydberg → eV conversion factor (CODATA 2018)
RY_TO_EV: float = 13.605693122994

#: Hartree → eV conversion factor (CODATA 2018)
HA_TO_EV: float = 27.211386245988

#: Bohr radius → Ångström conversion factor
BOHR_TO_ANG: float = 0.529177210903

#: eV/Å → Ry/Bohr conversion factor (for forces)
EV_ANG_TO_RY_BOHR: float = 1.0 / (RY_TO_EV / BOHR_TO_ANG)

#: GPa → kbar conversion factor
GPA_TO_KBAR: float = 10.0


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

#: Default log format used by :func:`setup_logging`.
LOG_FORMAT = "%(asctime)s  %(levelname)-8s  %(name)s — %(message)s"
LOG_FORMAT_SHORT = "%(levelname)-8s  %(message)s"

#: Date format used in log timestamps.
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(
    level: int | str = logging.INFO,
    log_file: Optional[Path] = None,
    fmt: str = LOG_FORMAT,
    date_fmt: str = LOG_DATE_FORMAT,
    silence_external: bool = True,
) -> logging.Logger:
    """Configure the root logger for the MACCAI battery pipeline.

    Sets up:
    - A ``StreamHandler`` writing to ``stderr`` at *level*.
    - An optional ``FileHandler`` writing to *log_file* at DEBUG level.

    This function is idempotent: calling it multiple times does not
    add duplicate handlers (existing ``maccai_battery`` handlers are
    removed before re-adding).

    Parameters
    ----------
    level : int or str
        Logging level for the console handler.
        May be an integer (e.g. ``logging.DEBUG``) or a string
        (e.g. ``"DEBUG"``, ``"INFO"``, ``"WARNING"``).
    log_file : Path, optional
        If given, all messages at DEBUG level and above are also written
        to this file.  Parent directories are created automatically.
    fmt : str
        Log record format string.
    date_fmt : str
        strftime-compatible date format for log timestamps.
    silence_external : bool
        If ``True`` (default), raises the log level of noisy third-party
        libraries (``ase``, ``pymatgen``, ``mattersim``, ``urllib3``)
        to ``WARNING`` so they don't clutter the console output.

    Returns
    -------
    logging.Logger
        The ``maccai_battery`` package logger.
    """
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    # Get the package-level logger
    pkg_logger = logging.getLogger("maccai_battery")
    pkg_logger.setLevel(logging.DEBUG)  # handlers control effective level

    # Remove any previously installed handlers to avoid duplicates
    for handler in list(pkg_logger.handlers):
        pkg_logger.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter(fmt=fmt, datefmt=date_fmt)

    # ----- Console handler -----
    console = logging.StreamHandler(sys.stderr)
    console.setLevel(level)
    console.setFormatter(logging.Formatter(fmt=LOG_FORMAT_SHORT, datefmt=date_fmt))
    pkg_logger.addHandler(console)

    # ----- File handler (optional) -----
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        pkg_logger.addHandler(file_handler)

    # ----- Silence noisy external libraries -----
    if silence_external:
        for lib in ("ase", "pymatgen", "mattersim", "urllib3", "matplotlib"):
            logging.getLogger(lib).setLevel(logging.WARNING)

    pkg_logger.propagate = False
    return pkg_logger


# ---------------------------------------------------------------------------
# Unit conversion helpers
# ---------------------------------------------------------------------------

def ry_to_ev(value_ry: float) -> float:
    """Convert an energy value from Rydberg to electron-volts.

    Parameters
    ----------
    value_ry : float
        Energy in Rydberg (Ry).

    Returns
    -------
    float
        Energy in electron-volts (eV).
    """
    return value_ry * RY_TO_EV


def ev_to_ry(value_ev: float) -> float:
    """Convert an energy value from electron-volts to Rydberg.

    Parameters
    ----------
    value_ev : float
        Energy in electron-volts (eV).

    Returns
    -------
    float
        Energy in Rydberg (Ry).
    """
    return value_ev / RY_TO_EV


def ha_to_ev(value_ha: float) -> float:
    """Convert an energy value from Hartree to electron-volts.

    Parameters
    ----------
    value_ha : float
        Energy in Hartree (Ha).

    Returns
    -------
    float
        Energy in electron-volts (eV).
    """
    return value_ha * HA_TO_EV


def bohr_to_ang(value_bohr: float) -> float:
    """Convert a length from Bohr radii to Ångström.

    Parameters
    ----------
    value_bohr : float
        Length in Bohr radii (a₀).

    Returns
    -------
    float
        Length in Ångström (Å).
    """
    return value_bohr * BOHR_TO_ANG


def ev_per_atom(energy_ev: float, n_atoms: int) -> float:
    """Compute energy per atom safely.

    Parameters
    ----------
    energy_ev : float
        Total energy in eV.
    n_atoms : int
        Number of atoms in the cell.

    Returns
    -------
    float
        Energy per atom (eV/atom).

    Raises
    ------
    ValueError
        If *n_atoms* is zero.
    """
    if n_atoms == 0:
        raise ValueError("n_atoms must be > 0.")
    return energy_ev / n_atoms


def ry_per_atom_to_ev_per_atom(energy_ry: float, n_atoms: int) -> float:
    """Convert total Rydberg energy to eV per atom.

    Convenience wrapper combining :func:`ry_to_ev` and :func:`ev_per_atom`.

    Parameters
    ----------
    energy_ry : float
        Total energy in Rydberg (Ry).
    n_atoms : int
        Number of atoms in the unit cell.

    Returns
    -------
    float
        Energy per atom in eV/atom.
    """
    return ev_per_atom(ry_to_ev(energy_ry), n_atoms)


# ---------------------------------------------------------------------------
# EXTXYZ parsing helpers
# ---------------------------------------------------------------------------

# Pre-compiled patterns for speed in tight loops
_ENERGY_PATTERN   = re.compile(r"(?:^|\s)energy=([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)")
_FORMULA_PATTERN  = re.compile(r'(?:^|\s)formula="?([A-Za-z0-9]+)"?')


def parse_extxyz_energy(file_path: Path) -> float:
    """Extract the ML total energy from the first frame of an EXTXYZ file.

    The energy is stored in the EXTXYZ comment line (line 2) as:
    ``energy=<float>``

    Parameters
    ----------
    file_path : Path
        Path to the EXTXYZ file (single- or multi-frame).

    Returns
    -------
    float
        The parsed total energy (units depend on what the writer stored;
        MatterSim stores eV).

    Raises
    ------
    FileNotFoundError
        If *file_path* does not exist.
    ValueError
        If no ``energy=`` field is found in the comment line.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"EXTXYZ file not found: {file_path}")

    with file_path.open("r") as fh:
        fh.readline()                   # line 1: atom count
        comment = fh.readline()         # line 2: comment line

    match = _ENERGY_PATTERN.search(comment)
    if not match:
        raise ValueError(
            f"No 'energy=' field found in EXTXYZ comment line of {file_path}.\n"
            f"Comment line was: {comment.strip()!r}"
        )

    return float(match.group(1))


def parse_extxyz_comment(file_path: Path) -> Dict[str, str]:
    """Parse all key=value pairs from the comment line of the first EXTXYZ frame.

    Handles both quoted (``key="value"``) and unquoted (``key=value``) forms.

    Parameters
    ----------
    file_path : Path
        Path to the EXTXYZ file.

    Returns
    -------
    dict of str → str
        All parsed key=value pairs from the comment line.
        Values are returned as raw strings; callers convert as needed.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"EXTXYZ file not found: {file_path}")

    with file_path.open("r") as fh:
        fh.readline()
        comment = fh.readline().strip()

    # Match key="quoted value" or key=unquoted_value
    pattern = re.compile(r'(\w+)=(?:"([^"]*?)"|(\S+))')
    result: Dict[str, str] = {}
    for m in pattern.finditer(comment):
        key   = m.group(1)
        value = m.group(2) if m.group(2) is not None else m.group(3)
        result[key] = value

    return result


def iter_extxyz_frames(file_path: Path) -> Iterator[Tuple[int, int, str]]:
    """Iterate over frames of a multi-frame EXTXYZ file without loading them.

    Yields lightweight metadata only — does not load atomic coordinates.
    Useful for quickly scanning a large EXTXYZ file.

    Parameters
    ----------
    file_path : Path
        Path to the EXTXYZ file.

    Yields
    ------
    (frame_index, n_atoms, comment_line)
        - ``frame_index`` : zero-based frame number
        - ``n_atoms``     : number of atoms in this frame
        - ``comment_line``: the raw comment string for this frame
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"EXTXYZ file not found: {file_path}")

    frame_idx = -1
    with file_path.open("r") as fh:
        while True:
            nat_line = fh.readline()
            if not nat_line:
                break                   # EOF
            nat_line = nat_line.strip()
            if not nat_line:
                continue
            if not nat_line.isdigit():
                continue

            n_atoms     = int(nat_line)
            comment     = fh.readline()
            frame_idx  += 1

            yield frame_idx, n_atoms, comment.rstrip()

            # Skip atom coordinate lines
            for _ in range(n_atoms):
                fh.readline()


def rank_extxyz_by_energy(file_path: Path) -> List[Tuple[int, float]]:
    """Rank all frames in an EXTXYZ file by their stored energy.

    Parameters
    ----------
    file_path : Path
        Path to the multi-frame EXTXYZ file.

    Returns
    -------
    list of (frame_index, energy)
        Sorted by ascending energy (lowest = most stable first).
        Frames whose comment line has no parseable ``energy=`` field are
        assigned ``float('inf')`` and placed at the end.
    """
    results: List[Tuple[int, float]] = []

    for frame_idx, n_atoms, comment in iter_extxyz_frames(file_path):
        match = _ENERGY_PATTERN.search(comment)
        energy = float(match.group(1)) if match else math.inf
        results.append((frame_idx, energy))

    results.sort(key=lambda x: x[1])
    return results


# ---------------------------------------------------------------------------
# Structure fingerprinting / deduplication
# ---------------------------------------------------------------------------

def structure_fingerprint(structure) -> str:
    """Compute a lightweight fingerprint string for a pymatgen Structure.

    The fingerprint encodes:
    - Reduced formula
    - Cell volume (rounded to 2 decimal places)
    - Density (rounded to 3 decimal places)

    This is NOT a cryptographic hash and is NOT guaranteed to be unique
    for all structures.  It is intended only to catch obvious duplicates
    (same formula + very similar cell) quickly without a full structure
    comparison.

    Parameters
    ----------
    structure : pymatgen.core.Structure

    Returns
    -------
    str
        A compact fingerprint string, e.g. ``"LiFePO4|v=151.23|d=3.456"``.
    """
    formula = structure.composition.reduced_formula
    volume  = round(structure.volume, 2)
    density = round(structure.density, 3)
    return f"{formula}|v={volume}|d={density}"


T = TypeVar("T")


def deduplicate_by_fingerprint(
    items: List[T],
    fingerprint_fn,
) -> Tuple[List[T], List[int]]:
    """Remove duplicate items using a fingerprinting function.

    Keeps the first occurrence of each unique fingerprint and discards
    subsequent duplicates.

    Parameters
    ----------
    items : list
        Items to deduplicate.
    fingerprint_fn : callable
        Function that takes one item and returns a hashable fingerprint.

    Returns
    -------
    (unique_items, duplicate_indices)
        - ``unique_items``      : deduplicated list (preserves order)
        - ``duplicate_indices`` : indices of removed items in the original list
    """
    seen: set          = set()
    unique: List[T]    = []
    dup_indices: List[int] = []

    for i, item in enumerate(items):
        fp = fingerprint_fn(item)
        if fp in seen:
            dup_indices.append(i)
        else:
            seen.add(fp)
            unique.append(item)

    return unique, dup_indices


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def ensure_dir(path: Path) -> Path:
    """Create *path* (and all parents) if it does not exist, then return it.

    Parameters
    ----------
    path : Path

    Returns
    -------
    Path
        The same *path* that was passed in, now guaranteed to exist.
    """
    Path(path).mkdir(parents=True, exist_ok=True)
    return Path(path)


def relative_path(target: Path, anchor: Path) -> Path:
    """Return *target* relative to *anchor*, or *target* unchanged if impossible.

    Unlike ``Path.relative_to()``, this function does not raise if *target*
    is not under *anchor* — it simply returns the original path.

    Parameters
    ----------
    target : Path
        The path to make relative.
    anchor : Path
        The directory to make *target* relative to.

    Returns
    -------
    Path
        Relative path, or the original *target* if it is outside *anchor*.
    """
    try:
        return target.relative_to(anchor)
    except ValueError:
        return target


def find_extxyz_files(directory: Path, recursive: bool = True) -> List[Path]:
    """Find all ``.extxyz`` files under *directory*.

    Parameters
    ----------
    directory : Path
        Root directory to search.
    recursive : bool
        If ``True`` (default), search recursively.

    Returns
    -------
    list of Path
        Sorted list of matching file paths.
    """
    directory = Path(directory)
    pattern   = "**/*.extxyz" if recursive else "*.extxyz"
    return sorted(directory.glob(pattern))


def find_ml_relaxed_files(ml_relaxed_dir: Path) -> List[Path]:
    """Find all ML-relaxed EXTXYZ files in *ml_relaxed_dir*.

    Expects files matching the pattern ``*_ml_relaxed.extxyz``.
    Results are sorted by frame number (extracted from filename) so that
    downstream tools process them in a consistent order.

    Parameters
    ----------
    ml_relaxed_dir : Path
        Directory containing ``*_ml_relaxed.extxyz`` files.

    Returns
    -------
    list of Path
        Paths sorted by ascending frame number.
    """
    files = list(Path(ml_relaxed_dir).glob("*_ml_relaxed.extxyz"))

    def _frame_number(p: Path) -> int:
        m = re.search(r"frame(\d+)_ml_relaxed", p.name)
        return int(m.group(1)) if m else 999999

    return sorted(files, key=_frame_number)


# ---------------------------------------------------------------------------
# Progress reporting
# ---------------------------------------------------------------------------

class ProgressLogger:
    """A lightweight progress reporter that logs to the standard logger.

    Useful for long loops where you want periodic progress messages
    without pulling in ``tqdm`` or similar dependencies.

    Parameters
    ----------
    total : int
        Total number of items to process.
    logger : logging.Logger
        Logger to write progress messages to.
    prefix : str
        Message prefix, e.g. ``"Relaxing structures"``.
    log_every : int
        Log a progress message every *log_every* items.

    Example
    -------
    >>> prog = ProgressLogger(64, logger, "Relaxing")
    >>> for i, atoms in enumerate(structures):
    ...     prog.update(i)
    ...     relax(atoms)
    >>> prog.done()
    """

    def __init__(
        self,
        total: int,
        logger: logging.Logger,
        prefix: str = "Progress",
        log_every: int = 5,
    ) -> None:
        self._total     = total
        self._logger    = logger
        self._prefix    = prefix
        self._log_every = max(1, log_every)
        self._t0        = time.monotonic()
        self._last_log  = -1

    def update(self, current: int) -> None:
        """Log a progress message for item *current* (zero-based index).

        A message is only printed every ``log_every`` items to avoid
        flooding the log.

        Parameters
        ----------
        current : int
            Zero-based index of the current item.
        """
        done = current + 1
        if done == 1 or done % self._log_every == 0 or done == self._total:
            if done != self._last_log:
                elapsed   = time.monotonic() - self._t0
                pct       = 100.0 * done / self._total if self._total else 0
                rate      = done / elapsed if elapsed > 0 else 0
                remaining = (self._total - done) / rate if rate > 0 else float("inf")
                eta_str   = f"{remaining:.0f}s" if math.isfinite(remaining) else "?"
                self._logger.info(
                    "%s: %d/%d (%.0f%%) | elapsed=%.1fs | eta=%s",
                    self._prefix, done, self._total, pct, elapsed, eta_str,
                )
                self._last_log = done

    def done(self) -> None:
        """Log a completion message with total wall-clock time."""
        elapsed = time.monotonic() - self._t0
        self._logger.info(
            "%s: complete. %d items in %.1fs (%.2f items/s).",
            self._prefix,
            self._total,
            elapsed,
            self._total / elapsed if elapsed > 0 else 0,
        )


# ---------------------------------------------------------------------------
# QE output parsing helpers
# ---------------------------------------------------------------------------

def parse_qe_total_energy_ry(output: str) -> Optional[float]:
    """Extract the final total energy (Ry) from a QE pw.x output string.

    Searches for the line::

        !    total energy              =    -XXXX.XXXXXXXX Ry

    The ``!`` prefix marks the converged SCF energy in QE output.

    Parameters
    ----------
    output : str
        Full text output from a ``pw.x`` run.

    Returns
    -------
    float or None
        Total energy in Rydberg, or ``None`` if not found.
    """
    pattern = re.compile(
        r"!\s+total energy\s+=\s+([-+]?\d+\.\d+(?:[eE][-+]?\d+)?)\s+Ry"
    )
    # Use findall so we get the LAST match (final SCF iteration)
    matches = pattern.findall(output)
    if not matches:
        return None
    return float(matches[-1])


def parse_qe_total_energy_ev(output: str) -> Optional[float]:
    """Extract and convert the final QE total energy from Ry to eV.

    Parameters
    ----------
    output : str
        Full text output from a ``pw.x`` run.

    Returns
    -------
    float or None
        Total energy in eV, or ``None`` if not found.
    """
    ry = parse_qe_total_energy_ry(output)
    return ry_to_ev(ry) if ry is not None else None


def parse_qe_n_atoms(output: str) -> Optional[int]:
    """Extract the number of atoms from a QE pw.x output string.

    Searches for the line::

        number of atoms/cell      =           NN

    Parameters
    ----------
    output : str
        Full text output from a ``pw.x`` run.

    Returns
    -------
    int or None
        Number of atoms, or ``None`` if not found.
    """
    pattern = re.compile(r"number of atoms/cell\s+=\s+(\d+)")
    match   = pattern.search(output)
    return int(match.group(1)) if match else None


def parse_qe_fermi_energy(output: str) -> Optional[float]:
    """Extract the Fermi energy (eV) from a QE pw.x output string.

    Parameters
    ----------
    output : str
        Full text output from a ``pw.x`` run.

    Returns
    -------
    float or None
        Fermi energy in eV, or ``None`` if not found.
    """
    pattern = re.compile(
        r"the Fermi energy is\s+([-+]?\d+\.\d+(?:[eE][-+]?\d+)?)\s+ev",
        re.IGNORECASE,
    )
    match = pattern.search(output)
    return float(match.group(1)) if match else None


def parse_qe_scf_converged(output: str) -> bool:
    """Check whether a QE pw.x SCF run converged.

    Parameters
    ----------
    output : str
        Full text output from a ``pw.x`` run.

    Returns
    -------
    bool
        ``True`` if the convergence marker ``"convergence has been achieved"``
        is present in the output.
    """
    return "convergence has been achieved" in output.lower()


def extract_scf_summary(output: str, n_atoms: Optional[int] = None) -> Dict[str, object]:
    """Extract a concise summary of key quantities from QE SCF output.

    Parameters
    ----------
    output : str
        Full text of a ``pw.x`` SCF run.
    n_atoms : int, optional
        Number of atoms (used to compute energy per atom).
        If ``None``, it is parsed from the output.

    Returns
    -------
    dict with keys:
        converged, energy_ry, energy_ev, energy_ev_per_atom, fermi_ev, n_atoms
    """
    nat    = n_atoms or parse_qe_n_atoms(output)
    e_ry   = parse_qe_total_energy_ry(output)
    e_ev   = ry_to_ev(e_ry) if e_ry is not None else None
    e_pa   = ev_per_atom(e_ev, nat) if (e_ev is not None and nat) else None
    fermi  = parse_qe_fermi_energy(output)
    conv   = parse_qe_scf_converged(output)

    return {
        "converged":           conv,
        "energy_ry":           e_ry,
        "energy_ev":           e_ev,
        "energy_ev_per_atom":  e_pa,
        "fermi_ev":            fermi,
        "n_atoms":             nat,
    }


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def format_energy_table(
    rows: Iterable[Tuple[str, float, int]],
    title: str = "Energy ranking",
) -> str:
    """Format a list of (name, energy_eV, n_atoms) tuples as a pretty table.

    Parameters
    ----------
    rows : iterable of (name, energy_eV, n_atoms)
        Data to display.
    title : str
        Table title.

    Returns
    -------
    str
        A formatted multi-line string ready to print.
    """
    rows = list(rows)
    if not rows:
        return f"{title}: (no data)"

    lines  = [f"\n{'='*72}", f"  {title}", f"{'='*72}"]
    header = f"  {'Rank':>4}  {'Name':<40}  {'nat':>5}  {'E (eV/atom)':>14}"
    lines.append(header)
    lines.append(f"  {'-'*66}")

    for rank, (name, e_ev, nat) in enumerate(rows, start=1):
        e_pa = e_ev / nat if nat else float("nan")
        lines.append(f"  {rank:>4}  {name:<40}  {nat:>5}  {e_pa:>14.6f}")

    lines.append(f"{'='*72}\n")
    return "\n".join(lines)


def human_size(n_bytes: int) -> str:
    """Return a human-readable file size string.

    Parameters
    ----------
    n_bytes : int
        File size in bytes.

    Returns
    -------
    str
        E.g. ``"1.23 MB"``, ``"456 KB"``, ``"12 B"``.
    """
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n_bytes < 1024:
            return f"{n_bytes:.0f} {unit}" if unit == "B" else f"{n_bytes:.2f} {unit}"
        n_bytes /= 1024
    return f"{n_bytes:.2f} PB"
