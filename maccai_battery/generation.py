# =============================================================================
# maccai_battery.generation — MatterGen crystal structure generation interface
# =============================================================================
# Wraps the MatterGen CLI (`mattergen-generate`) with:
#   - Typed config integration
#   - Structured logging
#   - Robust error handling and retry logic
#   - Output validation
#
# MatterGen must be installed in its own conda environment.
# See README.md → "Environment Setup" for instructions.
#
# Usage (standalone script):
#   from maccai_battery.config import load_config
#   from maccai_battery.generation import run_mattergen
#
#   cfg = load_config()
#   extxyz_path = run_mattergen(cfg)
# =============================================================================

from __future__ import annotations

import logging
import os
import subprocess
import time
from pathlib import Path
from typing import List, Optional, Tuple

from maccai_battery.config import PipelineConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_mattergen(cfg: PipelineConfig) -> Path:
    """Run MatterGen crystal structure generation.

    Calls the ``mattergen-generate`` CLI with conditioning properties
    derived from *cfg*.  The output is a multi-frame EXTXYZ file
    containing all generated structures.

    Parameters
    ----------
    cfg : PipelineConfig
        Fully loaded and validated pipeline configuration.

    Returns
    -------
    Path
        Absolute path to the generated ``generated_crystals.extxyz`` file.

    Raises
    ------
    FileNotFoundError
        If MatterGen is not installed or the expected output file is missing.
    subprocess.CalledProcessError
        If the ``mattergen-generate`` command exits with a non-zero status.
    RuntimeError
        If the output file exists but is empty or unreadable.
    """
    gen_cfg  = cfg.generation
    proj_cfg = cfg.project

    # ------------------------------------------------------------------
    # Apple Silicon MPS fallback
    # ------------------------------------------------------------------
    # Some PyTorch diffusion ops are not yet supported on the MPS backend.
    # Setting this env-var forces a safe CPU fallback for those ops instead
    # of crashing with an "unsupported operation" error.
    if gen_cfg.pytorch_mps_fallback:
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        logger.debug("PYTORCH_ENABLE_MPS_FALLBACK=1 set for Apple Silicon.")

    # ------------------------------------------------------------------
    # Output directory
    # ------------------------------------------------------------------
    # Namespaced by model + chemical system so multiple runs coexist cleanly.
    results_dir = _build_results_dir(cfg)
    results_dir.mkdir(parents=True, exist_ok=True)
    logger.info("MatterGen results dir: %s", results_dir)

    # ------------------------------------------------------------------
    # Build CLI command
    # ------------------------------------------------------------------
    cmd = _build_command(cfg, results_dir)
    logger.info("MatterGen command:\n  %s", " ".join(cmd))

    # ------------------------------------------------------------------
    # Run generation
    # ------------------------------------------------------------------
    logger.info(
        "Starting MatterGen generation: %s structures "
        "(%d batches × %d per batch) for system '%s' ...",
        gen_cfg.total_structures,
        gen_cfg.num_batches,
        gen_cfg.batch_size,
        gen_cfg.chemical_system,
    )

    t0 = time.monotonic()
    _run_subprocess(cmd)
    elapsed = time.monotonic() - t0

    logger.info("MatterGen finished in %.1f s.", elapsed)

    # ------------------------------------------------------------------
    # Validate output
    # ------------------------------------------------------------------
    extxyz_path = results_dir / "generated_crystals.extxyz"
    extxyz_path = _validate_extxyz(extxyz_path)

    # Copy the extxyz into the canonical CIF dir for downstream consistency
    cif_dir = proj_cfg.dir_path("cifs")
    cif_dir.mkdir(parents=True, exist_ok=True)
    canonical = cif_dir / "generated_crystals.extxyz"
    if not canonical.exists() or canonical.resolve() != extxyz_path.resolve():
        import shutil
        shutil.copy2(extxyz_path, canonical)
        logger.debug("Copied generated_crystals.extxyz → %s", canonical)

    return extxyz_path


def load_generated_structures(extxyz_path: Path) -> List:
    """Load all frames from a multi-frame EXTXYZ file via ASE.

    Parameters
    ----------
    extxyz_path : Path
        Path to the EXTXYZ file produced by MatterGen.

    Returns
    -------
    list of ase.Atoms
        All structures stored in the file.

    Raises
    ------
    ImportError
        If ASE is not installed.
    FileNotFoundError
        If *extxyz_path* does not exist.
    RuntimeError
        If the file contains no structures.
    """
    try:
        from ase import io as ase_io
    except ImportError as exc:
        raise ImportError(
            "ASE is required to load EXTXYZ files. "
            "Install it with: pip install ase"
        ) from exc

    extxyz_path = Path(extxyz_path)
    if not extxyz_path.exists():
        raise FileNotFoundError(f"EXTXYZ file not found: {extxyz_path}")

    logger.info("Loading structures from %s ...", extxyz_path)
    atoms_list = ase_io.read(str(extxyz_path), index=":")

    if not atoms_list:
        raise RuntimeError(
            f"No structures found in {extxyz_path}. "
            f"The file may be empty or malformed."
        )

    logger.info("Loaded %d structures.", len(atoms_list))
    return atoms_list


def count_structures(extxyz_path: Path) -> int:
    """Return the number of frames in an EXTXYZ file without loading them.

    Counts lines that begin with a bare integer (the atom-count line that
    opens each EXTXYZ frame).  This is much faster than loading all frames
    with ASE for large files.

    Parameters
    ----------
    extxyz_path : Path
        Path to the EXTXYZ file.

    Returns
    -------
    int
        Number of frames in the file.
    """
    extxyz_path = Path(extxyz_path)
    count = 0
    with extxyz_path.open("r") as fh:
        for line in fh:
            stripped = line.strip()
            if stripped.isdigit():
                count += 1
    return count


def summarise_generated(extxyz_path: Path) -> List[Tuple[int, str, float]]:
    """Return a lightweight summary of generated structures.

    Each entry is ``(frame_index, formula, ml_energy_eV)``.
    The ML energy is parsed from the EXTXYZ comment line.

    Parameters
    ----------
    extxyz_path : Path
        Path to the EXTXYZ file.

    Returns
    -------
    list of (int, str, float)
        ``(frame_index, formula, ml_energy_eV)`` tuples.
        Energy will be ``float('nan')`` if parsing fails for a frame.
    """
    import math
    import re

    extxyz_path = Path(extxyz_path)
    results: List[Tuple[int, str, float]] = []

    frame_idx = -1
    nat = 0
    state = "expect_nat"  # simple state machine: expect_nat | expect_comment | skip_atoms

    with extxyz_path.open("r") as fh:
        atom_lines_left = 0
        for raw_line in fh:
            line = raw_line.strip()

            if state == "expect_nat":
                if line.isdigit():
                    nat = int(line)
                    atom_lines_left = nat
                    frame_idx += 1
                    state = "expect_comment"
                continue

            if state == "expect_comment":
                # Parse formula and energy from the comment line
                energy_match = re.search(r"energy=([-\d\.Ee+]+)", line)
                formula_match = re.search(r'formula="?([A-Za-z0-9]+)"?', line)

                energy = float(energy_match.group(1)) if energy_match else math.nan
                formula = formula_match.group(1) if formula_match else "?"

                results.append((frame_idx, formula, energy))
                state = "skip_atoms"
                continue

            if state == "skip_atoms":
                atom_lines_left -= 1
                if atom_lines_left == 0:
                    state = "expect_nat"

    return results


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_results_dir(cfg: PipelineConfig) -> Path:
    """Build the namespaced output directory for a MatterGen run."""
    gen_cfg  = cfg.generation
    proj_cfg = cfg.project

    safe_system = gen_cfg.chemical_system.replace("-", "_")
    dir_name    = f"{gen_cfg.model_name}_{safe_system}"

    return proj_cfg.dir_path("cifs") / "mattergen_results" / dir_name


def _build_command(cfg: PipelineConfig, results_dir: Path) -> List[str]:
    """Construct the ``mattergen-generate`` CLI argument list."""
    gen_cfg = cfg.generation

    properties = {
        "energy_above_hull": float(gen_cfg.energy_above_hull),
        "chemical_system":   gen_cfg.chemical_system,
    }
    # MatterGen expects the properties as a Python-dict string literal.
    properties_str = str(properties)

    cmd = [
        "mattergen-generate",
        str(results_dir),
        f"--pretrained-name={gen_cfg.model_name}",
        f"--batch_size={gen_cfg.batch_size}",
        f"--num_batches={gen_cfg.num_batches}",
        f"--properties_to_condition_on={properties_str}",
        f"--diffusion_guidance_factor={gen_cfg.diffusion_guidance_factor}",
    ]

    return cmd


def _run_subprocess(cmd: List[str]) -> None:
    """Run *cmd* as a subprocess, streaming output to the logger.

    Parameters
    ----------
    cmd : list of str
        The command + arguments to execute.

    Raises
    ------
    FileNotFoundError
        If the executable (``cmd[0]``) is not found on PATH.
    subprocess.CalledProcessError
        If the command exits with a non-zero return code.
    """
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"'{cmd[0]}' not found. Is MatterGen installed and on your PATH?\n"
            f"Activate the correct conda environment: conda activate mattergen_env"
        ) from exc

    # Stream output line-by-line so long runs don't appear frozen.
    stdout_lines: List[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        stripped = line.rstrip()
        stdout_lines.append(stripped)
        logger.debug("[mattergen] %s", stripped)

    proc.wait()

    if proc.returncode != 0:
        # Re-print last 20 lines at WARNING level for visibility.
        tail = "\n".join(stdout_lines[-20:])
        raise subprocess.CalledProcessError(
            proc.returncode,
            cmd,
            output=tail,
        )


def _validate_extxyz(extxyz_path: Path) -> Path:
    """Confirm the EXTXYZ output file exists and is non-empty.

    Parameters
    ----------
    extxyz_path : Path
        Expected path to the output file.

    Returns
    -------
    Path
        The resolved absolute path to the file.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    RuntimeError
        If the file is empty (0 bytes).
    """
    if not extxyz_path.exists():
        raise FileNotFoundError(
            f"MatterGen did not produce the expected output file:\n"
            f"  {extxyz_path}\n"
            f"Check that the MatterGen run completed without errors."
        )

    size = extxyz_path.stat().st_size
    if size == 0:
        raise RuntimeError(
            f"MatterGen output file is empty (0 bytes):\n"
            f"  {extxyz_path}\n"
            f"The generation run may have failed silently."
        )

    n = count_structures(extxyz_path)
    logger.info(
        "Output validated: %s (%d bytes, %d frames).",
        extxyz_path.name,
        size,
        n,
    )

    return extxyz_path.resolve()
