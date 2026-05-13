# =============================================================================
# maccai_battery.dft.elemental_inputs — QE input generators for elemental refs
# =============================================================================
# Generates plain-text pw.x input files for the four elemental reference
# phases used in hull analysis:
#
#   Li  — BCC (Im-3m), 1-atom cell, a ≈ 3.510 Å
#   Fe  — BCC (Im-3m), 1-atom cell, a ≈ 2.831 Å, spin-polarised, DFT+U
#   P   — P4 molecule in a 10 Å cubic box (4 atoms)
#   O   — O2 molecule in a 15 Å cubic box (2 atoms, triplet state)
#
# Each function returns the full text of a pw.x input file as a string.
# The caller is responsible for writing it to disk and running pw.x.
#
# Reference energies to extract after running:
#   Li  → total energy / 1
#   Fe  → total energy / 1
#   P   → total energy / 4
#   O   → total energy / 2
#
# Usage:
#   from maccai_battery.dft.elemental_inputs import make_reference_inputs
#   inputs = make_reference_inputs(cfg)
#   for element, (text, n_atoms) in inputs.items():
#       (run_dir / "qe.in").write_text(text)
# =============================================================================

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Dict, Tuple

if TYPE_CHECKING:
    from maccai_battery.config import PipelineConfig

# n_atoms used to compute energy per atom after the run
REFERENCE_N_ATOMS: Dict[str, int] = {
    "Li": 1,
    "Fe": 1,
    "P":  4,
    "O":  2,
}


def _pseudo_dir(cfg: "PipelineConfig") -> str:
    pseudo_dir = Path(cfg.pseudopotentials.pseudo_dir)
    if not pseudo_dir.is_absolute():
        pseudo_dir = cfg.project._root / pseudo_dir
    return str(pseudo_dir.resolve())


def _ecutwfc(cfg: "PipelineConfig") -> float:
    return float(cfg.dft_screening.ecutwfc)


def _ecutrho(cfg: "PipelineConfig") -> float:
    return float(cfg.dft_screening.ecutrho)


# ---------------------------------------------------------------------------
# Li — BCC metal
# ---------------------------------------------------------------------------

def make_li_input(cfg: "PipelineConfig") -> str:
    """QE pw.x input for Li BCC reference (1 atom, ibrav=3)."""
    ecutwfc = _ecutwfc(cfg)
    ecutrho = _ecutrho(cfg)
    pseudo = cfg.pseudopotentials.get("Li")
    pseudo_dir = _pseudo_dir(cfg)

    return f"""\
&CONTROL
  calculation   = 'scf'
  prefix        = 'Li_ref'
  pseudo_dir    = '{pseudo_dir}'
  outdir        = './tmp'
  verbosity     = 'high'
/
&SYSTEM
  ibrav         = 3
  a             = 3.510
  nat           = 1
  ntyp          = 1
  ecutwfc       = {ecutwfc}
  ecutrho       = {ecutrho}
  occupations   = 'smearing'
  smearing      = 'mp'
  degauss       = 0.02
/
&ELECTRONS
  conv_thr      = 1.0d-8
  mixing_beta   = 0.3
/
ATOMIC_SPECIES
  Li  6.941  {pseudo}

ATOMIC_POSITIONS (crystal)
  Li  0.000  0.000  0.000

K_POINTS (automatic)
  8 8 8 0 0 0
"""


# ---------------------------------------------------------------------------
# Fe — BCC metal, spin-polarised, DFT+U
# ---------------------------------------------------------------------------

def make_fe_input(cfg: "PipelineConfig") -> str:
    """QE pw.x input for Fe BCC reference (1 atom, ibrav=3, spin-pol, DFT+U)."""
    ecutwfc = _ecutwfc(cfg)
    ecutrho = _ecutrho(cfg)
    pseudo = cfg.pseudopotentials.get("Fe")
    pseudo_dir = _pseudo_dir(cfg)

    # DFT+U from config if enabled
    hubbard_u_block = ""
    u_cfg = cfg.dft_screening.hubbard_u or {}
    if u_cfg.get("enabled") and "Fe" in (u_cfg.get("U") or {}):
        u_val = float(u_cfg["U"]["Fe"])
        lda_kind = int(u_cfg.get("lda_plus_u_kind", 0))
        hubbard_u_block = (
            f"  lda_plus_u        = .true.\n"
            f"  lda_plus_u_kind   = {lda_kind}\n"
            f"  Hubbard_U(1)      = {u_val}    ! Fe 3d (eV)\n"
        )

    return f"""\
&CONTROL
  calculation   = 'scf'
  prefix        = 'Fe_ref'
  pseudo_dir    = '{pseudo_dir}'
  outdir        = './tmp'
  verbosity     = 'high'
/
&SYSTEM
  ibrav         = 3
  a             = 2.831
  nat           = 1
  ntyp          = 1
  ecutwfc       = {ecutwfc}
  ecutrho       = {ecutrho}
  nspin         = 2
  starting_magnetization(1) = 1.0
  occupations   = 'smearing'
  smearing      = 'mp'
  degauss       = 0.02
{hubbard_u_block}/
&ELECTRONS
  conv_thr      = 1.0d-8
  mixing_beta   = 0.3
/
ATOMIC_SPECIES
  Fe  55.845  {pseudo}

ATOMIC_POSITIONS (crystal)
  Fe  0.000  0.000  0.000

K_POINTS (automatic)
  10 10 10 0 0 0
"""


# ---------------------------------------------------------------------------
# P — P4 molecule in a 10 Å cubic box
# ---------------------------------------------------------------------------

def make_p_input(cfg: "PipelineConfig") -> str:
    """QE pw.x input for P4 molecule reference (4 atoms, 10 Å box, ibrav=0)."""
    ecutwfc = _ecutwfc(cfg)
    ecutrho = _ecutrho(cfg)
    pseudo = cfg.pseudopotentials.get("P")
    pseudo_dir = _pseudo_dir(cfg)

    # Tetrahedral P4: bond length ~2.21 Å in a 10 Å box
    # Crystal coordinates for P4 tetrahedron centred at (0.5, 0.5, 0.5)
    # with bond ~2.21 Å → fractional offset ≈ 0.1105
    f = 0.1105
    c = 0.5
    p4_positions = (
        f"  P  {c+f:.4f}  {c+f:.4f}  {c+f:.4f}\n"
        f"  P  {c+f:.4f}  {c-f:.4f}  {c-f:.4f}\n"
        f"  P  {c-f:.4f}  {c+f:.4f}  {c-f:.4f}\n"
        f"  P  {c-f:.4f}  {c-f:.4f}  {c+f:.4f}"
    )

    return f"""\
&CONTROL
  calculation   = 'scf'
  prefix        = 'P_ref'
  pseudo_dir    = '{pseudo_dir}'
  outdir        = './tmp'
  verbosity     = 'high'
/
&SYSTEM
  ibrav         = 0
  nat           = 4
  ntyp          = 1
  ecutwfc       = {ecutwfc}
  ecutrho       = {ecutrho}
  occupations   = 'smearing'
  smearing      = 'mp'
  degauss       = 0.02
/
&ELECTRONS
  conv_thr      = 1.0d-8
  mixing_beta   = 0.3
/
ATOMIC_SPECIES
  P  30.974  {pseudo}

CELL_PARAMETERS (angstrom)
  10.000  0.000  0.000
   0.000 10.000  0.000
   0.000  0.000 10.000

ATOMIC_POSITIONS (crystal)
{p4_positions}

K_POINTS (automatic)
  1 1 1 0 0 0
"""


# ---------------------------------------------------------------------------
# O — O2 molecule in a 15 Å cubic box, triplet ground state
# ---------------------------------------------------------------------------

def make_o_input(cfg: "PipelineConfig") -> str:
    """QE pw.x input for O2 molecule reference (2 atoms, 15 Å box, ibrav=0, triplet)."""
    ecutwfc = _ecutwfc(cfg)
    ecutrho = _ecutrho(cfg)
    pseudo = cfg.pseudopotentials.get("O")
    pseudo_dir = _pseudo_dir(cfg)

    # O–O bond length ≈ 1.21 Å; bond half-length / box = 0.605/15 ≈ 0.0403
    offset = 0.0403
    c = 0.5
    o2_positions = (
        f"  O  {c:.4f}  {c:.4f}  {c - offset:.4f}\n"
        f"  O  {c:.4f}  {c:.4f}  {c + offset:.4f}"
    )

    return f"""\
&CONTROL
  calculation   = 'scf'
  prefix        = 'O_ref'
  pseudo_dir    = '{pseudo_dir}'
  outdir        = './tmp'
  verbosity     = 'high'
/
&SYSTEM
  ibrav         = 0
  nat           = 2
  ntyp          = 1
  ecutwfc       = {ecutwfc}
  ecutrho       = {ecutrho}
  nspin         = 2
  starting_magnetization(1) = 1.0
  tot_magnetization = 2
  occupations   = 'smearing'
  smearing      = 'mp'
  degauss       = 0.02
/
&ELECTRONS
  conv_thr      = 1.0d-8
  mixing_beta   = 0.3
/
ATOMIC_SPECIES
  O  15.999  {pseudo}

CELL_PARAMETERS (angstrom)
  15.000   0.000   0.000
   0.000  15.000   0.000
   0.000   0.000  15.000

ATOMIC_POSITIONS (crystal)
{o2_positions}

K_POINTS (automatic)
  1 1 1 0 0 0
"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

MAKERS = {
    "Li": make_li_input,
    "Fe": make_fe_input,
    "P":  make_p_input,
    "O":  make_o_input,
}


def make_reference_inputs(cfg: "PipelineConfig") -> Dict[str, Tuple[str, int]]:
    """Generate QE pw.x input text for all elemental reference phases.

    Returns
    -------
    dict
        Mapping element → (input_text, n_atoms).
        Divide the QE total energy by n_atoms to get energy per atom.
    """
    return {
        el: (maker(cfg), REFERENCE_N_ATOMS[el])
        for el, maker in MAKERS.items()
    }
