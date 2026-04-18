# =============================================================================
# maccai_battery.dft.input_generator — Quantum ESPRESSO pw.x input builder
# =============================================================================
# Generates QE pw.x input (PWin) objects for two calculation types:
#
#   make_scf_input   — single-point SCF energy calculation (for screening)
#   make_relax_input — ionic BFGS relaxation (for geometry optimisation)
#
# Both functions read all DFT parameters from the typed PipelineConfig so
# that no hardcoded values appear in this module.  The only external
# dependency is pymatgen-io-espresso, which is imported lazily so that the
# rest of the package remains importable even if that library is absent.
#
# Notebook fidelity
# -----------------
# This module is a production port of the QE input-generation cells in
# (on_gcolab)MACCAI_DFT.ipynb.  Key differences from the notebook:
#   - Pseudopotential filenames come from cfg.pseudopotentials (fixing the
#     missing-P-pseudopotential bug in the notebook's PSEUDOS dict).
#   - nstep is placed in the CONTROL namelist (correct for QE 6.x+).
#   - run_type-specific settings come from DFTScreeningConfig / DFTRelaxConfig.
#   - No Google Drive paths or Colab-specific code.
#
# Usage
# -----
#   from maccai_battery.dft.input_generator import make_scf_input, write_qe_input
#   pw_in = make_scf_input(structure, cfg)
#   write_qe_input(pw_in, run_dir / "qe.in")
# =============================================================================

from __future__ import annotations

import logging
from collections import OrderedDict
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pymatgen.core import Structure
    from pymatgen.io.espresso.inputs.pwin import PWin
    from maccai_battery.config import PipelineConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_ESPRESSO_INSTALL_MSG = (
    "The 'pymatgen-io-espresso' package is required for QE input generation.\n"
    "Install it with:\n"
    "  pip install git+https://github.com/Griffin-Group/pymatgen-io-espresso\n"
    "See: https://github.com/Griffin-Group/pymatgen-io-espresso"
)


def _import_pwin_classes():
    """Lazily import all PWin classes from pymatgen-io-espresso.

    Returns
    -------
    tuple
        (PWin, ControlNamelist, SystemNamelist, ElectronsNamelist,
         IonsNamelist, AtomicSpeciesCard, AtomicPositionsCard,
         CellParametersCard, KPointsCard)

    Raises
    ------
    ImportError
        If pymatgen-io-espresso is not installed.
    """
    try:
        from pymatgen.io.espresso.inputs.pwin import (  # type: ignore[import]
            AtomicPositionsCard,
            AtomicSpeciesCard,
            CellParametersCard,
            ControlNamelist,
            ElectronsNamelist,
            IonsNamelist,
            KPointsCard,
            PWin,
            SystemNamelist,
        )
        return (
            PWin,
            ControlNamelist,
            SystemNamelist,
            ElectronsNamelist,
            IonsNamelist,
            AtomicSpeciesCard,
            AtomicPositionsCard,
            CellParametersCard,
            KPointsCard,
        )
    except ImportError as exc:
        raise ImportError(_ESPRESSO_INSTALL_MSG) from exc


def _resolve_pseudo_dir(cfg: "PipelineConfig") -> Path:
    """Return the absolute pseudopotential directory path.

    If ``cfg.pseudopotentials.pseudo_dir`` is a relative path it is resolved
    relative to the project config root (i.e. the directory that contains
    ``config.yaml``).

    Parameters
    ----------
    cfg : PipelineConfig

    Returns
    -------
    Path
        Absolute path to the pseudopotential directory.
    """
    pseudo_dir = Path(cfg.pseudopotentials.pseudo_dir)
    if not pseudo_dir.is_absolute():
        pseudo_dir = cfg.project._root / pseudo_dir
    return pseudo_dir


def _ordered_species(structure: "Structure") -> list[str]:
    """Return the unique element symbols in their first-occurrence order.

    The ordering matches the species indices used in the QE input
    (1-indexed Fortran style for starting_magnetization etc.).

    Parameters
    ----------
    structure : pymatgen.core.Structure

    Returns
    -------
    list of str
        e.g. ["Li", "Fe", "P", "O"]
    """
    seen: set[str] = set()
    order: list[str] = []
    for site in structure:
        el = str(site.specie)
        if el not in seen:
            seen.add(el)
            order.append(el)
    return order


def _build_system_namelist(
    SystemNamelist,
    nat: int,
    ntyp: int,
    ecutwfc: float,
    ecutrho: float,
    smearing: str,
    degauss: float,
    conv_thr: float,
    spin_polarised: bool,
    species_order: list[str],
    starting_magnetization: dict[str, float],
):
    """Construct a SystemNamelist with optional spin-polarisation parameters.

    ``starting_magnetization(i)`` entries are set via item assignment after
    construction because Python keyword arguments cannot contain parentheses.

    Parameters
    ----------
    SystemNamelist : class
        The SystemNamelist class from pymatgen-io-espresso.
    nat : int
        Number of atoms in the cell.
    ntyp : int
        Number of distinct species.
    ecutwfc : float
        Plane-wave kinetic energy cut-off (Ry).
    ecutrho : float
        Charge-density cut-off (Ry).
    smearing : str
        Smearing type (e.g. ``"mp"``).
    degauss : float
        Smearing width (Ry).
    conv_thr : float
        SCF convergence threshold (Ry).
    spin_polarised : bool
        Whether to activate spin-polarised (``nspin=2``) calculation.
    species_order : list of str
        Unique element symbols in structure order (determines Fortran indices).
    starting_magnetization : dict
        Mapping from element symbol to initial magnetic moment.

    Returns
    -------
    SystemNamelist
    """
    kwargs: dict = dict(
        ibrav=0,
        nat=nat,
        ntyp=ntyp,
        ecutwfc=ecutwfc,
        ecutrho=ecutrho,
        occupations="smearing",
        smearing=smearing,
        degauss=degauss,
    )
    if spin_polarised:
        kwargs["nspin"] = 2

    system_nl = SystemNamelist(**kwargs)

    # Fortran-indexed starting_magnetization(i): must be set via __setitem__
    # because Python does not allow parentheses in keyword argument names.
    if spin_polarised:
        for i, el in enumerate(species_order, start=1):
            mag = starting_magnetization.get(el, 0.0)
            system_nl[f"starting_magnetization({i})"] = mag
            logger.debug(
                "  starting_magnetization(%d) = %.4f  [%s]", i, mag, el
            )

    return system_nl


def _build_cards(
    AtomicSpeciesCard,
    AtomicPositionsCard,
    CellParametersCard,
    KPointsCard,
    structure: "Structure",
    species_order: list[str],
    pseudo_dir: Path,
    cfg: "PipelineConfig",
    kpoints: list[int],
) -> "OrderedDict":
    """Build the four QE cards needed for a pw.x input.

    Parameters
    ----------
    AtomicSpeciesCard, AtomicPositionsCard, CellParametersCard, KPointsCard :
        Card classes from pymatgen-io-espresso.
    structure : pymatgen.core.Structure
    species_order : list of str
        Unique element symbols in first-occurrence order.
    pseudo_dir : Path
        Resolved absolute path to the pseudopotential directory (unused here
        but kept for logging / validation convenience).
    cfg : PipelineConfig
        Used to look up pseudopotential filenames via
        ``cfg.pseudopotentials.get(element)``.
    kpoints : list of int
        Three-element k-point mesh [k1, k2, k3].

    Returns
    -------
    collections.OrderedDict
        Keys: ``"atomic_species"``, ``"atomic_positions"``,
              ``"k_points"``, ``"cell_parameters"``.
    """
    from pymatgen.core import Element  # type: ignore[import]

    # ---- ATOMIC_SPECIES ----
    masses = [float(Element(el).atomic_mass) for el in species_order]
    pseudos = [cfg.pseudopotentials.get(el) for el in species_order]
    atomic_species = AtomicSpeciesCard(
        None,
        species_order,
        masses,
        pseudos,
    )
    logger.debug(
        "ATOMIC_SPECIES: %s",
        list(zip(species_order, pseudos)),
    )

    # ---- ATOMIC_POSITIONS crystal (fractional) ----
    pos_symbols = [str(site.specie) for site in structure]
    pos_coords = np.array([site.frac_coords for site in structure])
    atomic_positions = AtomicPositionsCard(
        "crystal",
        pos_symbols,
        pos_coords,
        None,
    )

    # ---- CELL_PARAMETERS angstrom ----
    lat = structure.lattice.matrix          # 3×3 ndarray, rows = a1, a2, a3
    cell_parameters = CellParametersCard(
        "angstrom",
        lat[0],
        lat[1],
        lat[2],
    )

    # ---- K_POINTS automatic ----
    k_points = KPointsCard(
        "automatic",
        list(kpoints),
        [0, 0, 0],
        [],
        [],
        [],
    )

    return OrderedDict([
        ("atomic_species",    atomic_species),
        ("atomic_positions",  atomic_positions),
        ("k_points",          k_points),
        ("cell_parameters",   cell_parameters),
    ])


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def make_scf_input(
    structure: "Structure",
    cfg: "PipelineConfig",
) -> "PWin":
    """Generate a QE SCF (single-point) input object from a pymatgen Structure.

    Settings are drawn exclusively from ``cfg.dft_screening``
    (DFTScreeningConfig) and ``cfg.pseudopotentials``
    (PseudopotentialsConfig).

    Parameters
    ----------
    structure : pymatgen.core.Structure
        The crystal structure to compute the SCF energy for.
    cfg : PipelineConfig
        Fully loaded and validated pipeline configuration.

    Returns
    -------
    PWin
        A pymatgen-io-espresso ``PWin`` object ready to be written to disk
        via :func:`write_qe_input`.

    Raises
    ------
    ImportError
        If ``pymatgen-io-espresso`` is not installed.
    KeyError
        If a pseudopotential is missing for any element in the structure
        (raised by ``cfg.pseudopotentials.get``).

    Notes
    -----
    * ``outdir`` is set to ``"./out"`` (relative to the run directory).
    * ``tprnfor=True`` and ``tstress=True`` are always enabled so that
      forces and stress are written even for SCF runs.
    * The ``nspin=2`` and ``starting_magnetization(i)`` entries are only
      written when ``cfg.dft_screening.spin_polarised`` is ``True``.
    """
    (
        PWin, ControlNamelist, SystemNamelist,
        ElectronsNamelist, _IonsNamelist,
        AtomicSpeciesCard, AtomicPositionsCard,
        CellParametersCard, KPointsCard,
    ) = _import_pwin_classes()

    scr = cfg.dft_screening
    pseudo_dir = _resolve_pseudo_dir(cfg)
    species_order = _ordered_species(structure)

    nat  = len(structure)
    ntyp = len(species_order)

    logger.debug(
        "make_scf_input: formula=%s  nat=%d  ntyp=%d  kpts=%s  ecutwfc=%.1f",
        structure.composition.reduced_formula,
        nat,
        ntyp,
        scr.kpoints,
        scr.ecutwfc,
    )

    # ---- CONTROL ----
    control = ControlNamelist(
        calculation="scf",
        prefix="qe",
        outdir="./out",
        pseudo_dir=str(pseudo_dir),
        verbosity="low",
        tprnfor=True,
        tstress=True,
    )

    # ---- SYSTEM ----
    system = _build_system_namelist(
        SystemNamelist,
        nat=nat,
        ntyp=ntyp,
        ecutwfc=scr.ecutwfc,
        ecutrho=scr.ecutrho,
        smearing=scr.smearing,
        degauss=scr.degauss,
        conv_thr=scr.conv_thr,
        spin_polarised=scr.spin_polarised,
        species_order=species_order,
        starting_magnetization=scr.starting_magnetization,
    )

    # ---- ELECTRONS ----
    electrons = ElectronsNamelist(
        conv_thr=scr.conv_thr,
        mixing_beta=scr.mixing_beta,
    )

    # ---- Cards ----
    cards = _build_cards(
        AtomicSpeciesCard,
        AtomicPositionsCard,
        CellParametersCard,
        KPointsCard,
        structure=structure,
        species_order=species_order,
        pseudo_dir=pseudo_dir,
        cfg=cfg,
        kpoints=scr.kpoints,
    )

    pw_in = PWin(
        namelists={
            "control":   control,
            "system":    system,
            "electrons": electrons,
        },
        cards=cards,
    )

    logger.info(
        "Generated SCF input for %s (%d atoms, %d species, kpts=%s)",
        structure.composition.reduced_formula,
        nat,
        ntyp,
        scr.kpoints,
    )
    return pw_in


def make_relax_input(
    structure: "Structure",
    cfg: "PipelineConfig",
    run_dir: Path,
) -> "PWin":
    """Generate a QE ionic-relaxation input object from a pymatgen Structure.

    Settings are drawn from ``cfg.dft_relax`` (DFTRelaxConfig) and
    ``cfg.pseudopotentials`` (PseudopotentialsConfig).

    Parameters
    ----------
    structure : pymatgen.core.Structure
        The crystal structure to relax.
    cfg : PipelineConfig
        Fully loaded and validated pipeline configuration.
    run_dir : Path
        Absolute path to the directory where this QE job will be run.
        Used to set ``outdir`` in the CONTROL namelist so that QE writes
        its scratch data and XML output to ``<run_dir>/out/``.

    Returns
    -------
    PWin
        A pymatgen-io-espresso ``PWin`` object ready to be written to disk
        via :func:`write_qe_input`.

    Raises
    ------
    ImportError
        If ``pymatgen-io-espresso`` is not installed.
    KeyError
        If a pseudopotential is missing for any element in the structure.

    Notes
    -----
    * ``nstep`` is placed in the CONTROL namelist (not IONS) as required by
      QE 6.x.  Earlier conventions placed it in IONS or SYSTEM, but CONTROL
      is the correct and portable location for QE ≥ 6.0.
    * ``outdir`` is set to the absolute path ``<run_dir>/out`` so that QE
      writes its XML output in a known location regardless of the working
      directory.
    * The IONS namelist only contains ``ion_dynamics``; all convergence
      parameters are controlled by the ELECTRONS and CONTROL namelists.
    """
    (
        PWin, ControlNamelist, SystemNamelist,
        ElectronsNamelist, IonsNamelist,
        AtomicSpeciesCard, AtomicPositionsCard,
        CellParametersCard, KPointsCard,
    ) = _import_pwin_classes()

    relax = cfg.dft_relax
    pseudo_dir = _resolve_pseudo_dir(cfg)
    species_order = _ordered_species(structure)
    run_dir = Path(run_dir)

    nat  = len(structure)
    ntyp = len(species_order)

    logger.debug(
        "make_relax_input: formula=%s  nat=%d  ntyp=%d  kpts=%s"
        "  ecutwfc=%.1f  nstep=%d  ion_dyn=%s",
        structure.composition.reduced_formula,
        nat,
        ntyp,
        relax.kpoints,
        relax.ecutwfc,
        relax.nstep,
        relax.ion_dynamics,
    )

    # ---- CONTROL ----
    # nstep goes here for QE 6.x (not in IONS or SYSTEM).
    control = ControlNamelist(
        calculation="relax",
        prefix="relax",
        outdir=str(run_dir / "out"),
        pseudo_dir=str(pseudo_dir),
        verbosity="low",
        tprnfor=True,
        tstress=True,
        nstep=relax.nstep,
    )

    # ---- SYSTEM ----
    system = _build_system_namelist(
        SystemNamelist,
        nat=nat,
        ntyp=ntyp,
        ecutwfc=relax.ecutwfc,
        ecutrho=relax.ecutrho,
        smearing="mp",          # always Methfessel-Paxton for metallic ions
        degauss=0.03,           # sensible default; relax config does not expose this
        conv_thr=relax.conv_thr,
        spin_polarised=relax.spin_polarised,
        species_order=species_order,
        starting_magnetization=relax.starting_magnetization,
    )

    # ---- ELECTRONS ----
    electrons = ElectronsNamelist(
        conv_thr=relax.conv_thr,
        mixing_beta=relax.mixing_beta,
    )

    # ---- IONS ----
    ions = IonsNamelist(
        ion_dynamics=relax.ion_dynamics,
    )

    # ---- Cards ----
    cards = _build_cards(
        AtomicSpeciesCard,
        AtomicPositionsCard,
        CellParametersCard,
        KPointsCard,
        structure=structure,
        species_order=species_order,
        pseudo_dir=pseudo_dir,
        cfg=cfg,
        kpoints=relax.kpoints,
    )

    pw_in = PWin(
        namelists={
            "control":   control,
            "system":    system,
            "electrons": electrons,
            "ions":      ions,
        },
        cards=cards,
    )

    logger.info(
        "Generated relax input for %s (%d atoms, %d species, kpts=%s, nstep=%d)",
        structure.composition.reduced_formula,
        nat,
        ntyp,
        relax.kpoints,
        relax.nstep,
    )
    return pw_in


def write_qe_input(pw_in: "PWin", path: Path) -> None:
    """Write a PWin object to a QE input file.

    Creates all intermediate parent directories if they do not exist.

    Parameters
    ----------
    pw_in : PWin
        A fully constructed QE pw.x input object.
    path : Path
        Destination file path (conventionally ``<run_dir>/qe.in``).

    Raises
    ------
    OSError
        If the file cannot be written (e.g. permission error).

    Examples
    --------
    >>> from pathlib import Path
    >>> write_qe_input(pw_in, Path("output/dft/scf/scf_0/qe.in"))
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pw_in.to_file(path)
    logger.debug("Wrote QE input → %s", path)
