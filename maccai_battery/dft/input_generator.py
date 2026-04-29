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
    pseudo_dir = Path(cfg.pseudopotentials.pseudo_dir)
    if not pseudo_dir.is_absolute():
        pseudo_dir = cfg.project._root / pseudo_dir
    return pseudo_dir


def _ordered_species(structure: "Structure") -> list[str]:
    seen: set[str] = set()
    order: list[str] = []
    for site in structure:
        el = str(site.specie)
        if el not in seen:
            seen.add(el)
            order.append(el)
    return order


# ---------------------------------------------------------------------------
# FIX: safe helper that reads a config value from either a Pydantic object
#      or a plain dict — avoids AttributeError / .get()-on-object crashes.
# ---------------------------------------------------------------------------
def _cfg_get(obj, key, default=None):
    """Read key from a Pydantic model or a dict, returning default if absent."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


# ---------------------------------------------------------------------------
# FIX: centralised DFT+U writer — used by both make_scf_input and
#      make_relax_input.  hub_cfg must come from the correct sub-section:
#        SCF   → cfg.dft_screening.hubbard_u
#        Relax → cfg.dft_relax.hubbard_u
# ---------------------------------------------------------------------------
def _apply_hubbard_u(system, hub_cfg, species_order: list[str]) -> None:
    """Write lda_plus_u / Hubbard_U entries into a SystemNamelist.

    Parameters
    ----------
    system : SystemNamelist
        The system namelist object to mutate.
    hub_cfg : object | dict | None
        The hubbard_u config section (cfg.dft_screening.hubbard_u or
        cfg.dft_relax.hubbard_u).  May be a Pydantic model or a dict.
    species_order : list of str
        Element symbols in the same order as ATOMIC_SPECIES (1-indexed).
    """
    if not _cfg_get(hub_cfg, "enabled", False):
        return

    system["lda_plus_u"] = True
    system["lda_plus_u_kind"] = _cfg_get(hub_cfg, "lda_plus_u_kind", 0)

    u_map = _cfg_get(hub_cfg, "U", {}) or {}
    for idx, sym in enumerate(species_order, start=1):
        # u_map may be a Pydantic model or a plain dict
        if isinstance(u_map, dict):
            u_val = u_map.get(sym, 0.0)
        else:
            u_val = getattr(u_map, sym, 0.0)
        system[f"Hubbard_U({idx})"] = u_val
        logger.debug("  Hubbard_U(%d) = %.2f  [%s]", idx, u_val, sym)

    logger.info(
        "DFT+U applied: lda_plus_u_kind=%d  U=%s",
        _cfg_get(hub_cfg, "lda_plus_u_kind", 0),
        {sym: (_cfg_get(u_map, sym, 0.0) if not isinstance(u_map, dict)
               else u_map.get(sym, 0.0))
         for sym in species_order},
    )


def _build_system_namelist(
    SystemNamelist,
    nat: int,
    ntyp: int,
    ecutwfc: float,
    ecutrho: float,
    smearing: str,
    degauss: float,
    spin_polarised: bool,
    species_order: list[str],
    starting_magnetization: dict[str, float],
):
    """Construct a SystemNamelist with optional spin-polarisation parameters."""
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
    """Build the four QE cards needed for a pw.x input."""
    from pymatgen.core import Element  # type: ignore[import]

    masses = [float(Element(el).atomic_mass) for el in species_order]
    pseudos = [cfg.pseudopotentials.get(el) for el in species_order]
    atomic_species = AtomicSpeciesCard(None, species_order, masses, pseudos)
    logger.debug("ATOMIC_SPECIES: %s", list(zip(species_order, pseudos)))

    pos_symbols = [str(site.specie) for site in structure]
    pos_coords = np.array([site.frac_coords for site in structure])
    atomic_positions = AtomicPositionsCard("crystal", pos_symbols, pos_coords, None)

    lat = structure.lattice.matrix
    cell_parameters = CellParametersCard("angstrom", lat[0], lat[1], lat[2])

    k_points = KPointsCard("automatic", list(kpoints), [0, 0, 0], [], [], [])

    return OrderedDict([
        ("atomic_species",   atomic_species),
        ("atomic_positions", atomic_positions),
        ("k_points",         k_points),
        ("cell_parameters",  cell_parameters),
    ])


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def make_scf_input(
    structure: "Structure",
    cfg: "PipelineConfig",
) -> "PWin":
    """Generate a QE SCF (single-point) input object from a pymatgen Structure."""
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
        structure.composition.reduced_formula, nat, ntyp, scr.kpoints, scr.ecutwfc,
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
        smearing=scr.smearing,          # from config — mv @ 0.01 Ry
        degauss=scr.degauss,
        spin_polarised=scr.spin_polarised,
        species_order=species_order,
        starting_magnetization=scr.starting_magnetization,
    )

    # ---- DFT+U (FIXED: read from cfg.dft_screening, not cfg) ----
    _apply_hubbard_u(
        system,
        hub_cfg=getattr(cfg.dft_screening, "hubbard_u", None),
        species_order=species_order,
    )

    # ---- ELECTRONS (FIXED: mixing_mode + nmix added) ----
    electrons = ElectronsNamelist(
        conv_thr=scr.conv_thr,
        mixing_beta=scr.mixing_beta,
        mixing_mode=getattr(scr, "mixing_mode", "plain"),
        nmix=getattr(scr, "nmix", 8),
    )

    # ---- Cards ----
    cards = _build_cards(
        AtomicSpeciesCard, AtomicPositionsCard, CellParametersCard, KPointsCard,
        structure=structure, species_order=species_order,
        pseudo_dir=pseudo_dir, cfg=cfg, kpoints=scr.kpoints,
    )

    pw_in = PWin(
        namelists={"control": control, "system": system, "electrons": electrons},
        cards=cards,
    )

    logger.info(
        "Generated SCF input for %s (%d atoms, %d species, kpts=%s)",
        structure.composition.reduced_formula, nat, ntyp, scr.kpoints,
    )
    return pw_in


def make_relax_input(
    structure: "Structure",
    cfg: "PipelineConfig",
    run_dir: Path,
) -> "PWin":
    """Generate a QE ionic-relaxation input object from a pymatgen Structure."""
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
        structure.composition.reduced_formula, nat, ntyp,
        relax.kpoints, relax.ecutwfc, relax.nstep, relax.ion_dynamics,
    )

    # ---- CONTROL ----
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

    # ---- SYSTEM (FIXED: smearing and degauss read from config, not hardcoded) ----
    system = _build_system_namelist(
        SystemNamelist,
        nat=nat,
        ntyp=ntyp,
        ecutwfc=relax.ecutwfc,
        ecutrho=relax.ecutrho,
        smearing=relax.smearing,        # FIXED: was hardcoded "mp"
        degauss=relax.degauss,          # FIXED: was hardcoded 0.03
        spin_polarised=relax.spin_polarised,
        species_order=species_order,
        starting_magnetization=relax.starting_magnetization,
    )

    # ---- DFT+U (FIXED: read from cfg.dft_relax, not cfg) ----
    _apply_hubbard_u(
        system,
        hub_cfg=getattr(cfg.dft_relax, "hubbard_u", None),
        species_order=species_order,
    )

    # ---- ELECTRONS (FIXED: mixing_mode + nmix added) ----
    electrons = ElectronsNamelist(
        conv_thr=relax.conv_thr,
        mixing_beta=relax.mixing_beta,
        mixing_mode=getattr(relax, "mixing_mode", "plain"),
        nmix=getattr(relax, "nmix", 8),
    )

    # ---- IONS ----
    ions = IonsNamelist(ion_dynamics=relax.ion_dynamics)

    # ---- Cards ----
    cards = _build_cards(
        AtomicSpeciesCard, AtomicPositionsCard, CellParametersCard, KPointsCard,
        structure=structure, species_order=species_order,
        pseudo_dir=pseudo_dir, cfg=cfg, kpoints=relax.kpoints,
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
        structure.composition.reduced_formula, nat, ntyp, relax.kpoints, relax.nstep,
    )
    return pw_in


def write_qe_input(pw_in: "PWin", path: Path) -> None:
    """Write a PWin object to a QE input file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pw_in.to_file(path)
    logger.debug("Wrote QE input → %s", path)