# =============================================================================
# maccai_battery.dft.parser — Quantum ESPRESSO output parsers
# =============================================================================
# Provides three parser entry-points:
#
#   parse_qe_xml         — parse a QE XML output file (data-file-schema.xml
#                          or <prefix>.xml) into a rich QEXMLResult dataclass.
#                          Extracts cell, positions, energies, forces, stress,
#                          magnetization, and Fermi energy.
#
#   parse_scf_stdout     — lightweight SCF stdout parser; re-uses the proven
#                          helpers from maccai_battery.utils.
#
#   parse_relax_stdout   — like parse_scf_stdout but also counts ionic steps.
#
# All unit conversions follow CODATA 2018 constants (same as maccai_battery.utils):
#   1 Bohr  = 0.529177210903 Å
#   1 Ha    = 27.211386245988 eV  (= 2 Ry)
#   1 Ry    = 13.605693122994 eV
#
# Implementation notes
#   • No external XML library is required — stdlib xml.etree.ElementTree only.
#   • XML namespace prefixes are stripped by splitting on "}" so that the code
#     is robust to any namespace URI QE chooses to emit.
#   • Cell vectors in the QE XML are in Bohr; lattice_a/b/c are returned in Å.
#   • Energies in the XML are in Hartree; eV values are derived on the fly.
#   • Forces are Ha/Bohr per atom (as stored by QE).
#   • Stress is Ha/Bohr³ (as stored by QE).
#
# Usage
# -----
#   from maccai_battery.dft.parser import parse_qe_xml, parse_scf_stdout
#
#   result = parse_qe_xml(Path("output/dft/scf/scf_0/out/qe.xml"))
#   print(result.final_energy_eV, result.lattice_a)
#
#   summary = parse_scf_stdout(Path("output/dft/scf/scf_0/qe.out").read_text())
#   print(summary["total_energy_eV"], summary["scf_converged"])
# =============================================================================

from __future__ import annotations

import logging
import math
import re
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal XML helpers
# ---------------------------------------------------------------------------


def _local(tag: str) -> str:
    """Strip XML namespace prefix from *tag*.

    QE XML files use namespace-prefixed tags of the form ``{namespace}tagname``.
    This function returns only the ``tagname`` part, making tag comparisons
    namespace-agnostic.

    Parameters
    ----------
    tag : str
        Raw XML tag string, e.g. ``"{http://www.quantum-espresso.org/ns/qes/...}cell"``.

    Returns
    -------
    str
        Local tag name, e.g. ``"cell"``.
    """
    return tag.split("}")[-1]


def _find_all_by_local(root: ET.Element, name: str) -> List[ET.Element]:
    """Find all elements anywhere under *root* whose local tag equals *name*.

    Parameters
    ----------
    root : xml.etree.ElementTree.Element
        Root or sub-tree element to search recursively.
    name : str
        Local (namespace-stripped) tag name to match.

    Returns
    -------
    list of Element
    """
    return [el for el in root.iter() if _local(el.tag) == name]


def _child_text(parent: ET.Element, local_name: str) -> Optional[str]:
    """Return the text of the first direct child of *parent* with local tag *local_name*.

    Parameters
    ----------
    parent : Element
    local_name : str

    Returns
    -------
    str or None
    """
    for child in parent:
        if _local(child.tag) == local_name:
            return child.text
    return None


def _safe_float(text: Optional[str]) -> Optional[float]:
    """Parse *text* as a float, returning ``None`` on any failure.

    Parameters
    ----------
    text : str or None

    Returns
    -------
    float or None
    """
    if not text:
        return None
    try:
        return float(text.strip())
    except (ValueError, AttributeError):
        return None


def _safe_float_list(text: Optional[str]) -> Optional[List[float]]:
    """Parse *text* as a whitespace-separated list of floats.

    Parameters
    ----------
    text : str or None

    Returns
    -------
    list of float or None
        ``None`` if *text* is empty or unparsable.
    """
    if not text or not text.strip():
        return None
    try:
        return [float(tok) for tok in text.split()]
    except (ValueError, AttributeError):
        return None


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class QEXMLResult:
    """Parsed result from a QE XML output file.

    This dataclass captures all quantities written to the QE
    ``data-file-schema.xml`` (or ``<prefix>.xml``) output, converting
    internal QE units (Bohr, Hartree) to the physical units noted below.

    The dataclass is split into *required* fields (always set by the parser,
    even if to empty collections) and *optional* fields that are populated
    only when the corresponding data is present in the XML.

    Parameters / Attributes
    -----------------------
    formula : str
        Reduced chemical formula derived from the atomic species in the XML,
        e.g. ``"Fe4Li4O16P4"``.
    chemical_system : str
        Dash-separated element string in alphabetical order,
        e.g. ``"Fe-Li-O-P"``.
    n_atoms : int
        Total number of atoms in the simulation cell.
    element_counts : dict[str, int]
        Mapping from element symbol to atom count,
        e.g. ``{"Li": 4, "Fe": 4, "P": 4, "O": 16}``.
    forces : list[list[float]]
        Per-atom forces ``[[fx, fy, fz], ...]`` in Ha/Bohr (QE native units).
        Empty list if no force data was found.
    atomic_positions_cartesian : list[tuple[str, list[float]]]
        Cartesian atomic positions ``[(element, [x, y, z]), ...]`` in **Bohr**.
    atomic_positions_fractional : list[tuple[str, list[float]]]
        Fractional (crystal) atomic positions ``[(element, [fx, fy, fz]), ...]``.
        Empty if no cell was found in the XML.

    Optional fields
    ---------------
    cell_matrix : np.ndarray or None
        3×3 lattice matrix in **Bohr**, rows = lattice vectors a1, a2, a3.
    lattice_a, lattice_b, lattice_c : float or None
        Lattice vector magnitudes in **Ångström**.
    lattice_alpha, lattice_beta, lattice_gamma : float or None
        Inter-vector angles in **degrees**.
    volume_A3 : float or None
        Unit-cell volume in **Å³**.
    final_energy_hartree : float or None
        Total energy from the last SCF / ionic step in **Hartree**.
    final_energy_eV : float or None
        Total energy converted to **eV**.
    final_energy_Ry : float or None
        Total energy converted to **Rydberg** (= 2 × Hartree).
    energy_per_atom_eV : float or None
        ``final_energy_eV / n_atoms``.
    magnetization_total : float or None
        Total magnetization (μ_B / cell).
    magnetization_absolute : float or None
        Absolute magnetization (μ_B / cell).
    fermi_energy_hartree : float or None
        Fermi energy in **Hartree**.
    stress_matrix : np.ndarray or None
        3×3 stress tensor in **Ha/Bohr³**.
    """

    # -----------------------------------------------------------------------
    # Required fields — always populated (may be empty collections)
    # -----------------------------------------------------------------------
    formula: str
    chemical_system: str
    n_atoms: int
    element_counts: Dict[str, int]
    forces: List[List[float]]
    atomic_positions_cartesian: List[Tuple[str, List[float]]]
    atomic_positions_fractional: List[Tuple[str, List[float]]]

    # -----------------------------------------------------------------------
    # Optional fields — present only when the XML contains the relevant data
    # -----------------------------------------------------------------------
    cell_matrix: Optional[np.ndarray] = None
    lattice_a: Optional[float] = None
    lattice_b: Optional[float] = None
    lattice_c: Optional[float] = None
    lattice_alpha: Optional[float] = None
    lattice_beta: Optional[float] = None
    lattice_gamma: Optional[float] = None
    volume_A3: Optional[float] = None
    final_energy_hartree: Optional[float] = None
    final_energy_eV: Optional[float] = None
    final_energy_Ry: Optional[float] = None
    energy_per_atom_eV: Optional[float] = None
    magnetization_total: Optional[float] = None
    magnetization_absolute: Optional[float] = None
    fermi_energy_hartree: Optional[float] = None
    stress_matrix: Optional[np.ndarray] = None

    # -----------------------------------------------------------------------
    # Methods
    # -----------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Serialise to a JSON-compatible dictionary.

        ``numpy`` arrays are converted to nested Python lists so that the
        result can be passed directly to ``json.dumps`` or stored in NDJSON.

        Returns
        -------
        dict
        """
        return {
            "formula": self.formula,
            "chemical_system": self.chemical_system,
            "n_atoms": self.n_atoms,
            "element_counts": self.element_counts,
            # Lattice
            "cell_matrix": (
                self.cell_matrix.tolist() if self.cell_matrix is not None else None
            ),
            "lattice_a": self.lattice_a,
            "lattice_b": self.lattice_b,
            "lattice_c": self.lattice_c,
            "lattice_alpha": self.lattice_alpha,
            "lattice_beta": self.lattice_beta,
            "lattice_gamma": self.lattice_gamma,
            "volume_A3": self.volume_A3,
            # Energies
            "final_energy_hartree": self.final_energy_hartree,
            "final_energy_eV": self.final_energy_eV,
            "final_energy_Ry": self.final_energy_Ry,
            "energy_per_atom_eV": self.energy_per_atom_eV,
            # Magnetism & Fermi level
            "magnetization_total": self.magnetization_total,
            "magnetization_absolute": self.magnetization_absolute,
            "fermi_energy_hartree": self.fermi_energy_hartree,
            # Forces & stress
            "forces": self.forces,
            "stress_matrix": (
                self.stress_matrix.tolist() if self.stress_matrix is not None else None
            ),
            # Positions
            "atomic_positions_cartesian": [
                [el, list(pos)] for el, pos in self.atomic_positions_cartesian
            ],
            "atomic_positions_fractional": [
                [el, list(pos)] for el, pos in self.atomic_positions_fractional
            ],
        }


# ---------------------------------------------------------------------------
# XML parser
# ---------------------------------------------------------------------------


def parse_qe_xml(xml_path: Path) -> QEXMLResult:
    """Parse a QE XML output file into a :class:`QEXMLResult`.

    Works with both ``data-file-schema.xml`` (the default QE ≥ 6.x output)
    and the older ``<prefix>.xml`` format.

    The parser uses :mod:`xml.etree.ElementTree` from the standard library —
    no additional XML dependencies are required.  All tag lookups strip XML
    namespace prefixes so that the code works regardless of which namespace
    URI QE has emitted.

    Unit conversions performed
    --------------------------
    * Cell vectors : Bohr → Å  (for ``lattice_a/b/c`` and ``volume_A3``)
    * Energies     : Hartree → eV  (for ``final_energy_eV``)
                     Hartree × 2   (for ``final_energy_Ry``)
    * Angles       : computed via arccos, returned in degrees
    * Fractional positions : derived from the Cartesian positions and
      the inverse-transposed cell matrix.

    Parameters
    ----------
    xml_path : Path
        Path to the QE XML file, e.g. ``<run_dir>/out/qe.xml`` or
        ``<run_dir>/out/data-file-schema.xml``.

    Returns
    -------
    QEXMLResult

    Raises
    ------
    FileNotFoundError
        If *xml_path* does not exist.
    xml.etree.ElementTree.ParseError
        If the XML is malformed.

    Notes
    -----
    The parser is intentionally lenient: most fields are treated as optional
    and default to ``None`` / empty list if the corresponding XML element is
    absent.  This makes the function usable even on incomplete or truncated
    XML files (e.g. from a crashed relax run).
    """
    from maccai_battery.utils import ha_to_ev  # avoid circular imports

    # Physical constant (same as maccai_battery.utils.BOHR_TO_ANG)
    _BOHR_TO_ANG: float = 0.529177210903

    xml_path = Path(xml_path)
    if not xml_path.exists():
        raise FileNotFoundError(
            f"QE XML output file not found: {xml_path}\n"
            "Ensure the QE job completed and wrote the XML file inside the 'out/' sub-directory."
        )

    logger.debug("Parsing QE XML: %s", xml_path)
    root = ET.parse(xml_path).getroot()

    # ------------------------------------------------------------------
    # 1) Cell vectors  (Bohr)
    # ------------------------------------------------------------------
    cell_matrix: Optional[np.ndarray] = None

    for cell_node in _find_all_by_local(root, "cell"):
        a1 = _safe_float_list(_child_text(cell_node, "a1"))
        a2 = _safe_float_list(_child_text(cell_node, "a2"))
        a3 = _safe_float_list(_child_text(cell_node, "a3"))
        if a1 and a2 and a3 and len(a1) == 3 and len(a2) == 3 and len(a3) == 3:
            cell_matrix = np.array([a1, a2, a3], dtype=float)  # rows = lattice vectors
            break

    # ------------------------------------------------------------------
    # 2) Derived lattice parameters
    # ------------------------------------------------------------------
    lattice_a: Optional[float] = None
    lattice_b: Optional[float] = None
    lattice_c: Optional[float] = None
    lattice_alpha: Optional[float] = None
    lattice_beta: Optional[float] = None
    lattice_gamma: Optional[float] = None
    volume_A3: Optional[float] = None

    if cell_matrix is not None:
        a_bohr = float(np.linalg.norm(cell_matrix[0]))
        b_bohr = float(np.linalg.norm(cell_matrix[1]))
        c_bohr = float(np.linalg.norm(cell_matrix[2]))

        lattice_a = a_bohr * _BOHR_TO_ANG
        lattice_b = b_bohr * _BOHR_TO_ANG
        lattice_c = c_bohr * _BOHR_TO_ANG

        def _safe_angle_deg(v1: np.ndarray, n1: float, v2: np.ndarray, n2: float) -> float:
            """Return the angle (degrees) between *v1* and *v2* with clipping."""
            cos_val = float(np.dot(v1, v2) / (n1 * n2))
            cos_val = max(-1.0, min(1.0, cos_val))  # guard against ±1.0000000001
            return math.degrees(math.acos(cos_val))

        lattice_alpha = _safe_angle_deg(cell_matrix[1], b_bohr, cell_matrix[2], c_bohr)
        lattice_beta  = _safe_angle_deg(cell_matrix[0], a_bohr, cell_matrix[2], c_bohr)
        lattice_gamma = _safe_angle_deg(cell_matrix[0], a_bohr, cell_matrix[1], b_bohr)

        vol_bohr3 = abs(float(np.dot(cell_matrix[0], np.cross(cell_matrix[1], cell_matrix[2]))))
        volume_A3 = vol_bohr3 * (_BOHR_TO_ANG ** 3)

        logger.debug(
            "Cell: a=%.4f Å  b=%.4f Å  c=%.4f Å  "
            "α=%.2f°  β=%.2f°  γ=%.2f°  V=%.2f Å³",
            lattice_a, lattice_b, lattice_c,
            lattice_alpha, lattice_beta, lattice_gamma,
            volume_A3,
        )

    # ------------------------------------------------------------------
    # 3) Atomic positions  (Bohr, Cartesian)
    # ------------------------------------------------------------------
    atoms_cartesian: List[Tuple[str, List[float]]] = []

    for atom_node in _find_all_by_local(root, "atom"):
        # Different QE versions use different attribute names for the element
        element = (
            atom_node.attrib.get("name")
            or atom_node.attrib.get("type")
            or atom_node.attrib.get("label")
        )
        if not element:
            continue
        element = element.strip()

        xyz = _safe_float_list(atom_node.text)
        if xyz and len(xyz) == 3:
            atoms_cartesian.append((element, xyz))

    nat = len(atoms_cartesian)
    logger.debug("Found %d atoms in XML", nat)

    # ------------------------------------------------------------------
    # 4) Chemical formula and element counts
    # ------------------------------------------------------------------
    elem_names = [el for el, _ in atoms_cartesian]
    counts = Counter(elem_names)

    # Alphabetically ordered formula, e.g. "Fe4Li4O16P4"
    formula = "".join(f"{el}{counts[el]}" for el in sorted(counts.keys()))
    chemical_system = "-".join(sorted(counts.keys()))
    element_counts: Dict[str, int] = dict(counts)

    logger.debug("Formula: %s  system: %s", formula, chemical_system)

    # ------------------------------------------------------------------
    # 5) Fractional coordinates (derived from Cartesian + cell)
    # ------------------------------------------------------------------
    atoms_fractional: List[Tuple[str, List[float]]] = []

    if cell_matrix is not None and nat > 0:
        # xyz_cart = cell.T @ frac  ⟹  frac = inv(cell.T) @ xyz_cart
        cell_inv = np.linalg.inv(cell_matrix.T)
        for el, xyz in atoms_cartesian:
            frac = (cell_inv @ np.array(xyz)).tolist()
            atoms_fractional.append((el, frac))

    # ------------------------------------------------------------------
    # 6) Total energies  (Hartree in XML)
    # ------------------------------------------------------------------
    energies_ha: List[float] = []

    for te_node in _find_all_by_local(root, "total_energy"):
        etot_text = _child_text(te_node, "etot")
        etot = _safe_float(etot_text)
        if etot is not None:
            energies_ha.append(etot)

    # Use the LAST entry — final SCF iteration (for relax: final ionic step)
    final_energy_hartree: Optional[float] = energies_ha[-1] if energies_ha else None
    final_energy_eV: Optional[float] = (
        ha_to_ev(final_energy_hartree) if final_energy_hartree is not None else None
    )
    # 1 Ha = 2 Ry
    final_energy_Ry: Optional[float] = (
        final_energy_hartree * 2.0 if final_energy_hartree is not None else None
    )
    energy_per_atom_eV: Optional[float] = (
        final_energy_eV / nat
        if (final_energy_eV is not None and nat > 0)
        else None
    )

    if final_energy_eV is not None:
        logger.debug(
            "Final energy: %.8f Ha  =  %.8f eV  =  %.8f Ry  (%.6f eV/atom)",
            final_energy_hartree,
            final_energy_eV,
            final_energy_Ry,
            energy_per_atom_eV or 0.0,
        )

    # ------------------------------------------------------------------
    # 7) Magnetization
    # ------------------------------------------------------------------
    magnetization_total: Optional[float] = None
    magnetization_absolute: Optional[float] = None

    for mag_node in _find_all_by_local(root, "magnetization"):
        for child in mag_node:
            tag = _local(child.tag)
            if tag == "total":
                magnetization_total = _safe_float(child.text)
            elif tag == "absolute":
                magnetization_absolute = _safe_float(child.text)

    # ------------------------------------------------------------------
    # 8) Fermi energy  (Hartree in XML)
    # ------------------------------------------------------------------
    fermi_energy_hartree: Optional[float] = None

    for fe_node in _find_all_by_local(root, "fermi_energy"):
        tmp = _safe_float(fe_node.text)
        if tmp is not None:
            fermi_energy_hartree = tmp
            # Keep the last / only occurrence

    # ------------------------------------------------------------------
    # 9) Per-atom forces  (Ha/Bohr)
    # ------------------------------------------------------------------
    forces: List[List[float]] = []

    for f_node in _find_all_by_local(root, "force"):
        fvals = _safe_float_list(f_node.text)
        if fvals and len(fvals) == 3:
            forces.append(fvals)

    # ------------------------------------------------------------------
    # 10) Stress tensor  (Ha/Bohr³, 3×3)
    # ------------------------------------------------------------------
    # Iterate all occurrences and keep the LAST valid entry.
    # For relax runs the XML contains one stress block per ionic step;
    # we want the final (most-relaxed) geometry's stress.
    stress_matrix: Optional[np.ndarray] = None

    for st_node in _find_all_by_local(root, "stress"):
        vals = _safe_float_list(st_node.text)
        if vals and len(vals) == 9:
            stress_matrix = np.array(vals, dtype=float).reshape(3, 3)

    logger.debug(
        "Parsed XML: nat=%d  forces=%d  has_stress=%s  has_mag=%s  E=%.6f eV",
        nat,
        len(forces),
        stress_matrix is not None,
        magnetization_total is not None,
        final_energy_eV or 0.0,
    )

    return QEXMLResult(
        formula=formula,
        chemical_system=chemical_system,
        n_atoms=nat,
        element_counts=element_counts,
        forces=forces,
        atomic_positions_cartesian=atoms_cartesian,
        atomic_positions_fractional=atoms_fractional,
        cell_matrix=cell_matrix,
        lattice_a=lattice_a,
        lattice_b=lattice_b,
        lattice_c=lattice_c,
        lattice_alpha=lattice_alpha,
        lattice_beta=lattice_beta,
        lattice_gamma=lattice_gamma,
        volume_A3=volume_A3,
        final_energy_hartree=final_energy_hartree,
        final_energy_eV=final_energy_eV,
        final_energy_Ry=final_energy_Ry,
        energy_per_atom_eV=energy_per_atom_eV,
        magnetization_total=magnetization_total,
        magnetization_absolute=magnetization_absolute,
        fermi_energy_hartree=fermi_energy_hartree,
        stress_matrix=stress_matrix,
    )


# ---------------------------------------------------------------------------
# Stdout parsers
# ---------------------------------------------------------------------------


def parse_scf_stdout(stdout: str) -> dict:
    """Parse QE SCF stdout and return a concise energy summary.

    Delegates to the proven helpers in :mod:`maccai_battery.utils` so that
    all energy extraction logic lives in one place.

    Parameters
    ----------
    stdout : str
        Full text of a ``pw.x`` SCF run (contents of ``qe.out``).

    Returns
    -------
    dict
        Keys:

        ``total_energy_Ry``
            Total energy from the last converged SCF iteration, in Ry.
            ``None`` if the ``!  total energy`` line was not found.
        ``total_energy_eV``
            Same energy in eV.
        ``n_atoms``
            Number of atoms parsed from ``"number of atoms/cell"``.
        ``energy_per_atom_eV``
            ``total_energy_eV / n_atoms``, or ``None`` if either is absent.
        ``scf_converged``
            ``True`` if ``"convergence has been achieved"`` appears in *stdout*.
        ``fermi_energy_Ry``
            Fermi energy in Ry (converted from the eV value printed by QE),
            or ``None`` if not found.

    Notes
    -----
    QE prints the Fermi energy in eV in its stdout
    (``"the Fermi energy is X ev"``).  The value is converted to Ry here
    so that all energy-like quantities in the returned dict use the same
    Ry unit as ``total_energy_Ry``.
    """
    from maccai_battery.utils import (
        ev_to_ry,
        parse_qe_fermi_energy,
        parse_qe_n_atoms,
        parse_qe_scf_converged,
        parse_qe_total_energy_ry,
        ry_to_ev,
    )

    e_ry = parse_qe_total_energy_ry(stdout)
    e_ev = ry_to_ev(e_ry) if e_ry is not None else None
    n_atoms = parse_qe_n_atoms(stdout)
    converged = parse_qe_scf_converged(stdout)

    # QE prints "the Fermi energy is X ev" → convert to Ry for consistency
    fermi_ev = parse_qe_fermi_energy(stdout)
    fermi_ry = ev_to_ry(fermi_ev) if fermi_ev is not None else None

    e_per_atom = (
        e_ev / n_atoms if (e_ev is not None and n_atoms) else None
    )

    logger.debug(
        "SCF stdout: E=%.6f Ry  n_atoms=%s  converged=%s",
        e_ry or 0.0,
        n_atoms,
        converged,
    )

    return {
        "total_energy_Ry":   e_ry,
        "total_energy_eV":   e_ev,
        "n_atoms":           n_atoms,
        "energy_per_atom_eV": e_per_atom,
        "scf_converged":     converged,
        "fermi_energy_Ry":   fermi_ry,
    }


def parse_relax_stdout(stdout: str) -> dict:
    """Parse QE ionic-relaxation stdout and return an energy + step summary.

    Extends :func:`parse_scf_stdout` with an ``n_ionic_steps`` count.
    The *final* total energy (the last ``!  total energy`` line) is returned,
    which corresponds to the energy of the last successfully converged ionic
    step.

    Parameters
    ----------
    stdout : str
        Full text of a ``pw.x`` relax run (contents of ``qe.out``).

    Returns
    -------
    dict
        All keys returned by :func:`parse_scf_stdout`, plus:

        ``n_ionic_steps``
            Number of ionic steps for which SCF converged.  Counted as the
            number of ``!    total energy`` lines in *stdout* (one per
            converged SCF cycle).

    Notes
    -----
    * ``total_energy_Ry`` / ``total_energy_eV`` reflect the **last** ionic
      step, i.e. the geometry closest to the (local) minimum.
    * A relax run that hit the ``nstep`` limit without completing the final
      step will still return useful energies from the last converged step.
    """
    # parse_scf_stdout already returns the LAST "! total energy" line
    # (parse_qe_total_energy_ry uses re.findall and picks matches[-1]).
    result = parse_scf_stdout(stdout)

    # Count converged ionic steps = number of "!    total energy" lines
    n_ionic_steps = len(
        re.findall(r"!\s+total energy\s+=\s+[-+]?\d+\.\d+", stdout)
    )
    result["n_ionic_steps"] = n_ionic_steps

    logger.debug(
        "Relax stdout: final E=%.6f Ry  ionic_steps=%d  converged=%s",
        result["total_energy_Ry"] or 0.0,
        n_ionic_steps,
        result["scf_converged"],
    )

    return result
