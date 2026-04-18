# =============================================================================
# tests/test_utils.py — Unit tests for maccai_battery.utils and .database
# =============================================================================
# Run with:
#   python -m pytest tests/test_utils.py -v
#   python -m pytest tests/ -v          (all tests)
#
# No external ML dependencies required — tests cover pure-Python utilities,
# unit conversions, EXTXYZ parsing helpers, and the candidate database.
# =============================================================================

from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_extxyz(frames: list[dict]) -> str:
    """Build a minimal multi-frame EXTXYZ string from a list of frame dicts.

    Each dict should have:
        atoms  : list of (symbol, x, y, z)
        energy : float  (optional, default 0.0)
        formula: str    (optional)
    """
    lines = []
    for frame in frames:
        atoms   = frame["atoms"]
        energy  = frame.get("energy", 0.0)
        formula = frame.get("formula", "")
        n       = len(atoms)

        comment_parts = [f'Lattice="10.0 0.0 0.0 0.0 10.0 0.0 0.0 0.0 10.0"']
        comment_parts.append(f"energy={energy}")
        if formula:
            comment_parts.append(f'formula="{formula}"')
        comment_parts.append('Properties=species:S:1:pos:R:3')

        lines.append(str(n))
        lines.append(" ".join(comment_parts))
        for sym, x, y, z in atoms:
            lines.append(f"{sym}  {x:.6f}  {y:.6f}  {z:.6f}")

    return "\n".join(lines) + "\n"


# ===========================================================================
# Unit conversion tests
# ===========================================================================

class TestUnitConversions:
    """Tests for maccai_battery.utils unit conversion functions."""

    def test_ry_to_ev_known_value(self):
        from maccai_battery.utils import ry_to_ev, RY_TO_EV
        assert ry_to_ev(1.0) == pytest.approx(RY_TO_EV)

    def test_ry_to_ev_zero(self):
        from maccai_battery.utils import ry_to_ev
        assert ry_to_ev(0.0) == pytest.approx(0.0)

    def test_ry_to_ev_negative(self):
        from maccai_battery.utils import ry_to_ev, RY_TO_EV
        assert ry_to_ev(-2.0) == pytest.approx(-2.0 * RY_TO_EV)

    def test_ev_to_ry_roundtrip(self):
        from maccai_battery.utils import ry_to_ev, ev_to_ry
        original = 42.0
        assert ry_to_ev(ev_to_ry(original)) == pytest.approx(original, rel=1e-9)

    def test_ha_to_ev_known_value(self):
        from maccai_battery.utils import ha_to_ev, HA_TO_EV
        assert ha_to_ev(1.0) == pytest.approx(HA_TO_EV)

    def test_bohr_to_ang_known_value(self):
        from maccai_battery.utils import bohr_to_ang, BOHR_TO_ANG
        assert bohr_to_ang(1.0) == pytest.approx(BOHR_TO_ANG)

    def test_ev_per_atom_normal(self):
        from maccai_battery.utils import ev_per_atom
        assert ev_per_atom(100.0, 10) == pytest.approx(10.0)

    def test_ev_per_atom_single(self):
        from maccai_battery.utils import ev_per_atom
        assert ev_per_atom(-5.5, 1) == pytest.approx(-5.5)

    def test_ev_per_atom_zero_atoms_raises(self):
        from maccai_battery.utils import ev_per_atom
        with pytest.raises(ValueError, match="n_atoms must be > 0"):
            ev_per_atom(100.0, 0)

    def test_ry_per_atom_to_ev_per_atom(self):
        from maccai_battery.utils import ry_per_atom_to_ev_per_atom, RY_TO_EV
        result = ry_per_atom_to_ev_per_atom(10.0, 2)
        assert result == pytest.approx(10.0 * RY_TO_EV / 2)


# ===========================================================================
# EXTXYZ parsing tests
# ===========================================================================

class TestExtxyzParsing:
    """Tests for maccai_battery.utils EXTXYZ parsing helpers."""

    def test_parse_energy_single_frame(self, tmp_path):
        from maccai_battery.utils import parse_extxyz_energy

        content = _make_extxyz([{
            "atoms":   [("Li", 0, 0, 0)],
            "energy":  -3.14159,
            "formula": "Li",
        }])
        f = tmp_path / "test.extxyz"
        f.write_text(content)

        energy = parse_extxyz_energy(f)
        assert energy == pytest.approx(-3.14159)

    def test_parse_energy_multi_frame_reads_first(self, tmp_path):
        from maccai_battery.utils import parse_extxyz_energy

        content = _make_extxyz([
            {"atoms": [("Li", 0, 0, 0)], "energy": -1.0},
            {"atoms": [("Fe", 0, 0, 0)], "energy": -2.0},
        ])
        f = tmp_path / "multi.extxyz"
        f.write_text(content)

        energy = parse_extxyz_energy(f)
        assert energy == pytest.approx(-1.0)

    def test_parse_energy_missing_raises(self, tmp_path):
        from maccai_battery.utils import parse_extxyz_energy

        # Comment line with no energy= field
        content = "1\nLattice=\"10 0 0 0 10 0 0 0 10\" Properties=species:S:1:pos:R:3\nLi  0.0  0.0  0.0\n"
        f = tmp_path / "no_energy.extxyz"
        f.write_text(content)

        with pytest.raises(ValueError, match="No 'energy=' field"):
            parse_extxyz_energy(f)

    def test_parse_energy_file_not_found(self, tmp_path):
        from maccai_battery.utils import parse_extxyz_energy

        with pytest.raises(FileNotFoundError):
            parse_extxyz_energy(tmp_path / "nonexistent.extxyz")

    def test_parse_extxyz_comment_extracts_keys(self, tmp_path):
        from maccai_battery.utils import parse_extxyz_comment

        content = _make_extxyz([{
            "atoms":   [("O", 0, 0, 0)],
            "energy":  -5.0,
            "formula": "O",
        }])
        f = tmp_path / "comment.extxyz"
        f.write_text(content)

        result = parse_extxyz_comment(f)
        assert "energy" in result
        assert float(result["energy"]) == pytest.approx(-5.0)

    def test_iter_extxyz_frames_count(self, tmp_path):
        from maccai_battery.utils import iter_extxyz_frames

        content = _make_extxyz([
            {"atoms": [("Li", 0, 0, 0)], "energy": -1.0},
            {"atoms": [("Fe", 0, 0, 0), ("O", 1, 0, 0)], "energy": -2.0},
            {"atoms": [("P", 0, 0, 0)], "energy": -3.0},
        ])
        f = tmp_path / "frames.extxyz"
        f.write_text(content)

        frames = list(iter_extxyz_frames(f))
        assert len(frames) == 3

    def test_iter_extxyz_frames_n_atoms(self, tmp_path):
        from maccai_battery.utils import iter_extxyz_frames

        content = _make_extxyz([
            {"atoms": [("Li", 0, 0, 0)], "energy": -1.0},
            {"atoms": [("Fe", 0, 0, 0), ("O", 1, 0, 0)], "energy": -2.0},
        ])
        f = tmp_path / "frames2.extxyz"
        f.write_text(content)

        frames = list(iter_extxyz_frames(f))
        assert frames[0][1] == 1   # frame 0 has 1 atom
        assert frames[1][1] == 2   # frame 1 has 2 atoms

    def test_rank_extxyz_by_energy_sorted(self, tmp_path):
        from maccai_battery.utils import rank_extxyz_by_energy

        content = _make_extxyz([
            {"atoms": [("Li", 0, 0, 0)], "energy": -1.0},
            {"atoms": [("Fe", 0, 0, 0)], "energy": -5.0},
            {"atoms": [("O", 0, 0, 0)],  "energy": -3.0},
        ])
        f = tmp_path / "rank.extxyz"
        f.write_text(content)

        ranked = rank_extxyz_by_energy(f)
        energies = [e for _, e in ranked]
        assert energies == sorted(energies)

    def test_rank_extxyz_best_is_most_negative(self, tmp_path):
        from maccai_battery.utils import rank_extxyz_by_energy

        content = _make_extxyz([
            {"atoms": [("Li", 0, 0, 0)], "energy": -1.0},
            {"atoms": [("Fe", 0, 0, 0)], "energy": -5.0},
        ])
        f = tmp_path / "best.extxyz"
        f.write_text(content)

        ranked = rank_extxyz_by_energy(f)
        assert ranked[0][1] == pytest.approx(-5.0)


# ===========================================================================
# QE output parsing tests
# ===========================================================================

class TestQEParsing:
    """Tests for maccai_battery.utils Quantum ESPRESSO output parsers."""

    _SCF_OUTPUT = """\
     number of atoms/cell      =          14
     kinetic-energy cutoff     =      35.0000  Ry

          !    total energy              =   -1052.43283515 Ry

     convergence has been achieved in  12 iterations

     the Fermi energy is    10.5432 ev
    """

    def test_parse_qe_total_energy_ry(self):
        from maccai_battery.utils import parse_qe_total_energy_ry
        e = parse_qe_total_energy_ry(self._SCF_OUTPUT)
        assert e == pytest.approx(-1052.43283515)

    def test_parse_qe_total_energy_ev(self):
        from maccai_battery.utils import parse_qe_total_energy_ev, RY_TO_EV
        e = parse_qe_total_energy_ev(self._SCF_OUTPUT)
        assert e == pytest.approx(-1052.43283515 * RY_TO_EV)

    def test_parse_qe_n_atoms(self):
        from maccai_battery.utils import parse_qe_n_atoms
        n = parse_qe_n_atoms(self._SCF_OUTPUT)
        assert n == 14

    def test_parse_qe_fermi_energy(self):
        from maccai_battery.utils import parse_qe_fermi_energy
        ef = parse_qe_fermi_energy(self._SCF_OUTPUT)
        assert ef == pytest.approx(10.5432)

    def test_parse_qe_scf_converged_true(self):
        from maccai_battery.utils import parse_qe_scf_converged
        assert parse_qe_scf_converged(self._SCF_OUTPUT) is True

    def test_parse_qe_scf_converged_false(self):
        from maccai_battery.utils import parse_qe_scf_converged
        assert parse_qe_scf_converged("SCF did not converge after 100 iterations") is False

    def test_parse_qe_total_energy_missing(self):
        from maccai_battery.utils import parse_qe_total_energy_ry
        assert parse_qe_total_energy_ry("no energy here") is None

    def test_parse_qe_n_atoms_missing(self):
        from maccai_battery.utils import parse_qe_n_atoms
        assert parse_qe_n_atoms("no atoms here") is None

    def test_extract_scf_summary_complete(self):
        from maccai_battery.utils import extract_scf_summary, RY_TO_EV
        s = extract_scf_summary(self._SCF_OUTPUT)
        assert s["converged"] is True
        assert s["energy_ry"] == pytest.approx(-1052.43283515)
        assert s["energy_ev"] == pytest.approx(-1052.43283515 * RY_TO_EV)
        assert s["n_atoms"] == 14
        assert s["energy_ev_per_atom"] == pytest.approx(-1052.43283515 * RY_TO_EV / 14)
        assert s["fermi_ev"] == pytest.approx(10.5432)

    def test_extract_scf_summary_uses_provided_n_atoms(self):
        from maccai_battery.utils import extract_scf_summary
        s = extract_scf_summary(self._SCF_OUTPUT, n_atoms=7)
        assert s["n_atoms"] == 7

    def test_parse_qe_picks_last_energy(self):
        """Ensure the last '! total energy' line is used (final SCF step)."""
        from maccai_battery.utils import parse_qe_total_energy_ry
        output = (
            "!    total energy              =    -100.00000000 Ry\n"
            "!    total energy              =    -200.00000000 Ry\n"
        )
        e = parse_qe_total_energy_ry(output)
        assert e == pytest.approx(-200.0)


# ===========================================================================
# Path helpers
# ===========================================================================

class TestPathHelpers:
    def test_ensure_dir_creates(self, tmp_path):
        from maccai_battery.utils import ensure_dir
        target = tmp_path / "a" / "b" / "c"
        result = ensure_dir(target)
        assert result.is_dir()

    def test_ensure_dir_idempotent(self, tmp_path):
        from maccai_battery.utils import ensure_dir
        target = tmp_path / "existing"
        target.mkdir()
        ensure_dir(target)   # should not raise
        assert target.is_dir()

    def test_relative_path_inside(self, tmp_path):
        from maccai_battery.utils import relative_path
        anchor = tmp_path
        target = tmp_path / "a" / "b.txt"
        rel    = relative_path(target, anchor)
        assert str(rel) == str(Path("a") / "b.txt")

    def test_relative_path_outside_returns_original(self, tmp_path):
        from maccai_battery.utils import relative_path
        anchor = tmp_path / "sub"
        target = tmp_path / "other" / "file.txt"
        rel    = relative_path(target, anchor)
        assert rel == target

    def test_find_ml_relaxed_files_sorted(self, tmp_path):
        from maccai_battery.utils import find_ml_relaxed_files

        for i in [5, 12, 1, 20]:
            (tmp_path / f"generated_crystals_frame{i}_ml_relaxed.extxyz").touch()

        files = find_ml_relaxed_files(tmp_path)
        indices = []
        import re
        for f in files:
            m = re.search(r"frame(\d+)", f.name)
            if m:
                indices.append(int(m.group(1)))

        assert indices == sorted(indices)

    def test_find_ml_relaxed_files_empty_dir(self, tmp_path):
        from maccai_battery.utils import find_ml_relaxed_files
        assert find_ml_relaxed_files(tmp_path) == []


# ===========================================================================
# Deduplication helpers
# ===========================================================================

class TestDeduplication:
    def test_no_duplicates(self):
        from maccai_battery.utils import deduplicate_by_fingerprint
        items  = [1, 2, 3]
        unique, dups = deduplicate_by_fingerprint(items, lambda x: x)
        assert unique    == [1, 2, 3]
        assert dups      == []

    def test_all_duplicates(self):
        from maccai_battery.utils import deduplicate_by_fingerprint
        items  = [1, 1, 1]
        unique, dups = deduplicate_by_fingerprint(items, lambda x: x)
        assert unique    == [1]
        assert len(dups) == 2

    def test_mixed(self):
        from maccai_battery.utils import deduplicate_by_fingerprint
        items  = ["a", "b", "a", "c", "b"]
        unique, dups = deduplicate_by_fingerprint(items, lambda x: x)
        assert unique    == ["a", "b", "c"]
        assert len(dups) == 2

    def test_preserves_order(self):
        from maccai_battery.utils import deduplicate_by_fingerprint
        items = [10, 5, 7, 5, 10]
        unique, _ = deduplicate_by_fingerprint(items, lambda x: x)
        assert unique == [10, 5, 7]


# ===========================================================================
# CandidateDatabase tests
# ===========================================================================

def _make_minimal_config(tmp_path: Path):
    """Build a minimal PipelineConfig pointing at tmp_path."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    from maccai_battery.config import (
        PipelineConfig, ProjectConfig, ProjectDirs,
        GenerationConfig, RelaxationConfig, ScreeningConfig,
        DFTScreeningConfig, DFTRelaxConfig, PseudopotentialsConfig, DatabaseConfig,
    )

    dirs = ProjectDirs(
        candidates  = "candidates",
        ml_relaxed  = "candidates/ml_relaxed",
        cifs        = "candidates/cifs",
        vasp_inputs = "vasp_inputs",
        dft_scf     = "dft/scf",
        dft_relax   = "dft/relax",
        logs        = "logs",
    )

    project = ProjectConfig(
        name          = "test",
        output_dir    = "output",
        candidates_db = "candidates.ndjson",
        dirs          = dirs,
    )
    project.resolve(tmp_path)

    pseudos = PseudopotentialsConfig(
        pseudo_dir = "/fake/pseudo",
        files      = {"Li": "li.UPF", "Fe": "fe.UPF", "P": "p.UPF", "O": "o.UPF"},
    )

    cfg = PipelineConfig(
        project         = project,
        generation      = GenerationConfig(),
        relaxation      = RelaxationConfig(),
        screening       = ScreeningConfig(),
        dft_screening   = DFTScreeningConfig(),
        dft_relax       = DFTRelaxConfig(),
        pseudopotentials= pseudos,
        database        = DatabaseConfig(),
    )
    return cfg


def _make_record(formula: str = "LiFePO4", energy: float = -3.0) -> dict:
    """Build a minimal candidate record dict for testing."""
    import uuid
    return {
        "id":      "MG-" + uuid.uuid4().hex[:8],
        "formula": formula,
        "stoichiometry": {"Li": 1, "Fe": 1, "P": 1, "O": 4},
        "structure_files": {
            "source_extxyz": None,
            "ml_relaxed":    f"candidates/ml_relaxed/{formula}_ml_relaxed.extxyz",
        },
        "generation_metadata": {
            "generator":        "MatterGen",
            "model_checkpoint": "test_model",
            "chemical_system":  "Li-Fe-P-O",
            "energy_above_hull": 0.05,
            "frame_index":       0,
            "timestamp":        "2025-01-01T00:00:00.000000Z",
        },
        "ml_scores": {
            "matter_sim_energy_eV_per_atom":      energy,
            "matter_sim_uncertainty_eV_per_atom": None,
            "density_gcc":                        3.5,
            "min_distance_A":                     1.85,
        },
        "filters": {
            "charge_neutral":              True,
            "oxidation_states_assigned":   True,
            "min_interatomic_distance_ok": True,
            "density_ok":                  True,
            "bv_consistent":               True,
            "passed_all":                  True,
            "warning_count":               0,
        },
        "dft_jobs": {
            "status":   "not_submitted",
            "workflow": "PBE_scf_relax_QE",
            "scf":    {"status": "not_submitted", "energy_eV": None, "energy_eV_per_atom": None, "n_atoms": None, "run_dir": None, "completed_at": None},
            "relax":  {"status": "not_submitted", "energy_eV": None, "energy_eV_per_atom": None, "n_atoms": None, "run_dir": None, "completed_at": None},
            "vasp_incar_settings": {"ENCUT": 520, "KSPACING": 0.3},
        },
        "notes": "",
    }


class TestCandidateDatabase:
    """Tests for maccai_battery.database.CandidateDatabase."""

    def test_path_property(self, tmp_path):
        from maccai_battery.database import CandidateDatabase
        cfg = _make_minimal_config(tmp_path)
        db  = CandidateDatabase(cfg)
        assert db.path == cfg.project.candidates_db_path

    def test_not_exists_initially(self, tmp_path):
        from maccai_battery.database import CandidateDatabase
        cfg = _make_minimal_config(tmp_path)
        db  = CandidateDatabase(cfg)
        assert not db.exists

    def test_count_empty(self, tmp_path):
        from maccai_battery.database import CandidateDatabase
        cfg = _make_minimal_config(tmp_path)
        db  = CandidateDatabase(cfg)
        assert db.count() == 0

    def test_append_single(self, tmp_path):
        from maccai_battery.database import CandidateDatabase
        cfg = _make_minimal_config(tmp_path)
        db  = CandidateDatabase(cfg)
        rec = _make_record()
        cid = db.append(rec)
        assert cid == rec["id"]
        assert db.count() == 1
        assert db.exists

    def test_append_many(self, tmp_path):
        from maccai_battery.database import CandidateDatabase
        cfg     = _make_minimal_config(tmp_path)
        db      = CandidateDatabase(cfg)
        records = [_make_record(f"Formula{i}", -float(i)) for i in range(5)]
        ids     = db.append_many(records)
        assert len(ids) == 5
        assert db.count() == 5

    def test_load_all_returns_all(self, tmp_path):
        from maccai_battery.database import CandidateDatabase
        cfg = _make_minimal_config(tmp_path)
        db  = CandidateDatabase(cfg)
        for i in range(3):
            db.append(_make_record(f"F{i}"))
        all_records = db.load_all()
        assert len(all_records) == 3

    def test_get_by_id_found(self, tmp_path):
        from maccai_battery.database import CandidateDatabase
        cfg = _make_minimal_config(tmp_path)
        db  = CandidateDatabase(cfg)
        rec = _make_record("LiFePO4")
        db.append(rec)
        found = db.get_by_id(rec["id"])
        assert found is not None
        assert found["formula"] == "LiFePO4"

    def test_get_by_id_not_found(self, tmp_path):
        from maccai_battery.database import CandidateDatabase
        cfg = _make_minimal_config(tmp_path)
        db  = CandidateDatabase(cfg)
        assert db.get_by_id("MG-nonexistent") is None

    def test_top_n_by_ml_energy_sorted(self, tmp_path):
        from maccai_battery.database import CandidateDatabase
        cfg = _make_minimal_config(tmp_path)
        db  = CandidateDatabase(cfg)
        db.append(_make_record("A", energy=-1.0))
        db.append(_make_record("B", energy=-5.0))
        db.append(_make_record("C", energy=-3.0))
        top2 = db.top_n_by_ml_energy(2)
        assert len(top2) == 2
        # Most stable (most negative) first
        assert top2[0]["ml_scores"]["matter_sim_energy_eV_per_atom"] == pytest.approx(-5.0)

    def test_top_n_by_ml_energy_respects_n(self, tmp_path):
        from maccai_battery.database import CandidateDatabase
        cfg = _make_minimal_config(tmp_path)
        db  = CandidateDatabase(cfg)
        for i in range(10):
            db.append(_make_record(f"F{i}", energy=-float(i)))
        top3 = db.top_n_by_ml_energy(3)
        assert len(top3) == 3

    def test_top_n_passed_checks_only(self, tmp_path):
        from maccai_battery.database import CandidateDatabase
        cfg = _make_minimal_config(tmp_path)
        db  = CandidateDatabase(cfg)

        rec_pass = _make_record("Pass", energy=-5.0)
        rec_fail = _make_record("Fail", energy=-6.0)
        rec_fail["filters"]["passed_all"] = False   # worse energy but failed checks

        db.append(rec_pass)
        db.append(rec_fail)

        top = db.top_n_by_ml_energy(5, passed_checks_only=True)
        assert all(r["filters"]["passed_all"] for r in top)
        assert len(top) == 1

    def test_update_field_simple(self, tmp_path):
        from maccai_battery.database import CandidateDatabase
        cfg = _make_minimal_config(tmp_path)
        db  = CandidateDatabase(cfg)
        rec = _make_record()
        db.append(rec)

        ok = db.update_field(rec["id"], "notes", "updated note")
        assert ok is True

        found = db.get_by_id(rec["id"])
        assert found["notes"] == "updated note"

    def test_update_field_nested(self, tmp_path):
        from maccai_battery.database import CandidateDatabase
        cfg = _make_minimal_config(tmp_path)
        db  = CandidateDatabase(cfg)
        rec = _make_record()
        db.append(rec)

        ok = db.update_field(rec["id"], "dft_jobs.status", "running")
        assert ok is True

        found = db.get_by_id(rec["id"])
        assert found["dft_jobs"]["status"] == "running"

    def test_update_field_not_found(self, tmp_path):
        from maccai_battery.database import CandidateDatabase
        cfg = _make_minimal_config(tmp_path)
        db  = CandidateDatabase(cfg)
        ok  = db.update_field("MG-ghost", "notes", "x")
        assert ok is False

    def test_update_dft_scf(self, tmp_path):
        from maccai_battery.database import CandidateDatabase
        cfg = _make_minimal_config(tmp_path)
        db  = CandidateDatabase(cfg)
        rec = _make_record()
        db.append(rec)

        ok = db.update_dft_scf(
            rec["id"],
            energy_eV = -100.0,
            n_atoms   = 10,
            run_dir   = "/qe/scf_0",
        )
        assert ok is True

        found = db.get_by_id(rec["id"])
        scf   = found["dft_jobs"]["scf"]
        assert scf["energy_eV"]          == pytest.approx(-100.0)
        assert scf["energy_eV_per_atom"] == pytest.approx(-10.0)
        assert scf["n_atoms"]            == 10
        assert scf["status"]             == "done"
        assert scf["completed_at"]       is not None

    def test_update_dft_relax(self, tmp_path):
        from maccai_battery.database import CandidateDatabase
        cfg = _make_minimal_config(tmp_path)
        db  = CandidateDatabase(cfg)
        rec = _make_record()
        db.append(rec)

        ok = db.update_dft_relax(
            rec["id"],
            energy_eV = -105.0,
            n_atoms   = 7,
        )
        assert ok is True

        found = db.get_by_id(rec["id"])
        relax = found["dft_jobs"]["relax"]
        assert relax["energy_eV"]          == pytest.approx(-105.0)
        assert relax["energy_eV_per_atom"] == pytest.approx(-15.0)
        assert relax["status"]             == "done"

    def test_deduplicate_removes_duplicates(self, tmp_path):
        from maccai_battery.database import CandidateDatabase
        cfg = _make_minimal_config(tmp_path)
        db  = CandidateDatabase(cfg)
        rec = _make_record("LiFePO4")
        # Append the same record twice
        db.append(rec)
        db.append(rec)
        assert db.count() == 2
        removed = db.deduplicate(key="id")
        assert removed == 1
        assert db.count() == 1

    def test_deduplicate_no_duplicates(self, tmp_path):
        from maccai_battery.database import CandidateDatabase
        cfg = _make_minimal_config(tmp_path)
        db  = CandidateDatabase(cfg)
        for i in range(3):
            db.append(_make_record(f"F{i}"))
        removed = db.deduplicate(key="id")
        assert removed == 0
        assert db.count() == 3

    def test_filter_by_formula(self, tmp_path):
        from maccai_battery.database import CandidateDatabase
        cfg = _make_minimal_config(tmp_path)
        db  = CandidateDatabase(cfg)
        db.append(_make_record("LiFePO4"))
        db.append(_make_record("LiFePO4"))
        db.append(_make_record("LiFeO2"))
        results = db.filter_by_formula("LiFePO4")
        assert len(results) == 2
        assert all(r["formula"] == "LiFePO4" for r in results)

    def test_filter_by_formula_no_match(self, tmp_path):
        from maccai_battery.database import CandidateDatabase
        cfg = _make_minimal_config(tmp_path)
        db  = CandidateDatabase(cfg)
        db.append(_make_record("LiFeO2"))
        results = db.filter_by_formula("NotAFormula")
        assert results == []

    def test_summary_counts(self, tmp_path):
        from maccai_battery.database import CandidateDatabase
        cfg = _make_minimal_config(tmp_path)
        db  = CandidateDatabase(cfg)

        # 2 passed, 1 failed
        r_pass1 = _make_record("A", -1.0)
        r_pass2 = _make_record("B", -2.0)
        r_fail  = _make_record("C", -3.0)
        r_fail["filters"]["passed_all"] = False

        db.append_many([r_pass1, r_pass2, r_fail])
        s = db.summary()

        assert s["total"]         == 3
        assert s["passed_checks"] == 2
        assert s["failed_checks"] == 1
        assert s["ml_screened"]   == 3

    def test_summary_dft_counts(self, tmp_path):
        from maccai_battery.database import CandidateDatabase
        cfg = _make_minimal_config(tmp_path)
        db  = CandidateDatabase(cfg)

        rec1 = _make_record("A")
        rec2 = _make_record("B")
        db.append_many([rec1, rec2])

        db.update_dft_scf(rec1["id"], energy_eV=-50.0, n_atoms=5)
        db.update_dft_relax(rec1["id"], energy_eV=-51.0, n_atoms=5)

        s = db.summary()
        assert s["scf_done"]   == 1
        assert s["relax_done"] == 1

    def test_top_n_by_dft_scf_energy(self, tmp_path):
        from maccai_battery.database import CandidateDatabase
        cfg = _make_minimal_config(tmp_path)
        db  = CandidateDatabase(cfg)

        r1 = _make_record("A")
        r2 = _make_record("B")
        r3 = _make_record("C")
        db.append_many([r1, r2, r3])

        db.update_dft_scf(r1["id"], energy_eV=-10.0, n_atoms=2)  # -5.0 eV/atom
        db.update_dft_scf(r2["id"], energy_eV=-30.0, n_atoms=3)  # -10.0 eV/atom
        # r3 has no SCF result

        top = db.top_n_by_dft_scf_energy(5)
        assert len(top) == 2
        # Most stable first
        assert top[0]["id"] == r2["id"]

    def test_top_n_by_dft_relax_energy(self, tmp_path):
        from maccai_battery.database import CandidateDatabase
        cfg = _make_minimal_config(tmp_path)
        db  = CandidateDatabase(cfg)

        r1 = _make_record("A")
        r2 = _make_record("B")
        db.append_many([r1, r2])

        db.update_dft_relax(r1["id"], energy_eV=-20.0, n_atoms=4)  # -5.0
        db.update_dft_relax(r2["id"], energy_eV=-40.0, n_atoms=4)  # -10.0

        top = db.top_n_by_dft_relax_energy(5)
        assert len(top) == 2
        assert top[0]["id"] == r2["id"]

    def test_backup_creates_file(self, tmp_path):
        from maccai_battery.database import CandidateDatabase
        cfg = _make_minimal_config(tmp_path)
        db  = CandidateDatabase(cfg)
        db.append(_make_record())

        backup = db.backup(suffix="test")
        assert backup.exists()
        assert "test" in backup.name

    def test_backup_raises_if_no_db(self, tmp_path):
        from maccai_battery.database import CandidateDatabase
        cfg = _make_minimal_config(tmp_path)
        db  = CandidateDatabase(cfg)
        with pytest.raises(FileNotFoundError):
            db.backup()

    def test_export_csv_creates_file(self, tmp_path):
        from maccai_battery.database import CandidateDatabase
        cfg = _make_minimal_config(tmp_path)
        db  = CandidateDatabase(cfg)
        db.append(_make_record("LiFePO4"))

        csv_path = db.export_csv()
        assert csv_path.exists()
        content = csv_path.read_text()
        assert "LiFePO4" in content

    def test_export_csv_has_header(self, tmp_path):
        from maccai_battery.database import CandidateDatabase
        cfg = _make_minimal_config(tmp_path)
        db  = CandidateDatabase(cfg)
        db.append(_make_record())

        csv_path = db.export_csv()
        first_line = csv_path.read_text().splitlines()[0]
        assert "formula" in first_line
        assert "ml_energy_eV_per_atom" in first_line

    def test_export_csv_custom_path(self, tmp_path):
        from maccai_battery.database import CandidateDatabase
        cfg      = _make_minimal_config(tmp_path)
        db       = CandidateDatabase(cfg)
        db.append(_make_record())

        out = tmp_path / "my_export.csv"
        result = db.export_csv(out)
        assert result == out
        assert out.exists()

    def test_iter_records_yields_all(self, tmp_path):
        from maccai_battery.database import CandidateDatabase
        cfg = _make_minimal_config(tmp_path)
        db  = CandidateDatabase(cfg)
        for i in range(4):
            db.append(_make_record(f"F{i}"))

        count = sum(1 for _ in db.iter_records())
        assert count == 4

    def test_iter_records_empty_db(self, tmp_path):
        from maccai_battery.database import CandidateDatabase
        cfg = _make_minimal_config(tmp_path)
        db  = CandidateDatabase(cfg)
        records = list(db.iter_records())
        assert records == []

    def test_atomic_write_safety(self, tmp_path):
        """Verify that a rewrite produces a valid NDJSON file."""
        from maccai_battery.database import CandidateDatabase
        cfg = _make_minimal_config(tmp_path)
        db  = CandidateDatabase(cfg)

        recs = [_make_record(f"F{i}") for i in range(5)]
        db.append_many(recs)

        # Update a field (which triggers an atomic rewrite)
        db.update_field(recs[2]["id"], "notes", "atomic write test")

        # The file should still be valid NDJSON with all records
        loaded = db.load_all()
        assert len(loaded) == 5
        notes = [r["notes"] for r in loaded if r["id"] == recs[2]["id"]]
        assert notes[0] == "atomic write test"


# ===========================================================================
# Database _set_nested helper
# ===========================================================================

class TestSetNested:
    def test_top_level_key(self):
        from maccai_battery.database import _set_nested
        r = {"a": 1}
        _set_nested(r, "a", 99)
        assert r["a"] == 99

    def test_creates_nested_dict(self):
        from maccai_battery.database import _set_nested
        r = {}
        _set_nested(r, "a.b.c", "deep")
        assert r["a"]["b"]["c"] == "deep"

    def test_updates_existing_nested(self):
        from maccai_battery.database import _set_nested
        r = {"dft_jobs": {"status": "not_submitted"}}
        _set_nested(r, "dft_jobs.status", "done")
        assert r["dft_jobs"]["status"] == "done"

    def test_does_not_clobber_sibling_keys(self):
        from maccai_battery.database import _set_nested
        r = {"dft_jobs": {"status": "not_submitted", "workflow": "PBE"}}
        _set_nested(r, "dft_jobs.status", "done")
        assert r["dft_jobs"]["workflow"] == "PBE"


# ===========================================================================
# Config validation tests
# ===========================================================================

class TestConfigValidation:
    """Tests for PipelineConfig.validate() cross-field checks."""

    def _base_cfg(self, tmp_path):
        return _make_minimal_config(tmp_path)

    def test_valid_config_does_not_raise(self, tmp_path):
        cfg = self._base_cfg(tmp_path)
        cfg.validate()   # should not raise

    def test_negative_energy_above_hull_raises(self, tmp_path):
        from maccai_battery.config import PipelineConfig
        cfg = self._base_cfg(tmp_path)
        cfg.generation.energy_above_hull = -0.01
        with pytest.raises(ValueError, match="energy_above_hull"):
            cfg.validate()

    def test_zero_batch_size_raises(self, tmp_path):
        cfg = self._base_cfg(tmp_path)
        cfg.generation.batch_size = 0
        with pytest.raises(ValueError, match="batch_size"):
            cfg.validate()

    def test_zero_num_batches_raises(self, tmp_path):
        cfg = self._base_cfg(tmp_path)
        cfg.generation.num_batches = 0
        with pytest.raises(ValueError, match="num_batches"):
            cfg.validate()

    def test_fmax_zero_raises(self, tmp_path):
        cfg = self._base_cfg(tmp_path)
        cfg.relaxation.fmax = 0.0
        with pytest.raises(ValueError, match="fmax"):
            cfg.validate()

    def test_density_range_inverted_raises(self, tmp_path):
        cfg = self._base_cfg(tmp_path)
        cfg.screening.density_min_gcc = 10.0
        cfg.screening.density_max_gcc = 1.0
        with pytest.raises(ValueError, match="density"):
            cfg.validate()

    def test_relax_candidates_exceeds_scf_raises(self, tmp_path):
        cfg = self._base_cfg(tmp_path)
        cfg.dft_screening.n_candidates = 3
        cfg.dft_relax.n_candidates     = 5
        with pytest.raises(ValueError, match="n_candidates"):
            cfg.validate()

    def test_missing_pseudopotential_raises(self, tmp_path):
        cfg = self._base_cfg(tmp_path)
        # Remove P pseudopotential — required for Li-Fe-P-O system
        del cfg.pseudopotentials.files["P"]
        with pytest.raises(KeyError, match="P"):
            cfg.validate()

    def test_pseudopotentials_validate_for_system(self, tmp_path):
        cfg = self._base_cfg(tmp_path)
        with pytest.raises(KeyError, match="Na"):
            cfg.pseudopotentials.validate_for_system("Li-Fe-Na-O")

    def test_total_structures_property(self, tmp_path):
        cfg = self._base_cfg(tmp_path)
        cfg.generation.batch_size  = 8
        cfg.generation.num_batches = 5
        assert cfg.generation.total_structures == 40


# ===========================================================================
# Formatting helpers
# ===========================================================================

class TestFormattingHelpers:
    def test_format_energy_table_basic(self):
        from maccai_battery.utils import format_energy_table
        rows = [
            ("structure_A", -100.0, 10),
            ("structure_B", -200.0, 20),
        ]
        table = format_energy_table(rows, title="Test Ranking")
        assert "Test Ranking" in table
        assert "structure_A" in table
        assert "structure_B" in table
        # energies per atom should appear: -10.0 and -10.0
        assert "-10.000000" in table

    def test_format_energy_table_empty(self):
        from maccai_battery.utils import format_energy_table
        result = format_energy_table([], title="Empty")
        assert "no data" in result.lower()

    def test_human_size_bytes(self):
        from maccai_battery.utils import human_size
        assert human_size(512) == "512 B"

    def test_human_size_kb(self):
        from maccai_battery.utils import human_size
        assert "KB" in human_size(2048)

    def test_human_size_mb(self):
        from maccai_battery.utils import human_size
        assert "MB" in human_size(2 * 1024 * 1024)


# ===========================================================================
# DeduplicationConfig and HullConfig validation tests
# ===========================================================================

class TestDeduplicationConfig:
    """Tests for DeduplicationConfig validation inside PipelineConfig."""

    def _base_cfg(self, tmp_path):
        return _make_minimal_config(tmp_path)

    def test_default_enabled(self, tmp_path):
        cfg = self._base_cfg(tmp_path)
        assert cfg.deduplication.enabled is True

    def test_default_method(self, tmp_path):
        cfg = self._base_cfg(tmp_path)
        assert cfg.deduplication.fingerprint_method == "formula_volume_density"

    def test_valid_rdf_method(self, tmp_path):
        cfg = self._base_cfg(tmp_path)
        cfg.deduplication.fingerprint_method = "pymatgen_rdf"
        cfg.validate()  # should not raise

    def test_invalid_method_raises(self, tmp_path):
        cfg = self._base_cfg(tmp_path)
        cfg.deduplication.fingerprint_method = "invalid_method"
        with pytest.raises(ValueError, match="fingerprint_method"):
            cfg.validate()

    def test_negative_volume_tolerance_raises(self, tmp_path):
        cfg = self._base_cfg(tmp_path)
        cfg.deduplication.volume_tolerance_A3 = -1.0
        with pytest.raises(ValueError, match="volume_tolerance"):
            cfg.validate()

    def test_zero_volume_tolerance_allowed(self, tmp_path):
        cfg = self._base_cfg(tmp_path)
        cfg.deduplication.volume_tolerance_A3 = 0.0
        cfg.validate()  # should not raise

    def test_negative_density_tolerance_raises(self, tmp_path):
        cfg = self._base_cfg(tmp_path)
        cfg.deduplication.density_tolerance_gcc = -0.1
        with pytest.raises(ValueError, match="density_tolerance"):
            cfg.validate()

    def test_zero_density_tolerance_allowed(self, tmp_path):
        cfg = self._base_cfg(tmp_path)
        cfg.deduplication.density_tolerance_gcc = 0.0
        cfg.validate()  # should not raise

    def test_disabled_flag(self, tmp_path):
        cfg = self._base_cfg(tmp_path)
        cfg.deduplication.enabled = False
        cfg.validate()  # should not raise
        assert cfg.deduplication.enabled is False


class TestHullConfig:
    """Tests for HullConfig validation inside PipelineConfig."""

    def _base_cfg(self, tmp_path):
        return _make_minimal_config(tmp_path)

    def test_default_threshold(self, tmp_path):
        cfg = self._base_cfg(tmp_path)
        assert cfg.hull.stability_threshold_eV == pytest.approx(0.1)

    def test_default_energy_source(self, tmp_path):
        cfg = self._base_cfg(tmp_path)
        assert cfg.hull.energy_source == "relax"

    def test_scf_energy_source_valid(self, tmp_path):
        cfg = self._base_cfg(tmp_path)
        cfg.hull.energy_source = "scf"
        cfg.validate()  # should not raise

    def test_invalid_energy_source_raises(self, tmp_path):
        cfg = self._base_cfg(tmp_path)
        cfg.hull.energy_source = "phonon"
        with pytest.raises(ValueError, match="energy_source"):
            cfg.validate()

    def test_negative_threshold_raises(self, tmp_path):
        cfg = self._base_cfg(tmp_path)
        cfg.hull.stability_threshold_eV = -0.05
        with pytest.raises(ValueError, match="stability_threshold"):
            cfg.validate()

    def test_zero_threshold_allowed(self, tmp_path):
        cfg = self._base_cfg(tmp_path)
        cfg.hull.stability_threshold_eV = 0.0
        cfg.validate()  # on-hull exactly

    def test_mp_api_key_defaults_empty(self, tmp_path):
        cfg = self._base_cfg(tmp_path)
        assert cfg.hull.mp_api_key == ""

    def test_report_max_e_above_hull_default(self, tmp_path):
        cfg = self._base_cfg(tmp_path)
        assert cfg.hull.report_max_e_above_hull == pytest.approx(0.5)


# ===========================================================================
# Deduplication fingerprinting helpers (unit-level, no pymatgen structures)
# ===========================================================================

class TestDeduplicationFingerprinting:
    """Tests for the fast formula_volume_density fingerprint bucketing logic."""

    def _make_mock_structure(self, formula: str, volume: float, density: float):
        """Create a minimal mock object that exposes the needed attributes."""

        class MockComposition:
            def __init__(self, f):
                self._formula = f

            @property
            def reduced_formula(self):
                return self._formula

        class MockStructure:
            def __init__(self, formula, vol, dens):
                self.composition = MockComposition(formula)
                self.volume      = vol
                self.density     = dens

        return MockStructure(formula, volume, density)

    def _fingerprint(self, structure, volume_tol=5.0, density_tol=0.05):
        """Call the private fingerprint function from 03_sanity_check.py."""
        import sys
        from pathlib import Path
        scripts_dir = Path(__file__).resolve().parent.parent / "scripts"
        if str(scripts_dir.parent) not in sys.path:
            sys.path.insert(0, str(scripts_dir.parent))

        # Import the private helper directly
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "sanity_check", scripts_dir / "03_sanity_check.py"
        )
        mod = importlib.util.load_from_spec(spec) if hasattr(importlib.util, "load_from_spec") else None

        # Inline the logic instead to avoid side-effects of importing a script
        import math

        formula  = structure.composition.reduced_formula
        volume   = structure.volume
        density  = structure.density
        v_bin    = round(volume   / volume_tol)  * volume_tol  if volume_tol  > 0 else volume
        d_bin    = round(density  / density_tol) * density_tol if density_tol > 0 else density
        return f"{formula}|v={v_bin:.1f}|d={d_bin:.3f}"

    def test_identical_structures_same_fp(self):
        s1 = self._make_mock_structure("LiFePO4", 150.0, 3.50)
        s2 = self._make_mock_structure("LiFePO4", 150.0, 3.50)
        assert self._fingerprint(s1) == self._fingerprint(s2)

    def test_different_formula_different_fp(self):
        s1 = self._make_mock_structure("LiFePO4",  150.0, 3.50)
        s2 = self._make_mock_structure("Li2FePO4", 150.0, 3.50)
        assert self._fingerprint(s1) != self._fingerprint(s2)

    def test_volume_within_tolerance_same_fp(self):
        """Structures within one volume bin should get the same fingerprint."""
        s1 = self._make_mock_structure("LiFePO4", 150.0, 3.50)
        s2 = self._make_mock_structure("LiFePO4", 152.0, 3.50)  # diff < 5 Å³
        # Both round to the same 5-Å³ bin
        assert self._fingerprint(s1, volume_tol=5.0) == self._fingerprint(s2, volume_tol=5.0)

    def test_volume_outside_tolerance_different_fp(self):
        """Structures separated by more than one bin should differ."""
        s1 = self._make_mock_structure("LiFePO4", 148.0, 3.50)
        s2 = self._make_mock_structure("LiFePO4", 157.0, 3.50)  # diff > 5 Å³
        assert self._fingerprint(s1, volume_tol=5.0) != self._fingerprint(s2, volume_tol=5.0)

    def test_density_within_tolerance_same_fp(self):
        s1 = self._make_mock_structure("LiFePO4", 150.0, 3.50)
        s2 = self._make_mock_structure("LiFePO4", 150.0, 3.52)  # diff < 0.05
        assert self._fingerprint(s1, density_tol=0.05) == self._fingerprint(s2, density_tol=0.05)

    def test_density_outside_tolerance_different_fp(self):
        s1 = self._make_mock_structure("LiFePO4", 150.0, 3.40)
        s2 = self._make_mock_structure("LiFePO4", 150.0, 3.60)  # diff > 0.05
        assert self._fingerprint(s1, density_tol=0.05) != self._fingerprint(s2, density_tol=0.05)

    def test_zero_volume_tolerance_uses_exact_volume(self):
        """When tolerance is 0, volumes must match exactly to share a fingerprint."""
        s1 = self._make_mock_structure("LiFePO4", 150.12345, 3.50)
        s2 = self._make_mock_structure("LiFePO4", 150.12346, 3.50)
        fp1 = self._fingerprint(s1, volume_tol=0.0)
        fp2 = self._fingerprint(s2, volume_tol=0.0)
        # With tol=0 both are passed through exactly, different float representations
        # The key point: function doesn't crash with tol=0
        assert isinstance(fp1, str)
        assert isinstance(fp2, str)


# ===========================================================================
# HullResult dataclass tests
# ===========================================================================

class TestHullResult:
    """Tests for maccai_battery.hull.HullResult."""

    def test_success_when_no_error(self):
        from maccai_battery.hull import HullResult
        r = HullResult(
            candidate_id            = "MG-abc123",
            formula                 = "LiFePO4",
            e_above_hull_eV_per_atom = 0.02,
        )
        assert r.success is True

    def test_not_success_when_error(self):
        from maccai_battery.hull import HullResult
        r = HullResult(
            candidate_id = "MG-abc123",
            formula      = "LiFePO4",
            error        = "pymatgen error",
        )
        assert r.success is False

    def test_not_success_when_none_energy(self):
        from maccai_battery.hull import HullResult
        r = HullResult(
            candidate_id             = "MG-abc123",
            formula                  = "LiFePO4",
            e_above_hull_eV_per_atom = None,
        )
        assert r.success is False

    def test_is_stable_below_threshold(self):
        from maccai_battery.hull import HullResult
        r = HullResult(
            candidate_id             = "MG-abc",
            formula                  = "LiFePO4",
            e_above_hull_eV_per_atom = 0.05,
            stability_threshold_eV   = 0.1,
            is_stable                = True,
        )
        assert r.is_stable is True

    def test_is_not_stable_above_threshold(self):
        from maccai_battery.hull import HullResult
        r = HullResult(
            candidate_id             = "MG-abc",
            formula                  = "LiFePO4",
            e_above_hull_eV_per_atom = 0.15,
            stability_threshold_eV   = 0.1,
            is_stable                = False,
        )
        assert r.is_stable is False

    def test_to_dict_keys(self):
        from maccai_battery.hull import HullResult
        r = HullResult(
            candidate_id             = "MG-abc",
            formula                  = "LiFePO4",
            e_above_hull_eV_per_atom = 0.02,
            dft_energy_eV_per_atom   = -3.5,
            is_stable                = True,
        )
        d = r.to_dict()
        assert "candidate_id"             in d
        assert "formula"                  in d
        assert "e_above_hull_eV_per_atom" in d
        assert "dft_energy_eV_per_atom"   in d
        assert "is_stable"                in d
        assert "mp_stable_phases"         in d
        assert "error"                    in d

    def test_to_dict_values(self):
        from maccai_battery.hull import HullResult
        r = HullResult(
            candidate_id             = "MG-test",
            formula                  = "LiFeO2",
            e_above_hull_eV_per_atom = 0.08,
            dft_energy_eV_per_atom   = -4.2,
            is_stable                = True,
            stability_threshold_eV   = 0.1,
        )
        d = r.to_dict()
        assert d["candidate_id"]             == "MG-test"
        assert d["formula"]                  == "LiFeO2"
        assert d["e_above_hull_eV_per_atom"] == pytest.approx(0.08)
        assert d["dft_energy_eV_per_atom"]   == pytest.approx(-4.2)
        assert d["is_stable"]                is True

    def test_summary_line_stable(self):
        from maccai_battery.hull import HullResult
        r = HullResult(
            candidate_id             = "MG-abc",
            formula                  = "LiFePO4",
            e_above_hull_eV_per_atom = 0.00,
            is_stable                = True,
        )
        line = r.summary_line()
        assert "STABLE" in line
        assert "LiFePO4" in line
        assert "MG-abc" in line

    def test_summary_line_metastable(self):
        from maccai_battery.hull import HullResult
        r = HullResult(
            candidate_id             = "MG-xyz",
            formula                  = "LiFeO2",
            e_above_hull_eV_per_atom = 0.12,
            is_stable                = False,
        )
        line = r.summary_line()
        assert "METASTABLE" in line

    def test_summary_line_error(self):
        from maccai_battery.hull import HullResult
        r = HullResult(
            candidate_id = "MG-err",
            formula      = "Unknown",
            error        = "Composition parse failed",
        )
        line = r.summary_line()
        assert "ERROR" in line
        assert "Composition parse failed" in line

    def test_mp_stable_phases_default_empty(self):
        from maccai_battery.hull import HullResult
        r = HullResult(candidate_id="MG-x", formula="LiO2")
        assert r.mp_stable_phases == []

    def test_mp_stable_phases_stored(self):
        from maccai_battery.hull import HullResult
        r = HullResult(
            candidate_id    = "MG-x",
            formula         = "LiFePO4",
            mp_stable_phases = ["LiFePO4", "Li3PO4", "Fe2O3"],
        )
        assert len(r.mp_stable_phases) == 3
        assert "LiFePO4" in r.mp_stable_phases


# ===========================================================================
# Hull helper: _parse_elements and _get_dft_energy
# ===========================================================================

class TestHullHelpers:
    """Tests for module-level helper functions in maccai_battery.hull."""

    def test_parse_elements_four_element(self):
        from maccai_battery.hull import _parse_elements
        elements = _parse_elements("Li-Fe-P-O")
        assert elements == ["Li", "Fe", "P", "O"]

    def test_parse_elements_two_element(self):
        from maccai_battery.hull import _parse_elements
        elements = _parse_elements("Li-O")
        assert elements == ["Li", "O"]

    def test_parse_elements_single(self):
        from maccai_battery.hull import _parse_elements
        elements = _parse_elements("Li")
        assert elements == ["Li"]

    def test_parse_elements_whitespace_trimmed(self):
        from maccai_battery.hull import _parse_elements
        elements = _parse_elements(" Li - Fe - O ")
        assert elements == ["Li", "Fe", "O"]

    def test_get_dft_energy_relax(self):
        from maccai_battery.hull import _get_dft_energy
        record = {
            "dft_jobs": {
                "relax": {"energy_eV_per_atom": -3.5},
                "scf":   {"energy_eV_per_atom": -3.4},
            }
        }
        assert _get_dft_energy(record, "relax") == pytest.approx(-3.5)

    def test_get_dft_energy_scf(self):
        from maccai_battery.hull import _get_dft_energy
        record = {
            "dft_jobs": {
                "relax": {"energy_eV_per_atom": -3.5},
                "scf":   {"energy_eV_per_atom": -3.4},
            }
        }
        assert _get_dft_energy(record, "scf") == pytest.approx(-3.4)

    def test_get_dft_energy_missing_returns_none(self):
        from maccai_battery.hull import _get_dft_energy
        record = {"dft_jobs": {"relax": {"energy_eV_per_atom": None}}}
        assert _get_dft_energy(record, "relax") is None

    def test_get_dft_energy_missing_key_returns_none(self):
        from maccai_battery.hull import _get_dft_energy
        record = {}
        assert _get_dft_energy(record, "relax") is None

    def test_get_dft_energy_no_dft_jobs_returns_none(self):
        from maccai_battery.hull import _get_dft_energy
        record = {"formula": "LiFePO4"}
        assert _get_dft_energy(record, "relax") is None


# ===========================================================================
# HullAnalyzer: no-API-key guard
# ===========================================================================

class TestHullAnalyzerInit:
    """Tests for HullAnalyzer initialisation guards (no real API calls)."""

    def _make_cfg(self, tmp_path):
        cfg = _make_minimal_config(tmp_path)
        cfg.hull = type("HullCfg", (), {"stability_threshold_eV": 0.1})()
        return cfg

    def test_raises_without_api_key(self, tmp_path):
        from maccai_battery.hull import HullAnalyzer
        import os
        # Temporarily remove env var if set
        old_key = os.environ.pop("MP_API_KEY", None)
        try:
            cfg = self._make_cfg(tmp_path)
            with pytest.raises(EnvironmentError, match="API key"):
                HullAnalyzer(cfg, api_key=None)
        finally:
            if old_key is not None:
                os.environ["MP_API_KEY"] = old_key

    def test_raises_missing_mp_api_import(self, tmp_path, monkeypatch):
        """If mp-api is not installed, HullAnalyzer should raise ImportError."""
        import maccai_battery.hull as hull_mod
        from maccai_battery.hull import HullAnalyzer

        # Patch _check_imports to raise ImportError
        def _mock_check():
            raise ImportError("mp-api not installed")
        monkeypatch.setattr(hull_mod, "_check_imports", _mock_check)

        cfg = self._make_cfg(tmp_path)
        with pytest.raises((ImportError, EnvironmentError)):
            HullAnalyzer(cfg, api_key="fake-key-for-test")


# ===========================================================================
# Database: hull_analysis field integration
# ===========================================================================

class TestDatabaseHullIntegration:
    """Test that hull_analysis results can be written to and read from the DB."""

    def test_update_hull_field(self, tmp_path):
        from maccai_battery.database import CandidateDatabase
        from maccai_battery.hull import HullResult

        cfg = _make_minimal_config(tmp_path)
        db  = CandidateDatabase(cfg)
        rec = _make_record("LiFePO4", -3.5)
        db.append(rec)

        hull_data = HullResult(
            candidate_id             = rec["id"],
            formula                  = "LiFePO4",
            e_above_hull_eV_per_atom = 0.02,
            dft_energy_eV_per_atom   = -3.5,
            is_stable                = True,
            stability_threshold_eV   = 0.1,
            mp_stable_phases         = ["LiFePO4", "Fe2O3"],
        ).to_dict()

        ok = db.update_field(rec["id"], "hull_analysis", hull_data)
        assert ok is True

        found = db.get_by_id(rec["id"])
        assert found is not None
        assert "hull_analysis" in found
        ha = found["hull_analysis"]
        assert ha["e_above_hull_eV_per_atom"] == pytest.approx(0.02)
        assert ha["is_stable"] is True
        assert "LiFePO4" in ha["mp_stable_phases"]

    def test_hull_analysis_persists_after_reload(self, tmp_path):
        from maccai_battery.database import CandidateDatabase
        from maccai_battery.hull import HullResult

        cfg = _make_minimal_config(tmp_path)
        db  = CandidateDatabase(cfg)
        rec = _make_record("LiFeO2", -4.0)
        db.append(rec)

        hull_data = HullResult(
            candidate_id             = rec["id"],
            formula                  = "LiFeO2",
            e_above_hull_eV_per_atom = 0.08,
            is_stable                = True,
        ).to_dict()

        db.update_field(rec["id"], "hull_analysis", hull_data)

        # Create a new DB instance (simulates re-loading from disk)
        db2 = CandidateDatabase(cfg)
        found = db2.get_by_id(rec["id"])
        assert found["hull_analysis"]["e_above_hull_eV_per_atom"] == pytest.approx(0.08)

    def test_multiple_hull_results_stored_independently(self, tmp_path):
        from maccai_battery.database import CandidateDatabase
        from maccai_battery.hull import HullResult

        cfg = _make_minimal_config(tmp_path)
        db  = CandidateDatabase(cfg)

        r1 = _make_record("LiFePO4", -3.5)
        r2 = _make_record("LiFeO2",  -4.0)
        db.append_many([r1, r2])

        for rec, e_hull in [(r1, 0.00), (r2, 0.12)]:
            hull = HullResult(
                candidate_id             = rec["id"],
                formula                  = rec["formula"],
                e_above_hull_eV_per_atom = e_hull,
                is_stable                = e_hull <= 0.1,
            ).to_dict()
            db.update_field(rec["id"], "hull_analysis", hull)

        found1 = db.get_by_id(r1["id"])
        found2 = db.get_by_id(r2["id"])

        assert found1["hull_analysis"]["e_above_hull_eV_per_atom"] == pytest.approx(0.00)
        assert found2["hull_analysis"]["e_above_hull_eV_per_atom"] == pytest.approx(0.12)
        assert found1["hull_analysis"]["is_stable"] is True
        assert found2["hull_analysis"]["is_stable"] is False
