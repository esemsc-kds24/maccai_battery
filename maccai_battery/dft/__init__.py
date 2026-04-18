# =============================================================================
# maccai_battery.dft — Quantum ESPRESSO DFT subpackage for MACCAI Battery
# =============================================================================
# This subpackage encapsulates the full DFT workflow used in the MACCAI
# Battery materials discovery pipeline:
#
#   1. Input generation  (input_generator.py)
#      - Build QE pw.x SCF and ionic-relaxation input files from pymatgen
#        Structure objects and a typed PipelineConfig.
#
#   2. Job runner        (runner.py)
#      - Launch pw.x (optionally via mpirun), stream output to logger,
#        write qe.out, and return a structured QERunResult.
#
#   3. Output parsing    (parser.py)
#      - Parse QE stdout for energies / convergence (parse_scf_stdout,
#        parse_relax_stdout).
#      - Parse QE XML (data-file-schema.xml) for full structural details
#        including forces, stress, magnetization, and lattice parameters
#        (parse_qe_xml → QEXMLResult).
#
#   4. Workflow          (workflow.py)
#      - DFTWorkflow orchestrates the two-stage pipeline:
#          run_scf_screening()  → List[SCFResult]
#          run_dft_relax()      → List[RelaxResult]
#          run()                → (scf_results, relax_results)
#
# Typical usage
# -------------
# >>> from maccai_battery.config import load_config
# >>> from maccai_battery.dft import DFTWorkflow
# >>> cfg = load_config()
# >>> wf  = DFTWorkflow(cfg)
# >>> scf_results, relax_results = wf.run()
#
# Or step-by-step:
# >>> scf_results   = wf.run_scf_screening()
# >>> relax_results = wf.run_dft_relax(scf_results)
#
# Requirements
# ------------
# Core:  numpy, pymatgen, ase
# QE I/O: pymatgen-io-espresso
#   Install: pip install git+https://github.com/Griffin-Group/pymatgen-io-espresso
# Runtime: Quantum ESPRESSO (pw.x in PATH, optionally mpirun)
# =============================================================================

from __future__ import annotations

from maccai_battery.dft.input_generator import (
    make_scf_input,
    make_relax_input,
    write_qe_input,
)
from maccai_battery.dft.runner import (
    run_qe_pw,
    QERunResult,
)
from maccai_battery.dft.parser import (
    QEXMLResult,
    parse_qe_xml,
    parse_scf_stdout,
    parse_relax_stdout,
)
from maccai_battery.dft.workflow import (
    DFTWorkflow,
    SCFResult,
    RelaxResult,
)

__all__ = [
    # ---- input_generator ------------------------------------------------
    "make_scf_input",       # Structure + cfg  → PWin (SCF)
    "make_relax_input",     # Structure + cfg  → PWin (ionic relax)
    "write_qe_input",       # PWin + path      → writes qe.in
    # ---- runner ---------------------------------------------------------
    "run_qe_pw",            # run_dir + opts   → QERunResult
    "QERunResult",          # dataclass: returncode, stdout, wall_time_s …
    # ---- parser ---------------------------------------------------------
    "QEXMLResult",          # dataclass: full QE XML parse result
    "parse_qe_xml",         # Path             → QEXMLResult
    "parse_scf_stdout",     # str              → dict of SCF summary
    "parse_relax_stdout",   # str              → dict of relax summary
    # ---- workflow -------------------------------------------------------
    "DFTWorkflow",          # orchestrates SCF screening + ionic relaxation
    "SCFResult",            # dataclass: per-structure SCF outcome
    "RelaxResult",          # dataclass: per-structure relax outcome
]
