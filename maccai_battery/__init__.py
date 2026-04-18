# =============================================================================
# maccai_battery — ML-to-DFT Battery Materials Discovery Package
# =============================================================================
# Pipeline:
#   MatterGen (generation) → MatterSim (ML relaxation) →
#   Deduplication → Sanity Checks → Candidate DB →
#   QE DFT (SCF + relax) → Merge Results → Hull Analysis (MP)
# =============================================================================

__version__ = "0.1.0"
__author__ = "MACCAI"
__description__ = "End-to-end ML-to-DFT pipeline for battery materials discovery"

from maccai_battery.config import (
    load_config,
    PipelineConfig,
    ProjectConfig,
    GenerationConfig,
    RelaxationConfig,
    ScreeningConfig,
    DeduplicationConfig,
    DFTScreeningConfig,
    DFTRelaxConfig,
    HullConfig,
    PseudopotentialsConfig,
    DatabaseConfig,
)
from maccai_battery.database import CandidateDatabase
from maccai_battery.generation import run_mattergen
from maccai_battery.relaxation import relax_structures
from maccai_battery.checks import run_sanity_checks
from maccai_battery.hull import HullAnalyzer, compute_hull_distances
from maccai_battery.dft import (
    DFTWorkflow,
    SCFResult,
    RelaxResult,
    make_scf_input,
    make_relax_input,
    run_qe_pw,
    QERunResult,
    parse_qe_xml,
    QEXMLResult,
)

__all__ = [
    # Config
    "load_config",
    "PipelineConfig",
    "ProjectConfig",
    "GenerationConfig",
    "RelaxationConfig",
    "ScreeningConfig",
    "DeduplicationConfig",
    "DFTScreeningConfig",
    "DFTRelaxConfig",
    "HullConfig",
    "PseudopotentialsConfig",
    "DatabaseConfig",
    # Pipeline steps
    "run_mattergen",
    "relax_structures",
    "run_sanity_checks",
    # Database
    "CandidateDatabase",
    # Hull analysis
    "HullAnalyzer",
    "compute_hull_distances",
    # DFT workflow
    "DFTWorkflow",
    "SCFResult",
    "RelaxResult",
    "make_scf_input",
    "make_relax_input",
    "run_qe_pw",
    "QERunResult",
    "parse_qe_xml",
    "QEXMLResult",
]
