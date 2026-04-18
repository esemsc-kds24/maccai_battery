# =============================================================================
# maccai_battery.config — Typed configuration loader
# =============================================================================
# Loads config.yaml into strongly-typed Python dataclasses so that:
#   - All parameters are validated at startup (not at runtime)
#   - IDE autocomplete works on config fields
#   - Config is passed around as a single object, not a raw dict
#
# Usage:
#   from maccai_battery.config import load_config
#   cfg = load_config()                        # loads config.yaml next to repo root
#   cfg = load_config("path/to/config.yaml")   # explicit path
# =============================================================================

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import yaml


# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------

@dataclass
class ProjectDirs:
    candidates: str    = "candidates"
    ml_relaxed: str    = "candidates/ml_relaxed"
    cifs: str          = "candidates/cifs"
    vasp_inputs: str   = "vasp_inputs"
    dft_scf: str       = "dft/scf"
    dft_relax: str     = "dft/relax"
    logs: str          = "logs"


@dataclass
class ProjectConfig:
    name: str                   = "MACCAI-Battery"
    output_dir: str             = "output"
    candidates_db: str          = "candidates.ndjson"
    dirs: ProjectDirs           = field(default_factory=ProjectDirs)

    # Resolved absolute Path objects (populated by load_config).
    _root: Path = field(default=Path("."), repr=False, compare=False)

    def resolve(self, root: Path) -> None:
        """Resolve all directory paths relative to *root*."""
        self._root = root

    @property
    def output_path(self) -> Path:
        return self._root / self.output_dir

    @property
    def candidates_db_path(self) -> Path:
        return self.output_path / self.candidates_db

    def dir_path(self, key: str) -> Path:
        """Return an absolute Path for a named sub-directory key."""
        sub = getattr(self.dirs, key)
        return self.output_path / sub


@dataclass
class GenerationConfig:
    chemical_system: str               = "Li-Fe-P-O"
    energy_above_hull: float           = 0.05
    model_name: str                    = "chemical_system_energy_above_hull"
    batch_size: int                    = 8
    num_batches: int                   = 8
    diffusion_guidance_factor: float   = 2.0
    pytorch_mps_fallback: bool         = True

    @property
    def total_structures(self) -> int:
        return self.batch_size * self.num_batches


@dataclass
class RelaxationConfig:
    device: str                 = "cpu"
    max_structures: Optional[int] = None
    fmax: float                 = 0.05
    max_steps: int              = 300
    relax_cell: bool            = True
    emt_fallback: bool          = True


@dataclass
class ScreeningConfig:
    min_distance_threshold_A: float  = 1.0
    density_min_gcc: float           = 1.0
    density_max_gcc: float           = 8.0
    assign_oxidation_states: bool    = True
    check_charge_neutrality: bool    = True
    check_bond_valence: bool         = True
    hard_filter: bool                = False


@dataclass
class DeduplicationConfig:
    """Configuration for structure deduplication before database insertion.

    Attributes
    ----------
    enabled : bool
        Whether to run deduplication in step 3.
    fingerprint_method : str
        ``"formula_volume_density"`` — fast formula + cell volume + density check.
        ``"pymatgen_rdf"``           — slower radial distribution function fingerprint.
    volume_tolerance_A3 : float
        Allowed volume difference (Å³) for ``formula_volume_density``.
    density_tolerance_gcc : float
        Allowed density difference (g/cm³) for ``formula_volume_density``.
    log_removed : bool
        Log a warning for each removed duplicate.
    """

    enabled: bool                    = True
    fingerprint_method: str          = "formula_volume_density"
    volume_tolerance_A3: float       = 5.0
    density_tolerance_gcc: float     = 0.05
    log_removed: bool                = True


@dataclass
class HullConfig:
    """Configuration for Materials Project convex hull analysis (step 5).

    Attributes
    ----------
    stability_threshold_eV : float
        Energy above hull (eV/atom) below which a candidate is labelled "stable".
    mp_api_key : str
        Materials Project API key.  Leave blank and set ``MP_API_KEY``
        environment variable instead (never commit keys to version control).
    energy_source : str
        ``"relax"`` — use DFT ionic-relaxation energies (recommended).
        ``"scf"``   — use DFT SCF single-point energies (less accurate).
    report_max_e_above_hull : float
        Maximum ΔH_hull to include in reports.
    """

    stability_threshold_eV: float    = 0.1
    mp_api_key: str                  = ""
    energy_source: str               = "relax"
    report_max_e_above_hull: float   = 0.5


@dataclass
class DFTScreeningConfig:
    n_candidates: int                        = 8
    nproc: int                               = 1
    kpoints: List[int]                       = field(default_factory=lambda: [2, 2, 2])
    ecutwfc: float                           = 35.0
    ecutrho: float                           = 280.0
    functional: str                          = "PBE"
    smearing: str                            = "mp"
    degauss: float                           = 0.03
    conv_thr: float                          = 1.0e-4
    mixing_beta: float                       = 0.3
    spin_polarised: bool                     = True
    starting_magnetization: Dict[str, float] = field(
        default_factory=lambda: {"Li": 0.0, "Fe": 0.3, "P": 0.0, "O": 0.0}
    )


@dataclass
class DFTRelaxConfig:
    n_candidates: int                        = 3
    nproc: int                               = 1
    kpoints: List[int]                       = field(default_factory=lambda: [3, 3, 3])
    ecutwfc: float                           = 45.0
    ecutrho: float                           = 360.0
    ion_dynamics: str                        = "bfgs"
    nstep: int                               = 10
    conv_thr: float                          = 1.0e-5
    mixing_beta: float                       = 0.3
    spin_polarised: bool                     = True
    starting_magnetization: Dict[str, float] = field(
        default_factory=lambda: {"Li": 0.0, "Fe": 0.4, "P": 0.0, "O": 0.0}
    )


@dataclass
class PseudopotentialsConfig:
    pseudo_dir: str                  = "pseudo"
    files: Dict[str, str]            = field(default_factory=dict)

    def get(self, element: str) -> str:
        """Return the pseudopotential filename for *element*.

        Raises
        ------
        KeyError
            If no pseudopotential is configured for the given element.
        """
        if element not in self.files:
            raise KeyError(
                f"No pseudopotential configured for element '{element}'. "
                f"Add it to the 'pseudopotentials.files' section of config.yaml."
            )
        return self.files[element]

    def validate_for_system(self, chemical_system: str) -> None:
        """Check that every element in *chemical_system* has a pseudopotential.

        Parameters
        ----------
        chemical_system : str
            Dash-separated element string, e.g. ``"Li-Fe-P-O"``.

        Raises
        ------
        KeyError
            If any element is missing a pseudopotential entry.
        """
        elements = chemical_system.split("-")
        missing = [el for el in elements if el not in self.files]
        if missing:
            raise KeyError(
                f"Missing pseudopotentials for: {missing}. "
                f"Add them to the 'pseudopotentials.files' section of config.yaml."
            )


@dataclass
class DatabaseConfig:
    generator: str                        = "MatterGen"
    dft_workflow: str                     = "PBE_scf_relax_QE"
    vasp_incar_defaults: Dict[str, object] = field(
        default_factory=lambda: {"ENCUT": 520, "KSPACING": 0.3}
    )


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    """Complete typed configuration for the MACCAI battery pipeline."""

    project: ProjectConfig                   = field(default_factory=ProjectConfig)
    generation: GenerationConfig             = field(default_factory=GenerationConfig)
    relaxation: RelaxationConfig             = field(default_factory=RelaxationConfig)
    screening: ScreeningConfig               = field(default_factory=ScreeningConfig)
    deduplication: DeduplicationConfig       = field(default_factory=DeduplicationConfig)
    dft_screening: DFTScreeningConfig        = field(default_factory=DFTScreeningConfig)
    dft_relax: DFTRelaxConfig                = field(default_factory=DFTRelaxConfig)
    hull: HullConfig                         = field(default_factory=HullConfig)
    pseudopotentials: PseudopotentialsConfig = field(default_factory=PseudopotentialsConfig)
    database: DatabaseConfig                 = field(default_factory=DatabaseConfig)

    def validate(self) -> None:
        """Run cross-field validation checks.

        Raises
        ------
        ValueError
            If any configuration is logically inconsistent.
        """
        # Generation
        if self.generation.energy_above_hull < 0:
            raise ValueError("energy_above_hull must be >= 0.")
        if self.generation.batch_size < 1:
            raise ValueError("batch_size must be >= 1.")
        if self.generation.num_batches < 1:
            raise ValueError("num_batches must be >= 1.")

        # Relaxation
        if self.relaxation.fmax <= 0:
            raise ValueError("relaxation.fmax must be > 0.")
        if self.relaxation.max_steps < 1:
            raise ValueError("relaxation.max_steps must be >= 1.")

        # Screening
        if self.screening.min_distance_threshold_A <= 0:
            raise ValueError("min_distance_threshold_A must be > 0.")
        if self.screening.density_min_gcc >= self.screening.density_max_gcc:
            raise ValueError("density_min_gcc must be < density_max_gcc.")

        # Deduplication
        valid_methods = {"formula_volume_density", "pymatgen_rdf"}
        if self.deduplication.fingerprint_method not in valid_methods:
            raise ValueError(
                f"deduplication.fingerprint_method must be one of {valid_methods}, "
                f"got '{self.deduplication.fingerprint_method}'."
            )
        if self.deduplication.volume_tolerance_A3 < 0:
            raise ValueError("deduplication.volume_tolerance_A3 must be >= 0.")
        if self.deduplication.density_tolerance_gcc < 0:
            raise ValueError("deduplication.density_tolerance_gcc must be >= 0.")

        # DFT
        if self.dft_screening.n_candidates < 1:
            raise ValueError("dft_screening.n_candidates must be >= 1.")
        if self.dft_relax.n_candidates < 1:
            raise ValueError("dft_relax.n_candidates must be >= 1.")
        if self.dft_relax.n_candidates > self.dft_screening.n_candidates:
            raise ValueError(
                "dft_relax.n_candidates cannot exceed dft_screening.n_candidates."
            )

        # Hull
        if self.hull.stability_threshold_eV < 0:
            raise ValueError("hull.stability_threshold_eV must be >= 0.")
        if self.hull.energy_source not in ("relax", "scf"):
            raise ValueError(
                f"hull.energy_source must be 'relax' or 'scf', "
                f"got '{self.hull.energy_source}'."
            )

        # Pseudopotentials
        self.pseudopotentials.validate_for_system(self.generation.chemical_system)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _dict_to_dataclass(cls, data: dict):
    """Recursively build a dataclass from a (possibly nested) dict.

    Only keys that correspond to dataclass fields are used;
    extra keys in *data* are silently ignored so that config.yaml
    can contain comments or future fields without breaking old code.
    """
    import dataclasses

    if not dataclasses.is_dataclass(cls):
        return data  # leaf value

    field_map = {f.name: f for f in dataclasses.fields(cls)
                 if not f.name.startswith("_")}
    kwargs: dict = {}

    for name, f in field_map.items():
        if name not in data:
            continue  # use dataclass default

        raw = data[name]

        # Recurse into nested dataclasses
        if dataclasses.is_dataclass(f.type):
            kwargs[name] = _dict_to_dataclass(f.type, raw)
        else:
            # Try to resolve string type hints (e.g. "ProjectDirs")
            origin = getattr(f.type, "__origin__", None)
            if origin is None and isinstance(f.type, str):
                # best-effort: look up in module globals
                resolved = globals().get(f.type)
                if resolved is not None and dataclasses.is_dataclass(resolved):
                    kwargs[name] = _dict_to_dataclass(resolved, raw)
                    continue
            kwargs[name] = raw

    return cls(**kwargs)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

# Default config file location: repo root / config.yaml
_DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"


def load_config(path: Optional[str | Path] = None) -> PipelineConfig:
    """Load and validate the pipeline configuration from a YAML file.

    Parameters
    ----------
    path : str or Path, optional
        Path to ``config.yaml``.  If omitted, the file is looked up in:
        1. The ``MACCAI_CONFIG`` environment variable.
        2. ``<repo_root>/config.yaml`` (default).

    Returns
    -------
    PipelineConfig
        A fully validated, typed configuration object.

    Raises
    ------
    FileNotFoundError
        If the config file cannot be found.
    ValueError
        If the configuration is logically inconsistent.
    """
    if path is None:
        env_path = os.environ.get("MACCAI_CONFIG")
        path = Path(env_path) if env_path else _DEFAULT_CONFIG_PATH

    path = Path(path).expanduser().resolve()

    if not path.exists():
        raise FileNotFoundError(
            f"Config file not found: {path}\n"
            f"Either create config.yaml or set the MACCAI_CONFIG environment variable."
        )

    with path.open("r") as fh:
        raw: dict = yaml.safe_load(fh) or {}

    cfg = _dict_to_dataclass(PipelineConfig, raw)

    # Resolve project paths relative to the config file's directory
    cfg.project.resolve(path.parent)

    # Validate cross-field consistency
    cfg.validate()

    return cfg
