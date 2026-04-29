# MACCAI Battery Materials Discovery Pipeline

> **End-to-end ML-to-DFT pipeline for computational battery materials discovery.**
> MatterGen → MatterSim → Sanity Checks → Quantum ESPRESSO DFT → Hull Analysis

---

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Pipeline Architecture](#pipeline-architecture)
4. [Project Structure](#project-structure)
5. [Setup & Installation](#setup--installation)
   - [Prerequisites](#prerequisites)
   - [Environment 1 — `mattergen` (Step 1)](#environment-1--mattergen-step-1)
   - [Environment 2 — `maccai` (Steps 2–6)](#environment-2--maccai-steps-26)
   - [Install MatterSim](#install-mattersim)
   - [Install Quantum ESPRESSO](#install-quantum-espresso)
   - [Install pymatgen-io-espresso](#install-pymatgen-io-espresso)
   - [Download Pseudopotentials](#download-pseudopotentials)
   - [Set Environment Variables](#set-environment-variables)
6. [Configuration](#configuration)
7. [Running the Pipeline](#running-the-pipeline)
   - [Full Pipeline via `main.py`](#full-pipeline-via-mainpy)
   - [Running Individual Steps](#running-individual-steps)
8. [Step-by-Step Reference](#step-by-step-reference)
   - [Step 1 — Crystal Generation (MatterGen)](#step-1--crystal-generation-mattergen)
   - [Step 2 — ML Relaxation (MatterSim)](#step-2--ml-relaxation-mattersim)
   - [Step 3 — Sanity Checks & Candidate Database](#step-3--sanity-checks--candidate-database)
   - [Step 4 — DFT Calculations (Quantum ESPRESSO)](#step-4--dft-calculations-quantum-espresso)
   - [Step 5 — Merge DFT Results](#step-5--merge-dft-results)
   - [Step 6 — Hull Stability Analysis](#step-6--hull-stability-analysis)
9. [Output Directory Structure](#output-directory-structure)
10. [Candidate Database Schema](#candidate-database-schema)
11. [DFT Notes & K-Point Convergence](#dft-notes--k-point-convergence)
12. [Testing](#testing)
13. [Troubleshooting](#troubleshooting)
14. [Citation](#citation)

---

## Overview

The **MACCAI Battery Materials Discovery Pipeline** is a modular, GPU-server-ready framework for
high-throughput computational discovery of novel battery electrode and electrolyte materials. It
automates the full screening funnel from generative AI crystal structure prediction through
machine-learning pre-relaxation, structural quality control, first-principles DFT validation, and
thermodynamic stability analysis against the Materials Project convex hull.

The pipeline was designed for Li-Fe-P-O chemistry (targeting LiFePO₄ analogs) but is straightforward
to adapt to any multi-element chemical system supported by MatterGen.

### What it does

```text
MatterGen (diffusion model)
    │  generates novel crystal structures conditioned on chemistry
    ▼
MatterSim (ML interatomic potential)
    │  fast geometry pre-relaxation, orders of magnitude cheaper than DFT
    ▼
Structural Sanity Checks (pymatgen)
    │  removes unphysical structures; deduplicates; populates candidate database
    ▼
Quantum ESPRESSO DFT — SCF + Ionic Relaxation
    │  high-accuracy single-point energies and structural optimisation
    ▼
Hull Analysis (Materials Project API)
    │  computes energy above the convex hull; flags thermodynamically stable candidates
    ▼
Ranked candidate list  →  candidates.ndjson  +  hull_analysis_report.txt
```

### Design principles

- **Single config file** — all parameters live in `config.yaml`; no magic numbers in code.
- **Append-only candidate database** — `candidates.ndjson` accumulates metadata from every step so
  no information is lost and the pipeline is fully resumable.
- **Two conda environments** — MatterGen's pinned PyTorch requirement is isolated from the rest of
  the stack, preventing dependency conflicts.
- **CPU-parallel DFT** — `pw.x` runs via MPI across all available CPU cores; GPU is used for
  MatterGen and MatterSim only.

---

## Quick Start

> For scientists who want to be running within 15 minutes on a GPU machine.

```shell
# ── 1. Clone the repository ───────────────────────────────────────────────
git clone https://github.com/your-org/maccai-battery.git
cd maccai-battery

# ── 2. Build environments ─────────────────────────────────────────────────
conda env create -f environment_mattergen.yml   # step 1 only
conda env create -f environment.yml             # steps 2–6

# ── 3. Install third-party tools (maccai env) ────────────────────────────
conda activate maccai

# MatterSim
cd /path/to/mattersim && pip install -e . && cd -

# Quantum ESPRESSO (pw.x must be on PATH)
conda install -c conda-forge qe

# pymatgen-io-espresso
pip install git+https://github.com/Griffin-Group/pymatgen-io-espresso

# ── 4. Download pseudopotentials ─────────────────────────────────────────
python scripts/download_pseudopotentials.py

# ── 5. Set required environment variable ─────────────────────────────────
export MP_API_KEY="your_materials_project_key"   # get one free at materialsproject.org/api

# ── 6a. Run Step 1 in the MatterGen environment ──────────────────────────
conda activate mattergen
python scripts/01_generate.py

# ── 6b. Run Steps 2–6 in the main environment ────────────────────────────
conda activate maccai
python main.py --from-step 2
```

The final stability report is written to `output/hull_analysis_report.txt`.

---

## Pipeline Architecture

| Step | Script | Tool | Conda Environment |
|------|--------|------|-------------------|
| 1 | `scripts/01_generate.py` | MatterGen (diffusion model) | `mattergen` |
| 2 | `scripts/02_relax.py` | MatterSim (ML potential via ASE) | `maccai` |
| 3 | `scripts/03_sanity_check.py` | pymatgen | `maccai` |
| 4 | `scripts/04_dft.py` | Quantum ESPRESSO `pw.x` | `maccai` |
| 5 | `scripts/05_merge_dft_results.py` | pymatgen + pandas | `maccai` |
| 6 | `scripts/06_hull_analysis.py` | Materials Project API + pymatgen | `maccai` |

### Why two environments?

MatterGen requires PyTorch 2.0–2.2 (pinned) with specific versions of `pytorch-lightning`,
`hydra-core`, and `torch-geometric`. MatterSim (used in step 2) requires PyTorch ≥ 2.3. These
requirements are mutually exclusive and cannot coexist in a single conda environment.

The solution is to isolate step 1 in the `mattergen` environment and run all subsequent steps in the
`maccai` environment. The handoff between them is the `output/candidates/cifs/` directory, which
persists on disk.

---

## Project Structure

```text
maccai_battery/
├── maccai_battery/               # Python package
│   ├── __init__.py
│   ├── config.py                 # Typed config loader (PipelineConfig)
│   ├── database.py               # Candidate NDJSON database (CandidateDatabase)
│   ├── generation.py             # MatterGen wrapper (run_mattergen)
│   ├── relaxation.py             # MatterSim wrapper (relax_structures)
│   ├── checks.py                 # Sanity checks (run_sanity_checks, CheckResult)
│   ├── hull.py                   # Hull analysis (HullAnalyzer)
│   ├── utils.py                  # Shared utilities (logging, parsers, converters)
│   └── dft/                      # DFT subpackage
│       ├── __init__.py
│       ├── input_generator.py    # QE input files (make_scf_input, make_relax_input)
│       ├── runner.py             # QE subprocess runner (run_qe_pw)
│       ├── parser.py             # QE output/XML parser (parse_qe_xml, QEXMLResult)
│       └── workflow.py           # Full DFT workflow (DFTWorkflow)
├── scripts/
│   ├── 01_generate.py            # Step 1: MatterGen
│   ├── 02_relax.py               # Step 2: MatterSim ML relaxation
│   ├── 03_sanity_check.py        # Step 3: Sanity checks + candidate database
│   ├── 04_dft.py                 # Step 4: Quantum ESPRESSO SCF + relax
│   ├── 05_merge_dft_results.py   # Step 5: Merge DFT results into database
│   ├── 06_hull_analysis.py       # Step 6: Convex hull stability analysis
│   └── download_pseudopotentials.py
├── tests/
│   ├── __init__.py
│   └── test_utils.py
├── config.yaml                   # Single source of truth for all parameters
├── environment.yml               # Main conda env (steps 2–6, GPU-ready)
├── environment_mattergen.yml     # MatterGen-only conda env (step 1)
├── requirements.txt
├── setup.py
├── pyproject.toml
└── main.py                       # End-to-end pipeline entry point
```

---

## Setup & Installation

### Prerequisites

Before creating the conda environments, ensure the following are available on your system:

| Requirement | Minimum Version | Notes |
|-------------|----------------|-------|
| [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or Anaconda | Any recent | `conda` must be on `PATH` |
| Python | 3.10 | Managed by conda |
| CUDA toolkit | 11.8+ | Required for GPU acceleration; not needed for CPU-only runs |
| git | Any recent | For cloning MatterGen and MatterSim |
| Quantum ESPRESSO `pw.x` | 7.0+ | Installed separately (see below) |

> **GPU Note:** The pipeline is designed for a Linux GPU server with NVIDIA hardware (A100, V100,
> RTX series). For CPU-only machines set `relaxation.device: "cpu"` in `config.yaml` and replace the
> `pytorch-cuda` conda dependency with `cpuonly`. MatterGen will still run but significantly slower.

---

### Environment 1 — `mattergen` (Step 1)

This environment is **only used for Step 1**. It contains MatterGen and its pinned dependencies.

```shell
# Create the environment
conda env create -f environment_mattergen.yml

# Activate it
conda activate mattergen

# Clone and install MatterGen from source
git clone https://github.com/microsoft/mattergen.git
cd mattergen
pip install -e . \
    --extra-index-url https://download.pytorch.org/whl/cu118 \
    --no-build-isolation
cd ..

# Verify
mattergen-generate --help
```

> **Note:** Do NOT install MatterSim inside the `mattergen` environment. The conflicting PyTorch
> versions will break one or both tools.

---

### Environment 2 — `maccai` (Steps 2–6)

This is the primary environment used for everything after generation. It includes MatterSim (for ML
relaxation), pymatgen, and the Materials Project API client.

```shell
# Create the environment (takes several minutes — CUDA packages are large)
conda env create -f environment.yml

# Activate it
conda activate maccai

# Install this package in editable mode (if not already done by environment.yml)
pip install -e .
```

---

### Install MatterSim

MatterSim is not available on PyPI or conda and must be installed from source. Do this inside the
`maccai` environment.

```shell
conda activate maccai

# Option A — install from your local clone
git clone https://github.com/microsoft/mattersim.git
cd mattersim
pip install -e .
cd ..

# Option B — install directly from GitHub
pip install git+https://github.com/microsoft/mattersim.git
```

Verify the installation:

```python
from mattersim.forcefield import MatterSimCalculator
print("MatterSim available ✓")
```

> **Fallback:** If MatterSim is unavailable, Step 2 will fall back to the ASE EMT potential
> (controlled by `relaxation.emt_fallback: true` in `config.yaml`). EMT is much less accurate and
> should only be used for pipeline testing.

---

### Install Quantum ESPRESSO

Quantum ESPRESSO's `pw.x` binary must be available on your `PATH` before running Step 4.

**Option A — conda (recommended, simplest):**

```shell
conda activate maccai
conda install -c conda-forge qe
```

**Option B — system package manager (Ubuntu/Debian):**

```shell
sudo apt-get update && sudo apt-get install quantum-espresso
```

**Option C — build from source:**

See the [official QE build instructions](https://www.quantum-espresso.org/Doc/user_guide/node9.html).
Building from source allows you to enable OpenMP and tune MPI settings for your hardware, but is
not required for this pipeline.

Verify the installation:

```shell
pw.x --version
```

> **Performance note:** `pw.x` is CPU-parallelised via MPI. It is **not** GPU-accelerated in the
> standard conda-forge build. On a machine with 16+ CPU cores, always pass `--nproc 16` (or however
> many cores you have) to Step 4 for maximum throughput.

---

### Install pymatgen-io-espresso

This library handles reading and writing Quantum ESPRESSO input/output files within pymatgen. It is
required for Step 4 and must be installed via pip from GitHub:

```shell
conda activate maccai
pip install git+https://github.com/Griffin-Group/pymatgen-io-espresso
```

---

### Download Pseudopotentials

Step 4 requires PBE USPP/PAW pseudopotential files (`.UPF` format) for each element in your
chemical system. For the default Li-Fe-P-O chemistry, run:

```shell
conda activate maccai
python scripts/download_pseudopotentials.py
```

This downloads the following files into `pseudo/` (configurable via `pseudopotentials.pseudo_dir`):

| Element | File | Type |
|---------|------|------|
| Li | `li_pbe_v1.4.uspp.F.UPF` | USPP |
| Fe | `Fe.pbe-spn-kjpaw_psl.0.2.1.UPF` | PAW |
| P | `P.pbe-n-kjpaw_psl.0.1.UPF` | PAW |
| O | `O.pbe-n-kjpaw_psl.0.1.UPF` | PAW |

For other chemical systems, add the corresponding element entries to `pseudopotentials.files` in
`config.yaml` and place (or download) the `.UPF` files in the `pseudo/` directory.

---

### Set Environment Variables

Two environment variables are used by the pipeline:

```shell
# Required for Step 6 (hull analysis)
# Get a free key at: https://materialsproject.org/api
export MP_API_KEY="your_key_here"

# Optional: override the config file location
export MACCAI_CONFIG="/absolute/path/to/config.yaml"
```

To make these persistent across terminal sessions, add them to your `~/.bashrc` or
`~/.bash_profile`:

```shell
echo 'export MP_API_KEY="your_key_here"' >> ~/.bashrc
source ~/.bashrc
```

> **Security:** Never paste your Materials Project API key into `config.yaml`. The `hull.mp_api_key`
> field in the config is intentionally left blank. Always use the environment variable.

---

## Configuration

All pipeline parameters are controlled by a single file: **`config.yaml`**. Edit this file rather
than touching the scripts directly.

The config is organised into the following sections:

```yaml
project:            # output directory, subdirectory layout
generation:         # MatterGen model, chemical system, batch settings
relaxation:         # MatterSim device, force thresholds, max steps
screening:          # sanity-check thresholds (distances, density, charge)
deduplication:      # fingerprinting method and tolerances
dft_screening:      # QE SCF settings (k-points, cutoffs, spin, nproc)
dft_relax:          # QE ionic relaxation settings
pseudopotentials:   # UPF file locations, element mapping
hull:               # stability threshold, energy source, MP API settings
database:           # generator tags, workflow labels
```

### Key parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `project.output_dir` | `output/` | Root for all pipeline outputs |
| `generation.chemical_system` | `"Li-Fe-P-O"` | Elements to condition MatterGen on |
| `generation.batch_size` | `8` | Structures per MatterGen batch |
| `generation.num_batches` | `8` | Number of batches (total = `batch_size × num_batches`) |
| `generation.energy_above_hull` | `0.05` | Max ΔH above hull for sampling (eV/atom) |
| `relaxation.device` | `"cpu"` | `"cpu"` or `"cuda"` for MatterSim |
| `relaxation.fmax` | `0.05` | Force convergence threshold (eV/Å) |
| `dft_screening.n_candidates` | `8` | Structures forwarded to QE SCF |
| `dft_screening.nproc` | `1` | MPI processes for `pw.x` (set to CPU core count) |
| `dft_screening.kpoints` | `[2,2,2]` | k-point grid for SCF screening (coarse) |
| `dft_relax.n_candidates` | `3` | Top-ranked SCF structures forwarded to ionic relax |
| `dft_relax.nproc` | `1` | MPI processes for `pw.x` relaxation |
| `dft_relax.kpoints` | `[3,3,3]` | k-point grid for ionic relaxation |
| `pseudopotentials.pseudo_dir` | `"pseudo/"` | Path to `.UPF` files |
| `hull.stability_threshold_eV` | `0.1` | ΔH above hull threshold to flag as "stable" |
| `hull.energy_source` | `"relax"` | Use `"relax"` or `"scf"` energies for hull comparison |
| `hull.mp_api_key` | `""` | Leave blank — set `MP_API_KEY` env var instead |

### Example: switching to GPU for MatterSim

```yaml
relaxation:
  device: "cuda"
```

### Example: expanding the candidate pool

```yaml
generation:
  batch_size: 16
  num_batches: 20     # generates 320 structures total

dft_screening:
  n_candidates: 20    # pass top 20 to SCF

dft_relax:
  n_candidates: 5     # pass top 5 to ionic relax
```

### Example: changing the chemical system

```yaml
generation:
  chemical_system: "Na-Mn-O"

pseudopotentials:
  files:
    Na: "Na.pbe-spn-kjpaw_psl.0.2.1.UPF"
    Mn: "Mn.pbe-spn-kjpaw_psl.0.3.1.UPF"
    O:  "O.pbe-n-kjpaw_psl.0.1.UPF"
```

---

## Running the Pipeline

### Full Pipeline via `main.py`

`main.py` is the recommended entry point for running the pipeline. It orchestrates all steps as
subprocesses, logs progress to the console, and stops cleanly if any step fails.

```shell
# Run all 6 steps end-to-end
python main.py

# Preview what would run without executing anything
python main.py --dry-run

# Resume from a specific step (e.g. after step 1 ran in the mattergen env)
python main.py --from-step 2

# Run only a specific range of steps
python main.py --steps 2 3 4

# Run a single step
python main.py --step 4

# Run up to and including step 3
python main.py --to-step 3

# Use an alternative config file
python main.py --config /path/to/my_config.yaml
```

#### Step-specific pass-through options

```shell
# Step 2 — override the inference device
python main.py --step 2 --device cuda

# Step 4 — parallel DFT with 16 MPI processes, SCF-only stage
python main.py --step 4 --nproc 16 --n-scf 10 --n-relax 4 --dft-stage scf

# Step 6 — use SCF energies with a tighter stability threshold
python main.py --step 6 --energy-source scf --threshold 0.05
```

---

### Running Individual Steps

Each pipeline step can also be run directly as a standalone script. This is useful for debugging,
testing config changes, or re-running a single step without the orchestrator overhead.

```shell
# Step 1 — crystal generation (run in mattergen env)
conda activate mattergen
python scripts/01_generate.py [--dry-run]

# Step 2 — ML relaxation
conda activate maccai
python scripts/02_relax.py [--device cuda] [--max-structures 20]

# Step 3 — sanity checks and candidate database
python scripts/03_sanity_check.py [--hard-filter]

# Step 4 — DFT (SCF and/or ionic relaxation)
python scripts/04_dft.py [--stage scf|relax|all] [--nproc N] [--n-scf N] [--n-relax M]

# Step 5 — merge DFT results into candidates.ndjson
python scripts/05_merge_dft_results.py [--dry-run]

# Step 6 — hull stability analysis
python scripts/06_hull_analysis.py [--threshold 0.05] [--energy-source scf|relax]

# Utility — download pseudopotentials
python scripts/download_pseudopotentials.py
```

---

## Step-by-Step Reference

### Step 1 — Crystal Generation (MatterGen)

**Environment:** `mattergen` | **Script:** `scripts/01_generate.py`

Uses Microsoft's MatterGen diffusion model to generate novel crystal structures conditioned on a
target chemical system and a maximum energy above the convex hull.

**Relevant config keys:**

```yaml
generation:
  chemical_system: "Li-Fe-P-O"
  energy_above_hull: 0.05          # eV/atom
  model_name: "chemical_system_energy_above_hull"
  batch_size: 8
  num_batches: 8                   # 64 structures total
  diffusion_guidance_factor: 2.0
  pytorch_mps_fallback: false      # set true only on Apple Silicon
```

**Output:** `output/candidates/cifs/generated_crystals.extxyz`

**Typical run time:** 5–30 minutes depending on GPU speed and batch settings.

```shell
conda activate mattergen
python scripts/01_generate.py

# Then hand off to the main environment
conda activate maccai
```

---

### Step 2 — ML Relaxation (MatterSim)

**Environment:** `maccai` | **Script:** `scripts/02_relax.py`

Performs geometry pre-relaxation of generated structures using MatterSim, a large-scale machine
learning interatomic potential trained by Microsoft Research. Using `FrechetCellFilter`, both atomic
positions and cell shape/volume are optimised. This removes the worst strain artefacts from
generation at a fraction of the cost of DFT.

**Relevant config keys:**

```yaml
relaxation:
  device: "cuda"         # or "cpu"
  fmax: 0.05             # force convergence (eV/Å)
  max_steps: 300
  relax_cell: true
  emt_fallback: true     # use ASE EMT if MatterSim unavailable (testing only)
```

**Output:**
- `output/candidates/ml_relaxed/*_ml_relaxed.extxyz`
- `output/candidates/ml_relaxed/ml_relaxed_summary.csv`

```shell
python scripts/02_relax.py --device cuda
# or with a cap on number of structures:
python scripts/02_relax.py --device cuda --max-structures 40
```

---

### Step 3 — Sanity Checks & Candidate Database

**Environment:** `maccai` | **Script:** `scripts/03_sanity_check.py`

Applies a series of structural quality filters to ML-relaxed structures using pymatgen:

- **Deduplication** — removes near-identical structures based on formula + cell volume + density
  fingerprints (configurable).
- **Minimum interatomic distance** — flags structures with unphysically short bonds.
- **Density range check** — flags structures with implausible cell densities.
- **Oxidation state assignment** — attempts pymatgen oxidation state decoration.
- **Charge neutrality check** — verifies electroneutrality of the formal charge sum.
- **Bond valence analysis** — estimates bond valence sums for sanity.

Structures that pass (or have soft warnings) are written to the candidate database
`output/candidates.ndjson`. By default, failed checks produce warnings rather than hard errors
(controlled by `screening.hard_filter`).

**Relevant config keys:**

```yaml
screening:
  min_distance_threshold_A: 1.0
  density_min_gcc: 1.0
  density_max_gcc: 8.0
  hard_filter: false

deduplication:
  enabled: true
  fingerprint_method: "formula_volume_density"
  volume_tolerance_A3: 5.0
  density_tolerance_gcc: 0.05
```

```shell
python scripts/03_sanity_check.py
# Treat failed checks as fatal errors:
python scripts/03_sanity_check.py --hard-filter
```

---

### Step 4 — DFT Calculations (Quantum ESPRESSO)

**Environment:** `maccai` | **Script:** `scripts/04_dft.py`

The most computationally intensive step. Runs Quantum ESPRESSO `pw.x` calculations in two stages:

**Stage 1 — SCF Screening:**
Performs self-consistent field (SCF) calculations on the top `dft_screening.n_candidates`
ML-relaxed structures (ranked by ML energy per atom). This gives single-point DFT energies using a
coarse k-point grid, suitable for ranking candidates but not for publication-quality thermodynamics.

**Stage 2 — Ionic Relaxation:**
Takes the top `dft_relax.n_candidates` SCF-ranked structures and performs full ionic relaxation
(`vc-relax` or `relax` calculation). This gives more accurate final geometries and energies.

**Relevant config keys:**

```yaml
dft_screening:
  n_candidates: 8
  nproc: 1               # set to CPU core count on your machine
  kpoints: [2, 2, 2]     # coarse grid — screening only
  ecutwfc: 35            # Ry
  ecutrho: 280           # Ry
  spin_polarised: true
  starting_magnetization:
    Li: 0.0
    Fe: 0.3              # high-spin Fe; adjust if SCF diverges
    P: 0.0
    O: 0.0

dft_relax:
  n_candidates: 3
  nproc: 1
  kpoints: [3, 3, 3]
  ecutwfc: 45
  ecutrho: 360
  ion_dynamics: "bfgs"
  nstep: 10
```

```shell
# Run both SCF and relax stages
python scripts/04_dft.py --stage all --nproc 16

# Run SCF only
python scripts/04_dft.py --stage scf --nproc 16 --n-scf 10

# Run ionic relax only (SCF must have been run first)
python scripts/04_dft.py --stage relax --nproc 16 --n-relax 4

# Preview jobs without submitting
python scripts/04_dft.py --dry-run
```

**Output structure:**

```text
output/dft/
├── scf/
│   ├── scf_0/  {qe.in, qe.out, out/}
│   ├── scf_1/  {qe.in, qe.out, out/}
│   └── ...
└── relax/
    ├── relax_0/  {qe.in, qe.out, out/}
    └── ...
```

Summary files are written to `output/dft_scf_summary.txt` and `output/dft_relax_summary.txt`.

---

### Step 5 — Merge DFT Results

**Environment:** `maccai` | **Script:** `scripts/05_merge_dft_results.py`

Parses the Quantum ESPRESSO XML output files, extracts total energies and convergence flags for
each SCF and relax calculation, and merges them back into the candidate database
(`output/candidates.ndjson`). Each candidate's `dft_jobs` record is populated:

```json
{
  "dft_jobs": {
    "status": "done",
    "scf":   {"energy_eV": -1234.5, "energy_eV_per_atom": -5.6, "converged": true},
    "relax": {"energy_eV": -1235.1, "energy_eV_per_atom": -5.61}
  }
}
```

```shell
python scripts/05_merge_dft_results.py
# Preview merges without writing:
python scripts/05_merge_dft_results.py --dry-run
```

---

### Step 6 — Hull Stability Analysis

**Environment:** `maccai` | **Script:** `scripts/06_hull_analysis.py`

Queries the Materials Project API to retrieve all known compounds in the target chemical system,
constructs the convex hull using pymatgen's `PhaseDiagram`, and computes the energy above the hull
(ΔH_hull) for each candidate.

Candidates with ΔH_hull ≤ `hull.stability_threshold_eV` are flagged as `is_stable: true` and
written to the `hull_analysis` field of their database record.

Results are written to `output/hull_analysis_report.txt`.

**Relevant config keys:**

```yaml
hull:
  stability_threshold_eV: 0.1   # eV/atom; 0.0–0.05 for high confidence
  energy_source: "relax"        # use DFT relax energies (recommended)
  report_max_e_above_hull: 0.5  # show candidates up to this threshold in report
```

```shell
# Use default settings
python scripts/06_hull_analysis.py

# Use SCF energies with a tighter stability criterion
python scripts/06_hull_analysis.py --energy-source scf --threshold 0.05
```

> **Important:** `MP_API_KEY` must be set as an environment variable before running this step.
> Get your free key at [materialsproject.org/api](https://materialsproject.org/api).

---

## Output Directory Structure

```text
output/
├── candidates.ndjson                  # Candidate database (append-only, one JSON per line)
├── candidates_summary.csv             # Flat CSV export of the database
├── dft_scf_summary.txt                # SCF energy ranking
├── dft_relax_summary.txt              # Ionic relax energy ranking
├── hull_analysis_report.txt           # Stability ranking and hull distances
├── candidates/
│   ├── cifs/
│   │   ├── generated_crystals.extxyz  # Raw MatterGen output (all batches)
│   │   └── mattergen_results/         # Per-batch metadata
│   └── ml_relaxed/
│       ├── *_ml_relaxed.extxyz        # Per-structure ML-relaxed geometries
│       └── ml_relaxed_summary.csv     # MatterSim energies and relaxation metadata
├── dft/
│   ├── scf/
│   │   ├── scf_0/ {qe.in, qe.out, out/}
│   │   ├── scf_1/ {qe.in, qe.out, out/}
│   │   └── ...
│   └── relax/
│       ├── relax_0/ {qe.in, qe.out, out/}
│       └── ...
└── logs/
    ├── 01_generate.log
    ├── 02_relax.log
    ├── 03_sanity_check.log
    ├── 04_dft.log
    ├── 05_merge_dft_results.log
    └── 06_hull_analysis.log
```

---

## Candidate Database Schema

The pipeline maintains a single NDJSON database: `output/candidates.ndjson`. Each line is a
self-contained JSON object representing one candidate structure. Fields are populated progressively
as the structure passes through pipeline steps.

```json
{
  "id": "MG-abc12345",
  "formula": "LiFePO4",
  "stoichiometry": {"Li": 1, "Fe": 1, "P": 1, "O": 4},
  "structure_files": {
    "source_extxyz": "candidates/cifs/generated_crystals.extxyz",
    "ml_relaxed":    "candidates/ml_relaxed/frame0_ml_relaxed.extxyz"
  },
  "generation_metadata": {
    "generator": "MatterGen",
    "frame_index": 0,
    "chemical_system": "Li-Fe-P-O",
    "energy_above_hull_target": 0.05
  },
  "ml_scores": {
    "matter_sim_energy_eV_per_atom": -3.12,
    "density_gcc": 3.45,
    "volume_A3": 291.4,
    "n_atoms": 28
  },
  "filters": {
    "charge_neutral": true,
    "min_distance_ok": true,
    "density_ok": true,
    "passed_all": true,
    "warning_count": 0
  },
  "dft_jobs": {
    "status": "done",
    "workflow": "PBE_scf_relax_QE",
    "scf": {
      "energy_eV": -1234.5,
      "energy_eV_per_atom": -5.6,
      "converged": true,
      "n_scf_cycles": 18,
      "qe_dir": "dft/scf/scf_0"
    },
    "relax": {
      "energy_eV": -1235.1,
      "energy_eV_per_atom": -5.61,
      "converged": true,
      "n_ionic_steps": 8,
      "qe_dir": "dft/relax/relax_0"
    }
  },
  "hull_analysis": {
    "e_above_hull_eV": 0.023,
    "is_stable": true,
    "energy_source": "relax",
    "stability_threshold_eV": 0.1
  }
}
```

The NDJSON format means the file can be read line-by-line without loading everything into memory,
and new records can be appended safely without rewriting the file.

---

## DFT Notes & K-Point Convergence

### QE is CPU-parallel, not GPU-accelerated

The standard conda-forge build of Quantum ESPRESSO does not use the GPU. All `pw.x` parallelism is
MPI over CPU cores. On a GPU machine with 16+ cores, always set `nproc` to the full CPU core count:

```shell
python scripts/04_dft.py --nproc 16
```

Or in `config.yaml`:

```yaml
dft_screening:
  nproc: 16
dft_relax:
  nproc: 16
```

### K-point convergence guidance

The default k-point grids in `config.yaml` are designed for **cost-effective screening**, not
publication-quality energies. The table below shows the intended use for each grid level (assuming a
~28-atom LiFePO₄ primitive cell):

| Stage | Grid | Purpose | Accuracy |
|-------|------|---------|----------|
| ML pre-screening | — | No DFT; ML energies only | ~50 meV/atom |
| SCF screening | `[2,2,2]` | Cheap ranking only | ~20 meV/atom |
| SCF medium | `[3,3,3]` | Better ranking | ~5 meV/atom |
| Ionic relaxation | `[3,3,3]` | Structural optimisation | ~5 meV/atom |
| Final static SCF | `[4,4,4]` | Production hull energies | ~1 meV/atom |

> **Warning:** The `[2,2,2]` grid used for screening is **not suitable** for computing hull
> distances or making thermodynamic stability claims. Always re-run with a converged k-grid (at
> least `[4,4,4]`) and verify convergence before including DFT energies in a publication.

### Convergence test protocol

For publication results, run a k-point convergence test on your most stable candidate:

```shell
# Run SCF at progressively finer grids and compare total energies
for kgrid in "2 2 2" "3 3 3" "4 4 4" "6 6 6"; do
    python scripts/04_dft.py --stage scf --nproc 16
done
```

Confirm that total energies converge to within 1 meV/atom between the last two grids. Use the
coarsest grid that meets this criterion for all final calculations.

### Spin polarisation and magnetism

The pipeline uses spin-polarised calculations (`nspin=2`) with an initial magnetic moment on Fe.
This is required for any system containing transition metals with partially filled d-shells. The
initial moments in `config.yaml` are starting guesses; the SCF cycle will find the ground-state
spin configuration.

If calculations for Fe-containing systems diverge, try:
- Lowering `mixing_beta` from `0.3` to `0.2` or `0.15`
- Adjusting the initial Fe moment from `0.3` to `0.5` (or vice versa)
- Ensuring you are using a PAW pseudopotential for Fe (not USPP)

### PBE vs. Materials Project energies

The pipeline uses PBE exchange-correlation (via pseudopotentials) to match the methodology of the
Materials Project. However, systematic offsets between Quantum ESPRESSO and VASP (used by MP) can
exist due to differences in pseudopotential construction, plane-wave basis conventions, and U-values
for correlated d-electron systems.

Treat hull distances from this pipeline as **relative screening indicators** rather than
high-accuracy absolute thermodynamic quantities. For final validation, consider re-running
calculations with VASP + MP-compatible POTCAR files and `LDAU` settings.

---

## Testing

The test suite uses `pytest`. All tests are in `tests/`.

```shell
conda activate maccai

# Run all tests
pytest tests/

# Verbose output with short tracebacks
pytest tests/ -v --tb=short

# Skip tests that require external APIs or hardware
pytest tests/ -m "not integration"

# Skip slow tests
pytest tests/ -m "not slow"

# Run with coverage report
pytest tests/ --cov=maccai_battery --cov-report=term-missing
```

Test markers are defined in `pyproject.toml`:

| Marker | Description |
|--------|-------------|
| `slow` | Tests that take more than a few seconds |
| `integration` | Tests that require external tools (QE, MP API, etc.) |
| `gpu` | Tests that require a CUDA-capable GPU |

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `mattergen-generate: command not found` | Switch to the `mattergen` environment: `conda activate mattergen`. MatterGen is not available in the `maccai` env. |
| `MatterSim not available — using EMT fallback` | Install MatterSim in the `maccai` env: `cd /path/to/mattersim && pip install -e .` |
| `pw.x not found on PATH` | Install Quantum ESPRESSO: `conda install -c conda-forge qe` |
| `No pseudopotential configured for element 'X'` | Add the element to `pseudopotentials.files` in `config.yaml` and run `python scripts/download_pseudopotentials.py` |
| `pymatgen-io-espresso` `ImportError` | `pip install git+https://github.com/Griffin-Group/pymatgen-io-espresso` |
| `MP_API_KEY environment variable is not set` | `export MP_API_KEY="your_key_here"` (get one free at materialsproject.org/api) |
| `Config file not found` | Create `config.yaml` in the project root, or set `MACCAI_CONFIG=/path/to/config.yaml` |
| QE SCF not converging | Lower `mixing_beta` to `0.2`; reduce `degauss` to `0.01` for insulators; check that pseudopotentials are correct for each element |
| Very slow QE on a GPU machine | Increase `nproc` to the number of available CPU cores: `python scripts/04_dft.py --nproc 16` |
| Step 2 runs but produces unrealistic structures | MatterSim EMT fallback is active — install the real MatterSim model weights |
| `torch-geometric` import error in mattergen env | Run `pip install torch-geometric` inside the `mattergen` env with the correct torch version pinned |
| Pipeline crashes mid-run | Use `python main.py --from-step N` to resume from the failed step after fixing the issue |
| Duplicate structures in database | Check `deduplication.enabled: true` in `config.yaml`; tighten `volume_tolerance_A3` if needed |
| Hull distances look systematically too high | Verify `hull.energy_source` is set to `"relax"` (not `"scf"`) and that DFT energies have been merged (Step 5) |

---

## Citation

If you use this pipeline in your research, please cite the underlying tools:

**MatterGen** — AI-driven crystal structure generation:
> Zeni, C. et al. *MatterGen: a generative model for inorganic materials design.*
> arXiv:2312.03687 (2023). Microsoft Research.

**MatterSim** — Machine learning interatomic potential:
> Han, J. et al. *MatterSim: A Deep Learning Atomistic Model Across Elements, Temperatures and Pressures.*
> arXiv:2405.04967 (2024). Microsoft Research.

**Quantum ESPRESSO** — First-principles DFT:
> Giannozzi, P. et al. *QUANTUM ESPRESSO: a modular and open-source software project for quantum
> simulations of materials.* J. Phys.: Condens. Matter **21**, 395502 (2009).
>
> Giannozzi, P. et al. *Advanced capabilities for materials modelling with Quantum ESPRESSO.*
> J. Phys.: Condens. Matter **29**, 465901 (2017).

**pymatgen** — Crystal structure analysis and phase diagrams:
> Ong, S. P. et al. *Python Materials Genomics (pymatgen): A robust, open-source python library for
> materials analysis.* Computational Materials Science **68**, 314–319 (2013).

**Materials Project** — Reference thermodynamic data and convex hull:
> Jain, A. et al. *Commentary: The Materials Project: A materials genome approach to accelerating
> materials innovation.* APL Materials **1**, 011002 (2013).

---

*MACCAI Battery Materials Discovery Pipeline — v0.1.0*