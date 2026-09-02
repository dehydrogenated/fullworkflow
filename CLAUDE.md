# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install (editable, dev deps)
pip install -e ".[dev]"

# Run all tests
pytest

# Run a single test file
pytest tests/test_pipeline.py

# Run a single test by name
pytest tests/test_pipeline.py::test_run_smoke -s

# Fetch a structure from Materials Project (requires mp-api)
python scripts/core/fetch_structure.py mp-2657

# Fetch all rutile structures at once
python scripts/core/fetch_rutiles.py

# Check which slab terminations exist for a material
python scripts/core/validate_materials.py

# Run a single pipeline stage (fastest iteration loop)
python scripts/core/run_stage.py bulk
python scripts/core/run_stage.py slab --thickness 20 --freeze 0.8
python scripts/core/run_stage.py vacancy --from runs/practice/slab/CONTCAR
python scripts/core/run_stage.py adsorbate --from runs/xyz/vacancy/site4_8d/CONTCAR --adsorbate O2

# Full benchmark: one candidate vs. reference
python -m oxide_workflow.pipeline --material mp-2657 --protocol full_pipeline

# Sweep all rutile family members
python scripts/core/run_family.py --dry-run          # plan + cost estimate
python scripts/core/run_family.py --adsorbate O2 --cap 6

# Report results from a completed run
python scripts/core/report.py runs/latest
```

### Script layout

- `scripts/core/` — the tools above: fetching structures, running stages, reporting. Stable, documented entry points.
- `scripts/studies/<theme>/` — one investigation per folder, holding **both** the `.py` and the
  `.slurm` that runs it, so a study is readable and re-runnable from one directory. Themes are
  `ovfe/` (vacancy-formation-energy convergence sweeps), `mo2/` (the rutile MO2 family sweep and
  its regression checks), and `molecular/` (per-adsorbate literature comparisons). Cross-cutting
  scripts belonging to no single study sit loose at `scripts/studies/`. Not part of the documented
  workflow; may assume a particular run already exists.
- `scripts/slurm/` — only the jobs that aren't tied to one study (they run `oxide_workflow.pipeline`
  or `scripts/core/run_stage.py`), plus `sync_sockeye_runs.sh`, which pulls results back to this
  laptop and is the one script here that isn't itself a submitted job.

A study's `.slurm` living beside its `.py` is deliberate and safe: SLURM only cares about the
**working directory at submission time** (must be `/scratch`), never where the job file lives.

Scripts resolve the repo root by walking up to the `pyproject.toml` marker, never by counting
parents — a hardcoded `parents[N]` silently resolves to the wrong directory the next time a file
moves, which has already happened twice in this tree.

## Architecture

The workflow benchmarks MLIP models against a reference model on oxide surfaces. The chain is always: **bulk → slab → vacancy → adsorbate**. Each stage produces an unrelaxed structure; a backend relaxes it and writes the result to disk.

### Core data flow

`pipeline.py` orchestrates everything:
1. The **reference** model runs the full chain once to produce ground-truth relaxed geometries.
2. Each **candidate** model runs against that shared reference under one or both protocols:
   - `full_pipeline`: each stage built from the candidate's own relaxed previous stage (realistic accumulated error).
   - `seeded`: each stage built from the reference's relaxed previous stage (isolates per-stage error).
3. Divergence between reference and candidate is recorded per-stage into `divergence.jsonl`.

### Key modules

- **`config.py`** — all chemistry/relaxation knobs as frozen dataclasses (`RunConfig`, `SlabConfig`, `RelaxConfig`, `AdsorbateConfig`). Edit defaults here; CLI flags override per-run. `ADSORBATE_FRAGMENTS` defines named molecule geometries.
- **`backends.py`** — the `Backend` dataclass and `relax()` function. Models run in isolated conda envs; the orchestrator writes a POSCAR + job spec, launches `worker_relax.py` in the model's env via subprocess, and reads the result back from disk. Model checkpoints are at `~/Desktop/mace_test/models/`. `REGISTRY` maps model names to backends; `ALL_CANDIDATES` is a convenience tuple of non-reference models.
- **`stages.py`** — structure building only (no relaxation). `make_slab()` cuts the slab; `oxygen_vacancy_candidates()` and `adsorbate_candidates()` return lists of unrelaxed `Candidate` objects with a `site_id` dict for identification.
- **`structures.py`** — resolves a material identifier (mp-id or path) to a `Structure`. mp-ids are read from `data/structures/` as CIF+JSON pairs saved by `scripts/core/fetch_structure.py`; a path is read and normalized directly, no registry involved.
- **`energetics.py`** — `adsorption_energy()`, `vacancy_formation_energy()`, `gas_reference_energy()`, `oxygen_chemical_potential()`. All energetics use same-calculator terms to cancel per-atom offsets.
- **`diverge.py`** — computes displacement statistics (mean/rmsd/max) between two relaxed structures via `StructureMatcher`, plus `energy_error`.
- **`records.py`** — writes the output file tree: POSCAR/CONTCAR/trajectory.xyz/OUTCAR per relaxation leaf, `rankings.csv` per funnel, `header.json` rollups, `divergence.jsonl`, `candidates.jsonl`, `summary.json`.
- **`checks.py`** — `placement_quality_flags()` post-relaxation sanity checks (adsorbate frozen, start distance too close, etc.).

### Output tree structure

```
runs/<run>/
  summary.json                  # top-level results and timing
  divergence.jsonl              # per-stage divergence rows (one per candidate/stage/protocol)
  candidates.jsonl              # per-site energies for ranking fidelity
  <model>/
    header.json                 # model-level timing rollup
    bulk/POSCAR, CONTCAR, trajectory.xyz, OUTCAR
    <protocol>/
      header.json
      slab/...
      vacancy/
        rankings.csv
        site<N>_<class>/POSCAR, CONTCAR, trajectory.xyz, OUTCAR
      adsorbate/
        rankings.csv
        site<N>_<class>/...
```

### Default configuration

- Reference model: `MACE-mh1-omat` (OMat24 PBE head of mace-mh-1)
- Default candidate: `MACE-mh1-oc20`
- Material: `mp-2657` (rutile TiO2, MP's relaxed cell, committed to `data/structures/`)
- Facet: (110), termination index 1 (stoichiometric, confirmed by `validate_materials.py`)
- Supercell: 4×2 = 192 atoms; shrink to 1×1 for smoke tests only
- fmax: 0.02 eV/Å; loosen to ~0.1 for quick checks (biggest speed lever)

### `divergence.jsonl` interpretation

`energy_error` is ~86 eV for many rows — this is a per-atom head offset between models, not a crash. `matched=false` means the StructureMatcher couldn't align the two structures (ranking divergence), not a failed relaxation.

### Adding a new model

Add a `Backend` entry to `REGISTRY` in `backends.py`. The model checkpoint must be locally accessible (Sockeye compute nodes have no outbound network). Use `_mace()` or `_uma()` helpers for the existing checkpoint files, or construct `Backend(...)` directly for a new loader.

### Adsorbate placement

Adsorbates are placed at approximately the covalent bond length above the surface (set by `seed_standoff=0` in `AdsorbateConfig`, meaning exactly at the covalent sum). When modifying placement distance, always verify the float-off distance remains physically reasonable and check `placement_quality_flags` output.

## Running on Sockeye (SLURM)

Key paths:
- `PROJECT=/arc/project/st-akkiraju-1/ssong18` — durable storage; this repo lives at `$PROJECT/fullworkflow`
- `/scratch/st-akkiraju-1/$USER` — fast, purged-on-a-timer scratch; **all job output must land here**, never `/arc/project`
- `$PROJECT/miniforge3` — conda base (`OXW_CONDA_BASE`). Verified via `conda env list` on
  the login node (2026-08-19): `oxw` (orchestrator, has `oxide_workflow`+pymatgen installed
  editable), `mace-clean`, `fairchem`, `sevenn`. **Not yet created: `orb`, `chgnet`, `esen`**
  — each model's env is created fresh with `conda create -n <name> python=3.11` then
  `pip install <package> ase` from the login node (needs network; compute nodes have none).
  Don't assume an env exists from this list without rechecking — it was wrong before
  (assumed `sevenn` was missing; it wasn't) and will drift again as more models get added.
- `$PROJECT/models` — checkpoints (`OXW_MODEL_DIR`): `mace-mh-1.model`, `uma-s-1p2.pt`, plus
  whatever's been hand-uploaded since (orb-v2-20241011.ckpt, esen_30m_oam.pt as of
  2026-08-19 per Sean) — verify presence on the login node before relying on any of these,
  same reasoning as the env list above.

Watch out: the home-ish project dir and the scratch dir both end in `ssong18` as their last path component — easy to be in the wrong one without noticing. Run `pwd` before trusting a relative `cd`; prefer absolute paths.

### Submitting a job

**Must submit from `/scratch`, never from `/arc/project`** — SLURM rejects it outright ("Submitting jobs from directories residing in /arc/project is not allowed"). The job *script* itself can still live in the repo under `/arc/project`; only the *working directory at submission time* matters:

```bash
cd /scratch/st-akkiraju-1/$USER
sbatch /arc/project/st-akkiraju-1/ssong18/fullworkflow/scripts/slurm/<job>.slurm
```

CPU is the `#SBATCH` default in every job script (`--account=st-akkiraju-1`, `--partition=cascade`). GPU needs a **different account** (the `-gpu` suffix) and explicit flags at submission time — command-line flags beat a script's `#SBATCH` defaults, so no editing is needed to switch:

```bash
sbatch --account=st-akkiraju-1-gpu --partition=gpu --gres=gpu:1 --time=6:00:00 \
       /arc/project/st-akkiraju-1/ssong18/fullworkflow/scripts/slurm/<job>.slurm
```

### Monitoring / canceling

```bash
squeue -u $USER                                          # running, pending, or gone?
tail -f /scratch/st-akkiraju-1/$USER/slurm-<jobid>.out    # live log (or slurm-<jobid>.err)
scancel <jobid>                                           # cancel one job
scancel -u $USER                                          # cancel everything you have running
```

### Why the job scripts look the way they do

Every job script (`sockeye_job.sh`, `sockeye_oc22.sh`, `sockeye_co2_retest.slurm`) carries the same environment contract, since compute nodes are far more restricted than the login node:

- **No outbound network** — `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1` stop anything that would try to reach out at runtime from hanging until the walltime runs out instead of failing fast. Checkpoints and pip installs have to be pre-staged from a login node beforehand.
- **`$HOME` and `/arc/project` are read-only on compute nodes** — only `/scratch` is writable. Any library that caches under `$HOME` by default (triton, torch inductor, huggingface, matplotlib via pymatgen, fairchem separately since it ignores `XDG_CACHE_HOME`, `cached_path` — pulled in by `orb-models`, ignores `XDG_CACHE_HOME` too and needs its own `CACHED_PATH_CACHE_ROOT`, confirmed via an actual `OSError: Read-only file system` on Sockeye, not assumed) has to be redirected there or it crashes mid-run — see any job script's cache-redirection block for the exact env vars. `cached_path` bit even when Orb-v2's `weights_path` pointed at an already-staged *local* checkpoint file — it unconditionally tries to create its cache directory before ever checking whether the path is local or remote, so passing a local file doesn't skip this.
- **Device detection from the scheduler, not a hardcoded flag** — `OXW_DEVICE` is set by checking whether `CUDA_VISIBLE_DEVICES` is actually populated, so a forgotten `--gres=gpu` fails loudly (wrong device) instead of silently paying for a GPU node and running on its CPUs.
- **Results land in scratch, not `/arc/project`** — scratch is purged on a timer, so every job script prints an `rsync` command at the end to copy results into `$PROJECT/runs/` for durable storage. That copy can only be run from the **login node**; `/arc/project` is read-only from inside a compute-node job.

### Common mistakes (all hit in practice)

- `.py` files aren't directly executable here — prefix with `python3`.
- `<placeholder>`-style text in an example command is not literal — substitute the real value (job ID, path, etc.) before running it.
- A results file opened in append mode (most of this repo's analysis scripts) will silently blend a new run's rows into an old run's leftover file at the same path — move or rename the old one first if the settings changed between runs.
