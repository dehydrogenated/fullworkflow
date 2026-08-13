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
python scripts/fetch_structure.py mp-2657

# Fetch all rutile structures at once
python scripts/fetch_rutiles.py

# Check which slab terminations exist for a material
python scripts/validate_materials.py

# Run a single pipeline stage (fastest iteration loop)
python scripts/run_stage.py bulk
python scripts/run_stage.py slab --thickness 20 --freeze 0.8
python scripts/run_stage.py vacancy --from runs/practice/slab/CONTCAR
python scripts/run_stage.py adsorbate --from runs/xyz/vacancy/site4_8d/CONTCAR --adsorbate O2

# Full benchmark: one candidate vs. reference
python -m oxide_workflow.pipeline --material mp-2657 --protocol full_pipeline

# Sweep all rutile family members
python scripts/run_family.py --dry-run          # plan + cost estimate
python scripts/run_family.py --adsorbate O2 --cap 6

# Report results from a completed run
python scripts/report.py runs/latest
```

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
- **`structures.py`** — resolves a material identifier (mp-id or path) to a `Structure`. mp-ids are read from `data/structures/` as CIF+JSON pairs saved by `scripts/fetch_structure.py`; a path is read and normalized directly, no registry involved.
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
