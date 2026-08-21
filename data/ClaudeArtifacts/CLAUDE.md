# ClaudeArtifacts conventions

Guidance for building benchmark artifacts (published HTML pages) from the data in this
directory's topic folders (`OVFE_Benchmark/`, `O_OH_Trendline/`, `CO_H_Ads/`, etc.). Read this
before starting a new one or updating an existing one — it's what keeps them looking and
behaving like one family instead of fifteen one-off reports.

## Folder layout

Each topic folder holds:
- `models/` (or a flat `<model>/` layout directly) — real relaxation output: `bulk/slab/vacancy(/adsorbate)`
  per model, each stage with `relax.json` (energy, elapsed_s, opt_log) and `CONTCAR`/`OUTCAR`/`trajectory.xyz`.
  This is ground truth — always compute metrics from these, never from a script's stdout recap.
- `*.pdf` — the literature reference(s) the benchmark compares against.
- `analysis/` — matplotlib PNGs from the existing `scripts/ClaudeScripts/*_report.py` / `*_plot.py`
  scripts. The HTML artifact supersedes these for presentation but the PNGs stay as a
  non-interactive fallback; don't delete them when building an artifact.
- `job.json` / `results.jsonl` — present for the newer results.jsonl-shaped scripts (mo2/co_h/h2o/o
  benchmarks); not present for the older run_stage.py-driven trees (OVFE), which use rankings.csv
  + relax.json instead. Know which shape a given folder is before writing a reader for it.

Shared caches, not per-topic: `Gas_Refs/` (gas-phase reference energies, one JSON per
model×fragment×fmax — see below) and the laptop's own `runs/_gas_refs/` (a *different*,
usually smaller, local-only cache — check both, prefer the artifact folder's copy, merge if
one has coverage the other lacks).

Don't assume a topic folder's structure is final or single — check for duplicates/stale nested
run trees (`--outdir` pointed at a subfolder instead of the topic root is the recurring failure
mode here) before reading from it.

## Literature-settings-vs-ours table

**Every benchmark artifact opens with this table**, above the fold, before any chart. It's the
single highest-value thing on the page: it's what lets a reader judge whether a MAE/RMSE number
means "the model is wrong" or "we compared it under different conditions." Pull the literature
side from the actual paper (PDF in the topic folder — extract text with `pypdf` if page
rendering isn't available; `pdftotext`/poppler is not guaranteed installed), not from memory or
from a script's docstring paraphrase. Rows worth including whenever the paper states them:
functional/method, pseudopotentials or basis, cutoff, k-points, spin treatment, force/energy
convergence, slab thickness + frozen fraction, vacuum, supercell + resulting defect/adsorbate
concentration, and how the paper computes its own gas-phase reference (many papers, including
Kowalski et al. 2009 for OVFE, do **not** use a raw DFT total energy for O2 — check for this
explicitly, see below). "Ours" side pulls from `oxide_workflow/config.py` defaults (`SlabConfig`,
`RelaxConfig`) plus whatever the specific benchmark script overrode (`SEED_STANDOFF`, `--fmax`,
etc. — check the script, defaults drift).

## Gas-phase reference convention (mu_O, and generally)

`oxide_workflow/energetics.py` has two O2 reference functions — check which one a given result
actually used before presenting a number, since the codebase's default has changed over time and
individual scripts can override it:

- `oxygen_chemical_potential` — **raw**, mu_O = E(O2)/2 from the model's own direct O2
  relaxation. This is what `pipeline.py` and `run_stage.py` call by default as of this writing.
- `oxygen_chemical_potential_corrected` — **corrected**, routes through the model's own H2 and
  H2O total energies plus the experimental water formation enthalpy
  (`WATER_FORMATION_ENTHALPY_EXP = 2.51` eV), sidestepping GGA-class O2 overbinding. This is now
  the *preferred* number to lead with for OVFE — it's also literally what Kowalski et al. (2009)
  themselves do (their own Sec. II explicitly avoids a raw DFT O2 total energy for the same
  overbinding reason), so it's the more apples-to-apples comparison, not just a nicer-looking one.

When a model has both, show both (a "does the correction actually help" delta chart is more
informative than either number alone — see `scripts/ClaudeScripts/ovfe_o2_correction.py` for the
existing matplotlib version of this chart). When a model only has the corrected pathway's gas
refs cached (H2/H2O but no O2), that's expected, not a gap to fill by re-running O2 — say so in
the UI (partial coverage badge / "n/a, not run" — never silently drop the model from a raw-vs-
corrected chart without marking it).

To recompute E_vac/E_ads independently of a script's cached `e_vac_eV`/`e_ads_eV` column (worth
doing as a sanity check — confirms which mu_O convention actually produced that number):
`E_vac = e_defective - e_pristine + mu_O`, all terms same-model, same-fmax gas ref.

## Chart choices for an N-model benchmark

With ~15 models, this repo's convention is:
- **Ranked horizontal bar** for MAE/RMSE per model (horizontal avoids label collision that a
  15-tick x-axis gets on a vertical bar — see dataviz skill's anti-patterns). Sort by the primary
  metric, not alphabetically or by model family, so "best model" is legible at a glance.
  Literature itself never gets a bar in an error chart (error vs. itself is 0 and meaningless) —
  it's the dashed reference line in the raw-value comparison chart instead.
- **Scatter (predicted vs. literature)** once there are enough models that a grouped bar per site
  gets cluttered — one point per (model, site), a y=x reference line, color by model or by site
  depending on which grouping is the point of that specific chart.
- **Walltime** as its own ranked horizontal bar, separate from accuracy — never force a dual-axis
  combining accuracy and speed on one chart (see dataviz skill's non-negotiables: one axis).
- Site/category colors follow the existing matplotlib scripts' own choices where one already
  exists (e.g. OVFE's O2c = blue slot 1, O3c = orange slot 2, from
  `ovfe_tio2_benchmark_plot.py`) — don't invent a new mapping for the same entity in the HTML
  version.
- Palette: this repo's matplotlib scripts already hand-pick the dataviz skill's validated
  categorical palette (`#2a78d6`, `#eb6834`, `#1baf7a`, ...) — reuse those exact hexes in HTML
  artifacts for continuity between the PNG and the interactive version, rather than re-deriving.

## Known duplicate-folder trap

Multiple sessions/backfills over time have produced sibling folders for the same benchmark
(`OVFE-TiO2` vs `OVFE_TiO2Benchmark`, consolidated 2026-08-20 into `OVFE_Benchmark/`) and nested
"full pipeline" trees inside what should be a flat per-model layout (one model run through
`oxide_workflow.pipeline` directly instead of the lighter benchmark script — look for a
`full_pipeline/` subdirectory as the tell). Before trusting a topic folder's apparent flatness,
`find <folder> -maxdepth 1 -type d` and sanity-check every entry is actually a model name, not
another run's wrapper directory.
