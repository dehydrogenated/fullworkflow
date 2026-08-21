"""Shared machinery for the four OVFE convergence sweeps (vacuum/supercell/height/freeze).

Each sweep script imports this module, defines its own sweep values, and varies exactly
one SlabConfig knob while holding the rest at the baseline used for the 15-model OVFE
benchmark (facet 110, termination 1, 4x2 supercell, 4 trilayers thick, 20 A vacuum, bottom
50% frozen). One variable at a time -- e.g. the supercell sweep always uses 4 trilayers
regardless of lateral size; the height sweep always uses the 4x2 supercell.

Models: the three top performers from the 15-model OVFE benchmark by corrected MAE
(UMA-omat 0.256 eV, eSEN-30M-OAM 0.188 eV, Orb-v2 0.477 eV) -- cheap enough per-relaxation
to sweep granularly, unlike the OC20/OC22-head models that either run slow or land far
outside the useful E_vac range for this system.

Trilayer thickness: measured directly off a 10-trilayer 1x1 cut of the relaxed mp-2657
bulk (not assumed) -- repeating O-Ti2O2-O spacing of 1.275/1.275/0.708 A = 3.258 A per
trilayer. This is what lets the height and freeze sweeps land exactly on trilayer
boundaries instead of an approximate fraction-of-height guess.

Gas reference: every E_vac here uses oxygen_chemical_potential_corrected (the H2/H2O
thermodynamic cycle + the experimental 2.51 eV water formation enthalpy), matching the
OVFE benchmark artifact's default -- not the raw E(O2)/2 reference. mu_O only depends on
(model, fmax), not on any slab-geometry knob, so it's computed once per model and reused
across every sweep point.
"""

from __future__ import annotations

import json
import time
from dataclasses import replace
from pathlib import Path

from oxide_workflow import pipeline
from oxide_workflow.backends import get_backend, relax
from oxide_workflow.config import RunConfig
from oxide_workflow.energetics import oxygen_chemical_potential_corrected
from oxide_workflow.stages import make_slab, oxygen_vacancy_candidates
from oxide_workflow.structures import get_structure

MODELS = ("UMA-omat", "eSEN-30M-OAM", "Orb-v2")

TRILAYER_A = 3.258  # A per trilayer, measured -- see module docstring
SITES_LIT = {"O2c": 3.02, "O3c": 4.00}  # Kowalski et al. 2009, Table I, triplet

MATERIAL = "mp-2657"
BASE_SLAB = dict(miller_index=(1, 1, 0), termination_index=1, lll_reduce=True, center_slab=True,
                  supercell=(4, 2), min_slab_size=12.0, min_vacuum_size=20.0,
                  freeze_bottom_fraction=0.5, vacancy_block_radius=1.3)
# min_slab_size=12.0 verbatim (not 4*TRILAYER_A): SlabGenerator rounds a requested
# min_slab_size UP to its own nearest achievable cut, and that boundary isn't at clean
# multiples of TRILAYER_A -- confirmed by hand (see ovfe_height_sweep.py's docstring).
#
# min_vacuum_size=20.0 -- SlabConfig's current default, per Sean (2026-08-20): use 20.0 as
# the held-constant baseline here going forward, not the 12.0 that actually generated the
# published OVFE_Benchmark artifact (confirmed via `git log -p oxide_workflow/config.py`:
# the default was 12.0 when that data was made, bumped to 20.0 afterward). Consequence:
# the supercell/height/freeze sweeps' baseline point (4x2 / 4 trilayers / 0.5 freeze) will
# NOT bit-for-bit reproduce the artifact's E_vac numbers -- deliberate, not a bug. The
# vacuum sweep itself is unaffected either way, since min_vacuum_size is what it varies.
FMAX = 0.02


def base_config() -> RunConfig:
    cfg = RunConfig()
    from oxide_workflow.config import SlabConfig
    cfg = replace(cfg, slab=SlabConfig(**BASE_SLAB), polymorph=MATERIAL)
    cfg = replace(cfg, relax=replace(cfg.relax, fmax=FMAX))
    return cfg


def relax_bulk(model: str, cfg: RunConfig, outdir: Path):
    """Relax the bulk once per model -- independent of every slab-geometry sweep knob, so
    every sweep point for this model reuses the same relaxed bulk structure."""
    backend = get_backend(model)
    start = get_structure(cfg.polymorph)
    out = pipeline._relax_record(
        start, backend, stage="bulk", protocol="reference", geometry_source="db",
        cfg=cfg, outdir=outdir, relax_cell=True, canonical=True,
    )
    print(f"  [{model}] bulk  E={out.energy:.4f} eV  {out.header['nsteps']} steps  "
          f"{out.elapsed_s:.0f}s", flush=True)
    return out.structure


def run_point(model: str, bulk_structure, cfg: RunConfig, outdir: Path, point_tag: str,
              sweep_param: str, sweep_value) -> dict:
    """Cut + relax the slab under cfg.slab, relax every surface O-vacancy candidate, and
    return one results.jsonl row: E_vac per site (corrected mu_O), error vs. literature,
    walltime, convergence. point_tag namespaces this sweep value's output directory so
    different sweep points for the same model don't overwrite each other's trajectories."""
    backend = get_backend(model)
    # _relax_record itself nests output under outdir/<backend.name>/<stage> -- don't
    # prepend model here too, or the tree doubles up (outdir/model/point_tag/model/slab/...).
    odir = outdir / point_tag
    t0 = time.time()

    slab_in = make_slab(bulk_structure, cfg.slab)
    n_atoms = len(slab_in)
    slab_out = pipeline._relax_record(
        slab_in, backend, stage="slab", protocol="reference",
        geometry_source="cut_from_relaxed_bulk", cfg=cfg, outdir=odir,
        relax_cell=False, canonical=True,
    )
    pristine_energy = slab_out.energy

    mu_o = oxygen_chemical_potential_corrected(backend, cfg, pipeline.relax)

    try:
        vacs = oxygen_vacancy_candidates(
            slab_out.structure, freeze_bottom_fraction=cfg.slab.freeze_bottom_fraction,
            block_radius=cfg.slab.vacancy_block_radius,
        )
    except RuntimeError as e:
        return {
            "model": model, "sweep_param": sweep_param, "sweep_value": sweep_value,
            "n_atoms": n_atoms, "failed": True, "error": f"no vacancy candidates: {e}"[:300],
            "sites": {}, "walltime_s": time.time() - t0,
        }

    sites = {}
    for cand in vacs:
        label = cand.site_id["site_label"]
        if label not in SITES_LIT:
            continue  # only the two literature-tabulated sites matter for this sweep
        res = relax(
            cand.structure, backend,
            workdir=odir / model / "vacancy" / f"site{cand.site_id['site_index']}_{label}",
            fmax=cfg.relax.fmax, max_steps=cfg.relax.max_steps, optimizer=cfg.relax.optimizer,
        )
        e_vac = res.energy - pristine_energy + mu_o
        sites[label] = {
            "e_vac_eV": e_vac, "lit_eV": SITES_LIT[label], "error_eV": e_vac - SITES_LIT[label],
            "converged": res.converged, "nsteps": res.nsteps,
        }

    elapsed = time.time() - t0
    errs = [s["error_eV"] for s in sites.values()]
    mae = sum(abs(e) for e in errs) / len(errs) if errs else None
    row = {
        "model": model, "sweep_param": sweep_param, "sweep_value": sweep_value,
        "n_atoms": n_atoms, "failed": False, "sites": sites, "mae_eV": mae,
        "walltime_s": elapsed, "slab_config": {
            "supercell": list(cfg.slab.supercell), "min_slab_size": cfg.slab.min_slab_size,
            "min_vacuum_size": cfg.slab.min_vacuum_size,
            "freeze_bottom_fraction": cfg.slab.freeze_bottom_fraction,
        },
    }
    site_str = "  ".join(f"{k}={v['e_vac_eV']:+.3f}(err {v['error_eV']:+.3f})" for k, v in sites.items())
    print(f"  [{model}] {sweep_param}={sweep_value}  {n_atoms} atoms  {site_str}  "
          f"MAE={mae:.3f}  {elapsed:.0f}s" if mae is not None else
          f"  [{model}] {sweep_param}={sweep_value}  {n_atoms} atoms  no sites  {elapsed:.0f}s",
          flush=True)
    return row


def _wait_for_gpu_memory(min_free_mib: int = 2000, timeout_s: float = 30.0, poll_s: float = 2.0) -> None:
    """Guard against a race where the previous model's last subprocess hasn't
    released its CUDA allocation by the time the next model's first subprocess
    launches. Observed directly on Sockeye: UMA-omat finished its full 8-point
    freeze sweep cleanly, then eSEN's very first relaxation OOMed with 31.56 of
    31.73 GiB already "in use" -- eSEN-30M-OAM is a 30M-param model, nowhere
    near enough to explain that on its own, and Orb-v2 ran clean immediately
    after eSEN's block ended, so the GPU wasn't externally starved for the
    whole job -- just for that one window right after UMA-omat. subprocess.run()
    returning doesn't guarantee the CUDA driver has actually reclaimed the
    memory yet. No-op if nvidia-smi isn't available (e.g. a CPU-only run)."""
    import shutil
    import subprocess as sp
    if shutil.which("nvidia-smi") is None:
        return
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            out = sp.run(
                ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10,
            )
            free_mib = int(out.stdout.strip().splitlines()[0])
        except Exception:
            return  # can't query -- don't block the sweep over a diagnostics failure
        if free_mib >= min_free_mib:
            return
        print(f"    (GPU has only {free_mib} MiB free -- waiting for the previous "
              f"model's subprocess to release memory)", flush=True)
        time.sleep(poll_s)


def run_sweep(sweep_param: str, values: list, apply_value, outdir: Path) -> None:
    """Drives one full sweep: relax bulk once per model, then one slab+vacancy funnel per
    (model, value). apply_value(cfg, value) -> cfg with exactly that one SlabConfig field
    changed; every other field stays at BASE_SLAB."""
    outdir.mkdir(parents=True, exist_ok=True)
    results_path = outdir / "results.jsonl"
    job_path = outdir / "job.json"
    job_path.write_text(json.dumps({
        "sweep_param": sweep_param, "values": [v if not isinstance(v, tuple) else list(v) for v in values],
        "models": list(MODELS), "base_slab": {k: (list(v) if isinstance(v, tuple) else v) for k, v in BASE_SLAB.items()},
        "fmax": FMAX, "mu_o_reference": "oxygen_chemical_potential_corrected (H2/H2O cycle + 2.51 eV)",
    }, indent=2))

    for model in MODELS:
        _wait_for_gpu_memory()
        cfg = base_config()
        try:
            bulk = relax_bulk(model, cfg, outdir)
        except Exception as e:
            # A missing/unloadable checkpoint kills this one model, not the whole sweep --
            # every other model still gets its full run. See backends.py's REGISTRY / the
            # local models/ dir for what's actually staged on this machine right now.
            print(f"  [{model}] BULK FAILED, skipping this model entirely: {e}", flush=True)
            for value in values:
                row = {"model": model, "sweep_param": sweep_param, "sweep_value": value,
                       "failed": True, "error": f"bulk: {e}"[:500], "sites": {}}
                with results_path.open("a") as f:
                    f.write(json.dumps(row) + "\n")
            continue
        for value in values:
            # Not just between models -- observed directly on the height sweep: eSEN
            # succeeded at 96/192/288 atoms (wall-clock climbing steeply: 214s, 346s,
            # 535s) then OOMed at 384 atoms. A leftover allocation between successive
            # points of the SAME model is just as real a risk as between models.
            _wait_for_gpu_memory()
            point_cfg = apply_value(cfg, value)
            tag = f"{sweep_param}_{value}".replace(" ", "").replace(",", "x").replace("(", "").replace(")", "")
            try:
                row = run_point(model, bulk, point_cfg, outdir, tag, sweep_param, value)
            except Exception as e:
                print(f"  [{model}] {sweep_param}={value} FAILED: {e}", flush=True)
                row = {"model": model, "sweep_param": sweep_param, "sweep_value": value,
                       "failed": True, "error": str(e)[:500], "sites": {}}
            with results_path.open("a") as f:
                f.write(json.dumps(row) + "\n")

    print(f"\nwrote {results_path}")
