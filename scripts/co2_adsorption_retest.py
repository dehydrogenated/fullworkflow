"""CO2 adsorption retest: does bending appear once placement stops biasing it away?

The first pass (co2_adsorption_benchmark.py) placed CO2 perfectly vertically (a symmetric
starting orientation that can sit exactly on a linear ridge and never discover a bent
minimum) and with a standoff close enough to the surface that the winning site every
single time was flagged "placed on binding site... no room to relax into the well" by
checks.py -- i.e. most reported energies were near an initial guess, not a genuinely
explored minimum. Two of the three ontop sites per oxide also desorbed outright.

This retest fixes both:

  1. Larger seed_standoff (SEED_STANDOFF below) so there is real room to relax into
     whatever well exists, instead of starting on top of it.
  2. For every candidate site, try the vertical baseline PLUS N_AZIMUTH evenly-spaced
     azimuthal tilts at a fixed polar angle (POLAR_TILT_DEG) -- a rigid rotation of the
     whole linear CO2 unit about its anchor atom, so bond lengths/linearity are preserved
     at t=0 and only the approach direction changes. Evenly spacing N points on a circle is
     closed-form: phi_i = i * (360/N) degrees -- no golden-angle/spiral needed, that solves
     the harder problem of spreading points over a *sphere's surface*, not a single ring.

Also turns on the new early-stop desorption check (desorb_early_stop_step in
AdsorbateConfig): a trajectory that has net-drifted away from the surface by step 100 is
flagged and abandoned instead of burning the rest of a 300-step budget on a site that was
never going to bind.

Bulk and slab relaxations are resumed for free from an existing run at the same --outdir
(pipeline._relax_record's own resume-from-disk logic) -- only the adsorbate stage repeats,
since standoff/orientation only affect adsorbate placement. Point --outdir at a copy of the
original run's directory to skip re-relaxing bulk/slab on Sockeye.

    python scripts/co2_adsorption_retest.py runs/co2_ads_benchmark

Writes runs/co2_ads_benchmark/results_retest.jsonl: EVERY (model, oxide, site, orientation)
attempt, not just winners, so the full picture is inspectable afterward.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np

from oxide_workflow import pipeline
from oxide_workflow.backends import get_backend, relax
from oxide_workflow.config import ADSORBATE_FRAGMENTS, RunConfig
from oxide_workflow.energetics import adsorption_energy, gas_reference_energy
from oxide_workflow.stages import adsorbate_candidates, make_slab
from oxide_workflow.structures import get_structure

MODELS = ["UMA-omat", "MACE-mh1-omat"]

KCAL_TO_EV = 0.0433641
OXIDES = {
    "TiO2": {"mp_id": "mp-2657", "lit_e_ads_kcal": -30.450, "lit_angle": 129.2},
    "GeO2": {"mp_id": "mp-470", "lit_e_ads_kcal": -19.540, "lit_angle": 142.9},
    "SnO2": {"mp_id": "mp-856", "lit_e_ads_kcal": -17.113, "lit_angle": 136.3},
    "IrO2": {"mp_id": "mp-2723", "lit_e_ads_kcal": -4.671, "lit_angle": 178.6},
    "PbO2": {"mp_id": "mp-20725", "lit_e_ads_kcal": -9.663, "lit_angle": 173.9},
}
for _o in OXIDES.values():
    _o["lit_e_ads_eV"] = _o["lit_e_ads_kcal"] * KCAL_TO_EV

POLAR_TILT_DEG = 25.0
N_AZIMUTH = 5
DESORB_CHECK_STEP = 100
SEED_STANDOFF = 1.2  # Å beyond covalent-radii sum; default AdsorbateConfig is 0.2


def azimuthal_orientations(structure, anchor_idx: int, n_ads: int):
    """[(label, Structure), ...]: vertical baseline + N_AZIMUTH evenly-spaced tilts.

    Rigid rotation of the fragment about its anchor atom (index ``anchor_idx``, unmoved):
    every other fragment atom is placed at its own original anchor-distance along the same
    (theta, phi) direction, which keeps the linear CO2 unit's bond lengths and colinearity
    intact at the new starting orientation.
    """
    anchor = structure[anchor_idx].coords.copy()
    radii = [
        float(np.linalg.norm(structure[anchor_idx + k].coords - anchor))
        for k in range(1, n_ads)
    ]
    variants = [("vertical", structure.copy())]
    theta = math.radians(POLAR_TILT_DEG)
    for i in range(N_AZIMUTH):
        phi = math.radians(i * 360.0 / N_AZIMUTH)
        direction = np.array([
            math.sin(theta) * math.cos(phi),
            math.sin(theta) * math.sin(phi),
            math.cos(theta),
        ])
        tilted = structure.copy()
        for k, r in zip(range(1, n_ads), radii):
            tilted[anchor_idx + k].coords = anchor + r * direction
        variants.append((f"tilt{i}_az{i * 360 // N_AZIMUTH}", tilted))
    return variants


def run_one(model: str, oxide: str, outdir: Path, cfg: RunConfig) -> list[dict]:
    backend = get_backend(model)
    mp_id = OXIDES[oxide]["mp_id"]
    odir = outdir / oxide

    def relax_record(structure, stage, source_desc, relax_cell=False):
        out = pipeline._relax_record(
            structure, backend, stage=stage, protocol="reference",
            geometry_source=source_desc, cfg=cfg, outdir=odir,
            relax_cell=relax_cell, canonical=True,
        )
        print(f"    {stage:14s} E={out.energy:.4f} eV  {out.header['nsteps']} steps  "
              f"{out.elapsed_s:.0f}s", flush=True)
        return out

    print(f"  [{model} / {oxide}]")
    start = get_structure(mp_id)
    bulk = relax_record(start, "bulk", "db", relax_cell=True).structure

    slab_in = make_slab(bulk, cfg.slab)
    slab_out = relax_record(slab_in, "slab", "cut_from_relaxed_bulk")
    pristine_energy = slab_out.energy
    pristine_structure = slab_out.structure

    co2_species, co2_coords = ADSORBATE_FRAGMENTS["CO2"]
    n_ads = len(co2_species)
    e_gas = gas_reference_energy(
        backend, cfg, pipeline.relax, species=co2_species, coords=co2_coords,
    )

    candidates = adsorbate_candidates(
        pristine_structure, cfg.adsorbate, freeze_bottom_fraction=cfg.slab.freeze_bottom_fraction,
    )
    print(f"    {len(candidates)} ontop site(s) x {1 + N_AZIMUTH} orientation(s) "
          f"= {len(candidates) * (1 + N_AZIMUTH)} relaxations", flush=True)

    rows = []
    for cand in candidates:
        anchor_idx = len(cand.structure) - n_ads
        for label, oriented in azimuthal_orientations(cand.structure, anchor_idx, n_ads):
            res = relax(
                oriented, backend,
                fmax=cfg.relax.fmax, max_steps=cfg.relax.max_steps, optimizer=cfg.relax.optimizer,
                desorb_check_n_ads=n_ads, desorb_check_step=DESORB_CHECK_STEP,
            )
            e_ads = adsorption_energy(res.energy, pristine_energy, e_gas)
            angle = res.structure.get_angle(anchor_idx, anchor_idx + 1, anchor_idx + n_ads - 1)
            desorbing = bool(res.meta.get("early_stopped_desorbing"))
            row = {
                "model": model, "oxide": oxide, "site_index": cand.site_id["site_index"],
                "symmetry_class": cand.site_id["symmetry_class"], "orientation": label,
                "e_ads_eV": e_ads, "oco_angle_deg": angle, "desorbing": desorbing,
                "converged": res.converged, "nsteps": res.nsteps,
            }
            rows.append(row)
            print(f"    site{row['site_index']} {label:14s} E_ads={e_ads:+.4f} eV  "
                  f"angle={angle:6.1f}  {'DESORBING' if desorbing else 'OK':10s}"
                  f"nsteps={res.nsteps}", flush=True)
    return rows


def main(outdir: Path, fmax: float) -> None:
    co2_species, co2_coords = ADSORBATE_FRAGMENTS["CO2"]
    cfg = RunConfig()
    cfg = replace(cfg, adsorbate=replace(
        cfg.adsorbate, species=co2_species, coords=co2_coords,
        positions=("ontop",), max_per_position=3,
        seed_standoff=SEED_STANDOFF, desorb_early_stop_step=DESORB_CHECK_STEP,
    ))
    cfg = replace(cfg, relax=replace(cfg.relax, fmax=fmax))
    outdir.mkdir(parents=True, exist_ok=True)
    results_path = outdir / "results_retest.jsonl"
    for model in MODELS:
        for oxide in OXIDES:
            rows = run_one(model, oxide, outdir, cfg)
            with results_path.open("a") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")
    print(f"\nwrote {results_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("outdir", type=Path)
    ap.add_argument("--fmax", type=float, default=0.05)
    a = ap.parse_args()
    main(a.outdir, a.fmax)
