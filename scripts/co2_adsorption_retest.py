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

Sites: the original 3 ontop candidates (undercoordinated metal, O2c, 6-fold metal) had a
10/10 record across the first sweep -- the undercoordinated metal bound every time, the
other two desorbed every time. undercoordinated_metal_sites() below spends the orientation
budget on that one proven site instead: ontop PLUS the nearest bridge and hollow sites
anchored at the same undercoordinated-metal atom (stages.py labels bridge/hollow candidates
by nearest exposed neighbor, so this is a direct site_label match, not a new distance
calculation). 3 sites total, same as before -- broader than one placement, without paying
for a blind bridge/hollow search over the whole surface.

Also turns on two symmetric relaxation-length checks (both in AdsorbateConfig):

  - desorb_early_stop_step: a trajectory that has net-drifted away from the surface by
    step 100 is flagged and abandoned instead of burning the rest of the step budget on a
    site that was never going to bind.
  - extend_if_approaching: the reverse case -- literature starts CO2 5 A from the surface
    and lets a full optimization bring it in, farther than SEED_STANDOFF affords here for a
    fixed step budget, so a trajectory that's still net-closing the gap (not oscillating)
    when max_steps runs out gets one extension of EXTEND_STEPS more rather than being
    called unconverged on a trajectory that hadn't actually finished.

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
import re
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
EXTEND_STEPS = 100  # one extension of this many steps if still net-approaching at max_steps
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


def _coordination_of(label: str) -> int | None:
    m = re.match(r"^[A-Za-z]+(\d+)c$", label or "")
    return int(m.group(1)) if m else None


def undercoordinated_metal_sites(candidates):
    """4 candidates: ontop/bridge/hollow anchored at the single most undercoordinated metal
    atom, PLUS the ontop bridging-oxygen (O2c) site -- stages.py labels bridge/hollow sites
    by their nearest exposed neighbor, so "site_label == the metal's label" is exactly "this
    bridge/hollow sits next to that atom".

    The undercoordinated-metal ontop site won 10/10 of the original sweep's (model, oxide)
    pairs outright; O2c and the 6-fold metal both desorbed in all 10 -- but that was under
    the too-close, symmetric-start placement this retest exists to fix, so O2c (the literal
    "Os" of the paper's secondary C-Os bending interaction) still gets a fair fresh attempt
    here rather than being assumed dead. O3c (in-plane oxygen, never in the paper's
    mechanism and never tested even in the original sweep) is deliberately left out.
    """
    by_type: dict[str, list] = {"ontop": [], "bridge": [], "hollow": []}
    for c in candidates:
        by_type.setdefault(c.site_id["symmetry_class"], []).append(c)

    metal_ontop = [
        c for c in by_type["ontop"]
        if c.site_id["site_label"] and not c.site_id["site_label"].startswith("O")
    ]
    if not metal_ontop:
        raise RuntimeError("no metal ontop site found among candidates")
    target = min(metal_ontop, key=lambda c: _coordination_of(c.site_id["site_label"]) or 99)
    label = target.site_id["site_label"]

    picked = [target]
    for ptype in ("bridge", "hollow"):
        match = next((c for c in by_type[ptype] if c.site_id["site_label"] == label), None)
        if match is not None:
            picked.append(match)
        else:
            print(f"    no {ptype} site anchored at {label} found among candidates -- skipping")

    o2c = next((c for c in by_type["ontop"] if c.site_id["site_label"] == "O2c"), None)
    if o2c is not None:
        picked.append(o2c)
    else:
        print("    no ontop O2c site found among candidates -- skipping")
    return picked, label


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

    all_candidates = adsorbate_candidates(
        pristine_structure, cfg.adsorbate, freeze_bottom_fraction=cfg.slab.freeze_bottom_fraction,
    )
    candidates, metal_label = undercoordinated_metal_sites(all_candidates)
    print(f"    {len(candidates)} site(s) (ontop/bridge/hollow near {metal_label}) x "
          f"{1 + N_AZIMUTH} orientation(s) = {len(candidates) * (1 + N_AZIMUTH)} relaxations",
          flush=True)

    rows = []
    for cand in candidates:
        anchor_idx = len(cand.structure) - n_ads
        for label, oriented in azimuthal_orientations(cand.structure, anchor_idx, n_ads):
            res = relax(
                oriented, backend,
                fmax=cfg.relax.fmax, max_steps=cfg.relax.max_steps, optimizer=cfg.relax.optimizer,
                desorb_check_n_ads=n_ads, desorb_check_step=DESORB_CHECK_STEP,
                extend_if_approaching=True, extend_steps=EXTEND_STEPS,
            )
            e_ads = adsorption_energy(res.energy, pristine_energy, e_gas)
            angle = res.structure.get_angle(anchor_idx, anchor_idx + 1, anchor_idx + n_ads - 1)
            desorbing = bool(res.meta.get("early_stopped_desorbing"))
            extended = bool(res.meta.get("extended"))
            row = {
                "model": model, "oxide": oxide, "site_index": cand.site_id["site_index"],
                "symmetry_class": cand.site_id["symmetry_class"], "orientation": label,
                "e_ads_eV": e_ads, "oco_angle_deg": angle, "desorbing": desorbing,
                "extended": extended, "converged": res.converged, "nsteps": res.nsteps,
            }
            rows.append(row)
            status = "DESORBING" if desorbing else ("EXTENDED" if extended else "OK")
            print(f"    site{row['site_index']} {label:14s} E_ads={e_ads:+.4f} eV  "
                  f"angle={angle:6.1f}  {status:10s}"
                  f"nsteps={res.nsteps}", flush=True)
    return rows


def main(outdir: Path, fmax: float) -> None:
    co2_species, co2_coords = ADSORBATE_FRAGMENTS["CO2"]
    cfg = RunConfig()
    cfg = replace(cfg, adsorbate=replace(
        cfg.adsorbate, species=co2_species, coords=co2_coords,
        # Enumerate all three position types uncapped; undercoordinated_metal_sites() then
        # keeps only the ontop/bridge/hollow trio anchored at the undercoordinated metal,
        # so nothing here is spent relaxing sites with no binding track record.
        positions=("ontop", "bridge", "hollow"), max_per_position=None,
        seed_standoff=SEED_STANDOFF, desorb_early_stop_step=DESORB_CHECK_STEP,
        extend_if_approaching=True, extend_steps=EXTEND_STEPS,
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
