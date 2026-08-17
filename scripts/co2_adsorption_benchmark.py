"""CO2 adsorption energy across the rutile family: chemical-trend transferability test.

Runs bulk -> slab -> CO2 adsorbate (skipping the vacancy stage -- CO2 goes on the
PRISTINE surface, matching the literature setup) for each (model, oxide) pair, restricted
to metal-ontop sites only, since that is the mechanism the reference paper reports (CO2's
oxygen anchors to the surface metal cation M1, then may bend toward a surface oxygen as a
second interaction -- an outcome of relaxation, not a distinct site type to search).

    python scripts/co2_adsorption_benchmark.py runs/co2_ads_benchmark

Reference: Chavez-Rocha et al., Molecules 2023, 28, 1776 (PBEsol-D3/TZP, ADF/BAND),
Table 1 (E_ads, (110) plane) and Table 2 (O-C-O angle, (110) plane).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from oxide_workflow import pipeline
from oxide_workflow.backends import get_backend
from oxide_workflow.config import ADSORBATE_FRAGMENTS, RunConfig
from oxide_workflow.energetics import adsorption_energy, gas_reference_energy
from oxide_workflow.stages import adsorbate_candidates, make_slab
from oxide_workflow.structures import get_structure

MODELS = ["UMA-omat", "MACE-mh1-omat"]

# oxide -> (mp-id, literature E_ads (110) kcal/mol, literature O-C-O angle (110) deg)
# Chavez-Rocha et al. 2023, Table 1 & Table 2.
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


def run_one(model: str, oxide: str, outdir: Path, cfg: RunConfig) -> dict:
    backend = get_backend(model)
    mp_id = OXIDES[oxide]["mp_id"]
    odir = outdir / oxide

    def relax_one(structure, stage, source_desc, relax_cell=False):
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
    bulk = relax_one(start, "bulk", "db", relax_cell=True).structure

    slab_in = make_slab(bulk, cfg.slab)
    print(f"    slab built     {len(slab_in)} atoms")
    slab_out = relax_one(slab_in, "slab", "cut_from_relaxed_bulk")
    pristine_energy = slab_out.energy
    pristine_structure = slab_out.structure

    co2_species, co2_coords = ADSORBATE_FRAGMENTS["CO2"]
    e_gas = gas_reference_energy(
        backend, cfg, pipeline.relax, species=co2_species, coords=co2_coords,
    )

    ads = adsorbate_candidates(
        pristine_structure, cfg.adsorbate, freeze_bottom_fraction=cfg.slab.freeze_bottom_fraction,
    )
    print(f"    {len(ads)} ontop CO2 site(s)")
    cand_table = odir / "candidates.jsonl"
    out = pipeline._run_funnel(
        ads, backend, stage="adsorbate", protocol="reference",
        geometry_source="placed_on_relaxed_substrate", cfg=cfg, outdir=odir,
        candidates_table=cand_table,
        e_ads_reference=(pristine_energy, e_gas),
    )
    best_key = min(out, key=lambda si: out[si].energy)
    best = out[best_key]
    print(f"    best site      site{best_key}  E_ads={best.e_ads:+.4f} eV")

    n_ads = len(co2_species)
    struct = best.structure
    i_o1, i_c, i_o2 = len(struct) - n_ads, len(struct) - n_ads + 1, len(struct) - 1
    angle = struct.get_angle(i_o1, i_c, i_o2)
    print(f"    O-C-O angle    {angle:.1f} deg")

    return {
        "model": model, "oxide": oxide, "e_ads_eV": best.e_ads, "oco_angle_deg": angle,
        "site_index": best_key,
    }


def main(outdir: Path, fmax: float) -> None:
    co2_species, co2_coords = ADSORBATE_FRAGMENTS["CO2"]
    cfg = RunConfig()
    cfg = replace(cfg, adsorbate=replace(
        cfg.adsorbate, species=co2_species, coords=co2_coords,
        positions=("ontop",), max_per_position=3,
    ))
    cfg = replace(cfg, relax=replace(cfg.relax, fmax=fmax))
    outdir.mkdir(parents=True, exist_ok=True)
    results = []
    results_path = outdir / "results.jsonl"
    for model in MODELS:
        for oxide in OXIDES:
            r = run_one(model, oxide, outdir, cfg)
            results.append(r)
            with results_path.open("a") as f:
                f.write(json.dumps(r) + "\n")
    print(f"\nwrote {results_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("outdir", type=Path)
    ap.add_argument("--fmax", type=float, default=0.05)
    a = ap.parse_args()
    main(a.outdir, a.fmax)
