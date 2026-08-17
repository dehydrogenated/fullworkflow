"""Fast standoff sweep: TiO2, UMA-oc22, single ontop site (undercoordinated metal),
vertical orientation only. One relaxation per standoff. Ti-O covalent sum ~2.26 A, so
SEED_STANDOFFS below give total placement distances of ~2.76/3.26/3.76 A.

    python scripts/co2_standoff_sweep/run.py runs/co2_standoff_sweep
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import replace
from pathlib import Path

from oxide_workflow import pipeline
from oxide_workflow.backends import get_backend, relax
from oxide_workflow.config import ADSORBATE_FRAGMENTS, RunConfig
from oxide_workflow.energetics import adsorption_energy, gas_reference_energy
from oxide_workflow.pipeline import _adsorbate_anchor_distance
from oxide_workflow.stages import adsorbate_candidates, make_slab
from oxide_workflow.structures import get_structure

MODEL = "UMA-oc22"
MP_ID = "mp-2657"
SEED_STANDOFFS = [0.5, 1.0, 1.5]
DESORB_TOL = 2.0


def undercoordinated_metal_site(candidates):
    metal_ontop = [
        c for c in candidates
        if c.site_id["symmetry_class"] == "ontop"
        and c.site_id["site_label"] and not c.site_id["site_label"].startswith("O")
    ]
    if not metal_ontop:
        raise RuntimeError("no metal ontop site found")

    def coord(c):
        m = re.match(r"^[A-Za-z]+(\d+)c$", c.site_id["site_label"])
        return int(m.group(1)) if m else 99

    return min(metal_ontop, key=coord)


def main(outdir: Path) -> None:
    backend = get_backend(MODEL)
    cfg = RunConfig()
    co2_species, co2_coords = ADSORBATE_FRAGMENTS["CO2"]
    n_ads = len(co2_species)

    start = get_structure(MP_ID)
    bulk = pipeline._relax_record(
        start, backend, stage="bulk", protocol="reference", geometry_source="db",
        cfg=cfg, outdir=outdir, relax_cell=True, canonical=True,
    ).structure
    slab_in = make_slab(bulk, cfg.slab)
    slab_out = pipeline._relax_record(
        slab_in, backend, stage="slab", protocol="reference",
        geometry_source="cut_from_relaxed_bulk", cfg=cfg, outdir=outdir, canonical=True,
    )
    pristine_energy = slab_out.energy
    pristine_structure = slab_out.structure

    e_gas = gas_reference_energy(
        backend, cfg, pipeline.relax, species=co2_species, coords=co2_coords,
    )

    results = []
    for standoff in SEED_STANDOFFS:
        ads_cfg = replace(
            cfg.adsorbate, species=co2_species, coords=co2_coords,
            positions=("ontop",), max_per_position=None, seed_standoff=standoff,
        )
        candidates = adsorbate_candidates(
            pristine_structure, ads_cfg, freeze_bottom_fraction=cfg.slab.freeze_bottom_fraction,
        )
        cand = undercoordinated_metal_site(candidates)
        anchor_idx = len(cand.structure) - n_ads

        site_dir = outdir / MODEL / "adsorbate" / f"standoff_{standoff}"
        res = relax(
            cand.structure, backend, workdir=site_dir,
            fmax=cfg.relax.fmax, max_steps=cfg.relax.max_steps, optimizer=cfg.relax.optimizer,
        )
        e_ads = adsorption_energy(res.energy, pristine_energy, e_gas)
        angle = res.structure.get_angle(anchor_idx, anchor_idx + 1, anchor_idx + n_ads - 1)
        end_dist, bond_len = _adsorbate_anchor_distance(res.structure, n_ads)
        desorbed = (
            end_dist is not None and bond_len is not None and end_dist >= DESORB_TOL * bond_len
        )

        row = {
            "standoff": standoff, "site": cand.site_id["site_label"],
            "e_ads_eV": e_ads, "oco_angle_deg": angle, "desorbed": desorbed,
            "end_anchor_distance_A": end_dist, "converged": res.converged, "nsteps": res.nsteps,
        }
        results.append(row)
        print(f"standoff={standoff:<4} E_ads={e_ads:+.4f} eV  angle={angle:6.1f}  "
              f"desorbed={desorbed}  nsteps={res.nsteps}", flush=True)

    (outdir / "results.jsonl").write_text("\n".join(json.dumps(r) for r in results) + "\n")
    print(f"\nwrote {outdir / 'results.jsonl'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("outdir", type=Path)
    a = ap.parse_args()
    a.outdir.mkdir(parents=True, exist_ok=True)
    main(a.outdir)
