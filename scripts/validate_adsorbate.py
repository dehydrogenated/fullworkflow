"""Targeted validation of the densified + covalent-radii adsorbate sampling.

Reuses the reference model's already-relaxed slab (no re-relaxing bulk/slab/vacancy) and
runs ONLY the adsorbate stage with the two-arm site generator. For each placement it
reports the quality flags and — crucially — where the adsorbate's binding atom actually
landed after relaxation (nearest surface atom + distance), so we can see which surface
site each fragment prefers and whether the previously-missing Ti5c site is sampled.

Parametrized by adsorbate (``H`` default, plus ``O`` and ``CO``): run as
``python scripts/validate_adsorbate.py CO``. Writes each leaf under
runs/adsorbate_validation_<adsorbate>/ as it goes (inspectable mid-run) and a summary
table at the end.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
from pymatgen.core import Structure

from oxide_workflow.backends import get_backend, relax
from oxide_workflow.checks import placement_quality_flags
from oxide_workflow.config import RunConfig
from oxide_workflow.pipeline import _adsorbate_max_disp, _adsorbate_start_distance
from oxide_workflow.stages import adsorbate_candidates

SLAB = Path("runs/mace_sweep/MACE-mh1-omat/slab/CONTCAR")

# Adsorbate fragments to sweep. coords[0] is the *binding* atom (placed at each site); for a
# molecule the rest hang off it in the given geometry.
ADSORBATES = {
    "H": {"species": ("H",), "coords": ((0.0, 0.0, 0.0),)},
    "O": {"species": ("O",), "coords": ((0.0, 0.0, 0.0),)},
    # CO binds carbon-down on most surfaces; O rides 1.13 Å above C (the CO bond length).
    "CO": {"species": ("C", "O"), "coords": ((0.0, 0.0, 0.0), (0.0, 0.0, 1.13))},
}


def _nearest_surface_atom(final: Structure, n_ads: int):
    """Element + distance (Å) of the slab atom closest to the adsorbate's binding atom."""
    ads_i = len(final) - n_ads  # first adsorbate atom (the binding atom)
    L = final.lattice
    best = (None, 1e9)
    for j in range(len(final) - n_ads):
        d = final.frac_coords[ads_i] - final.frac_coords[j]
        d -= np.round(d)
        dist = float(np.linalg.norm(d @ L.matrix))
        if dist < best[1]:
            best = (str(final[j].specie), dist)
    return best


def main(adsorbate: str = "H") -> None:
    spec = ADSORBATES[adsorbate]
    cfg = RunConfig()
    ads_cfg = replace(cfg.adsorbate, species=spec["species"], coords=spec["coords"])
    out = Path(f"runs/adsorbate_validation_{adsorbate}")
    backend = get_backend(cfg.reference)
    slab = Structure.from_file(SLAB)
    cands = adsorbate_candidates(
        slab, ads_cfg, freeze_bottom_fraction=cfg.slab.freeze_bottom_fraction
    )
    n_ads = len(ads_cfg.species)
    out.mkdir(parents=True, exist_ok=True)
    print(
        f"adsorbate: {adsorbate} {spec['species']} | substrate: {SLAB} ({len(slab)} atoms) "
        f"| candidates: {len(cands)}",
        flush=True,
    )

    rows = []
    for c in cands:
        sid = c.site_id
        tag = f"site{sid['site_index']}_{sid['symmetry_class']}"
        leaf = out / tag
        print(f"\n>>> relaxing {tag} ...", flush=True)
        res = relax(c.structure, backend, workdir=leaf, relax_cell=False)
        disp = _adsorbate_max_disp(c.structure, res.structure, n_ads)
        start_dist, bond_len = _adsorbate_start_distance(c.structure, n_ads)
        flags = placement_quality_flags(
            start_fmax=res.start_fmax,
            fmax_target=cfg.relax.fmax,
            nsteps=res.nsteps,
            adsorbate_max_disp=disp,
            start_ads_distance=start_dist,
            ads_bond_length=bond_len,
        )
        neighbor, ndist = _nearest_surface_atom(res.structure, n_ads)
        row = {
            "tag": tag,
            "energy": res.energy,
            "start_fmax": round(res.start_fmax, 4),
            "nsteps": res.nsteps,
            "ads_max_disp": round(disp, 3) if disp is not None else None,
            "start_dist": round(start_dist, 3) if start_dist is not None else None,
            "bonded_to": neighbor,
            "bond_dist": round(ndist, 3),
            "flags": flags,
        }
        rows.append(row)
        print(json.dumps(row), flush=True)

    rows.sort(key=lambda r: r["energy"])
    e0 = rows[0]["energy"]
    print("\n==== ranking (relative energy, eV) ====", flush=True)
    for r in rows:
        print(
            f"  {r['tag']:22s} dE={r['energy'] - e0:+.3f}  "
            f"bonded_to={r['bonded_to']}@{r['bond_dist']}A  "
            f"disp={r['ads_max_disp']}  flags={len(r['flags'])}",
            flush=True,
        )
    (out / "summary.json").write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {out / 'summary.json'}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("adsorbate", nargs="?", default="H", choices=list(ADSORBATES))
    main(ap.parse_args().adsorbate)
