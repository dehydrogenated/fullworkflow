"""Propane and propene (propylene) adsorption on rutile TiO2/RuO2/IrO2(110), single site
per pair, vs. van Hout, Loveday, Morales-Vidal, Morandi & Lopez 2026 (GAME-Net-Ox, Digital
Discovery 5, 407-414, DOI 10.1039/D5DD00331H). Not a thorough multi-site screen -- one
orientation each, chosen to match a specific literature config, just enough to have a
number to point to.

Their DFT: VASP 5.4.4, PBE+D3, PAW, 450 eV cutoff, 1e-6 eV electronic / 0.025 eV/A force
convergence. Slab: p(2x2) rutile MO2(110), 5 tri-layers, top 2 relaxed / bottom 3 fixed,
15 A vacuum. Adsorbates placed with DockOnSurf on 5 named sites (top_M5c, top_Obr,
bridge_Obr, hollow_M5c-Obr, hollow_M5c-Oip), up to 3 configs each.

Orientation: propane physisorbs weakly and mostly lies flat in their dataset, but a few
configs per oxide are genuine standing/vertical geometries, and a few propene configs have
their C=C genuinely flat/close to the surface (Dewar-Chatt-Duncanson pi-complex geometry)
rather than tilted -- identified by downloading their own relaxed CONTCARs from ioChem-BD
(doi:10.19061/iochem-bd-1-396) and, for every config of that molecule on that oxide (all 5
sites, not just top_M5c), measuring how close its geometry sits to OUR OWN fragment's exact
target: propane's 3-carbon z-spread (ours = 2.106 A, from ADSORBATE_FRAGMENTS["Propane"] in
config.py) or propene's C=C height difference (ours = 0.000 A, both alkene carbons built at
identical z in ADSORBATE_FRAGMENTS["Propene"]). The literature config with the SMALLEST
|difference| from our own target wins the comparison slot for that oxide/molecule -- not
their strongest binder, not necessarily top_M5c (an earlier pass of this search was
restricted to top_M5c on the a priori assumption both signatures would land there; three of
the six pairs below turned out to have an even closer geometric match on a different site
once the search was widened to all 5).

Literature values: the config identified above for each oxide/molecule (not their global
minimum across all 5 sites, and not always top_M5c -- see paragraph above). Not tabulated as
numbers in the paper's text; computed from their own relaxed energies via their Eq. 1
(E_ads = E_tot - E_slab - E_mol):
- TiO2(110): Propane -0.501 eV (top_M5c config_2), Propene -0.933 eV (hollow_M5c-Oip config_2)
- RuO2(110): Propane -0.539 eV (top_M5c config_2), Propene -1.328 eV (hollow_M5c-Oip config_1)
- IrO2(110): Propane -0.904 eV (hollow_M5c-Obr config_1), Propene -2.127 eV (top_M5c config_1)

Slab size: their p(2x2), 5 tri-layers (120 atoms), 15 A vacuum vs. our 4x2 supercell,
min_slab_size 12 A (~4 trilayers), bottom 50% frozen -- deliberately bigger laterally (a
1x1-cell smoke test during development produced a wildly unbound false positive from
adsorbate self-image clash; see conversation). --vacuum defaults to 40 A here specifically
(vs. this repo's usual 20 A default elsewhere): vertical propane stands up to ~3.6 A off
its own anchor atom (see config.py's Propane fragment comment) plus seed_standoff plus
relaxation headroom, so the usual 20 A leaves less clearance above a standing molecule than
intended. seed_standoff=0.5 (see the constant's own comment below -- NOT the paper's
1.5-3 A figure, that's a different quantity in this codebase's placement convention).
fmax=0.01, tighter than this repo's usual 0.02, since the first run's shallow physisorption
wells need a stricter force threshold to be sure FIRE actually found the minimum rather
than stopping early in a nearly-flat region.

    python scripts/studies/molecular/propane_propene_adsorption_benchmark.py runs/propane_propene_ads
    python scripts/studies/molecular/propane_propene_adsorption_benchmark.py runs/propane_propene_ads --oxides TiO2
"""

from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import replace
from pathlib import Path

from oxide_workflow import pipeline
from oxide_workflow.backends import get_backend, relax
from oxide_workflow.config import ADSORBATE_FRAGMENTS, RunConfig
from oxide_workflow.energetics import adsorption_energy, gas_reference_energy
from oxide_workflow.pipeline import _adsorbate_anchor_distance
from oxide_workflow.stages import adsorbate_candidates, make_slab
from oxide_workflow.structures import get_structure

LIT_SOURCE = (
    "van Hout, Loveday, Morales-Vidal, Morandi & Lopez 2026, Digital Discovery 5, 407-414 "
    "(GAME-Net-Ox), DOI 10.1039/D5DD00331H, PBE+D3; config identified from their own relaxed "
    "CONTCARs (doi:10.19061/iochem-bd-1-396) as the closest geometric match -- across all 5 "
    "sites, not just top_M5c -- to our own fixed vertical-propane / side-on-propene "
    "orientation, not necessarily their global best or on the same site every time, see "
    "module docstring"
)

OXIDES = {
    "TiO2": {
        "mp_id": "mp-2657",
        "adsorbates": {
            "Propane": {"lit_e_ads_eV": -0.5013, "lit_config": "top_M5c config_2"},
            "Propene": {"lit_e_ads_eV": -0.9329, "lit_config": "hollow_M5c-Oip config_2"},
        },
    },
    "RuO2": {
        "mp_id": "mp-825",
        "adsorbates": {
            "Propane": {"lit_e_ads_eV": -0.5394, "lit_config": "top_M5c config_2"},
            "Propene": {"lit_e_ads_eV": -1.3277, "lit_config": "hollow_M5c-Oip config_1"},
        },
    },
    "IrO2": {
        "mp_id": "mp-2723",
        "adsorbates": {
            "Propane": {"lit_e_ads_eV": -0.9043, "lit_config": "hollow_M5c-Obr config_1"},
            "Propene": {"lit_e_ads_eV": -2.1267, "lit_config": "top_M5c config_1"},
        },
    },
}
# UMA-M-* last -- same slowest-model-last convention as every other benchmark in this repo.
MODELS = ["Orb-v2", "eSEN-30M-OAM", "UMA-M-oc20"]
# NOT the paper's own 1.5-3 A range -- that figure is a TOTAL distance in their paper, but
# in this codebase seed_standoff is ADDED ON TOP of a covalent-radius-sum estimate
# (_target_distance() in stages.py), so 1.5 here put every anchor ~3.2-3.4 A out -- right at
# the edge of (or past) a physisorption well that only extends to ~3.0-3.5 A, which is why
# the first run of this script mostly just sat there and called it "converged" without ever
# sliding in (confirmed: 3.41/3.27/3.22 A observed starts match Ti/Ru/Ir-H covalent sums +
# 1.5 exactly). 0.5 matches co_h_adsorption_benchmark.py's own validated value instead.
SEED_STANDOFF = 0.5
FMAX = 0.01
EXTEND_STEPS = 100
MAX_EXTENSIONS = 3


def undercoordinated_metal_site(candidates):
    """top_M5c equivalent: the single ontop site on the least-coordinated surface metal
    cation. Identical logic to co_h_adsorption_benchmark.py's own helper of the same name
    (duplicated rather than imported -- that module is itself a standalone script, not a
    shared library)."""
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


def relax_bulk_slab(model: str, oxide: str, outdir: Path, cfg: RunConfig):
    """Bulk + slab relaxation for one (model, oxide) -- shared by Propane and Propene."""
    backend = get_backend(model)
    mp_id = OXIDES[oxide]["mp_id"]
    odir = outdir / oxide / model

    def relax_record(structure, stage, source_desc, relax_cell=False):
        out = pipeline._relax_record(
            structure, backend, stage=stage, protocol="reference",
            geometry_source=source_desc, cfg=cfg, outdir=odir,
            relax_cell=relax_cell, canonical=True,
        )
        print(f"    {stage:14s} E={out.energy:.4f} eV  {out.header['nsteps']} steps  "
              f"{out.elapsed_s:.0f}s", flush=True)
        return out

    print(f"[{oxide} / {model}]", flush=True)
    start = get_structure(mp_id)
    bulk = relax_record(start, "bulk", "db", relax_cell=True).structure
    slab_in = make_slab(bulk, cfg.slab)
    slab_out = relax_record(slab_in, "slab", "cut_from_relaxed_bulk")
    return slab_out.structure, slab_out.energy


def run_adsorbate(
    model: str, oxide: str, adsorbate: str, pristine_structure, pristine_energy: float,
    outdir: Path, cfg: RunConfig,
) -> dict:
    backend = get_backend(model)
    info = OXIDES[oxide]["adsorbates"][adsorbate]
    odir = outdir / oxide / adsorbate / model

    print(f"  -> {adsorbate}", flush=True)
    ads_species, ads_coords = ADSORBATE_FRAGMENTS[adsorbate]
    n_ads = len(ads_species)
    e_gas = gas_reference_energy(backend, cfg, pipeline.relax, species=ads_species, coords=ads_coords)

    candidates = adsorbate_candidates(
        pristine_structure,
        replace(cfg.adsorbate, species=ads_species, coords=ads_coords,
                positions=("ontop",), max_per_position=None, seed_standoff=SEED_STANDOFF),
        freeze_bottom_fraction=cfg.slab.freeze_bottom_fraction,
    )
    cand = undercoordinated_metal_site(candidates)
    start_dist, _ = _adsorbate_anchor_distance(cand.structure, n_ads)

    t0 = time.time()
    res = relax(
        cand.structure, backend, workdir=odir / "adsorbate",
        fmax=cfg.relax.fmax, max_steps=cfg.relax.max_steps, optimizer=cfg.relax.optimizer,
        desorb_check_n_ads=n_ads,
        extend_if_approaching=True, extend_steps=EXTEND_STEPS, max_extensions=MAX_EXTENSIONS,
    )
    elapsed_s = time.time() - t0
    e_ads = adsorption_energy(res.energy, pristine_energy, e_gas)
    end_dist, _ = _adsorbate_anchor_distance(res.structure, n_ads)
    desorbing = end_dist is not None and start_dist is not None and end_dist > start_dist
    adsorbed = not desorbing

    lit_str = f"{info['lit_e_ads_eV']:+.3f}"
    print(f"    E_ads={e_ads:+.4f} eV (lit {lit_str}, {info['lit_config']})  adsorbed={adsorbed} "
          f"(start_dist={start_dist:.3f} A, end_dist={end_dist:.3f} A)  converged={res.converged}  "
          f"extended={bool(res.meta.get('extended'))}  nsteps={res.nsteps}  {elapsed_s:.0f}s", flush=True)

    return {
        "model": model, "oxide": oxide, "adsorbate": adsorbate, "failed": False,
        "site": cand.site_id["site_label"],
        "e_ads_eV": e_ads, "lit_e_ads_eV": info["lit_e_ads_eV"], "lit_config": info["lit_config"],
        "lit_source": LIT_SOURCE,
        "adsorbed": adsorbed, "start_dist_A": start_dist, "end_dist_A": end_dist,
        "converged": res.converged, "nsteps": res.nsteps, "elapsed_s": elapsed_s,
        "extended": bool(res.meta.get("extended")), "extensions_used": res.meta.get("extensions_used", 0),
    }


def main(outdir: Path, oxides: list[str], adsorbates: list[str], models: list[str], fmax: float, vacuum: float) -> None:
    cfg = RunConfig()
    cfg = replace(cfg, relax=replace(cfg.relax, fmax=fmax))
    cfg = replace(cfg, slab=replace(cfg.slab, min_vacuum_size=vacuum))
    outdir.mkdir(parents=True, exist_ok=True)
    results_path = outdir / "results.jsonl"

    job_path = outdir / "job.json"
    job_path.write_text(json.dumps({
        "oxides": oxides, "adsorbates": adsorbates, "models": models,
        "seed_standoff": SEED_STANDOFF, "fmax": fmax, "vacuum_A": vacuum,
        "site": "top_M5c equivalent (ontop the undercoordinated surface metal cation)",
        "e_ads_reference": "E(system) - E(pristine slab) - E(gas-phase molecule), own relaxed geometries",
        "literature_source": LIT_SOURCE,
    }, indent=2))

    pair_failures: list[str] = []
    for model in models:
        for oxide in oxides:
            wanted = [a for a in adsorbates if a in OXIDES[oxide]["adsorbates"]]
            if not wanted:
                continue
            tag = f"{oxide}/{model}"
            try:
                pristine_structure, pristine_energy = relax_bulk_slab(model, oxide, outdir, cfg)
            except Exception as e:
                print(f"  [{tag}] BULK/SLAB FAILED: {e}", flush=True)
                pair_failures.append(f"{tag} (bulk/slab): {str(e)[:300]}")
                for adsorbate in wanted:
                    row = {
                        "model": model, "oxide": oxide, "adsorbate": adsorbate, "failed": True,
                        "error": f"bulk/slab: {e}"[:2000], "e_ads_eV": None, "adsorbed": None,
                        "converged": None, "nsteps": None, "elapsed_s": None,
                    }
                    with results_path.open("a") as f:
                        f.write(json.dumps(row) + "\n")
                continue

            for adsorbate in wanted:
                atag = f"{tag}/{adsorbate}"
                try:
                    row = run_adsorbate(model, oxide, adsorbate, pristine_structure, pristine_energy, outdir, cfg)
                except Exception as e:
                    print(f"  [{atag}] PAIR FAILED: {e}", flush=True)
                    pair_failures.append(f"{atag}: {str(e)[:300]}")
                    row = {
                        "model": model, "oxide": oxide, "adsorbate": adsorbate, "failed": True,
                        "error": f"{e}"[:2000], "e_ads_eV": None, "adsorbed": None,
                        "converged": None, "nsteps": None, "elapsed_s": None,
                    }
                with results_path.open("a") as f:
                    f.write(json.dumps(row) + "\n")

    print(f"\nwrote {results_path}")
    if pair_failures:
        print(f"\n{len(pair_failures)} pair(s) failed:")
        for p in pair_failures:
            print(f"  {p}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("outdir", type=Path)
    ap.add_argument("--oxides", nargs="+", default=list(OXIDES), choices=list(OXIDES))
    ap.add_argument("--adsorbates", nargs="+", default=["Propane", "Propene"],
                     choices=["Propane", "Propene"])
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--fmax", type=float, default=FMAX)
    ap.add_argument("--vacuum", type=float, default=40.0,
                     help="A; bigger than the 20 A repo default -- propane stands up to "
                          "~3.6 A off its own anchor (see config.py's Propane fragment), "
                          "plus seed_standoff, plus relaxation headroom")
    a = ap.parse_args()
    main(a.outdir, a.oxides, a.adsorbates, a.models, a.fmax, a.vacuum)
