"""CO and H adsorption on rutile TiO2(110) vs. Andriuc, Siron & Persson (Surface Science
758, 2025, 122745), RPBE+D3.

Single site only: C-down for CO (perpendicular to the surface), monoatomic for H, both
ontop the most undercoordinated surface Ti -- the paper's own placement (Delaunay
triangulation + adsorbate-surface distance optimization, same lineage this repo's own
site-finder descends from, both via Montoya & Persson 2017) actually searches every site
type and keeps the best; this script deliberately narrows to the single Ti5c site already
used everywhere else in this repo (Zhao & Kulik's O*, Comer's O*/OH*), not a full
site-search replication -- a smoke test that things adsorb at all, not a site-fidelity study.

seed_standoff=0.5, not looser: a prior local probe (runs/COtest_july29/) pushed CO in
tighter than this (Ti-C ~2.2 A) and the relaxation pushed it straight back out to ~4 A
instead of settling into a stronger bound state -- a real, observed failure mode, not a
hypothetical one. 0.5 is also o_adsorption_benchmark.py's own validated default, reused
here for consistency.

Reference: CO -- DeltaE_ads = E(CO*) - E(*) - E(CO, gas), its own relaxed gas energy
directly (no derived formula). H -- DeltaE_ads = E(H*) - E(*) - 0.5*E(H2, gas), the
paper's own eq. 1 (half of the H2 gas energy, not an isolated single H atom's own DFT
energy -- deliberately avoids the spin/multiplicity awkwardness of computing an isolated
unpaired H directly, same convention already used for O*/OH* in mo2_adsorption_benchmark.py).

Literature: CO on rutile TiO2(110) = -0.42 eV (RPBE+D3), cross-validated by the paper
itself against thermal desorption experiments at -0.43 eV (Linsebigler et al. 1995) -- a
real, weak-binding system, so a small negative E_ads here is a correct result, not a
failure. The paper does not tabulate a TiO2(110)-specific H value in the text (only in
Fig. 3's scatter, not as a number) -- H's lit_e_ads_eV is left None rather than eyeballing
a plot.

    python scripts/ClaudeScripts/co_h_adsorption_benchmark.py runs/co_h_ads_benchmark
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

OXIDES = {"TiO2": "mp-2657"}
ADSORBATES = {
    "CO": {"lit_e_ads_eV": -0.42, "lit_source": "Andriuc et al. 2025, RPBE+D3, TiO2(110)"},
    "H": {"lit_e_ads_eV": None, "lit_source": "not tabulated for TiO2(110) in the paper's text"},
}
MODELS = ["MACE-mh1-omat", "UMA-omat", "UMA-oc22", "SevenNet-omni-omat24"]
SEED_STANDOFF = 0.5  # see module docstring -- validated, don't push tighter
FMAX = 0.01
DESORB_TOL = 2.0
DESORB_CHECK_STEP = 100
DESORB_TREND_WINDOW = 20
EXTEND_STEPS = 100
MAX_EXTENSIONS = 3


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


def run_one(model: str, oxide: str, adsorbate: str, outdir: Path, cfg: RunConfig) -> dict:
    backend = get_backend(model)
    mp_id = OXIDES[oxide]
    info = ADSORBATES[adsorbate]
    odir = outdir / oxide / adsorbate / model

    def relax_record(structure, stage, source_desc, relax_cell=False):
        out = pipeline._relax_record(
            structure, backend, stage=stage, protocol="reference",
            geometry_source=source_desc, cfg=cfg, outdir=odir,
            relax_cell=relax_cell, canonical=True,
        )
        print(f"    {stage:14s} E={out.energy:.4f} eV  {out.header['nsteps']} steps  "
              f"{out.elapsed_s:.0f}s", flush=True)
        return out

    print(f"[{oxide} / {adsorbate} / {model}]", flush=True)
    start = get_structure(mp_id)
    bulk = relax_record(start, "bulk", "db", relax_cell=True).structure
    slab_in = make_slab(bulk, cfg.slab)
    slab_out = relax_record(slab_in, "slab", "cut_from_relaxed_bulk")
    pristine_energy = slab_out.energy
    pristine_structure = slab_out.structure

    ads_species, ads_coords = ADSORBATE_FRAGMENTS[adsorbate]
    n_ads = len(ads_species)

    # CO references directly against its own gas energy; H references against half of
    # H2's -- see module docstring for why H doesn't use its own single-atom fragment.
    if adsorbate == "H":
        h2_species, h2_coords = ADSORBATE_FRAGMENTS["H2"]
        e_gas = 0.5 * gas_reference_energy(backend, cfg, pipeline.relax, species=h2_species, coords=h2_coords)
    else:
        e_gas = gas_reference_energy(backend, cfg, pipeline.relax, species=ads_species, coords=ads_coords)

    candidates = adsorbate_candidates(
        pristine_structure,
        replace(cfg.adsorbate, species=ads_species, coords=ads_coords,
                positions=("ontop",), max_per_position=None, seed_standoff=SEED_STANDOFF),
        freeze_bottom_fraction=cfg.slab.freeze_bottom_fraction,
    )
    cand = undercoordinated_metal_site(candidates)

    res = relax(
        cand.structure, backend, workdir=odir / "adsorbate",
        fmax=cfg.relax.fmax, max_steps=cfg.relax.max_steps, optimizer=cfg.relax.optimizer,
        desorb_check_n_ads=n_ads, desorb_check_step=DESORB_CHECK_STEP,
        desorb_trend_window=DESORB_TREND_WINDOW,
        extend_if_approaching=True, extend_steps=EXTEND_STEPS, max_extensions=MAX_EXTENSIONS,
    )
    e_ads = adsorption_energy(res.energy, pristine_energy, e_gas)
    end_dist, bond_len = _adsorbate_anchor_distance(res.structure, n_ads)
    adsorbed = bool(end_dist is not None and bond_len is not None and end_dist < DESORB_TOL * bond_len)

    lit_str = f"{info['lit_e_ads_eV']:+.2f}" if info["lit_e_ads_eV"] is not None else "n/a"
    print(f"    E_ads={e_ads:+.4f} eV (lit {lit_str})  adsorbed={adsorbed} "
          f"(end_dist={end_dist:.3f} A, bond~{bond_len:.3f} A)  converged={res.converged}  "
          f"nsteps={res.nsteps}", flush=True)

    return {
        "model": model, "oxide": oxide, "adsorbate": adsorbate, "failed": False,
        "site": cand.site_id["site_label"],
        "e_ads_eV": e_ads, "lit_e_ads_eV": info["lit_e_ads_eV"], "lit_source": info["lit_source"],
        "adsorbed": adsorbed, "end_dist_A": end_dist, "bond_len_A": bond_len,
        "converged": res.converged, "nsteps": res.nsteps,
    }


def main(outdir: Path, oxides: list[str], adsorbates: list[str], models: list[str], fmax: float) -> None:
    cfg = RunConfig()
    cfg = replace(cfg, relax=replace(cfg.relax, fmax=fmax))
    outdir.mkdir(parents=True, exist_ok=True)
    results_path = outdir / "results.jsonl"

    job_path = outdir / "job.json"
    job_path.write_text(json.dumps({
        "oxides": oxides, "adsorbates": adsorbates, "models": models,
        "seed_standoff": SEED_STANDOFF, "fmax": fmax,
        "e_ads_reference": "CO: E(CO,gas) direct; H: 0.5*E(H2,gas) -- Andriuc et al. 2025 eq. 1",
        "literature_source": "Andriuc, Siron & Persson 2025, Surface Science 758, 122745",
    }, indent=2))

    pair_failures: list[str] = []
    for oxide in oxides:
        for adsorbate in adsorbates:
            for model in models:
                tag = f"{oxide}/{adsorbate}/{model}"
                try:
                    row = run_one(model, oxide, adsorbate, outdir, cfg)
                except Exception as e:
                    print(f"  [{tag}] PAIR FAILED: {e}", flush=True)
                    pair_failures.append(f"{tag}: {str(e)[:300]}")
                    row = {
                        "model": model, "oxide": oxide, "adsorbate": adsorbate, "failed": True,
                        "error": f"{e}"[:2000], "e_ads_eV": None, "adsorbed": None,
                        "converged": None, "nsteps": None,
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
    ap.add_argument("--adsorbates", nargs="+", default=list(ADSORBATES), choices=list(ADSORBATES))
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--fmax", type=float, default=FMAX)
    a = ap.parse_args()
    main(a.outdir, a.oxides, a.adsorbates, a.models, a.fmax)
