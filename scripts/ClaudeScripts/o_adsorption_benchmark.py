"""O-atom adsorption energy across 7 rutile transition metal oxides vs. Zhao & Kulik
(J. Phys. Chem. Lett. 2019, 10, 5090-5098), PBE/plane-wave column.

Single site only: ontop the most undercoordinated surface metal cation. The paper never
varies site type or orientation for O (single atom, no orientation d.o.f.; SI Table S9
names exactly one adsorption site, "MO1" -- the rest are neighboring metal atoms tracked
for density analysis, not alternate candidates) -- unlike the CO2 paper, which explicitly
tested multiple orientations. So this mirrors their methodology directly.

Reference is their eq. 2: DeltaE_O = E(O*) - E(*) - (E(H2O) - E(H2)), raw model energies,
no added experimental constant -- NOT the same as energetics.oxygen_chemical_potential_
corrected(), which adds the Kowalski O2-overbinding correction for a different purpose.

Standoff/fmax validated by a one-off local test (UMA-omat/TiO2/Ti5c, seed_standoff=0.5,
fmax=0.02): converged in 84 real steps, final Ti-O = 1.637 A vs. literature PBE 1.716 A.
That 0.5 A standoff is the default here and is reused for any --adsorbate; the literature
columns (lit_e_ads_eV, lit_bond_A, lit_e_sigma) and the eq-2 water-splitting reference are
specific to O and are only populated when --adsorbate O (the default) is used -- swapping
in another fragment gets a plain isolated-fragment gas reference and blank literature
columns instead of silently mislabeled O numbers.

    python scripts/o_adsorption_benchmark.py runs/o_ads_benchmark
    python scripts/o_adsorption_benchmark.py runs/co_ads_benchmark --adsorbate CO

Each run writes job.json (adsorbate, standoff, fmax, models, oxides) into outdir for
provenance. Use a fresh outdir per adsorbate: results.jsonl is append-only, so rerunning
into the same outdir with a different --adsorbate blends both into one file (each row is
tagged with "adsorbate" so it stays interpretable) and job.json will only describe the
most recent invocation.
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

MODELS = ["MACE-mh1-omat", "UMA-omat", "UMA-oc22"]
# Opt-in via --models; not run by default so the original 3-model comparison stays stable.
EXTRA_MODELS = [
    "UMA-oc20", "MACE-mh1-oc20",
    "SevenNet-omni-oc22", "SevenNet-omni-oc20", "SevenNet-omni-omat24", "SevenNet-omni-mpa",
    "UMA-M-oc20", "UMA-M-omat",
]
SEED_STANDOFF = 0.5
DESORB_TOL = 2.0
DESORB_CHECK_STEP = 100
DESORB_TREND_WINDOW = 20
EXTEND_STEPS = 100
MAX_EXTENSIONS = 3
TRIVIAL_START_TOL = 1.0

# oxide -> (mp-id, literature Delta_E_O (PBE/PW, eV), literature E_sigma (PBE/PW, eV/(110)u.a.),
# literature M-O bond length (PBE, A)). Zhao & Kulik 2019, Fig. S3 (PW legend), Fig. S1
# (PW legend), Table S3.
OXIDES = {
    "TiO2": {"mp_id": "mp-2657", "lit_e_ads_eV": 4.43, "lit_e_sigma": 0.77, "lit_bond_A": 1.716},
    "VO2": {"mp_id": "mp-19094", "lit_e_ads_eV": -0.60, "lit_e_sigma": 0.60, "lit_bond_A": 1.586},
    "MoO2": {"mp_id": "mp-510536", "lit_e_ads_eV": -0.29, "lit_e_sigma": 1.51, "lit_bond_A": 1.725},
    "RuO2": {"mp_id": "mp-825", "lit_e_ads_eV": 2.28, "lit_e_sigma": 1.39, "lit_bond_A": 1.761},
    "RhO2": {"mp_id": "mp-725", "lit_e_ads_eV": 3.32, "lit_e_sigma": 1.44, "lit_bond_A": 1.818},
    "IrO2": {"mp_id": "mp-2723", "lit_e_ads_eV": 1.84, "lit_e_sigma": 1.77, "lit_bond_A": 1.789},
    "PtO2": {"mp_id": "mp-1077716", "lit_e_ads_eV": 3.45, "lit_e_sigma": 1.06, "lit_bond_A": 1.855},
}
ALL_OXIDES = tuple(OXIDES)


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


def run_one(model: str, oxide: str, outdir: Path, cfg: RunConfig, adsorbate: str) -> dict:
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

    print(f"  [{model} / {oxide}]", flush=True)
    start = get_structure(mp_id)
    bulk = relax_record(start, "bulk", "db", relax_cell=True).structure
    slab_in = make_slab(bulk, cfg.slab)
    slab_out = relax_record(slab_in, "slab", "cut_from_relaxed_bulk")
    pristine_energy = slab_out.energy
    pristine_structure = slab_out.structure

    ads_species, ads_coords = ADSORBATE_FRAGMENTS[adsorbate]
    n_ads = len(ads_species)
    if adsorbate == "O":
        # Zhao & Kulik eq. 2: raw model energies, no added constant.
        h2_species, h2_coords = ADSORBATE_FRAGMENTS["H2"]
        h2o_species, h2o_coords = ADSORBATE_FRAGMENTS["H2O"]
        e_h2 = gas_reference_energy(backend, cfg, pipeline.relax, species=h2_species, coords=h2_coords)
        e_h2o = gas_reference_energy(backend, cfg, pipeline.relax, species=h2o_species, coords=h2o_coords)
        e_ads_ref = e_h2o - e_h2
    else:
        # No paper-specific reaction reference for this fragment -- fall back to the
        # plain isolated-fragment gas energy (cfg.adsorbate is already set to it).
        e_ads_ref = gas_reference_energy(backend, cfg, pipeline.relax)

    candidates = adsorbate_candidates(
        pristine_structure, cfg.adsorbate, freeze_bottom_fraction=cfg.slab.freeze_bottom_fraction,
    )
    cand = undercoordinated_metal_site(candidates)
    anchor_idx = len(cand.structure) - n_ads
    site_dir = odir / model / "adsorbate" / f"site{cand.site_id['site_index']}_{cand.site_id['symmetry_class']}"

    row = {
        "model": model, "oxide": oxide, "adsorbate": adsorbate, "site": cand.site_id["site_label"],
        # Zhao & Kulik literature values are O-specific; blank for any other adsorbate
        # rather than attaching O's numbers to a different fragment's row.
        "lit_e_ads_eV": OXIDES[oxide]["lit_e_ads_eV"] if adsorbate == "O" else None,
        "lit_e_sigma": OXIDES[oxide]["lit_e_sigma"] if adsorbate == "O" else None,
        "lit_bond_A": OXIDES[oxide]["lit_bond_A"] if adsorbate == "O" else None,
    }
    try:
        res = relax(
            cand.structure, backend, workdir=site_dir,
            fmax=cfg.relax.fmax, max_steps=cfg.relax.max_steps, optimizer=cfg.relax.optimizer,
            desorb_check_n_ads=n_ads, desorb_check_step=DESORB_CHECK_STEP,
            desorb_trend_window=DESORB_TREND_WINDOW,
            extend_if_approaching=True, extend_steps=EXTEND_STEPS, max_extensions=MAX_EXTENSIONS,
        )
    except Exception as e:
        row.update({
            "failed": True, "error": str(e)[:500], "e_ads_eV": None, "bond_A": None,
            "desorbing": None, "converged": None, "nsteps": None,
        })
        print(f"    RELAX FAILED: {row['error'][:200]}", flush=True)
        return row

    e_ads = adsorption_energy(res.energy, pristine_energy, e_ads_ref)
    end_dist, bond_len = _adsorbate_anchor_distance(res.structure, n_ads)
    early_stopped = bool(res.meta.get("early_stopped_desorbing"))
    desorbed_final = (
        end_dist is not None and bond_len is not None and end_dist >= DESORB_TOL * bond_len
    )
    desorbing = early_stopped or desorbed_final
    trivial_start = res.start_fmax <= TRIVIAL_START_TOL * cfg.relax.fmax
    row.update({
        "failed": False, "error": None, "e_ads_eV": e_ads, "bond_A": end_dist,
        "desorbing": desorbing, "desorbing_early_stop": early_stopped,
        "desorbing_final_geometry": desorbed_final, "trivial_start": trivial_start,
        "start_fmax": res.start_fmax, "extended": bool(res.meta.get("extended")),
        "extensions_used": res.meta.get("extensions_used", 0),
        "converged": res.converged, "nsteps": res.nsteps,
    })
    status = "DESORBING" if desorbing else ("EXTENDED" if row["extended"] else "OK")
    lit_e_str = f"{row['lit_e_ads_eV']:+.2f}" if row["lit_e_ads_eV"] is not None else "n/a"
    lit_bond_str = f"{row['lit_bond_A']:.3f}" if row["lit_bond_A"] is not None else "n/a"
    print(f"    E_ads={e_ads:+.4f} eV (lit {lit_e_str})  "
          f"bond={end_dist:.3f} A (lit {lit_bond_str})  {status}  "
          f"nsteps={res.nsteps}", flush=True)
    return row


def main(
    outdir: Path, fmax: float, oxides: list[str] = None, models: list[str] = None,
    adsorbate: str = "O", seed_standoff: float = SEED_STANDOFF,
) -> None:
    oxides = oxides or list(OXIDES)
    models = models or MODELS
    ads_species, ads_coords = ADSORBATE_FRAGMENTS[adsorbate]
    cfg = RunConfig()
    cfg = replace(cfg, adsorbate=replace(
        cfg.adsorbate, species=ads_species, coords=ads_coords,
        positions=("ontop",), max_per_position=None, seed_standoff=seed_standoff,
    ))
    cfg = replace(cfg, relax=replace(cfg.relax, fmax=fmax))
    outdir.mkdir(parents=True, exist_ok=True)
    results_path = outdir / "results.jsonl"

    job_path = outdir / "job.json"
    job_path.write_text(json.dumps({
        "adsorbate": adsorbate, "seed_standoff": seed_standoff, "fmax": fmax,
        "positions": list(cfg.adsorbate.positions), "oxides": oxides, "models": models,
        "e_ads_reference": (
            "Zhao & Kulik eq. 2 (E(H2O) - E(H2), raw model energies)" if adsorbate == "O"
            else "isolated-fragment gas reference (gas_reference_energy)"
        ),
    }, indent=2))

    pair_failures: list[tuple[str, str, str]] = []
    for model in models:
        for oxide in oxides:
            try:
                row = run_one(model, oxide, outdir, cfg, adsorbate)
            except Exception as e:
                print(f"  [{model} / {oxide}] PAIR FAILED before relax: {e}", flush=True)
                pair_failures.append((model, oxide, str(e)[:500]))
                row = {
                    "model": model, "oxide": oxide, "adsorbate": adsorbate, "site": None,
                    "failed": True, "error": f"pair-level: {e}"[:500], "e_ads_eV": None,
                    "bond_A": None, "desorbing": None, "converged": None, "nsteps": None,
                }
            with results_path.open("a") as f:
                f.write(json.dumps(row) + "\n")

    print(f"\nwrote {results_path}")
    if pair_failures:
        print(f"\n{len(pair_failures)} (model, oxide) pair(s) failed before relaxing:")
        for model, oxide, err in pair_failures:
            print(f"  {model:16s}{oxide:8s}{err[:150]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("outdir", type=Path)
    ap.add_argument("--fmax", type=float, default=0.02)
    ap.add_argument("--oxides", nargs="+", choices=ALL_OXIDES, default=list(ALL_OXIDES))
    ap.add_argument("--models", nargs="+", choices=MODELS + EXTRA_MODELS, default=MODELS)
    ap.add_argument("--adsorbate", choices=list(ADSORBATE_FRAGMENTS), default="O")
    ap.add_argument("--seed-standoff", type=float, default=SEED_STANDOFF)
    a = ap.parse_args()
    main(a.outdir, a.fmax, a.oxides, a.models, a.adsorbate, a.seed_standoff)
