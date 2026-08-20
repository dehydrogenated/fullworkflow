"""H2O adsorption on TiO2/RuO2/IrO2(110) vs. Gonzalez, Heras-Domingo, Pantaleone, Rimola,
Rodriguez-Santiago, Solans-Monfort & Sodupe (ACS Omega 2019, 4, 2989-2999), PBE-D2.

Single site only: water's O ontop the most undercoordinated surface metal cation (M5c),
oriented so the H-O-H bisector tips toward the nearest bridging O2c -- one H forming a
hydrogen bond to it, matching the paper's own described molecular-adsorption geometry
("Adsorption of Isolated Water Molecules": O binds M5c, one H hydrogen-bonds to the
nearest Obr). One relaxation per (oxide, model) -- no dissociation probing, no nudge-apart
re-relaxation to test whether the outcome is a genuine minimum. That machinery lives in
h2o_dissociation_probe.py; this script only wants the adsorption energy, the same way the
paper reports it, not a mechanistic study of the transfer pathway.

Reference: DeltaE_ads = E(H2O*) - E(*) - E(H2O, gas) -- the paper's own eq. 2. No
water-splitting cycle (unlike O*/OH* in mo2_adsorption_benchmark.py): water isn't being
split here, the whole molecule adsorbs.

Literature values are Table 3's *mol* row per oxide, converted kJ/mol -> eV (/96.485) --
except IrO2(110), where the paper found no molecular minimum at all: every attempt to
optimize the molecular form collapsed to the dissociated one. That row's lit_e_ads_eV is
their *diss* value instead, and lit_form records which -- a model landing in the
dissociated basin for IrO2 is matching the paper's own finding, not a failure. Every
result row's `dissociated` flag exists so analysis can bucket by final state (matches the
paper's own mol/diss split) rather than compare energies across mismatched configurations.

Placement (seed_standoff, orient_toward's "bisector" mode, DIAGONAL_NUDGE) reuses
h2o_dissociation_probe.py's already-validated recipe: single_h_ti_facing repulsed instead
of adsorbing on TiO2 (its pre-bend mirror step starts both H's pointed into the surface),
bisector adsorbed. Rotation alone couldn't close the O-H...O2c gap (a full relaxation left
it at 1.95 A, down from ~2.9-3.6 A at the start -- a distance problem, not an orientation
one), hence the added rigid translate.

    python scripts/ClaudeScripts/h2o_adsorption_benchmark.py runs/h2o_ads_benchmark
    python scripts/ClaudeScripts/h2o_adsorption_benchmark.py runs/h2o_ads_benchmark --oxides RuO2 IrO2 --models UMA-omat
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from h2o_dissociation_probe import (  # noqa: E402
    find_ti_and_o2c_anchors, is_dissociated, min_image_vector, orient_toward,
    translate_toward, undercoordinated_metal_site,
)

import numpy as np  # noqa: E402

from oxide_workflow import pipeline  # noqa: E402
from oxide_workflow.backends import get_backend, relax  # noqa: E402
from oxide_workflow.config import ADSORBATE_FRAGMENTS, RunConfig  # noqa: E402
from oxide_workflow.energetics import adsorption_energy, gas_reference_energy  # noqa: E402
from oxide_workflow.pipeline import _adsorbate_anchor_distance  # noqa: E402
from oxide_workflow.stages import adsorbate_candidates, make_slab  # noqa: E402
from oxide_workflow.structures import get_structure  # noqa: E402

# Same threshold o_adsorption_benchmark.py uses for its own desorb check: end distance to
# the nearest surface atom, in units of the covalent-radii bond length for that pair.
DESORB_TOL = 2.0

KJ_PER_MOL_TO_EV = 96.485

# Table 3, (110) column. IrO2 has no *mol* row in the paper -- diss is the only minimum
# they located there, so lit_form flags that row as not a like-for-like comparison target.
OXIDES = {
    "TiO2": {"mp_id": "mp-2657", "lit_e_ads_eV": -86.9 / KJ_PER_MOL_TO_EV, "lit_form": "mol"},
    "RuO2": {"mp_id": "mp-825", "lit_e_ads_eV": -131.3 / KJ_PER_MOL_TO_EV, "lit_form": "mol"},
    "IrO2": {
        "mp_id": "mp-2723", "lit_e_ads_eV": -211.5 / KJ_PER_MOL_TO_EV,
        "lit_form": "diss (paper found no mol minimum)",
    },
}
MODELS = ["MACE-mh1-omat", "UMA-omat", "UMA-oc22", "SevenNet-omni-omat24"]
SEED_STANDOFF = 1.2  # matches h2o_dissociation_probe.py -- see its own comment
DIAGONAL_NUDGE = 1.0  # matches h2o_dissociation_probe.py -- see its own comment
FMAX = 0.02
DESORB_CHECK_STEP = 100
DESORB_TREND_WINDOW = 20
EXTEND_STEPS = 100
MAX_EXTENSIONS = 3


def run_one(model: str, oxide: str, outdir: Path, cfg: RunConfig, nudge: float) -> dict:
    backend = get_backend(model)
    odir = outdir / oxide / model
    info = OXIDES[oxide]

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
    start = get_structure(info["mp_id"])
    bulk = relax_record(start, "bulk", "db", relax_cell=True).structure
    slab_in = make_slab(bulk, cfg.slab)
    slab_out = relax_record(slab_in, "slab", "cut_from_relaxed_bulk")
    pristine_energy = slab_out.energy
    pristine_structure = slab_out.structure

    ti_idx, o2c_idx = find_ti_and_o2c_anchors(pristine_structure)

    h2o_species, h2o_coords = ADSORBATE_FRAGMENTS["H2O"]
    n_ads = len(h2o_species)
    e_gas = gas_reference_energy(backend, cfg, pipeline.relax, species=h2o_species, coords=h2o_coords)

    candidates = adsorbate_candidates(
        pristine_structure,
        replace(cfg.adsorbate, species=h2o_species, coords=h2o_coords, positions=("ontop",),
                seed_standoff=SEED_STANDOFF),
        freeze_bottom_fraction=cfg.slab.freeze_bottom_fraction,
    )
    cand = undercoordinated_metal_site(candidates)
    anchor_idx = len(cand.structure) - n_ads  # O
    h_near_idx = anchor_idx + 1
    h_far_idx = anchor_idx + 2

    target_dir = min_image_vector(
        pristine_structure.lattice, pristine_structure[o2c_idx].coords, pristine_structure[ti_idx].coords,
    )
    target_dir = target_dir / np.linalg.norm(target_dir)

    oriented = orient_toward(cand.structure, anchor_idx, n_ads, target_dir, mode="bisector")
    oriented = translate_toward(oriented, anchor_idx, n_ads, target_dir, nudge)
    start_o_ti = oriented.get_distance(anchor_idx, ti_idx)
    # Both H's, not just the one orient_toward aimed -- the two are chemically equivalent
    # and can swap which one ends up closer to O2c during relaxation (H_near/H_far are seed
    # labels, not fixed identities the optimizer has to respect).
    start_h_o2c = min(oriented.get_distance(h_near_idx, o2c_idx), oriented.get_distance(h_far_idx, o2c_idx))
    print(f"    seed: O-M5c={start_o_ti:.3f} A, min(H...O2c)={start_h_o2c:.3f} A", flush=True)

    t0 = time.time()
    res = relax(
        oriented, backend, workdir=odir / "adsorbate",
        fmax=cfg.relax.fmax, max_steps=cfg.relax.max_steps, optimizer=cfg.relax.optimizer,
        desorb_check_n_ads=n_ads, desorb_check_step=DESORB_CHECK_STEP,
        desorb_trend_window=DESORB_TREND_WINDOW,
        extend_if_approaching=True, extend_steps=EXTEND_STEPS, max_extensions=MAX_EXTENSIONS,
    )
    elapsed_s = time.time() - t0

    e_ads = adsorption_energy(res.energy, pristine_energy, e_gas)
    dissociated = is_dissociated(res.structure, h_near_idx, anchor_idx, o2c_idx)
    end_o_ti = res.structure.get_distance(anchor_idx, ti_idx)
    end_h_o2c = min(res.structure.get_distance(h_near_idx, o2c_idx), res.structure.get_distance(h_far_idx, o2c_idx))

    # Is it adsorbed onto the metal surface at all -- same check o_adsorption_benchmark.py
    # uses: the O's distance to its nearest surface atom post-relaxation, vs. the covalent
    # bond length for that pair. This is independent of whether the H-Obr bond formed --
    # it only asks "did the water stay bound to the surface, or drift away."
    end_dist, bond_len = _adsorbate_anchor_distance(res.structure, n_ads)
    adsorbed = end_dist is not None and bond_len is not None and end_dist < DESORB_TOL * bond_len
    h_bonded = end_h_o2c < 2.3  # standard H-bond cutoff; informational, separate question

    print(f"    E_ads={e_ads:+.4f} eV (lit {info['lit_e_ads_eV']:+.4f} eV, {info['lit_form']})  "
          f"adsorbed={adsorbed} (O-anchor={end_dist:.3f} A, bond~{bond_len:.3f} A)  "
          f"h_bonded={h_bonded} (min H...O2c={end_h_o2c:.3f} A)  dissociated={dissociated}  "
          f"converged={res.converged}  nsteps={res.nsteps}  {elapsed_s:.0f}s", flush=True)

    return {
        "model": model, "oxide": oxide, "failed": False,
        "site": cand.site_id["site_label"], "nudge_A": nudge,
        "e_ads_eV": e_ads, "lit_e_ads_eV": info["lit_e_ads_eV"], "lit_form": info["lit_form"],
        "adsorbed": adsorbed, "h_bonded": h_bonded, "dissociated": dissociated,
        "start_o_m5c_A": start_o_ti, "start_h_o2c_A": start_h_o2c,
        "end_o_m5c_A": end_o_ti, "end_h_o2c_A": end_h_o2c,
        "converged": res.converged, "nsteps": res.nsteps, "elapsed_s": elapsed_s,
    }


def main(outdir: Path, oxides: list[str], models: list[str], fmax: float, nudge: float) -> None:
    cfg = RunConfig()
    cfg = replace(cfg, relax=replace(cfg.relax, fmax=fmax))
    outdir.mkdir(parents=True, exist_ok=True)
    results_path = outdir / "results.jsonl"

    job_path = outdir / "job.json"
    job_path.write_text(json.dumps({
        "oxides": oxides, "models": models, "seed_standoff": SEED_STANDOFF,
        "diagonal_nudge": nudge, "fmax": fmax,
        "e_ads_reference": "E(H2O*) - E(*) - E(H2O, gas), Gonzalez et al. 2019 eq. 2",
        "literature_source": "Gonzalez et al. 2019, ACS Omega 4, 2989-2999, Table 3 (110)",
    }, indent=2))

    pair_failures: list[str] = []
    for oxide in oxides:
        for model in models:
            tag = f"{oxide}/{model}"
            try:
                row = run_one(model, oxide, outdir, cfg, nudge)
            except Exception as e:
                print(f"  [{tag}] PAIR FAILED: {e}", flush=True)
                pair_failures.append(f"{tag}: {str(e)[:300]}")
                row = {
                    "model": model, "oxide": oxide, "failed": True, "error": f"{e}"[:2000],
                    "e_ads_eV": None, "adsorbed": None, "dissociated": None,
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
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--fmax", type=float, default=FMAX)
    ap.add_argument("--nudge", type=float, default=DIAGONAL_NUDGE,
                     help="rigid translate of the seed toward O2c, in Angstrom")
    a = ap.parse_args()
    main(a.outdir, a.oxides, a.models, a.fmax, a.nudge)
