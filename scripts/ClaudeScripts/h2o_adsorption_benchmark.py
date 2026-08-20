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
bisector adsorbed. Rotation alone couldn't close the O-H...O2c gap on its own -- the nudge
translates the seed toward O2c, restricted to the surface-plane (lateral) direction only
(Ti->O2c is not lateral -- the bridging O protrudes above the M5c plane, so translating
along the raw 3D direction pushes the seed higher with every nudge instead of sliding it
sideways; orient_toward still uses the full 3D vector for the H tilt, only the rigid
translate is restricted). DIAGONAL_NUDGE=1.0 is validated as the sweet spot: it's the only
value across a 1.0/1.5/2.0/2.5 sweep that actually converges within budget on TiO2/RuO2/IrO2
-- everything above it moves the seed too far off-axis to re-approach the surface in time.
On IrO2 specifically, this seed found the paper's own dissociated basin unprompted with no
dissociation-probing machinery at all -- a real, unforced match to their finding that
IrO2(110) has no molecular minimum.

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
# Full model roster (Sean's original list). Orb-v2/CHGNet-0.3.0 now run directly on GPU
# -- their "orb"/"chgnet" envs previously pulled a torch+cu130 build with no kernels for
# Sockeye's V100s (CC 7.0), fixed by reinstalling both with a CUDA-compatible torch;
# sockeye_h2o_adsorption_benchmark_cpu.slurm is no longer needed for new runs but is left
# in place as a fallback.
# UMA-M-* last on purpose: gated uma-m-1p1.pt checkpoint, and noticeably slower per
# relaxation than the rest of the roster (confirmed on Sockeye, not just a documented
# risk -- backends.py's fairchem#2095 note describes a predictor-construction hang some
# setups hit, but on this cluster it's just slow, not stuck). Keeping it last costs
# nothing and means the rest of the roster's rows are already written before it's done.
MODELS = [
    "MACE-mh1-omat", "MACE-mh1-oc20", "MACE-mh1-matpes",
    "UMA-omat", "UMA-oc22",
    "SevenNet-omni-omat24", "SevenNet-omni-oc20", "SevenNet-omni-oc22", "SevenNet-omni-mpa",
    "eSEN-30M-OAM", "CHGNet-0.3.0", "Orb-v2",
    "UMA-M-omat", "UMA-M-oc20",
]
SEED_STANDOFF = 1.2  # matches h2o_dissociation_probe.py -- see its own comment
DIAGONAL_NUDGE = 1.0  # validated: the only nudge value that actually converges (see
# h2o_dissociation_probe.py's history) -- 1.5/2.0/2.5 all failed to converge within budget
FMAX = 0.01  # validated on TiO2/RuO2/IrO2 -- SevenNet TiO2 adsorbed cleanly, IrO2 found
# the literature's own dissociated basin unprompted
DESORB_CHECK_STEP = 100
DESORB_TREND_WINDOW = 20
EXTEND_STEPS = 100
MAX_EXTENSIONS = 3


def relax_bulk_slab(model: str, oxide: str, outdir: Path, cfg: RunConfig):
    """Bulk + slab relaxation for one (model, oxide) -- shared by every nudge value in a
    sweep, since they all start from the identical pristine slab. Splitting this out
    avoids re-relaxing the same slab once per nudge value (same reasoning as
    mo2_adsorption_benchmark.py's own relax_bulk_slab: real duplicated compute at sweep
    scale, not a micro-optimization)."""
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
    return slab_out.structure, slab_out.energy


def run_adsorbate(
    model: str, oxide: str, pristine_structure, pristine_energy: float,
    outdir: Path, cfg: RunConfig, nudge: float,
) -> dict:
    backend = get_backend(model)
    odir = outdir / oxide / model
    info = OXIDES[oxide]

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

    # Ti->O2c is not lateral -- the bridging O protrudes above the M5c plane (confirmed on
    # this TiO2(110) slab: unit vector ~(0, -0.91, 0.42), i.e. 42% straight up), so
    # translating the *whole seed* along target_dir pushes it higher with every nudge
    # instead of just sliding it sideways -- confirmed empirically: water visibly floating
    # further off the surface as nudge increased. orient_toward still gets the full 3D
    # target_dir (the H really should tilt up toward O2c), but the rigid translate uses
    # only the lateral (surface-plane) component, so nudge changes reach toward Obr without
    # also changing standoff -- that's SEED_STANDOFF's job, not this one's.
    lateral_dir = np.array([target_dir[0], target_dir[1], 0.0])
    lateral_norm = np.linalg.norm(lateral_dir)
    lateral_dir = lateral_dir / lateral_norm if lateral_norm > 1e-8 else target_dir

    oriented = orient_toward(cand.structure, anchor_idx, n_ads, target_dir, mode="bisector")
    oriented = translate_toward(oriented, anchor_idx, n_ads, lateral_dir, nudge)
    start_o_ti = oriented.get_distance(anchor_idx, ti_idx)
    # Both H's, not just the one orient_toward aimed -- the two are chemically equivalent
    # and can swap which one ends up closer to O2c during relaxation (H_near/H_far are seed
    # labels, not fixed identities the optimizer has to respect).
    start_h_o2c = min(oriented.get_distance(h_near_idx, o2c_idx), oriented.get_distance(h_far_idx, o2c_idx))
    print(f"    [nudge={nudge}] seed: O-M5c={start_o_ti:.3f} A, min(H...O2c)={start_h_o2c:.3f} A", flush=True)

    t0 = time.time()
    res = relax(
        # Namespaced by nudge -- otherwise a sweep's later nudge values overwrite the
        # trajectory/relaxed.vasp of earlier ones sharing the same (oxide, model) odir.
        oriented, backend, workdir=odir / "adsorbate" / f"nudge_{nudge}",
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
    # bool(...): end_h_o2c/end_dist/bond_len trace back through Structure.get_distance(),
    # which returns numpy floats -- comparisons against them are numpy.bool_, and
    # json.dumps() rejects that (confirmed: TypeError on Sockeye without this wrap, since
    # is_dissociated() had the exact same gap -- see its own comment).
    adsorbed = bool(end_dist is not None and bond_len is not None and end_dist < DESORB_TOL * bond_len)
    h_bonded = bool(end_h_o2c < 2.3)  # standard H-bond cutoff; informational, separate question

    print(f"    [nudge={nudge}] E_ads={e_ads:+.4f} eV (lit {info['lit_e_ads_eV']:+.4f} eV, {info['lit_form']})  "
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


def main(
    outdir: Path, oxides: list[str], models: list[str], fmax: float, nudges: list[float],
) -> None:
    cfg = RunConfig()
    cfg = replace(cfg, relax=replace(cfg.relax, fmax=fmax))
    outdir.mkdir(parents=True, exist_ok=True)
    results_path = outdir / "results.jsonl"

    job_path = outdir / "job.json"
    job_path.write_text(json.dumps({
        "oxides": oxides, "models": models, "seed_standoff": SEED_STANDOFF,
        "nudges": nudges, "fmax": fmax,
        "e_ads_reference": "E(H2O*) - E(*) - E(H2O, gas), Gonzalez et al. 2019 eq. 2",
        "literature_source": "Gonzalez et al. 2019, ACS Omega 4, 2989-2999, Table 3 (110)",
    }, indent=2))

    # Model-outer, oxide-inner on purpose -- put any noticeably slower model (e.g.
    # UMA-M-*) LAST in --models. With model as the outer loop, every other model finishes
    # across ALL oxides before the slow one even starts, so its rows land last without
    # delaying RuO2/IrO2 data for models that already ran (oxide-outer would let a slow
    # model on TiO2 delay every later oxide too, even for models with nothing to do with it).
    pair_failures: list[str] = []
    for model in models:
        for oxide in oxides:
            tag = f"{oxide}/{model}"
            try:
                pristine_structure, pristine_energy = relax_bulk_slab(model, oxide, outdir, cfg)
            except Exception as e:
                print(f"  [{tag}] BULK/SLAB FAILED: {e}", flush=True)
                pair_failures.append(f"{tag} (bulk/slab): {str(e)[:300]}")
                for nudge in nudges:
                    row = {
                        "model": model, "oxide": oxide, "nudge_A": nudge, "failed": True,
                        "error": f"bulk/slab: {e}"[:2000], "e_ads_eV": None,
                        "adsorbed": None, "dissociated": None, "converged": None, "nsteps": None,
                    }
                    with results_path.open("a") as f:
                        f.write(json.dumps(row) + "\n")
                continue

            for nudge in nudges:
                ntag = f"{tag}/nudge={nudge}"
                try:
                    row = run_adsorbate(model, oxide, pristine_structure, pristine_energy, outdir, cfg, nudge)
                except Exception as e:
                    print(f"  [{ntag}] PAIR FAILED: {e}", flush=True)
                    pair_failures.append(f"{ntag}: {str(e)[:300]}")
                    row = {
                        "model": model, "oxide": oxide, "nudge_A": nudge, "failed": True,
                        "error": f"{e}"[:2000], "e_ads_eV": None, "adsorbed": None,
                        "dissociated": None, "converged": None, "nsteps": None,
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
    ap.add_argument("--nudges", type=float, nargs="+", default=[DIAGONAL_NUDGE],
                     help="one or more rigid-translate distances toward O2c, in Angstrom "
                          "(bulk/slab relax once per model, reused across every nudge value)")
    a = ap.parse_args()
    main(a.outdir, a.oxides, a.models, a.fmax, a.nudges)
