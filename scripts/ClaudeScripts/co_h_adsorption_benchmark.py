"""CO and H adsorption vs. two independent literature sources, single site per pair.

TiO2(110): vs. Andriuc, Siron & Persson (Surface Science 758, 2025, 122745), RPBE+D3,
fmax=0.05 eV/A, minimum-dimension slab (12 A vacuum, 7 A min height, 10 A min lateral).
IrO2(110): vs. Pope, Yun et al. (Surface Science 751, 2025, 122619), plain PBE (no
dispersion correction), fmax=0.03 eV/A, explicit 4x2 supercell, 4 layers (~12 A thick),
25 A vacuum, bottom 2 layers fixed. Different functional, different fmax, different slab
convention between the two papers -- noted here rather than reconciled, since our own
compute settings (fmax=0.01, seed_standoff=0.5) stay fixed across both oxides regardless
of what each paper individually used. Both papers agree on the site type that matters for
CO though: ontop the single most undercoordinated surface metal cation (Ti5c / Ircus) --
on IrO2(110)'s *stoichiometric* surface specifically, this isn't just the preferred site,
it's the only one physically available at all (no O-vacancy for a bridging site to exist
on a pristine slab), so this script's existing single-metal-site convention is exactly
right for IrO2/CO, not a simplification the way it is for TiO2.

H is placed differently from CO: ontop the bridging O2c site, not the metal cation.
Andriuc et al.'s own finding (Section 3: "the shortest H-surface distances, corresponding
to chemisorption, occur *exclusively* in cases in which H is closest to a surface O
site") -- placing H on the metal cation the same way CO is placed gave every model a
consistent positive (unfavorable) E_ads, which is real chemistry (H correctly doesn't
want the metal site) but not comparable to any literature number, since the paper's own
best H site is O, not Ti. IrO2 doesn't get an H row here (not requested, and the Pope et
al. paper doesn't study H at all -- it's a CO oxidation paper).

seed_standoff=0.5, not looser: a prior local probe (runs/COtest_july29/) pushed CO in
tighter than this (Ti-C ~2.2 A) and the relaxation pushed it straight back out to ~4 A
instead of settling into a stronger bound state -- a real, observed failure mode, not a
hypothetical one. 0.5 is also o_adsorption_benchmark.py's own validated default.

Reference: CO -- DeltaE_ads = E(CO*) - E(*) - E(CO, gas), its own relaxed gas energy
directly. H -- DeltaE_ads = E(H*) - E(*) - 0.5*E(H2, gas), Andriuc et al.'s eq. 1 (half of
H2's gas energy, not an isolated single H atom's own DFT energy -- avoids the
spin/multiplicity awkwardness of an isolated unpaired H, same convention already used for
O*/OH* in mo2_adsorption_benchmark.py).

Literature values:
- TiO2(110) CO = -0.42 eV (RPBE+D3), cross-validated by Andriuc et al. against thermal
  desorption experiments at -0.43 eV (Linsebigler et al. 1995) -- genuinely weak binding,
  so a small negative E_ads is a correct result, not a failure.
- TiO2(110) H: not tabulated as a number anywhere in the paper's text (only in a Fig. 3
  scatter plot) -- left None rather than eyeballed.
- IrO2(110) CO = -2.311 eV (223 kJ/mol desorption energy, PBE, converted and sign-flipped
  to this repo's E_ads convention: their Ed = -(our E_ads)). The paper itself flags PBE as
  likely overestimating this -- single-point HSE06 gives 216 kJ/mol (-2.239 eV), and their
  own TPRS-based kinetic estimate puts the *true* binding energy closer to 170-190 kJ/mol
  (-1.76 to -1.97 eV). -2.311 eV is recorded as the primary comparison point since it's
  the paper's own directly-computed DFT number, with the corrected range as context.

Default MODELS is the full roster (14): CHGNet-0.3.0 and Orb-v2 included directly now
that their envs pull a CUDA-compatible torch build (previously incompatible with
Sockeye's V100s, which needed a separate CPU-forced job -- not needed anymore).
UMA-M-* stays last since it's still the slowest.

    python scripts/ClaudeScripts/co_h_adsorption_benchmark.py runs/co_h_ads_benchmark
    python scripts/ClaudeScripts/co_h_adsorption_benchmark.py runs/co_h_ads_benchmark --oxides IrO2
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

KJ_PER_MOL_TO_EV = 96.485

OXIDES = {
    "TiO2": {
        "mp_id": "mp-2657",
        "adsorbates": {
            "CO": {"lit_e_ads_eV": -0.42, "lit_source": "Andriuc et al. 2025, RPBE+D3, TiO2(110)"},
            "H": {"lit_e_ads_eV": None, "lit_source": "not tabulated for TiO2(110) in the paper's text"},
        },
    },
    "IrO2": {
        "mp_id": "mp-2723",
        "adsorbates": {
            "CO": {
                "lit_e_ads_eV": -223 / KJ_PER_MOL_TO_EV,
                "lit_source": "Pope, Yun et al. 2025, PBE (no dispersion), IrO2(110), COt on "
                               "s-IrO2(110) -- paper itself flags PBE as an overestimate; "
                               "HSE06 single-point = -2.239 eV, TPRS-inferred true value "
                               "~-1.76 to -1.97 eV",
            },
        },
    },
}
# UMA-M-* last on purpose (see h2o_adsorption_benchmark.py's identical comment): it's the
# slowest model in the roster, so keeping it last means every other model's rows are
# already written before it's done, regardless of how long it takes.
MODELS = [
    "MACE-mh1-omat", "MACE-mh1-oc20", "MACE-mh1-matpes",
    "UMA-omat", "UMA-oc22",
    "SevenNet-omni-omat24", "SevenNet-omni-oc20", "SevenNet-omni-oc22", "SevenNet-omni-mpa",
    "eSEN-30M-OAM", "CHGNet-0.3.0", "Orb-v2",
    "UMA-M-omat", "UMA-M-oc20",
]
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


def undercoordinated_oxygen_site(candidates):
    """The bridging O2c-equivalent ontop site -- H's real preferred site per Andriuc et
    al.'s own finding (see module docstring), mirroring undercoordinated_metal_site()'s
    logic but filtering for O-labeled sites instead of metal ones."""
    oxygen_ontop = [
        c for c in candidates
        if c.site_id["symmetry_class"] == "ontop"
        and c.site_id["site_label"] and c.site_id["site_label"].startswith("O")
    ]
    if not oxygen_ontop:
        raise RuntimeError("no oxygen ontop site found")

    def coord(c):
        m = re.match(r"^[A-Za-z]+(\d+)c$", c.site_id["site_label"])
        return int(m.group(1)) if m else 99

    return min(oxygen_ontop, key=coord)


def relax_bulk_slab(model: str, oxide: str, outdir: Path, cfg: RunConfig):
    """Bulk + slab relaxation for one (model, oxide) -- shared by every adsorbate tested
    on that oxide (CO and H both start from the identical pristine TiO2 slab), so this
    avoids re-relaxing it once per adsorbate. Same reasoning as
    h2o_adsorption_benchmark.py's own relax_bulk_slab()."""
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
    # H goes on the bridging O2c site (its real preferred site per Andriuc et al.);
    # everything else (CO here) goes on the undercoordinated metal cation.
    cand = undercoordinated_oxygen_site(candidates) if adsorbate == "H" else undercoordinated_metal_site(candidates)

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
        "e_ads_reference": "CO: E(CO,gas) direct; H: 0.5*E(H2,gas)",
        "literature_sources": {
            "TiO2": "Andriuc, Siron & Persson 2025, Surface Science 758, 122745 (RPBE+D3)",
            "IrO2": "Pope, Yun et al. 2025, Surface Science 751, 122619 (PBE, no dispersion)",
        },
    }, indent=2))

    # Model-outer, oxide-inner, adsorbate-innermost: every other model finishes across
    # both oxides (and both adsorbates, sharing bulk/slab) before the slowest one
    # (UMA-M-*, last in MODELS) even starts. See h2o_adsorption_benchmark.py's identical
    # comment for the full reasoning.
    # (oxide, adsorbate) combos not defined for that oxide (e.g. IrO2/H) -- computed once,
    # independent of which models are running, so it isn't repeated per model below.
    wanted_by_oxide = {o: [a for a in adsorbates if a in OXIDES[o]["adsorbates"]] for o in oxides}
    skipped = [
        f"{o}/{a}" for o in oxides for a in adsorbates if a not in OXIDES[o]["adsorbates"]
    ]

    pair_failures: list[str] = []
    for model in models:
        for oxide in oxides:
            wanted = wanted_by_oxide[oxide]
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
                        "converged": None, "nsteps": None,
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
                        "converged": None, "nsteps": None,
                    }
                with results_path.open("a") as f:
                    f.write(json.dumps(row) + "\n")

    print(f"\nwrote {results_path}")
    if skipped:
        print(f"\nskipped (adsorbate not defined for that oxide): {', '.join(skipped)}")
    if pair_failures:
        print(f"\n{len(pair_failures)} pair(s) failed:")
        for p in pair_failures:
            print(f"  {p}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("outdir", type=Path)
    ap.add_argument("--oxides", nargs="+", default=list(OXIDES), choices=list(OXIDES))
    ap.add_argument("--adsorbates", nargs="+", default=["CO", "H"], choices=["CO", "H"],
                     help="a combo not defined for a given oxide (e.g. IrO2/H) is skipped, not an error")
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--fmax", type=float, default=FMAX)
    a = ap.parse_args()
    main(a.outdir, a.oxides, a.adsorbates, a.models, a.fmax)
