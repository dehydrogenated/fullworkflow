"""O* and OH* adsorption on the MO2 rutile transition-metal-oxide family vs. Comer, Li,
Abild-Pedersen, Bajdich & Winther (J. Phys. Chem. C 2022, 126, 7903-7909), DFT+U/PBE.

Same water-splitting reference the paper uses (its Table S3 reactions), raw model energies,
no added experimental constant:

    H2O(g) - H2(g)     + * -> O*   =>  DeltaE_ads = E(O*)  - E(*) - [E(H2O) - E(H2)]
    H2O(g) - 0.5 H2(g) + * -> HO*  =>  DeltaE_ads = E(HO*) - E(*) - [E(H2O) - 0.5 E(H2)]

Single site only, same choice as scripts/o_adsorption_benchmark.py and for the same reason:
the paper places O*/OH* ontop the single most undercoordinated surface metal cation, no
orientation search.

Literature values come from data/literature/comer2022_mo2_adsorption/adsorption_energies.csv
(all 33 metals x {110,100} x {O,OH} -- see metadata.json there for citation/caveats,
notably the constrained-magnetism assumption that this repo's spin-less MLIPs can't match).
Structures come from data/structures/ (fetch missing ones with scripts/core/fetch_rutiles.py
--formulas <FORMULA>); an oxide with no local structure is skipped, not fatal.

    python scripts/ClaudeScripts/mo2_adsorption_benchmark.py runs/mo2_pilot \\
        --oxides TiO2 CrO2 --facets 110 100 --adsorbates O OH --models Orb-v2

Writes job.json (facets, adsorbates, standoff, fmax, models, oxides) into outdir for
provenance, same convention as o_adsorption_benchmark.py. Use a fresh outdir per sweep:
results.jsonl is append-only.

``--desorb-check-step`` defaults to 200, not o_adsorption_benchmark.py's 100: a 12-way
orientation/standoff sweep on TiO2(110)/OH showed every variant converging to the *same*
final bond length and energy regardless of starting geometry, but the ones flagged
"desorbing" all did so at exactly step 100 while the rest settled cleanly by step 116-255
(well under the 300-step budget) -- a false positive from FIRE's approach dynamics
wobbling outward briefly around that checkpoint, not real desorption. O* alone never
needed more than ~60 steps in this same sweep, so the tighter check stays fine there;
OH*'s extra internal degree of freedom (the O-H bond can librate while approaching) is
what needed the room.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from o_adsorption_benchmark import (  # noqa: E402
    DESORB_TOL, DESORB_TREND_WINDOW, EXTEND_STEPS, MAX_EXTENSIONS,
    TRIVIAL_START_TOL, undercoordinated_metal_site,
)

DESORB_CHECK_STEP = 200  # see module docstring -- o_adsorption_benchmark.py's 100 is too
# tight for OH*'s extra librational d.o.f.; O* alone never needed past ~60 steps here anyway.

from oxide_workflow import pipeline  # noqa: E402
from oxide_workflow.backends import get_backend, relax  # noqa: E402
from oxide_workflow.config import ADSORBATE_FRAGMENTS, RunConfig  # noqa: E402
from oxide_workflow.energetics import adsorption_energy, gas_reference_energy  # noqa: E402
from oxide_workflow.pipeline import _adsorbate_anchor_distance, _cli_miller  # noqa: E402
from oxide_workflow.stages import adsorbate_candidates, make_slab  # noqa: E402
from oxide_workflow.structures import STRUCTURE_DIR, get_structure  # noqa: E402

# OH seed orientation is facet-dependent, not universal -- a 12-way tilt/standoff sweep on
# each facet's already-relaxed TiO2 slab showed the (110) Ti5c site's approach dynamics are
# oscillatory at 0deg/40deg tilt (tripping the desorb check even at 2x patience) but clean
# at 20deg/60deg, while the (100) Ti5c site is the opposite: every non-zero tilt desorbs for
# real at the production standoff (0.5 A) -- its nearest neighbor ends up a surface O, not
# the nominal Ti anchor, so a lateral tilt walks the H into that O rather than away from it.
# Vertical is the only orientation that converged cleanly across all 3 standoffs on (100).
OH_COORDS_BY_FACET = {
    "110": ADSORBATE_FRAGMENTS["OH"][1],  # 20deg tilt, config.py's default
    "100": ((0.0, 0.0, 0.0), (0.0, 0.0, 0.9573)),  # vertical
}

SEED_STANDOFF = 0.5  # validated on TiO2(110)/Ti5c -- see o_adsorption_benchmark.py docstring
# repo root is two levels up from scripts/ClaudeScripts/, not one -- this script isn't at
# the top of scripts/ anymore.
LIT_CSV = Path(__file__).resolve().parent.parent.parent / "data" / "literature" / \
    "comer2022_mo2_adsorption" / "adsorption_energies.csv"


def load_literature() -> dict[tuple[str, str, str], dict]:
    """(oxide formula, facet, adsorbate) -> {delta_E_ads_eV, delta_G_ads_eV}."""
    with LIT_CSV.open() as f:
        rows = list(csv.DictReader(f))
    return {
        (r["oxide"], r["facet"], r["adsorbate"]): {
            "delta_E_ads_eV": float(r["delta_E_ads_eV"]),
            "delta_G_ads_eV": float(r["delta_G_ads_eV"]),
        }
        for r in rows
    }


def mp_id_for_formula(formula: str) -> str | None:
    """Resolve an oxide formula (e.g. "FeO2") to something get_structure() accepts.

    Prefers data/structures/O_OH_AdsStructures/<formula>/<formula>.cif -- Comer et al.'s own
    DFT+U-relaxed cell, full-fidelity to the literature reference values, per the decision to
    use their structures for all 33 oxides rather than mixing sources. Falls back to the
    data/structures/<mp-id>_<formula>/ Materials Project convention if that's not there.
    get_structure() accepts a raw path directly (see oxide_workflow/structures.py's own
    docstring), so returning the .cif path works exactly like returning an mp-id string.
    """
    comer_cif = STRUCTURE_DIR / "O_OH_AdsStructures" / formula / f"{formula}.cif"
    if comer_cif.exists():
        return str(comer_cif)
    matches = sorted(STRUCTURE_DIR.glob(f"*_{formula}"))
    return matches[0].name.split("_")[0] if matches else None


def recenter_on_anchor(structure, anchor_idx: int):
    """Shift the whole cell laterally (a,b only -- z stays as SlabConfig.center_slab already
    set it) so the given site sits at (0.5, 0.5), then wraps everyone back into [0,1).

    Purely cosmetic: a constant translation of a periodic cell changes no distance, no
    energy, no force -- the physics is identical either way. What it fixes is *rendering*:
    the site-finder has no reason to prefer cell-interior sites, and a FIRE trajectory can
    drift an atom's fractional coordinate past 1.0 without ever wrapping it back, so an
    adsorbate ontop an edge site can end up looking chopped off or floating outside the box
    in a plain structure viewer (confirmed on TiO2(100): the OH anchor's real Ti neighbor
    was one full cell over, `image=(0,1,0)`, invisible to a viewer that doesn't draw across
    periodic boundaries). Recentering on the adsorbate's own anchor site before writing
    input.vasp/relaxed.vasp means every future POSCAR opens looking like what it is.
    """
    a, b, _ = structure[anchor_idx].frac_coords
    structure.translate_sites(range(len(structure)), (0.5 - a, 0.5 - b, 0.0),
                               frac_coords=True, to_unit_cell=True)
    return structure


def run_one(model: str, oxide: str, facet: str, adsorbate: str, mp_id: str,
            outdir: Path, cfg: RunConfig, lit: dict, desorb_check_step: int) -> dict:
    backend = get_backend(model)
    odir = outdir / oxide / facet

    def relax_record(structure, stage, source_desc, relax_cell=False):
        out = pipeline._relax_record(
            structure, backend, stage=stage, protocol="reference",
            geometry_source=source_desc, cfg=cfg, outdir=odir,
            relax_cell=relax_cell, canonical=True,
        )
        print(f"    {stage:14s} E={out.energy:.4f} eV  {out.header['nsteps']} steps  "
              f"{out.elapsed_s:.0f}s", flush=True)
        return out

    print(f"  [{model} / {oxide} / {facet} / {adsorbate}]", flush=True)
    start = get_structure(mp_id)
    bulk = relax_record(start, "bulk", "db", relax_cell=True).structure
    slab_in = make_slab(bulk, cfg.slab)
    slab_out = relax_record(slab_in, "slab", "cut_from_relaxed_bulk")
    pristine_energy = slab_out.energy
    pristine_structure = slab_out.structure

    ads_species, ads_coords = ADSORBATE_FRAGMENTS[adsorbate]
    n_ads = len(ads_species)
    h2_species, h2_coords = ADSORBATE_FRAGMENTS["H2"]
    h2o_species, h2o_coords = ADSORBATE_FRAGMENTS["H2O"]
    e_h2 = gas_reference_energy(backend, cfg, pipeline.relax, species=h2_species, coords=h2_coords)
    e_h2o = gas_reference_energy(backend, cfg, pipeline.relax, species=h2o_species, coords=h2o_coords)
    # Comer et al. Table S3: O* against full H2 splitting, HO* against half (one H stays put).
    e_ads_ref = e_h2o - e_h2 if adsorbate == "O" else e_h2o - 0.5 * e_h2

    candidates = adsorbate_candidates(
        pristine_structure, cfg.adsorbate, freeze_bottom_fraction=cfg.slab.freeze_bottom_fraction,
    )
    lit_row = lit.get((oxide, facet, adsorbate))
    row = {
        "model": model, "oxide": oxide, "facet": facet, "adsorbate": adsorbate,
        "lit_e_ads_eV": lit_row["delta_E_ads_eV"] if lit_row else None,
        "lit_g_ads_eV": lit_row["delta_G_ads_eV"] if lit_row else None,
    }
    if not candidates:
        row.update({
            "site": None, "failed": True, "error": "no adsorbate candidates for this facet",
            "e_ads_eV": None, "bond_A": None, "desorbing": None, "converged": None, "nsteps": None,
        })
        print(f"    NO SITES for facet {facet}", flush=True)
        return row

    cand = undercoordinated_metal_site(candidates)
    site_dir = odir / model / adsorbate / f"site{cand.site_id['site_index']}_{cand.site_id['symmetry_class']}"
    row["site"] = cand.site_id["site_label"]
    recenter_on_anchor(cand.structure, len(cand.structure) - n_ads)

    try:
        res = relax(
            cand.structure, backend, workdir=site_dir,
            fmax=cfg.relax.fmax, max_steps=cfg.relax.max_steps, optimizer=cfg.relax.optimizer,
            desorb_check_n_ads=n_ads, desorb_check_step=desorb_check_step,
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
    lit_str = f"{row['lit_e_ads_eV']:+.2f}" if row["lit_e_ads_eV"] is not None else "n/a"
    print(f"    E_ads={e_ads:+.4f} eV (lit {lit_str})  bond={end_dist:.3f} A  {status}  "
          f"nsteps={res.nsteps}", flush=True)
    return row


def main(
    outdir: Path, fmax: float, oxides: list[str], facets: list[str], adsorbates: list[str],
    models: list[str], seed_standoff: float, desorb_check_step: int,
) -> None:
    lit = load_literature()
    outdir.mkdir(parents=True, exist_ok=True)
    results_path = outdir / "results.jsonl"

    resolved: dict[str, str | None] = {ox: mp_id_for_formula(ox) for ox in oxides}
    missing = [ox for ox, mp_id in resolved.items() if mp_id is None]
    if missing:
        print(f"skipping (no local structure, fetch with scripts/core/fetch_rutiles.py "
              f"--formulas {' '.join(missing)}): {', '.join(missing)}\n", flush=True)

    job_path = outdir / "job.json"
    job_path.write_text(json.dumps({
        "oxides": oxides, "resolved_mp_ids": resolved, "facets": facets,
        "adsorbates": adsorbates, "models": models, "seed_standoff": seed_standoff, "fmax": fmax,
        "desorb_check_step": desorb_check_step,
        "e_ads_reference": "Comer et al. 2022 Table S3 (H2O/H2 water-splitting, half-H2 for OH*)",
        "literature_source": str(LIT_CSV),
    }, indent=2))

    pair_failures: list[str] = []
    for model in models:
        cfg = RunConfig()
        cfg = replace(cfg, adsorbate=replace(
            cfg.adsorbate, positions=("ontop",), max_per_position=None,
            seed_standoff=seed_standoff,
        ))
        cfg = replace(cfg, relax=replace(cfg.relax, fmax=fmax))
        for oxide in oxides:
            mp_id = resolved[oxide]
            if mp_id is None:
                continue
            for facet in facets:
                fcfg = replace(cfg, slab=replace(cfg.slab, miller_index=_cli_miller(facet)))
                for adsorbate in adsorbates:
                    species, coords = ADSORBATE_FRAGMENTS[adsorbate]
                    if adsorbate == "OH":
                        coords = OH_COORDS_BY_FACET.get(facet, coords)
                    acfg = replace(fcfg, adsorbate=replace(
                        fcfg.adsorbate, species=species, coords=coords,
                    ))
                    tag = f"{model}/{oxide}/{facet}/{adsorbate}"
                    try:
                        row = run_one(
                            model, oxide, facet, adsorbate, mp_id, outdir, acfg, lit,
                            desorb_check_step,
                        )
                    except Exception as e:
                        print(f"  [{tag}] PAIR FAILED before relax: {e}", flush=True)
                        pair_failures.append(f"{tag}: {str(e)[:300]}")
                        row = {
                            "model": model, "oxide": oxide, "facet": facet, "adsorbate": adsorbate,
                            "site": None, "failed": True, "error": f"pair-level: {e}"[:500],
                            "e_ads_eV": None, "bond_A": None, "desorbing": None,
                            "converged": None, "nsteps": None,
                        }
                    with results_path.open("a") as f:
                        f.write(json.dumps(row) + "\n")

    print(f"\nwrote {results_path}")
    if pair_failures:
        print(f"\n{len(pair_failures)} pair(s) failed before relaxing:")
        for p in pair_failures:
            print(f"  {p}")


if __name__ == "__main__":
    all_oxides = sorted({r[0] for r in load_literature()})
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("outdir", type=Path)
    ap.add_argument("--fmax", type=float, default=0.02)
    ap.add_argument("--oxides", nargs="+", choices=all_oxides, default=all_oxides)
    ap.add_argument("--facets", nargs="+", choices=["110", "100"], default=["110", "100"])
    ap.add_argument("--adsorbates", nargs="+", choices=["O", "OH"], default=["O", "OH"])
    ap.add_argument("--models", nargs="+", default=["Orb-v2"])
    ap.add_argument("--seed-standoff", type=float, default=SEED_STANDOFF)
    ap.add_argument("--desorb-check-step", type=int, default=DESORB_CHECK_STEP)
    a = ap.parse_args()
    main(
        a.outdir, a.fmax, a.oxides, a.facets, a.adsorbates, a.models, a.seed_standoff,
        a.desorb_check_step,
    )
