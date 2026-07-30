"""Full bulk -> slab -> vacancy -> adsorbate benchmark, with the adsorbate pinned to Ti5c.

The adsorbate funnel normally relaxes every symmetry-distinct site and keeps the minimum,
which is ~85% of a full run's wall time. Here it is restricted to the single ontop site
above a five-coordinate surface Ti — the literature binding site on rutile (110) — so the
whole chain finishes in a demo-sized window. Everything else (bulk, slab, the vacancy
funnel, both seeded and full-pipeline protocols, the divergence table) is unchanged.

    python scripts/demo_pipeline_ti5c.py                       # RunConfig's adsorbate
    python scripts/demo_pipeline_ti5c.py --adsorbate CO
    python scripts/demo_pipeline_ti5c.py --adsorbate H2O --candidates MACE-mh1-oc20

The vacancy funnel is the remaining cost (~4 min/site/model). Trim it with
``--max-vacancy-sites`` if the demo window is tight.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from oxide_workflow import pipeline
from oxide_workflow.config import ADSORBATE_FRAGMENTS, RunConfig
from oxide_workflow.stages import adsorbate_candidates, oxygen_vacancy_candidates

TI5C_LATERAL_TOL = 0.6  # A; how close an ontop site must sit above a Ti5c to count


def _ti5c_indices(substrate) -> list[int]:
    """Surface Ti atoms with only five O neighbours — the coordinatively unsaturated site.

    A bulk Ti in rutile has six O neighbours; the (110) surface cuts one away. That missing
    neighbour is what an adsorbate coordinates to, so this is the site the demo pins.
    """
    ztop = max(s.coords[2] for s in substrate)
    return [
        i
        for i, s in enumerate(substrate)
        if s.specie.symbol == "Ti"
        and ztop - s.coords[2] < 3.5
        and len([nb for nb in substrate.get_neighbors(s, 2.6) if nb.specie.symbol == "O"]) == 5
    ]


def ti5c_only_candidates(substrate, config, freeze_bottom_fraction: float = 0.0):
    """``adsorbate_candidates`` restricted to the ontop site above a Ti5c.

    Drop-in for the real generator: the pipeline only needs objects exposing ``.structure``
    and ``.site_id``, and re-indexing the survivor to ``site_index=0`` keeps the seeded
    protocol's per-site matching (which pairs reference and candidate by index) working.
    """
    cands = adsorbate_candidates(
        substrate, replace(config, positions=("ontop",)), freeze_bottom_fraction
    )
    ti5c = _ti5c_indices(substrate)
    if not ti5c:
        raise SystemExit("no Ti5c on this substrate — is it rutile (110)?")

    L = substrate.lattice

    def lateral(cand) -> float:
        coord = L.get_cartesian_coords(cand.site_id["frac_coord"])
        best = 1e9
        for j in ti5c:
            f = L.get_fractional_coords(substrate[j].coords - coord)
            f[2] = 0.0
            f -= np.round(f)
            best = min(best, float(np.linalg.norm(f @ L.matrix)))
        return best

    best = min(cands, key=lateral)
    if lateral(best) > TI5C_LATERAL_TOL:
        raise SystemExit(f"nearest ontop site is {lateral(best):.2f} A from any Ti5c")
    best.site_id["site_index"] = 0
    best.site_id["symmetry_class"] = "ontop_ti5c"
    return [best]


def _capped_vacancies(limit: int):
    def gen(slab, symprec: float = 0.1, freeze_bottom_fraction: float = 0.0):
        return oxygen_vacancy_candidates(slab, symprec, freeze_bottom_fraction)[:limit]

    return gen


def _report(outdir: Path) -> None:
    table = outdir / "divergence.jsonl"
    if not table.exists():
        return
    print(f"\n{'protocol':15s}{'stage':11s}{'rmsd (A)':>11s}{'max disp':>11s}{'dE (eV)':>11s}")
    for line in table.read_text().splitlines():
        r = json.loads(line)
        fmt = lambda v: f"{v:11.4f}" if isinstance(v, (int, float)) else f"{'-':>11s}"
        print(
            f"{r['protocol']:15s}{r['stage']:11s}"
            f"{fmt(r['rmsd'])}{fmt(r['max_displacement'])}{fmt(r['energy_error'])}"
        )
    print(f"\nfull tree: {outdir}")


def main(args) -> None:
    cfg = RunConfig()
    if args.material:
        cfg = replace(cfg, polymorph=args.material)
    if args.adsorbate:
        species, coords = ADSORBATE_FRAGMENTS[args.adsorbate]
        cfg = replace(cfg, adsorbate=replace(cfg.adsorbate, species=species, coords=coords))

    # Swap the adsorbate generator the pipeline calls; the pipeline itself is untouched.
    pipeline.adsorbate_candidates = ti5c_only_candidates
    if args.max_vacancy_sites:
        pipeline.oxygen_vacancy_candidates = _capped_vacancies(args.max_vacancy_sites)

    candidates = args.candidates or [cfg.candidate]
    print(f"material   {cfg.polymorph}  facet {''.join(map(str, cfg.slab.miller_index))}")
    print(f"adsorbate  {''.join(cfg.adsorbate.species)}  -> pinned to ontop Ti5c (1 site)")
    print(f"reference  {cfg.reference}")
    print(f"candidates {', '.join(candidates)}")
    print(f"relax      fmax={cfg.relax.fmax} max_steps={cfg.relax.max_steps} "
          f"{cfg.relax.optimizer}\n")

    summary = pipeline.run_sweep(candidates, cfg=cfg, outdir=args.outdir)
    print(f"\ndone in {summary['total_elapsed_s'] / 60:.1f} min")
    _report(Path(args.outdir))


if __name__ == "__main__":
    cfg = RunConfig()
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--adsorbate", choices=sorted(ADSORBATE_FRAGMENTS),
                    help="fragment to place (default: whatever AdsorbateConfig pins)")
    ap.add_argument("--candidates", nargs="+", help=f"models to benchmark (default: {cfg.candidate})")
    ap.add_argument("--material", help="mp-id or alias (default: RunConfig.polymorph)")
    ap.add_argument("--max-vacancy-sites", type=int,
                    help="cap the vacancy funnel — the remaining cost driver")
    ap.add_argument("--outdir", default="runs/demo_ti5c")
    main(ap.parse_args())

'''
python scripts/demo_pipeline_ti5c.py \
    --adsorbate H \
    --candidates MACE-mh1-oc20 \
    --outdir runs/H_oc20_0729 2>&1 | tee runs/H_oc20_0729.log
'''