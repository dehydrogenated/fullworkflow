"""Run one adsorbate across the whole rutile family in a single command.

    python scripts/run_family.py --dry-run          # plan + cost, runs nothing
    python scripts/run_family.py
    python scripts/run_family.py --adsorbate O2 --cap 6 --candidates MACE-mh1-oc20

Restartable by design: a family sweep runs for days, so a material that already has a
``summary.json`` is skipped and a material that raises is recorded in ``batch.json`` and
stepped over. Rerunning the same command retries only what is missing.

Why the sampling cap is on by default
-------------------------------------
Uncapped, ``find_adsorption_sites`` returns a different number of sites per material for
reasons that are not chemical: identical 192-atom slabs with identical 40-atom exposed
surfaces yield 63 sites for TiO2 but 116 for RuO2, purely because pymatgen's symmetry
reduction merges fewer near-equivalent sites on some vacancy substrates. Left alone, that
gives some materials a denser search than others, so they find lower minima for reasons
unrelated to their chemistry — a systematic artifact sitting directly on top of the
quantity being measured. A fixed cap gives every material the same sampling density
(``--cap 6`` -> 18 sites everywhere), which is what makes cross-material E_ads comparable.

The cap is not a plain truncation: ``_spread_sites`` seeds one site per distinct surface
environment, most coordinatively unsaturated first, before filling the rest by
farthest-point spread. See ``--cap 0`` to disable it and sample everything.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from oxide_workflow import pipeline
from oxide_workflow.config import ADSORBATE_FRAGMENTS, RunConfig

# The well-behaved rutile family, as fetched by scripts/fetch_rutiles.py and confirmed by
# scripts/validate_materials.py (termination 1 is stoichiometric, non-polar and symmetric
# for every one). "rutile-tio2" is deliberately absent: it is the same material as mp-2657
# from a hand-written cell, so including it would buy a duplicate run.
RUTILES = {
    "mp-2657": "TiO2",
    "mp-856": "SnO2",
    "mp-470": "GeO2",
    "mp-825": "RuO2",
    "mp-2723": "IrO2",
    "mp-20725": "PbO2",
    "mp-6947": "SiO2",
}

# Wall-clock model, calibrated on this machine from two measured points:
#   192-atom slab, 90 steps -> 292 s   (runs/COtest_july29 adsorbate leaf)
#    24-atom slab, 20 steps ->  12.7 s (direct timing, MACE-mh1-omat on CPU)
# giving t ~= FIXED + PER_ATOM_STEP * n_atoms * steps. A flat per-relaxation constant is
# badly wrong the moment the supercell or step cap changes, which is exactly what a smoke
# test does — hence the explicit atom/step dependence.
FIXED_S = 4.7          # process launch + model load
PER_ATOM_STEP_S = 0.0166
# Relaxations usually converge well before max_steps (COtest averaged ~90), so a raw
# max_steps of 300 would overestimate by ~3x. Cap the estimate at the observed typical.
TYPICAL_STEPS = 90


def _estimate_seconds(n_atoms: int, max_steps: int) -> float:
    return FIXED_S + PER_ATOM_STEP_S * n_atoms * min(max_steps, TYPICAL_STEPS)


def _plan(materials: list[str], cfg: RunConfig, protocols: tuple[str, ...],
          n_candidates: int) -> list[dict]:
    """Per-material relaxation counts, without relaxing anything (same walk as validate)."""
    from oxide_workflow.stages import (
        adsorbate_candidates, make_slab, oxygen_vacancy_candidates,
    )
    from oxide_workflow.structures import get_structure

    rows = []
    for m in materials:
        try:
            slab = make_slab(get_structure(m), cfg.slab)
            vacs = oxygen_vacancy_candidates(
                slab, freeze_bottom_fraction=cfg.slab.freeze_bottom_fraction,
                max_sites=cfg.slab.max_vacancy_sites,
            )
            ads = adsorbate_candidates(
                vacs[0].structure, cfg.adsorbate,
                freeze_bottom_fraction=cfg.slab.freeze_bottom_fraction,
            )
            # reference chain: bulk + slab + vacancies + adsorbates.
            ref = 2 + len(vacs) + len(ads)
            # each candidate: one shared bulk, then per protocol slab + funnels
            # (+1 bare-substrate relaxation for seeded, which needs its own E_ads reference).
            per_proto = 1 + len(vacs) + len(ads)
            cand = 1 + sum(per_proto + (1 if p == "seeded" else 0) for p in protocols)
            total = ref + n_candidates * cand
            rows.append({"material": m, "formula": RUTILES.get(m, "?"),
                         "n_vac": len(vacs), "n_ads": len(ads), "relaxations": total,
                         "n_atoms": len(slab),
                         "seconds": total * _estimate_seconds(len(slab), cfg.relax.max_steps)})
        except Exception as exc:  # noqa: BLE001
            rows.append({"material": m, "formula": RUTILES.get(m, "?"), "error": str(exc)})
    return rows


def _results_table(outdir: Path, summaries: dict) -> None:
    """Cross-material headline: the minimum adsorption energy each model found."""
    print(f"\n{'=' * 78}\nMIN E_ads BY MATERIAL (eV, negative = bound)\n{'=' * 78}")
    print(f"{'material':11s}{'formula':9s}{'reference':>13s}   candidates")
    for m, s in summaries.items():
        if not s:
            continue
        ref = s.get("reference_min_e_ads")
        cands = "  ".join(
            f"{name}: " + ", ".join(f"{p}={v:+.3f}" for p, v in (d.get("min_e_ads") or {}).items())
            for name, d in (s.get("per_candidate") or {}).items()
        )
        ref_s = f"{ref:+.3f}" if isinstance(ref, (int, float)) else "-"
        print(f"{m:11s}{RUTILES.get(m, '?'):9s}{ref_s:>13s}   {cands}")
    print(f"\nper-material detail: {outdir}/<material>/summary.json")
    print(f"site rankings:       {outdir}/<material>/<model>/adsorbate/rankings.csv")


def main(a) -> None:
    materials = a.materials or list(RUTILES)
    protocols = pipeline.PROTOCOLS if a.protocol == "both" else (a.protocol,)
    species, coords = ADSORBATE_FRAGMENTS[a.adsorbate]

    cfg = RunConfig()
    cfg = replace(
        cfg,
        adsorbate=replace(cfg.adsorbate, species=species, coords=coords,
                          max_per_position=a.cap or None),
        relax=replace(cfg.relax, fmax=a.fmax, max_steps=a.max_steps),
    )
    if a.max_vacancy_sites:
        cfg = replace(cfg, slab=replace(cfg.slab, max_vacancy_sites=a.max_vacancy_sites))
    if a.supercell:
        # The dominant cost knob: a 4x2 slab is 192 atoms, a 1x1 is 24, and relaxation
        # cost scales with atom count *per step*. Shrinking it is the only way to make a
        # smoke pass take minutes. It also raises defect/adsorbate coverage to unphysical
        # levels, so a shrunken cell is for plumbing checks only, never for numbers.
        nx, ny = (int(v) for v in a.supercell.split(","))
        cfg = replace(cfg, slab=replace(cfg.slab, supercell=(nx, ny)))

    candidates = a.candidates or [cfg.candidate]
    outdir = Path(a.outdir)

    print(f"adsorbate   {a.adsorbate}   facet {''.join(map(str, cfg.slab.miller_index))}")
    print(f"reference   {cfg.reference}")
    print(f"candidates  {', '.join(candidates)}")
    print(f"protocols   {', '.join(protocols)}")
    print(f"cap         {a.cap or 'none (all sites)'} per position type")
    print(f"relax       fmax={cfg.relax.fmax} max_steps={cfg.relax.max_steps}")
    print(f"outdir      {outdir}\n")

    rows = _plan(materials, cfg, protocols, len(candidates))
    print(f"{'material':11s}{'formula':9s}{'atoms':>6s}{'vac':>5s}{'ads':>5s}"
          f"{'relaxations':>13s}{'est':>9s}")
    total, seconds = 0, 0.0
    for r in rows:
        if "error" in r:
            print(f"{r['material']:11s}{r['formula']:9s}  ERROR {r['error'][:44]}")
            continue
        total += r["relaxations"]
        seconds += r["seconds"]
        est = (f"{r['seconds'] / 60:.0f} min" if r["seconds"] < 5400
               else f"{r['seconds'] / 3600:.1f} h")
        print(f"{r['material']:11s}{r['formula']:9s}{r['n_atoms']:>6d}{r['n_vac']:>5d}"
              f"{r['n_ads']:>5d}{r['relaxations']:>13d}{est:>9s}")
    total_est = f"{seconds / 60:.0f} min" if seconds < 5400 else f"{seconds / 3600:.1f} h"
    print(f"\n{total} relaxations total, estimated {total_est} "
          f"(rough — scales with slab size and step cap)")

    already = [m for m in materials if (outdir / m / "summary.json").exists()]
    if already:
        print(f"\n{len(already)} already complete and will be skipped: {' '.join(already)}")

    if a.dry_run:
        print("\n--dry-run: nothing was run")
        return

    print("\nstarting — rerun this exact command to resume if it stops\n")
    summaries = pipeline.run_batch(
        materials, cfg=cfg, outdir=outdir, protocols=protocols,
        resume=True, continue_on_error=True,
    )

    index = json.loads((outdir / "batch.json").read_text())
    if index.get("failed"):
        print(f"\n{len(index['failed'])} material(s) FAILED:")
        for m, err in index["failed"].items():
            print(f"   {m}: {err}")
        print("rerun the same command to retry only those")
    _results_table(outdir, summaries)


if __name__ == "__main__":
    base = RunConfig()
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("materials", nargs="*", help=f"default: the {len(RUTILES)} rutiles")
    ap.add_argument("--adsorbate", choices=sorted(ADSORBATE_FRAGMENTS), default="O2")
    ap.add_argument("--candidates", nargs="+", help=f"default: {base.candidate}")
    ap.add_argument("--protocol", choices=("full_pipeline", "seeded", "both"),
                    default="full_pipeline")
    ap.add_argument("--cap", type=int, default=6,
                    help="adsorbate sites per position type; 0 = uncapped. Keeps sampling "
                         "density equal across materials (default: 6 -> 18 sites)")
    ap.add_argument("--max-vacancy-sites", type=int, help="cap the vacancy funnel too")
    ap.add_argument("--supercell", help='lateral slab replication as "nx,ny" (default 4,2 '
                                        '= 192 atoms). "1,1" = 24 atoms, for smoke tests only')
    ap.add_argument("--fmax", type=float, default=base.relax.fmax)
    ap.add_argument("--max-steps", type=int, default=base.relax.max_steps)
    ap.add_argument("--outdir", default="runs/rutile_family")
    ap.add_argument("--dry-run", action="store_true", help="print the plan and stop")
    main(ap.parse_args())
