"""Run ONE stage of the chain, on your choice of starting structure — the play-around tool.

The full pipeline always runs bulk -> slab -> vacancy -> adsorbate. This runs a single
stage, either chaining from the configured material or starting from a structure file you
already have, so you can change one knob and see its effect in minutes instead of hours.

    # quick bulk relax
    python scripts/run_stage.py bulk

    # quick slab relax, thicker and with 80% of the bottom frozen
    python scripts/run_stage.py slab --thickness 20 --freeze 0.8

    # a different facet, and a different termination of it
    python scripts/run_stage.py slab --miller 101
    python scripts/run_stage.py slab --termination 0

    # vacancy screening on a slab that is ALREADY relaxed
    python scripts/run_stage.py vacancy --from runs/practice/slab/CONTCAR

    # adsorbate screening on an already-relaxed vacancy slab
    python scripts/run_stage.py adsorbate --from runs/xyz/vacancy/site4_8d/CONTCAR --adsorbate O2

``--from`` accepts any CIF/POSCAR/CONTCAR, and skips every stage before the one requested.
Without it, the stage chains from ``RunConfig.polymorph``, relaxing whatever it needs
first (so ``slab`` relaxes the bulk first, ``vacancy`` relaxes bulk then slab, and so on).

Output uses the same tree as a full run — POSCAR/CONTCAR/trajectory.xyz/OUTCAR per
relaxation, plus ``rankings.csv`` for the screening stages — so anything that reads a run
directory reads this too.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from pymatgen.core import Structure

from oxide_workflow import pipeline
from oxide_workflow.backends import REGISTRY, get_backend
from oxide_workflow.config import ADSORBATE_FRAGMENTS, RunConfig
from oxide_workflow.energetics import gas_reference_energy
from oxide_workflow.stages import adsorbate_candidates, make_slab, oxygen_vacancy_candidates
from oxide_workflow.structures import get_structure

STAGES = ("bulk", "slab", "vacancy", "adsorbate")


def _load_start(source: str | None, cfg: RunConfig) -> tuple[Structure, str]:
    """(structure, description) for the starting geometry."""
    if source:
        p = Path(source)
        if not p.exists():
            raise SystemExit(f"no such structure file: {p}")
        return Structure.from_file(str(p)), f"file {p}"
    return get_structure(cfg.polymorph), f"material {cfg.polymorph}"


def main(a) -> None:
    cfg = RunConfig()
    if a.miller:
        cfg = replace(cfg, slab=replace(cfg.slab, miller_index=pipeline._cli_miller(a.miller)))
    slab_over = {}
    if a.termination is not None:
        slab_over["termination_index"] = a.termination
    if a.thickness is not None:
        slab_over["min_slab_size"] = a.thickness
    if a.vacuum is not None:
        slab_over["min_vacuum_size"] = a.vacuum
    if a.freeze is not None:
        slab_over["freeze_bottom_fraction"] = a.freeze
    if a.supercell:
        nx, ny = (int(v) for v in a.supercell.split(","))
        slab_over["supercell"] = (nx, ny)
    if a.max_vacancy_sites:
        slab_over["max_vacancy_sites"] = a.max_vacancy_sites
    if slab_over:
        cfg = replace(cfg, slab=replace(cfg.slab, **slab_over))

    cfg = replace(cfg, relax=replace(cfg.relax, fmax=a.fmax, max_steps=a.max_steps))
    if a.adsorbate:
        species, coords = ADSORBATE_FRAGMENTS[a.adsorbate]
        cfg = replace(cfg, adsorbate=replace(cfg.adsorbate, species=species, coords=coords))
    if a.max_sites:
        cfg = replace(cfg, adsorbate=replace(cfg.adsorbate, max_per_position=a.max_sites))
    if a.material:
        cfg = replace(cfg, polymorph=a.material)

    model = a.model or cfg.reference
    backend = get_backend(model)
    outdir = Path(a.outdir)
    start, described = _load_start(a.source, cfg)

    print(f"stage      {a.stage}")
    print(f"start      {described}  ({len(start)} atoms)")
    print(f"model      {model}  (env {backend.env}, {backend.device})")
    print(f"slab       facet {''.join(map(str, cfg.slab.miller_index))}  "
          f"termination {cfg.slab.termination_index}  {cfg.slab.min_slab_size} A thick  "
          f"vacuum {cfg.slab.min_vacuum_size} A")
    print(f"           supercell {cfg.slab.supercell}  freeze {cfg.slab.freeze_bottom_fraction:.0%} "
          f"of the bottom")
    if a.stage == "adsorbate":
        print(f"adsorbate  {pipeline._fragment_label(cfg.adsorbate)}  "
              f"cap {cfg.adsorbate.max_per_position or 'none'} per position type")
    print(f"relax      fmax={cfg.relax.fmax} max_steps={cfg.relax.max_steps} "
          f"{cfg.relax.optimizer}")
    print(f"outdir     {outdir}\n")

    frozen = cfg.slab.freeze_bottom_fraction
    want = STAGES.index(a.stage)
    # --from supplies the *previous* stage's relaxed output, so every earlier stage is
    # skipped and we enter the loop at the requested stage. Without it we start at bulk and
    # relax forward. One index does the whole thing.
    first = want if a.source else 0
    current = start
    substrate_energy: float | None = None

    outdir.mkdir(parents=True, exist_ok=True)
    cand_table = outdir / "candidates.jsonl"
    if cand_table.exists():
        cand_table.unlink()

    def _relax_one(structure, stage, source_desc, relax_cell=False):
        out = pipeline._relax_record(
            structure, backend, stage=stage, protocol="reference",
            geometry_source=source_desc, cfg=cfg, outdir=outdir,
            relax_cell=relax_cell, canonical=True,
        )
        print(f"  {stage:14s} E={out.energy:.4f} eV  fmax {out.start_fmax:.3f}->"
              f"{out.header['fmax']:.3f}  {out.header['nsteps']} steps  "
              f"{out.elapsed_s:.0f}s", flush=True)
        return out

    for idx in range(first, want + 1):
        stage = STAGES[idx]

        if stage == "bulk":
            current = _relax_one(current, "bulk", "db", relax_cell=True).structure

        elif stage == "slab":
            slab_in = make_slab(current, cfg.slab)
            print(f"  slab built     {len(slab_in)} atoms")
            current = _relax_one(slab_in, "slab", "cut_from_relaxed_bulk").structure

        elif stage == "vacancy":
            vacs = oxygen_vacancy_candidates(
                current, freeze_bottom_fraction=frozen, max_sites=cfg.slab.max_vacancy_sites
            )
            print(f"  {len(vacs)} symmetry-distinct O vacancy site(s)")
            out = pipeline._run_funnel(
                vacs, backend, stage="vacancy", protocol="reference",
                geometry_source="from_relaxed_slab", cfg=cfg, outdir=outdir,
                candidates_table=cand_table,
            )
            best = min(out, key=lambda si: out[si].energy)
            print(f"  best vacancy   site{best} ({out[best].site_id['symmetry_class']})  "
                  f"E={out[best].energy:.4f} eV")
            print(f"  rankings       {outdir / backend.name / 'vacancy' / 'rankings.csv'}")
            current = out[best].structure
            # The winner's energy IS this model's bare-substrate energy for the next stage.
            substrate_energy = out[best].energy

        elif stage == "adsorbate":
            if substrate_energy is None:
                # Entered directly via --from: the geometry came off disk with no energy
                # attached. Relax it once with THIS model — borrowing another model's
                # energy would leave the per-atom offset uncancelled and swamp E_ads.
                bare = _relax_one(current, "bare_substrate", "bare_reference_state")
                substrate_energy = bare.energy
            e_gas = gas_reference_energy(backend, cfg, pipeline.relax)
            ads = adsorbate_candidates(current, cfg.adsorbate, freeze_bottom_fraction=frozen)
            print(f"  {len(ads)} adsorption site(s)")
            out = pipeline._run_funnel(
                ads, backend, stage="adsorbate", protocol="reference",
                geometry_source="placed_on_relaxed_substrate", cfg=cfg, outdir=outdir,
                candidates_table=cand_table,
                e_ads_reference=(substrate_energy, e_gas),
            )
            best = min(out, key=lambda si: out[si].energy)
            print(f"\n  best site      site{best} ({out[best].site_id['symmetry_class']})")
            print(f"  min E_ads      {out[best].e_ads:+.4f} eV")
            print(f"  rankings       {outdir / backend.name / 'adsorbate' / 'rankings.csv'}")

    print(f"\ndone -> {outdir}")


if __name__ == "__main__":
    base = RunConfig()
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="every knob below overrides the matching value in oxide_workflow/config.py",
    )
    ap.add_argument("stage", choices=STAGES)
    ap.add_argument("--from", dest="source", metavar="FILE",
                    help="start from this CIF/POSCAR/CONTCAR instead of chaining from the "
                         "material; skips every earlier stage")
    ap.add_argument("--material", help=f"mp-id, alias, or path (default {base.polymorph})")
    ap.add_argument("--model", choices=sorted(REGISTRY), help=f"default {base.reference}")

    g = ap.add_argument_group("slab geometry")
    g.add_argument("--miller", help='facet as "110" or "1,1,-1"')
    g.add_argument("--termination", type=int,
                   help=f"which termination of the facet (default {base.slab.termination_index})")
    g.add_argument("--thickness", type=float, metavar="A",
                   help=f"min slab thickness in A (default {base.slab.min_slab_size})")
    g.add_argument("--vacuum", type=float, metavar="A",
                   help=f"min vacuum gap in A (default {base.slab.min_vacuum_size})")
    g.add_argument("--freeze", type=float, metavar="FRAC",
                   help=f"fraction of slab thickness held at bulk positions, 0-1 "
                        f"(default {base.slab.freeze_bottom_fraction})")
    g.add_argument("--supercell", help=f'lateral replication "nx,ny" '
                                       f'(default {base.slab.supercell[0]},{base.slab.supercell[1]})')

    s = ap.add_argument_group("screening")
    s.add_argument("--adsorbate", choices=sorted(ADSORBATE_FRAGMENTS))
    s.add_argument("--max-sites", type=int, help="adsorbate sites per position type")
    s.add_argument("--max-vacancy-sites", type=int, help="cap the vacancy funnel")

    r = ap.add_argument_group("relaxation")
    r.add_argument("--fmax", type=float, default=base.relax.fmax)
    r.add_argument("--max-steps", type=int, default=base.relax.max_steps)

    ap.add_argument("--outdir", default="runs/stage")
    main(ap.parse_args())
