"""Relax the configured adsorbate on the Ti5c ontop site — the fast config feedback loop.

Skips bulk/slab/vacancy by loading an already-relaxed vacancy slab off disk, so one model
is ~1-2 min instead of hours. Every placement knob comes from RunConfig, so edit
oxide_workflow/config.py, rerun, and watch the start distance and E_ads move.

    python scripts/probe_ti5c.py                          # RunConfig reference + candidate
    python scripts/probe_ti5c.py MACE-mh1-omat MACE-mh1-oc20
    python scripts/probe_ti5c.py --adsorbate CO

Reference energies (bare slab, gas-phase molecule) are relaxed once per model+fragment and
cached in the outdir, so reruns after a config tweak only pay for the adsorbate relaxation.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
from pymatgen.analysis.adsorption import AdsorbateSiteFinder
from pymatgen.core import Lattice, Molecule, Structure

from oxide_workflow.backends import get_backend, relax
from oxide_workflow.config import ADSORBATE_FRAGMENTS, RunConfig
from oxide_workflow.stages import _placement_z, apply_bottom_freeze, exposed_surface_atoms

VACANCY = Path("runs/COtest_july29/MACE-mh1-omat/vacancy/site184_8d/CONTCAR")


def find_ti5c_ontop(slab: Structure, cfg, exposed: set[int]) -> np.ndarray:
    """The ontop site sitting over a five-coordinate surface Ti.

    Ti5c is the reactive site on rutile (110) — a surface Ti missing one of its six bulk
    O neighbours, so it has a dangling coordination an adsorbate can fill. pymatgen's site
    finder is purely geometric and does not distinguish it from the saturated Ti6c, hence
    the explicit O-coordination count here.
    """
    ztop = max(s.coords[2] for s in slab)
    cn = {
        i: len([nb for nb in slab.get_neighbors(s, 2.6) if nb.specie.symbol == "O"])
        for i, s in enumerate(slab)
        if s.specie.symbol == "Ti"
    }
    ti5c = [i for i, c in cn.items() if c == 5 and ztop - slab[i].coords[2] < 3.5]
    if not ti5c:
        raise SystemExit("no Ti5c found — is this a rutile (110) vacancy slab?")

    props = ["surface" if i in exposed else "subsurface" for i in range(len(slab))]
    asf = AdsorbateSiteFinder(slab.copy(site_properties={"surface_properties": props}))
    sites = asf.find_adsorption_sites(symm_reduce=cfg.symm_reduce, positions=["ontop"])

    L = slab.lattice

    def lateral(coord) -> float:
        best = 1e9
        for j in ti5c:
            f = L.get_fractional_coords(slab[j].coords - coord)
            f[2] = 0.0
            f -= np.round(f)
            best = min(best, float(np.linalg.norm(f @ L.matrix)))
        return best

    over_ti5c = [c for c in sites["ontop"] if lateral(c) < 0.6]
    if not over_ti5c:
        raise SystemExit("site finder returned no ontop site above a Ti5c")
    return over_ti5c[0], asf


def _relax(structure, backend, cfg, workdir, relax_cell=False):
    return relax(
        structure, backend, workdir=workdir, relax_cell=relax_cell,
        fmax=cfg.relax.fmax, max_steps=cfg.relax.max_steps, optimizer=cfg.relax.optimizer,
    )


def reference_energies(model: str, slab: Structure, cfg, outdir: Path) -> tuple[float, float]:
    """(bare slab energy, gas-phase fragment energy) for this model — cached on disk.

    E_ads needs both, and neither depends on the placement knobs being tuned, so they are
    computed once and reused across config edits.
    """
    frag = "".join(cfg.adsorbate.species)
    # fmax is in the key: E_ads mixes three energies, so a reference converged to a looser
    # tolerance than the adsorbate run would leak that difference straight into E_ads.
    cache = outdir / f"refs_{model}_{frag}_fmax{cfg.relax.fmax}.json"
    if cache.exists():
        d = json.loads(cache.read_text())
        return d["e_slab"], d["e_gas"]

    backend = get_backend(model)
    print(f"  [{model}] caching reference energies (bare slab + gas {frag})...", flush=True)
    e_slab = _relax(slab, backend, cfg, outdir / f"_bare_{model}").energy

    box = Structure(
        Lattice.cubic(15.0), list(cfg.adsorbate.species),
        [np.array(c) / 15.0 + 0.5 for c in cfg.adsorbate.coords], coords_are_cartesian=False,
    )
    e_gas = _relax(box, backend, cfg, outdir / f"_gas_{model}_{frag}").energy

    cache.write_text(json.dumps({"e_slab": e_slab, "e_gas": e_gas}, indent=2))
    return e_slab, e_gas


def main(models: list[str], adsorbate: str | None, vacancy: Path, outdir: Path) -> None:
    cfg = RunConfig()
    if adsorbate:
        species, coords = ADSORBATE_FRAGMENTS[adsorbate]
        cfg = replace(cfg, adsorbate=replace(cfg.adsorbate, species=species, coords=coords))
    acfg = cfg.adsorbate
    frag = "".join(acfg.species)
    outdir.mkdir(parents=True, exist_ok=True)

    slab = Structure.from_file(str(vacancy))
    print(f"slab      {vacancy} ({len(slab)} atoms)")
    print(f"adsorbate {frag}  species={acfg.species}")
    print(f"placement seed_standoff={acfg.seed_standoff} min_normal_height={acfg.min_normal_height} "
          f"min_clearance={acfg.min_clearance}")
    print(f"relax     fmax={cfg.relax.fmax} max_steps={cfg.relax.max_steps} {cfg.relax.optimizer}")

    exposed = exposed_surface_atoms(slab, acfg.surface_depth, acfg.exposure_block_radius)
    target, asf = find_ti5c_ontop(slab, acfg, exposed)
    z, _ = _placement_z(slab, target, exposed, acfg.species[0], acfg)

    mol = Molecule(list(acfg.species), [list(c) for c in acfg.coords])
    placed = asf.add_adsorbate(mol, np.array([target[0], target[1], z]))
    start = Structure(placed.lattice, placed.species, placed.frac_coords)
    n_ads = len(acfg.species)
    bind = len(start) - n_ads  # binding atom: first of the appended fragment
    apply_bottom_freeze(
        start, cfg.slab.freeze_bottom_fraction,
        always_free=set(range(len(start) - n_ads, len(start))),
    )

    def ti_dist(s: Structure) -> float:
        return min(
            s.get_distance(bind, j)
            for j, site in enumerate(s[:bind])
            if site.specie.symbol == "Ti"
        )

    d0 = ti_dist(start)
    print(f"\nTi5c ontop xy = {np.round(target[:2], 2)}   start Ti-{acfg.species[0]} = {d0:.2f} A\n")

    for model in models:
        e_slab, e_gas = reference_energies(model, slab, cfg, outdir)
        res = _relax(start, get_backend(model), cfg, outdir / f"{frag}_{model}")
        final = res.structure
        e_ads = res.energy - e_slab - e_gas
        extra = ""
        if n_ads > 1:
            extra = f"  {acfg.species[0]}-{acfg.species[1]}={final.get_distance(bind, bind + 1):.2f}"
        print(
            f"{model:18s} Ti-{acfg.species[0]} {d0:.2f}->{ti_dist(final):.2f} A  "
            f"E={res.energy:.4f}  E_ads={e_ads:+.3f} eV{extra}  "
            f"({res.nsteps} steps, {res.meta['elapsed_s']:.0f} s)",
            flush=True,
        )


if __name__ == "__main__":
    cfg = RunConfig()
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("models", nargs="*", default=[cfg.reference, cfg.candidate],
                    help="models to relax with (default: RunConfig reference + candidate)")
    ap.add_argument("--adsorbate", choices=sorted(ADSORBATE_FRAGMENTS),
                    help="override the fragment pinned in AdsorbateConfig")
    ap.add_argument("--vacancy", type=Path, default=VACANCY, help="relaxed vacancy slab CONTCAR")
    ap.add_argument("--outdir", type=Path, default=Path("runs/probe_ti5c"))
    args = ap.parse_args()
    main(args.models or [cfg.reference, cfg.candidate], args.adsorbate, args.vacancy, args.outdir)
