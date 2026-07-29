"""Relax just bulk → slab with one model — the short practice loop.

Skips vacancy + adsorbate, which were ~97% of the last full run (bulk 2 s, slab 129 s vs
vacancy 1434 s, adsorbate 7901 s). Every number comes from RunConfig, so edit
oxide_workflow/config.py, rerun, and watch the printed knobs and atom counts move.

    python scripts/relax_bulk_slab.py                  # RunConfig.reference
    python scripts/relax_bulk_slab.py MACE-mh1-oc20    # any registered model
"""

from __future__ import annotations

import argparse
from pathlib import Path

from oxide_workflow.backends import get_backend, relax
from oxide_workflow.config import RunConfig
from oxide_workflow.stages import make_slab
from oxide_workflow.structures import get_structure


def _relax(structure, backend, cfg, workdir, relax_cell):
    """Mirrors pipeline._relax_record: RelaxConfig has to be forwarded explicitly, or
    relax() falls back to the Backend defaults (max_steps=500, not RelaxConfig's 300)."""
    return relax(
        structure,
        backend,
        workdir=workdir,
        relax_cell=relax_cell,
        fmax=cfg.relax.fmax,
        max_steps=cfg.relax.max_steps,
        optimizer=cfg.relax.optimizer,
    )


def main(model: str | None, outdir: str) -> None:
    cfg = RunConfig()
    backend = get_backend(model or cfg.reference)
    out = Path(outdir)

    print(f"model     {backend.name} (env {backend.env}, {backend.device})")
    print(f"material  {cfg.polymorph}")
    print(f"slab      facet {''.join(map(str, cfg.slab.miller_index))}, "
          f"termination {cfg.slab.termination_index}, supercell {cfg.slab.supercell}, "
          f"{cfg.slab.min_slab_size} A thick")
    print(f"relax     fmax={cfg.relax.fmax} max_steps={cfg.relax.max_steps} "
          f"{cfg.relax.optimizer}")

    bulk_in = get_structure(cfg.polymorph)
    print(f"\nbulk  in  {len(bulk_in)} atoms, a={bulk_in.lattice.a:.4f} A", flush=True)
    bulk = _relax(bulk_in, backend, cfg, out / "bulk", relax_cell=True)
    print(f"bulk  out a={bulk.structure.lattice.a:.4f} A  E={bulk.energy:.4f} eV  "
          f"fmax {bulk.start_fmax:.3f}->{bulk.fmax:.3f}  {bulk.nsteps} steps")

    slab_in = make_slab(bulk.structure, cfg.slab)
    print(f"\nslab  in  {len(slab_in)} atoms", flush=True)
    slab = _relax(slab_in, backend, cfg, out / "slab", relax_cell=False)
    print(f"slab  out E={slab.energy:.4f} eV  fmax {slab.start_fmax:.3f}->{slab.fmax:.3f}  "
          f"{slab.nsteps} steps")

    print(f"\nwrote {out}/bulk and {out}/slab "
          "(input.vasp, relaxed.vasp, trajectory.xyz, opt.log, result.json)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("model", nargs="?", help="model name (default: RunConfig.reference)")
    ap.add_argument("--outdir", default="runs/practice", help="where to write the two leaves")
    args = ap.parse_args()
    main(args.model, args.outdir)
