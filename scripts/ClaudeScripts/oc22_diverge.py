"""Seeded divergence against OC22: relax from the DFT initial geometry, compare endpoints.

    python scripts/oc22_diverge.py                              # all seeds, reference model
    python scripts/oc22_diverge.py --model UMA-oc22 --fmax 0.05
    python scripts/oc22_diverge.py --only IrO2 --dry-run

Run ``scripts/oc22_seed.py`` first to produce the seeds. Runs in the orchestrator env;
the model itself runs in its own conda env via ``backends.relax`` as usual.

PROTOCOL. Matched to OC22 rather than to this repo's defaults, so the divergence
measures the model and not a difference of convention:
  - unconstrained. OC22's ``fixed`` is all-zero; the seeds carry no selective_dynamics,
    so ASE builds no constraints. Equivalent to --freeze 0.
  - fmax 0.05 eV/A, matching their VASP ediffg=0.05, not this repo's default 0.02.
  - cell pinned (relax_cell=False) — OC22 relaxed positions only.

DISPLACEMENTS. Two corrections, both load-bearing:
  - minimum image. Differencing raw cartesian coordinates counts an atom that crossed a
    periodic boundary as having moved a whole cell vector. Measured on IrO2(211)+OH this
    inflated the mean displacement 16x, from 0.170 A to 2.657 A.
  - rigid-body translation. Nothing pins an unconstrained slab, so a uniform shift costs
    no energy and is not divergence. Removed as the mean shift of the SLAB atoms only —
    using all atoms would let the adsorbate's motion define its own reference frame.

DECOMPOSITION. By OC22's own ``tags`` (0 subsurface, 1 surface, 2 adsorbate) rather than
re-deriving regions from z, which would invent a second disagreeing definition of
"surface". This is not cosmetic: on IrO2(211)+OH the aggregate mean is 0.083 A while the
adsorbate alone is 1.009 A — 2 atoms in 64 dilute their own failure 32:1.

VALIDITY. Reconstruction and dissociation are categorical failures, not divergence
values, and are flagged rather than averaged in. A slab that diverges far more with the
adsorbate than without it has found a different minimum: on PbO2(213)+OH the slab moved
0.314 A mean against 0.027 A for the same slab clean, and its energy came out 2.9 eV
below the DFT reference. That number describes a different structure, not a model error.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from pymatgen.core import Structure

from oxide_workflow.backends import get_backend, relax

REPO = Path(__file__).resolve().parent.parent
OH_BOND_MAX = 1.3  # A; beyond this an O-H is dissociated, not stretched
RECONSTRUCTION_SLAB_MEAN = 0.25  # A; slab-mean divergence above this is a different minimum


def min_image_disp(a: np.ndarray, b: np.ndarray, cell: np.ndarray,
                   slab: np.ndarray) -> np.ndarray:
    """Per-atom displacement of ``a`` from ``b``, minimum image, slab translation removed."""
    frac = (a - b) @ np.linalg.inv(cell)
    frac -= np.round(frac)
    cart = frac @ cell
    return cart - cart[slab].mean(axis=0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seeds", default="data/oc22/oh_systems")
    ap.add_argument("--out", default="runs/oc22_divergence")
    ap.add_argument("--model", default="MACE-mh1-omat")
    ap.add_argument("--fmax", type=float, default=0.05, help="matches OC22 ediffg=0.05")
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--only", help="substring filter on system name")
    ap.add_argument("--dry-run", action="store_true", help="list what would run, relax nothing")
    ap.add_argument("--reuse", action="store_true",
                    help="reuse an existing relaxation in --out instead of recomputing; "
                         "for re-analysis after changing a metric or filter")
    args = ap.parse_args()

    seeds = REPO / args.seeds
    summary = json.loads((seeds / "summary.json").read_text())
    ref = np.load(seeds / "reference.npz")
    names = sorted(n for n in summary if not args.only or args.only in n)
    if not names:
        raise SystemExit(f"no systems match --only {args.only!r}")

    out = REPO / args.out
    out.mkdir(parents=True, exist_ok=True)
    print(f"model={args.model} fmax={args.fmax} unconstrained  |  {len(names)} system(s)")
    if args.dry_run:
        for n in names:
            print(f"  {n}: {summary[n]['natoms']} atoms, tags={summary[n]['tag_counts']}")
        return

    backend = get_backend(args.model)
    rows = []
    for name in names:
        info = summary[name]
        struct = Structure.from_file(seeds / f"{name}_initial.vasp")
        if "selective_dynamics" in struct.site_properties:
            raise SystemExit(f"{name}: seed is constrained; OC22 relaxed unconstrained")

        t0 = time.time()
        prev, done = out / name / "mlip_relaxed.vasp", out / name / "result.json"
        if args.reuse and prev.exists() and done.exists():
            r = json.loads(done.read_text())
            relaxed = Structure.from_file(prev)
            res = SimpleNamespace(structure=relaxed, energy=r["energy"], fmax=r["fmax"],
                                  start_fmax=r["start_fmax"], nsteps=r["nsteps"],
                                  converged=r["converged"])
            print(f"  [reused existing relaxation in {(out / name).relative_to(REPO)}]")
        else:
            res = relax(struct, backend, workdir=out / name, relax_cell=False,
                        fmax=args.fmax, max_steps=args.max_steps, optimizer="FIRE")
            res.structure.to(filename=str(prev), fmt="poscar")

        # An atom-order change through the POSCAR round-trip would make every per-atom
        # number below a comparison between unrelated atoms.
        if list(res.structure.atomic_numbers) != list(struct.atomic_numbers):
            raise SystemExit(f"{name}: atom order changed through the relaxation round-trip")

        cell, tags = ref[f"{name}_cell"], ref[f"{name}_tags"]
        slab = tags != 2
        mag = np.linalg.norm(
            min_image_disp(res.structure.cart_coords, ref[f"{name}_posR"], cell, slab), axis=1)

        row = {
            "system": name, "model": args.model, "sid": info["sid"], "split": info["split"],
            "ads": info["ads"], "natoms": info["natoms"],
            "e_mlip": res.energy, "e_dft": info["y_relaxed"],
            "start_fmax": res.start_fmax, "final_fmax": res.fmax,
            "nsteps": res.nsteps, "converged": res.converged,
            "mean_displacement": float(mag.mean()),
            "rmsd": float(np.sqrt((mag ** 2).mean())),
            "max_displacement": float(mag.max()),
            "elapsed_s": round(time.time() - t0, 1),
            "flags": [],
        }
        for tag, label in ((0, "subsurface"), (1, "surface"), (2, "adsorbate")):
            m = tags == tag
            if m.any():
                row[f"mean_{label}"] = float(mag[m].mean())
                row[f"max_{label}"] = float(mag[m].max())
        if not res.converged:
            row["flags"].append("not_converged")
        if row.get("mean_surface", 0) > RECONSTRUCTION_SLAB_MEAN:
            row["flags"].append("possible_reconstruction")
        # adsorbate integrity: any O-H stretched past OH_BOND_MAX has dissociated
        ads = np.where(tags == 2)[0]
        if len(ads) == 2:
            P = res.structure.cart_coords
            v = P[ads[0]] - P[ads[1]]
            v -= np.round(v @ np.linalg.inv(cell)) @ cell
            row["adsorbate_bond_A"] = float(np.linalg.norm(v))
            if row["adsorbate_bond_A"] > OH_BOND_MAX:
                row["flags"].append("dissociated")

        rows.append(row)
        print(f"\n=== {name} ({info['natoms']} atoms) ===")
        print(f"  E={res.energy:.4f} vs DFT {info['y_relaxed']:.4f}  "
              f"fmax {res.start_fmax:.2f}->{res.fmax:.3f}  steps={res.nsteps}")
        print(f"  divergence mean={row['mean_displacement']:.3f} "
              f"rmsd={row['rmsd']:.3f} max={row['max_displacement']:.3f} A")
        for label in ("subsurface", "surface", "adsorbate"):
            if f"mean_{label}" in row:
                print(f"     {label:11s} mean={row[f'mean_{label}']:.3f} "
                      f"max={row[f'max_{label}']:.3f} A")
        print(f"  flags: {row['flags'] or 'none'}")

    path = out / "divergence.jsonl"
    path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))
    print(f"\nwrote {len(rows)} rows -> {path.relative_to(REPO)}")

    # Ranking: E(system) - E(clean slab) for a FIXED adsorbate. The gas-phase term is a
    # per-species constant that shifts every surface equally, so it cancels from the
    # ordering — which is the only ranking OC22 supports, since it ships no gas refs.
    # Pair on slab_sid, not on name: an adsorbate system and its clean slab carry
    # different sids, so any name-derived key silently fails to match.
    by_sid = {r["sid"]: r for r in rows}
    ranked, orphans = [], []
    for r in rows:
        if r["ads"] is None:
            continue
        base = by_sid.get(summary[r["system"]].get("slab_sid"))
        if base is None:
            orphans.append(r["system"])
            continue
        ranked.append((r["system"], r["ads"], r["e_mlip"] - base["e_mlip"],
                       r["e_dft"] - base["e_dft"], r["flags"]))
    if orphans:
        print(f"\n  no clean slab seeded for: {orphans} -> excluded from ranking")
    # Only a fixed adsorbate is rankable; across species the gas reference does not cancel.
    for species in sorted({x[1] for x in ranked}):
        group = [x for x in ranked if x[1] == species]
        if len(group) < 2:
            continue
        print(f"\n--- ranking set {species!r} ({len(group)} surfaces) ---")
        print(f"{'system':34s}{'dE_MLIP':>10s}{'dE_DFT':>10s}{'error':>9s}  flags")
        for s, _, dm, dd, fl in group:
            print(f"{s:34s}{dm:>10.3f}{dd:>10.3f}{dm - dd:>9.3f}  {','.join(fl) or '-'}")
        valid = [x for x in group if not x[4]]
        if len(valid) < 2:
            print(f"  only {len(valid)} of {len(group)} pass the validity filters — "
                  "no ranking claim (a reconstructed system describes a different structure)")
        else:
            by_mlip = [x[0] for x in sorted(valid, key=lambda x: x[2])]
            by_dft = [x[0] for x in sorted(valid, key=lambda x: x[3])]
            gap_m = abs(valid[0][2] - valid[-1][2])
            gap_d = abs(valid[0][3] - valid[-1][3])
            print(f"  ranking reproduced: {by_mlip == by_dft} (n={len(valid)}; "
                  f"MLIP spread {gap_m:.3f} eV vs DFT {gap_d:.3f} eV)")


if __name__ == "__main__":
    main()
