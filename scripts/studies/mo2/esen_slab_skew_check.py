#!/usr/bin/env python
"""Bulk+slab-only regression check for the lattice-shear fix (stages.py's
get_orthogonal_c_slab in make_slab), run with eSEN specifically since it produced the worst
observed case pre-fix (TiO2(110): b_z=-1.19 A, vs Orb-v2's ~-0.003 A on the same facet).

No vacancy, no adsorbate -- just bulk relax (cell free) -> make_slab() -> slab relax
(cell fixed) -> read back the lattice matrix. Cheap on purpose: this is breadth across
chemistries, not depth on any one of them, since the shear pymatgen's SlabGenerator
introduces pre-fix is chemistry-dependent (driven by each oxide's own converged bulk
lattice constants), so confirming the fix holds means confirming it across many oxides'
worth of different lattice constants, not proving it once on TiO2 again.

Usage:
    python esen_slab_skew_check.py OUTDIR [--oxides ...] [--facets 110 100]
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mo2_adsorption_benchmark import load_literature, mp_id_for_formula, relax_bulk_slab  # noqa: E402

sys.path.insert(0, str(next(  # repo root by marker, not by parent count
    d for d in Path(__file__).resolve().parents if (d / "pyproject.toml").exists())))
from oxide_workflow.config import RunConfig  # noqa: E402
from oxide_workflow.pipeline import _cli_miller  # noqa: E402

MODEL = "eSEN-30M-OAM"
SHEAR_TOL = 1e-6  # A; anything above this after the fix is a real regression, not noise


def main(outdir: Path, oxides: list[str], facets: list[str]) -> None:
    cfg = RunConfig()
    rows = []
    for oxide in oxides:
        mp_id = mp_id_for_formula(oxide)
        if mp_id is None:
            print(f"skip {oxide}: no structure found", flush=True)
            continue
        for facet in facets:
            fcfg = replace(cfg, slab=replace(cfg.slab, miller_index=_cli_miller(facet)))
            try:
                structure, energy = relax_bulk_slab(MODEL, oxide, facet, mp_id, outdir, fcfg)
            except Exception as exc:  # noqa: BLE001 -- breadth run, one bad oxide shouldn't kill the rest
                print(f"  FAILED {oxide}/{facet}: {exc}", flush=True)
                rows.append({"oxide": oxide, "facet": facet, "failed": True, "error": str(exc)})
                continue
            m = structure.lattice.matrix
            a_z, b_z = float(m[0][2]), float(m[1][2])
            skewed = abs(a_z) > SHEAR_TOL or abs(b_z) > SHEAR_TOL
            print(f"  -> a_z={a_z:.8f}  b_z={b_z:.8f}  {'SKEWED' if skewed else 'clean'}", flush=True)
            rows.append({
                "oxide": oxide, "facet": facet, "failed": False,
                "a_z": a_z, "b_z": b_z, "skewed": skewed, "slab_energy_eV": energy,
            })

    outdir.mkdir(parents=True, exist_ok=True)
    report = outdir / "skew_report.jsonl"
    with open(report, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    skewed = [r for r in rows if r.get("skewed")]
    failed = [r for r in rows if r.get("failed")]
    print(f"\n{len(rows)} slabs checked, {len(skewed)} skewed, {len(failed)} failed")
    if skewed:
        print("SKEWED:", [(r["oxide"], r["facet"]) for r in skewed])
    print(f"wrote {report}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir", type=Path)
    ap.add_argument("--oxides", nargs="+", default=None, help="default: all 33 Comer oxides")
    ap.add_argument("--facets", nargs="+", choices=["110", "100"], default=["110", "100"])
    args = ap.parse_args()

    oxides = args.oxides or sorted({r[0] for r in load_literature()})
    main(args.outdir, oxides, args.facets)
