"""Recompute desorption from the FINAL relaxed geometry of every persisted CO2 retest
attempt, instead of trusting co2_adsorption_retest.py's in-loop early-stop flag alone.

The in-loop desorb_check_step check only fires at one specific step (150); a slow drift
that crosses "farther than start" only after that step, or that settles into a low-force
but still-far-away final position, sails through as "converged" without ever being
flagged. This reads back every persisted relaxed.vasp (co2_adsorption_retest.py now writes
one per attempt, under outdir/<oxide>/<model>/adsorbate/<site>/<orientation>/) and applies
the same final-geometry check pipeline.py uses everywhere else in this codebase --
_adsorbate_anchor_distance() + checks.py's desorb_tol=2.0 convention: the adsorbate's
anchor atom, in the FINAL structure, compared against its nearest surface neighbor's
covalent bond length. No relaxations are rerun -- this is pure post-processing of
structures already on disk.

    python scripts/co2_retest_reclassify.py runs/co2_ads_benchmark
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pymatgen.core import Structure

from oxide_workflow.pipeline import _adsorbate_anchor_distance

N_ADS = 3  # CO2: O, C, O
DESORB_TOL = 2.0  # matches checks.py's placement_quality_flags default


def reclassify(adsorbate_dir: Path) -> list[dict]:
    rows = []
    for relaxed in sorted(adsorbate_dir.glob("*/*/relaxed.vasp")):
        site_dir, orientation = relaxed.parent.parent.name, relaxed.parent.name
        final = Structure.from_file(str(relaxed))
        end_dist, bond = _adsorbate_anchor_distance(final, N_ADS)
        ratio = (end_dist / bond) if (end_dist is not None and bond) else None
        desorbed = ratio is not None and ratio >= DESORB_TOL
        rows.append({
            "site": site_dir, "orientation": orientation,
            "end_distance_A": end_dist, "bond_length_A": bond, "ratio": ratio,
            "desorbed_final_geometry": desorbed,
        })
    return rows


def main(rundir: Path) -> None:
    total = desorbed_total = 0
    for adsorbate_dir in sorted(rundir.glob("*/*/adsorbate")):
        oxide, model = adsorbate_dir.parent.parent.name, adsorbate_dir.parent.name
        rows = reclassify(adsorbate_dir)
        if not rows:
            continue
        n_desorbed = sum(1 for r in rows if r["desorbed_final_geometry"])
        total += len(rows)
        desorbed_total += n_desorbed
        print(f"{model:16s}{oxide:8s}{len(rows):>4d} attempt(s)   "
              f"{n_desorbed:>4d} desorbed by final geometry")
        worst = sorted(rows, key=lambda r: r["ratio"] or 0, reverse=True)[:3]
        for r in worst:
            if r["ratio"] is None:
                continue
            tag = "DESORBED" if r["desorbed_final_geometry"] else "bound"
            print(f"    {r['site']:20s}{r['orientation']:16s}"
                  f"end_dist={r['end_distance_A']:.2f} A  bond={r['bond_length_A']:.2f} A  "
                  f"ratio={r['ratio']:.2f}  {tag}")

    if total == 0:
        print("no persisted adsorbate/*/*/relaxed.vasp found under this rundir")
        return
    print(f"\nTOTAL: {desorbed_total}/{total} desorbed by final-geometry check "
          f"({100 * desorbed_total / total:.0f}%)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("rundir", type=Path)
    main(ap.parse_args().rundir)
