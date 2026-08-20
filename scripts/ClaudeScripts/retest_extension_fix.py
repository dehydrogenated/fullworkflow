#!/usr/bin/env python
"""Re-runs exactly the legs that didn't converge in the medium test, after the
worker_relax.py extension-logic fix (compare against start distance, not a rolling
per-window baseline -- see the commit "Fix extension logic to compare against start
distance, not a rolling window"). Explicit list, not a cross-product sweep: the point is
to see whether these SPECIFIC previously-stuck legs now extend further / converge, not to
re-run the whole medium test.

The 7 non-converged legs from runs/mo2_medium_test/results.jsonl, with their old numbers
for comparison (all OH -- matches the observed pattern that every non-convergent leg in
that run was an OH leg):

    model                   oxide  facet  old_nsteps  old_extended  old_ext_used
    MACE-mh1-omat           ZrO2   110    300         False         0
    MACE-mh1-matpes         ZrO2   110    300         False         0
    UMA-oc22                ZrO2   100    400         True          1
    UMA-M-omat              ZrO2   100    300         False         0
    UMA-M-omat              RuO2   110    300         False         0
    SevenNet-omni-omat24    ZrO2   110    300         False         0
    SevenNet-omni-mpa       ZrO2   110    300         False         0

Note most show extensions_used=0 despite not converging -- the OLD windowed check failed
on its very first look, never extending even once. If the fix works, these should now show
extensions_used > 0 (using the budget instead of stopping cold), and ideally converge.

Point OUTDIR at the medium test's OWN output directory (e.g.
runs/mo2_medium_test), not a fresh one: relax_bulk_slab() goes through
pipeline._relax_record(), which resumes an already-converged, matching bulk/slab leaf from
disk instead of recomputing it -- free reuse of work already done, since the fix touches
only the adsorbate stage's worker_relax.py code path (n_ads/extend_if_approaching are never
set for bulk/slab). run_adsorbate() itself calls backends.relax() directly with no resume
check, so it always relaxes fresh regardless -- exactly what's needed to pick up the fix.

This appends 7 more rows to that directory's results.jsonl, one per already-present
(model, oxide, facet, adsorbate) combo -- when analyzing, dedupe by keeping the LAST row
per combo, same as the CPU/GPU append-mode duplicate from earlier in this run.

Usage:
    python retest_extension_fix.py OUTDIR   # OUTDIR = runs/mo2_medium_test
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mo2_adsorption_benchmark import (  # noqa: E402
    ADSORBATE_FRAGMENTS, OH_COORDS_BY_FACET, SEED_STANDOFF,
    load_literature, mp_id_for_formula, relax_bulk_slab, run_adsorbate,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from oxide_workflow.config import RunConfig  # noqa: E402
from oxide_workflow.pipeline import _cli_miller  # noqa: E402

# (model, oxide, facet, adsorbate, old_nsteps, old_extended, old_extensions_used)
FAILED_LEGS = [
    ("MACE-mh1-omat", "ZrO2", "110", "OH", 300, False, 0),
    ("MACE-mh1-matpes", "ZrO2", "110", "OH", 300, False, 0),
    ("UMA-oc22", "ZrO2", "100", "OH", 400, True, 1),
    ("UMA-M-omat", "ZrO2", "100", "OH", 300, False, 0),
    ("UMA-M-omat", "RuO2", "110", "OH", 300, False, 0),
    ("SevenNet-omni-omat24", "ZrO2", "110", "OH", 300, False, 0),
    ("SevenNet-omni-mpa", "ZrO2", "110", "OH", 300, False, 0),
]


def main(outdir: Path) -> None:
    lit = load_literature()
    outdir.mkdir(parents=True, exist_ok=True)
    results_path = outdir / "results.jsonl"

    base_cfg = RunConfig()
    base_cfg = replace(base_cfg, adsorbate=replace(
        base_cfg.adsorbate, positions=("ontop",), max_per_position=None,
        seed_standoff=SEED_STANDOFF,
    ))
    base_cfg = replace(base_cfg, relax=replace(base_cfg.relax, fmax=0.02))

    print(f"{'model':24s}{'oxide':8s}{'facet':7s}{'old_nsteps':>11s}{'old_ext':>9s}"
          f"{'new_nsteps':>11s}{'new_ext':>9s}{'new_conv':>10s}{'verdict':>10s}")

    for model, oxide, facet, adsorbate, old_nsteps, old_extended, old_ext_used in FAILED_LEGS:
        mp_id = mp_id_for_formula(oxide)
        fcfg = replace(base_cfg, slab=replace(base_cfg.slab, miller_index=_cli_miller(facet)))

        # Resumes from outdir's existing bulk/slab leaf if it matches (see module docstring)
        # -- no separate caching needed here, relax_bulk_slab already does it via _resume.
        pristine_structure, pristine_energy = relax_bulk_slab(model, oxide, facet, mp_id, outdir, fcfg)

        species, coords = ADSORBATE_FRAGMENTS[adsorbate]
        if adsorbate == "OH":
            coords = OH_COORDS_BY_FACET.get(facet, coords)
        acfg = replace(fcfg, adsorbate=replace(fcfg.adsorbate, species=species, coords=coords))

        row = run_adsorbate(
            model, oxide, facet, adsorbate, pristine_structure, pristine_energy,
            outdir, acfg, lit,
        )
        with results_path.open("a") as f:
            f.write(json.dumps(row) + "\n")

        new_ext_used = row.get("extensions_used", 0)
        verdict = "BETTER" if (row["converged"] or new_ext_used > old_ext_used) else (
            "same" if new_ext_used == old_ext_used else "worse"
        )
        print(f"{model:24s}{oxide:8s}{facet:7s}{old_nsteps:>11d}{str(old_extended):>9s}"
              f"{row['nsteps']:>11d}{new_ext_used:>9d}{str(row['converged']):>10s}{verdict:>10s}")

    print(f"\nwrote {results_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir", type=Path)
    args = ap.parse_args()
    main(args.outdir)
