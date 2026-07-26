"""Step 5 integration: the pseudo-reference plumbing test end-to-end.

Slow (runs many real relaxations across two conda envs). Opt in with
``OXIDE_RUN_SLOW=1``. Verifies the pipeline produces well-formed seeded and
full-pipeline divergence tables plus the per-candidate ranking by-product.
"""

from __future__ import annotations

import json
import os
from collections import Counter

import pytest

from oxide_workflow.backends import get_backend
from oxide_workflow.config import RelaxConfig, RunConfig
from oxide_workflow.records import read_divergence_table

pytestmark = pytest.mark.skipif(
    os.environ.get("OXIDE_RUN_SLOW") != "1"
    or not os.path.exists(get_backend("MACE-OMAT24").interpreter())
    or not os.path.exists(get_backend("UMA-s").interpreter()),
    reason="set OXIDE_RUN_SLOW=1 and have both model envs to run the full pipeline",
)


def test_pipeline_produces_both_mode_tables(tmp_path):
    from oxide_workflow.pipeline import run

    cfg = RunConfig(relax=RelaxConfig(fmax=0.1, max_steps=120))
    summary = run(cfg, outdir=tmp_path)

    rows = read_divergence_table(tmp_path / "divergence.jsonl")
    protocols = {(r.stage, r.protocol) for r in rows}
    # bulk (shared), plus seeded and full-pipeline at slab, vacancy, and adsorbate.
    assert ("bulk", "seeded") in protocols
    assert ("slab", "seeded") in protocols
    assert ("slab", "full_pipeline") in protocols
    assert ("vacancy", "seeded") in protocols
    assert ("vacancy", "full_pipeline") in protocols
    assert ("adsorbate", "seeded") in protocols
    assert ("adsorbate", "full_pipeline") in protocols

    for r in rows:
        assert r.model == cfg.candidate
        assert r.energy_error is not None
        if r.stage != "bulk":
            # Unrelaxed-by-construction: the stage input carries real force.
            assert r.start_fmax_at_ref_geom > 0.5
        # Geometric metrics exist when the matcher aligned the pair. A matcher miss is a
        # valid, informative outcome (§5) — expected mainly if an adsorbate migrates.
        if r.meta["matched"]:
            assert r.rmsd is not None and r.rmsd >= 0
            assert r.mean_displacement <= r.rmsd <= r.max_displacement + 1e-9

    # Ranking by-product: every enumerated candidate energy was recorded, per stage.
    cand = [json.loads(ln) for ln in (tmp_path / "candidates.jsonl").read_text().splitlines()]
    by_group = Counter((c["stage"], c["protocol"]) for c in cand)
    # Vacancy: reference + seeded + full-pipeline funnels, all over the same site count.
    assert by_group[("vacancy", "reference")] == summary["n_vacancy_sites"]
    assert by_group[("vacancy", "seeded")] == summary["n_vacancy_sites"]
    assert by_group[("vacancy", "full_pipeline")] == summary["n_vacancy_sites"]
    # Adsorbate: one funnel per protocol, each 1..len(positions) site types (≤3).
    for proto in ("reference", "seeded", "full_pipeline"):
        assert 1 <= by_group[("adsorbate", proto)] <= 3
