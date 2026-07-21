"""Step 5 integration: the pseudo-reference plumbing test end-to-end.

Slow (runs many real relaxations across two conda envs). Opt in with
``OXIDE_RUN_SLOW=1``. Verifies the pipeline produces well-formed seeded and
full-pipeline divergence tables plus the per-candidate ranking by-product.
"""

from __future__ import annotations

import os

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
    # bulk (shared), plus seeded and full-pipeline at slab and vacancy.
    assert ("bulk", "seeded") in protocols
    assert ("slab", "seeded") in protocols
    assert ("slab", "full_pipeline") in protocols
    assert ("vacancy", "seeded") in protocols
    assert ("vacancy", "full_pipeline") in protocols

    for r in rows:
        assert r.model == cfg.candidate
        assert r.meta["matched"] is True
        assert r.rmsd is not None and r.rmsd >= 0
        assert r.energy_error is not None
        assert r.mean_displacement <= r.rmsd <= r.max_displacement + 1e-9
        if r.stage != "bulk":
            # Unrelaxed-by-construction: the stage input carries real force.
            assert r.start_fmax_at_ref_geom > 0.5

    # Ranking by-product: every reference vacancy site has a candidate counterpart.
    cand_lines = (tmp_path / "candidates.jsonl").read_text().splitlines()
    assert len(cand_lines) == 3 * summary["n_vacancy_sites"]  # ref + seeded + full
