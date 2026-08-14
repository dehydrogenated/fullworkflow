"""Step 3 green light: rutile bulk relaxes with MACE via subprocess isolation.

Integration test — requires the ``mace-clean`` conda env and the cached MACE-mh1-omat
model. The orchestrator (this process) never imports mace; it launches the model env's
interpreter on the worker and reads results from disk.
"""

from __future__ import annotations

import math
import shutil

import pytest

from oxide_workflow.backends import get_backend, relax
from oxide_workflow.structures import get_structure

pytestmark = pytest.mark.skipif(
    not shutil.os.path.exists(get_backend("MACE-mh1-omat").interpreter()),
    reason="mace-clean env not available",
)


def test_rutile_bulk_relaxes_with_mace(tmp_path):
    struct = get_structure("mp-2657")
    res = relax(
        struct, get_backend("MACE-mh1-omat"), workdir=tmp_path, relax_cell=True,
        fmax=0.03, max_steps=500, optimizer="FIRE",
    )

    assert res.converged, f"did not converge: fmax={res.fmax}"
    assert res.fmax <= 0.05
    assert res.fmax < res.start_fmax  # relaxation reduced forces
    assert math.isfinite(res.energy)
    assert res.energy / len(res.structure) < 0  # bound oxide, negative energy/atom
    assert res.structure.composition.reduced_formula == "TiO2"
    assert len(res.structure) == len(struct)
    # subprocess wrote its handoff files to disk
    assert (tmp_path / "relaxed.vasp").exists()
    assert (tmp_path / "result.json").exists()
    # journey artifacts: extended-xyz trajectory (≥1 frame) + the per-step optimizer log
    traj = tmp_path / "trajectory.xyz"
    assert traj.exists()
    assert sum(1 for ln in traj.read_text().splitlines() if ln.strip().isdigit()) >= 1
    assert (tmp_path / "opt.log").exists()
