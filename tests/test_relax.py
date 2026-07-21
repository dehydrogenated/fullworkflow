"""Step 3 green light: rutile bulk relaxes with MACE via subprocess isolation.

Integration test — requires the ``mace-clean`` conda env and the cached MACE-OMAT24
model. The orchestrator (this process) never imports mace; it launches the model env's
interpreter on the worker and reads results from disk.
"""

from __future__ import annotations

import math
import shutil

import pytest

from oxide_workflow.backends import get_backend, relax
from oxide_workflow.structures import rutile_tio2

pytestmark = pytest.mark.skipif(
    not shutil.os.path.exists(get_backend("MACE-OMAT24").interpreter()),
    reason="mace-clean env not available",
)


def test_rutile_bulk_relaxes_with_mace(tmp_path):
    struct = rutile_tio2()
    res = relax(struct, get_backend("MACE-OMAT24"), workdir=tmp_path, relax_cell=True)

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
