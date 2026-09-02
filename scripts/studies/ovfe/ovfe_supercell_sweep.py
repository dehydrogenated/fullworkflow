"""OVFE convergence vs. lateral supercell size, on the 3 top-performing OVFE benchmark
models (UMA-omat, eSEN-30M-OAM, Orb-v2). Everything else held at the 15-model benchmark
baseline (4 trilayers thick, 20 A vacuum, bottom 50% frozen) -- lateral size is the only
thing that changes. This is the expensive axis: atom count scales with nx*ny, from 192
atoms at the 4x2 baseline up to ~1500 at 8x8 -- still fine for an MLIP, just the slowest
of the four sweeps, so it's ordered smallest-first (partial results stay usable if a run
is cut short).

    python scripts/studies/ovfe/ovfe_supercell_sweep.py runs/ovfe_supercell_sweep
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ovfe_convergence_common import run_sweep  # noqa: E402

VALUES = [
    (1, 1), (1, 2), (2, 2), (3, 2), (3, 3), (4, 2),  # 4x2 = current OVFE-benchmark default
    (4, 3), (4, 4), (5, 5), (6, 4), (6, 6), (7, 7), (8, 8),
]


def apply_value(cfg, value):
    return replace(cfg, slab=replace(cfg.slab, supercell=tuple(value)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("outdir", type=Path)
    a = ap.parse_args()
    run_sweep("supercell", VALUES, apply_value, a.outdir)
