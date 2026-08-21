"""OVFE convergence vs. slab thickness (trilayer count), on the 3 top-performing OVFE
benchmark models (UMA-omat, eSEN-30M-OAM, Orb-v2). Everything else held at the 15-model
benchmark baseline (4x2 supercell, 20 A vacuum) -- only the number of trilayers changes.

Trilayer counts are restricted to even numbers so freeze_bottom_fraction=0.5 (the existing
codebase default, kept fixed here rather than switched to an absolute frozen depth) always
lands exactly on a trilayer boundary -- half of an even trilayer count is always a whole
number of trilayers, so there's no ambiguity about a fractional layer being half-frozen.

min_slab_size per trilayer count is NOT trilayer_thickness*k (confirmed that overshoots
SlabGenerator's own achievable-thickness boundary near k=4, changing which cut it returns
-- see ovfe_convergence_common.py's BASE_SLAB comment). Values below are the midpoint
between the measured min_slab_size that first reaches k trilayers and the one that first
reaches k+1, on the unrelaxed mp-2657 bulk at 1x1 -- a safety margin against small
per-model lattice-constant shifts landing on the wrong side of a boundary, not the
boundary itself.

    python scripts/ClaudeScripts/ovfe_height_sweep.py runs/ovfe_height_sweep
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ovfe_convergence_common import run_sweep  # noqa: E402

# trilayers -> min_slab_size (A), midpoint of that trilayer count's achievable-thickness
# window (see module docstring). 4 = current OVFE-benchmark default (9.80-13.05 window).
MIN_SLAB_SIZE_BY_TRILAYERS = {2: 4.9, 4: 11.4, 6: 17.9, 8: 24.4, 10: 30.9}
VALUES = sorted(MIN_SLAB_SIZE_BY_TRILAYERS)


def apply_value(cfg, value):
    return replace(cfg, slab=replace(cfg.slab, min_slab_size=MIN_SLAB_SIZE_BY_TRILAYERS[value]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("outdir", type=Path)
    a = ap.parse_args()
    run_sweep("height_trilayers", VALUES, apply_value, a.outdir)
