"""OVFE convergence vs. how much of the slab is frozen at bulk positions, on the 3
top-performing OVFE benchmark models (UMA-omat, eSEN-30M-OAM, Orb-v2). Slab thickness is
fixed at 8 trilayers for this sweep (see ovfe_height_sweep.py for the separate thickness
study) and 4x2/20 A stay at the benchmark baseline -- only freeze_bottom_fraction changes,
in exact single-trilayer steps (k/8 for k=0..7) since 8 trilayers divides evenly.

Stops at 7/8 (one trilayer left free), not 8/8: freezing every atom in the slab would
freeze the vacancy site's own neighbors too, so the defective structure could never
relax and E_vac would just be the unrelaxed removal energy -- not a meaningful convergence
point, just a degenerate one.

    python scripts/ClaudeScripts/ovfe_freeze_sweep.py runs/ovfe_freeze_sweep
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ovfe_convergence_common import run_sweep  # noqa: E402

HEIGHT_TRILAYERS = 8
# Midpoint of the measured 8-trilayer achievable-thickness window (22.80-26.05 A) -- see
# ovfe_height_sweep.py's docstring for how this was derived; same value it uses for k=8.
MIN_SLAB_SIZE_FOR_8_TRILAYERS = 24.4
VALUES = list(range(0, 8))  # k of 8 trilayers frozen, k=0..7


def apply_value(cfg, value):
    return replace(cfg, slab=replace(
        cfg.slab, min_slab_size=MIN_SLAB_SIZE_FOR_8_TRILAYERS,
        freeze_bottom_fraction=value / HEIGHT_TRILAYERS,
    ))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("outdir", type=Path)
    a = ap.parse_args()
    run_sweep("frozen_trilayers_of_8", VALUES, apply_value, a.outdir)
