"""OVFE convergence vs. vacuum thickness, on the 3 top-performing OVFE benchmark models
(UMA-omat, eSEN-30M-OAM, Orb-v2). Everything else held at the 15-model benchmark baseline
(4x2 supercell, 4 trilayers, bottom 50% frozen) -- vacuum is the only thing that changes.

    python scripts/studies/ovfe/ovfe_vacuum_sweep.py runs/ovfe_vacuum_sweep
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ovfe_convergence_common import run_sweep  # noqa: E402

VALUES = [5, 8, 13, 16, 20, 25, 30, 35, 40, 50]  # A; 13 A matches Kowalski et al., 20 A is our default


def apply_value(cfg, value):
    return replace(cfg, slab=replace(cfg.slab, min_vacuum_size=float(value)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("outdir", type=Path)
    a = ap.parse_args()
    run_sweep("vacuum_A", VALUES, apply_value, a.outdir)
