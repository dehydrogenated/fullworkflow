"""Prototype input structures.

The design's bulk stage is "MP-ID input" (mp-2657). Until an MP/OPTIMADE source is
wired in (deferred, §8), this module provides the canonical rutile TiO2 cell directly
as the prototype's warm-start bulk input. This is *input data*, not pipeline logic —
the pipeline stays material-agnostic.
"""

from __future__ import annotations

from pymatgen.core import Lattice, Structure


def rutile_tio2() -> Structure:
    """Canonical rutile TiO2 (space group P4_2/mnm), ~experimental lattice.

    Prototype bulk warm start for mp-2657. Re-relaxed by every backend, so exact
    lattice constants are not load-bearing.
    """
    a, c, u = 4.5937, 2.9587, 0.3050
    lattice = Lattice.tetragonal(a, c)
    species = ["Ti", "Ti", "O", "O", "O", "O"]
    coords = [
        [0.0, 0.0, 0.0],
        [0.5, 0.5, 0.5],
        [u, u, 0.0],
        [-u, -u, 0.0],
        [0.5 + u, 0.5 - u, 0.5],
        [0.5 - u, 0.5 + u, 0.5],
    ]
    return Structure(lattice, species, coords)
