"""Prototype input structures + material resolution.

The design's bulk stage is "MP-ID input" (mp-2657). Until an MP/OPTIMADE source is
wired in (deferred, §8), this module resolves a material *identifier* (mp-id or alias)
to a bulk warm-start ``Structure`` via a small local registry. The single prototype
(rutile TiO2, ``mp-2657``) is the default; a batch run simply passes several identifiers,
each of which must resolve here (or, later, be fetched from MP/OPTIMADE).

This is *input data*, not pipeline logic — the pipeline stays material-agnostic and asks
only for ``get_structure(identifier)``.
"""

from __future__ import annotations

from typing import Callable

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


# --- Material registry: identifier -> bulk Structure factory -----------------------------
#
# The lookup key is the same string carried as ``RunConfig.polymorph`` (an mp-id or a
# human alias). A batch run outlines several of these and iterates. New prototypes are
# added here (or, eventually, resolved through an MP/OPTIMADE fetch hook) without touching
# the pipeline. Mirrors the ``backends.REGISTRY`` pattern.

STRUCTURE_REGISTRY: dict[str, Callable[[], Structure]] = {
    "mp-2657": rutile_tio2,  # canonical rutile TiO2 prototype
    "rutile-tio2": rutile_tio2,  # human-friendly alias for the same cell
}


def register_structure(identifier: str, factory: Callable[[], Structure]) -> None:
    """Register a material identifier -> Structure factory (e.g. a local POSCAR/CIF loader).

    Lets a caller add batch members at runtime without editing this module — the escape
    hatch until MP/OPTIMADE fetching lands (§8).
    """
    STRUCTURE_REGISTRY[identifier] = factory


def get_structure(identifier: str) -> Structure:
    """Resolve a material identifier (mp-id or alias) to its bulk warm-start Structure.

    Local registry only for now; MP/OPTIMADE fetching is deferred (§8). Raises ``KeyError``
    with the set of known identifiers if the material is not registered.
    """
    try:
        factory = STRUCTURE_REGISTRY[identifier]
    except KeyError:
        raise KeyError(
            f"unknown material {identifier!r}; registered: {sorted(STRUCTURE_REGISTRY)}. "
            "Register it via register_structure() or wire an MP/OPTIMADE fetch (deferred, §8)."
        ) from None
    return factory()
