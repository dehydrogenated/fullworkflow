"""Pinned configuration for the prototype chain (design §4: facets pinned by config).

Facet/termination/slab geometry are *config*, not search — facet/Wulff screening is
deferred (§8). Kept material-agnostic: these are knobs, not rutile-specific logic.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SlabConfig:
    miller_index: tuple[int, int, int] = (1, 1, 0)  # rutile (110), the pinned default
    min_slab_size: float = 6.0  # Å of slab material, change?
    min_vacuum_size: float = 12.0  # Å of vacuum
    termination_index: int = 0  # which generated termination to pin
    lll_reduce: bool = True
    center_slab: bool = True


@dataclass(frozen=True)
class RelaxConfig:
    fmax: float = 0.05  # eV/Å
    max_steps: int = 300
    optimizer: str = "FIRE"


@dataclass(frozen=True)
class AdsorbateConfig:
    """Adsorbate placement knobs (design §4). Kept material-agnostic: the adsorbate is a
    generic fragment specified here, never hardcoded in stage logic."""

    species: tuple[str, ...] = ("H",)  # generic fragment; not material logic
    coords: tuple[tuple[float, float, float], ...] = ((0.0, 0.0, 0.0),)  # molecule geom, Å
    positions: tuple[str, ...] = ("ontop", "bridge", "hollow")  # one representative each
    site_distance: float = 2.0  # placement height above surface, Å
    symm_reduce: float = 0.01  # symmetry tol for site enumeration


@dataclass(frozen=True)
class RunConfig:
    reference: str = "MACE-OMAT24"  # backend name that generates the reference chain
    candidate: str = "UMA-s"  # backend name benchmarked against the reference
    slab: SlabConfig = SlabConfig()
    relax: RelaxConfig = RelaxConfig()
    adsorbate: AdsorbateConfig = AdsorbateConfig()
    polymorph: str = "mp-2657"  # provenance label for records only
