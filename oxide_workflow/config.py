"""Pinned configuration for the prototype chain (design §4: facets pinned by config).

Facet/termination/slab geometry are *config*, not search — facet/Wulff screening is
deferred (§8). Kept material-agnostic: these are knobs, not rutile-specific logic.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SlabConfig:
    """Rutile-oxide slab baseline (design §4), sized to mirror standard DFT surface models.

    Defaults reproduce the common rutile (110) convention: a ~4-trilayer slab in a 4×2
    lateral supercell with the bottom ~2 trilayers frozen at bulk positions. For the
    canonical rutile TiO2 cell this yields a 192-atom slab (Ti64O128).
    """

    miller_index: tuple[int, int, int] = (1, 1, 0)  # rutile (110), the pinned default
    min_slab_size: float = 12.0  # Å; ≈4 trilayers of rutile (110) slab material
    min_vacuum_size: float = 12.0  # Å of vacuum
    termination_index: int = 1  # standard rutile(110): the bridging-O termination (term 0 is the non-standard coplanar-Ti cut)
    lll_reduce: bool = True
    center_slab: bool = True
    supercell: tuple[int, int] = (4, 2)  # lateral replication (along a, b); dilutes defect images
    freeze_bottom_fraction: float = 0.5  # fix the bottom fraction of slab thickness at bulk positions


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
    reference: str = "MACE-mh1-omat"  # mace-mh-1 OMat24/PBE head — the updated reference
    candidate: str = "UMA-oc22"  # default candidate; oc22 is the prime oxide-cat head
    slab: SlabConfig = SlabConfig()
    relax: RelaxConfig = RelaxConfig()
    adsorbate: AdsorbateConfig = AdsorbateConfig()
    polymorph: str = "mp-2657"  # provenance label for records only
