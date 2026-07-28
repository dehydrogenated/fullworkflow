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
    site_distance: float = 1.5  # pymatgen site-enumeration distance (Å); also the fallback
    # placement height when a site's coordinating species isn't in site_distance_by_species.
    # 1.5 Å is the middle ground between two failure modes. At 2.0 Å a small fragment
    # (e.g. H) starts *outside* bonding range, feels a force already below the relax fmax,
    # and "converges" frozen at its placement (no work). At 1.0 Å it starts essentially at
    # the bond length — placed on the answer, so it barely relaxes (a real but uninformative
    # trajectory, and spurious "barely moved" flags). 1.5 Å starts inside the attractive
    # basin with room to relax *down* into the well: a genuine binding trajectory.

    # Target adsorbate–surface *bond distance* (Å) to the coordinating surface atom, keyed
    # by that atom's element. This is a distance, NOT a vertical offset: the surface-normal
    # placement height is *solved* from it and the site's actual lateral offset (see
    # stages._normal_height), so a bridge/hollow site scales correctly with the lattice — an
    # adsorbate laterally displaced from its coordinating atom sits lower to keep the bond
    # length. Chemistry sets the target (H–O ≈ 1 Å hydroxyl; larger for a metal); lattice
    # geometry sets the height. Elements *not* listed fall back to a covalent-radii estimate
    # + seed_standoff — a fully automated, material-agnostic default, so a batch sweep over
    # many compositions needs no per-material tuning. Pin only what you want to override,
    # e.g. (("O", 1.0), ("Ti", 1.9)).
    site_distance_by_species: tuple[tuple[str, float], ...] = (("O", 1.4),)
    # 1.4 Å (vs the ~0.97 Å O–H bond) starts the adsorbate *outside* the well with room to
    # relax down into it — a genuine binding trajectory. Placing at ~1.0 (the bond length)
    # lands on the answer: the site's energy is right but the trajectory is uninformative,
    # which the on-site placement flag now catches.
    min_normal_height: float = 0.5  # Å floor on the solved normal height, for sites whose
    # lateral offset exceeds the target bond length (adsorbate can't reach it vertically).
    surface_layer_tol: float = 0.7  # Å; thickness of the *topmost* surface layer used to
    # pick the coordinating atom for arm-1 geometric sites. A geometric bridge/hollow site
    # sits laterally over the recessed cation row, so a naive nearest-atom search would key
    # its height off that subsurface metal; restricting to the top layer keeps it referenced
    # to the flanking top-layer atoms (the bridging O's on rutile(110)). Recessed-but-exposed
    # cations like Ti5c are still sampled — arm 2 seeds them directly.
    symm_reduce: float = 0.01  # symmetry tol for site enumeration

    # Arm 2 — covalent-radii seeding over symmetry-distinct *exposed surface atoms*.
    # pymatgen's geometric finder detects "surface" atoms with a narrow height window, so
    # on a rumpled oxide it misses recessed-but-exposed cations (e.g. the 5-fold Ti on
    # rutile(110), the canonical adsorption site, sits ~1.3 Å below the bridging-O row and
    # never becomes an ``ontop`` candidate). This arm seeds a placement directly above each
    # symmetry-distinct exposed atom at a height set by covalent radii — material-agnostic,
    # element-driven, no per-surface logic.
    seed_surface_atoms: bool = True
    surface_depth: float = 3.0  # Å below the topmost atom to scan for exposed surface atoms
    exposure_block_radius: float = 1.3  # Å; a higher atom within this lateral radius occludes
    dedup_tol: float = 0.75  # Å; merge arm-1/arm-2 placements whose anchor points coincide
    seed_standoff: float = 0.5  # Å above the covalent bond length for arm-2 seeds, so the
    # adsorbate starts inside the attractive basin but with room to relax *down* (mirrors the
    # 1.5-vs-1.0 reasoning for site_distance — don't place exactly on the answer).


@dataclass(frozen=True)
class RunConfig:
    reference: str = "MACE-mh1-omat"  # mace-mh-1 OMat24/PBE head — the updated reference
    candidate: str = "UMA-oc22"  # default candidate; oc22 is the prime oxide-cat head
    slab: SlabConfig = SlabConfig()
    relax: RelaxConfig = RelaxConfig()
    adsorbate: AdsorbateConfig = AdsorbateConfig()
    polymorph: str = "mp-2657"  # provenance label for records only
