"""
Every chemistry and relaxation knob.

Defaults settings tuned for the stoichiometric rutile TiO2 (110) surface (pymatgen's
termination index 1). ``scripts/validate_materials.py`` re-checks this for any new material.

Every value below has a matching command-line flag. Edit this file to change the
default for everything; pass the flag to change one run. The flag always wins.
"""

from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class SlabConfig:
    miller_index: tuple[int, int, int] = (1, 1, 0)  
    min_slab_size: float = 12.0  # Å; ≈4 trilayers of rutile (110) 
    min_vacuum_size: float = 12.0  
    termination_index: int = 1  # standard rutile(110)
    lll_reduce: bool = True # Allows equivalent basis with shorter closer-to-perpendicular lattice vectors, easier to work with
    center_slab: bool = True # Centres slab on Z, meaning slabGenerator can easily find highest atom for surface
    supercell: tuple[int, int] = (4, 2)  # lateral replication (along a, b); dilutes defect images, decrease no. vacancies when multiplied
    freeze_bottom_fraction: float = 0.5  # fix the bottom fraction of slab thickness at bulk positions

    # How deep below the topmost atom an O can sit and still count as a surface vacancy.
    # None enumerates every symmetry-distinct O in the unfrozen region — mostly subsurface
    # bulk O, and a different question from surface chemistry.
    #
    # This is not just a cost knob. The funnel's winner becomes the substrate for the
    # adsorbate stage, and uncapped on IrO2 a vacancy 3.19 A down beat the surface O2c by
    # 0.33 eV — so the adsorbate stage ran on a surface with no vacancy in it at all.
    #
    # 1.8 A keeps the first two O shells. Measured across all seven rutile(110) slabs, the
    # shells sit at a fixed spacing with one symmetry class each:
    #
    #     shell 0   O2c   0.00 A            bridging, the classic defect
    #     shell 1   O3c   1.16 - 1.36 A     in-plane surface
    #     shell 2   O3c   2.31 - 2.72 A     first bulk-like layer  <- excluded
    #
    # so any cut in (1.36, 2.31) gives the same two candidates on every one; 1.8 is the
    # middle of that window. Keeping shell 1 is deliberate: with the bridging O alone the
    # funnel has nothing to rank, and the O3c is the comparison that shows the model
    # actually prefers the bridging site rather than being assumed to.
    vacancy_surface_depth: float | None = 1.8
    max_vacancy_sites: int | None = None  # cap the vacancy funnel (None = all); smoke tests only — keeps the first N by site index, which may drop the real minimum

@dataclass(frozen=True)
class RelaxConfig:
    fmax: float = 0.02  # eV/Å; water's soft librations need tighter than CO's 0.05
    max_steps: int = 300
    optimizer: str = "FIRE"

# We use Pymatgen to geometrically find sites based on all surface atoms, then add guardrails to prevent spawning too far or in atoms
@dataclass(frozen=True)
class AdsorbateConfig:
    species: tuple[str, ...] = ("H",)  # Dictates which species to place, tuple because larger molecules will have multiple entries
    coords: tuple[tuple[float, float, float], ...] = ((0.0, 0.0, 0.0),)  # relative molecule geom, Å, molecules are oriented at z=0 closest to surface
    positions: tuple[str, ...] = ("ontop", "bridge", "hollow")  # Keys for Pymatgen's Delaunay Triangulation
    
    # Which atoms the vacuum can see from one axis — these become the triangulation vertices
    surface_depth: float = 3.0  # Å below the topmost atom to scan for exposed surface atoms
    exposure_block_radius: float = 1.3  # Å; nothing sits 1.3 A sideways from it
    seed_standoff: float = 0.2
    min_normal_height: float = 0.5  # Å floor when the site is further from its atom sideways than the bond length, so there is no vertical solution
    min_clearance: float = 0.8  # reject a placement closer than this x the covalent bond length to any slab atom — the guard against spawning inside a surface atom
    symm_reduce: float = 0.01  # pymatgen: how close (fractional coords) before two sites merge
    max_per_position: int | None = None  # cap sites kept per position type (None = all); for smoke tests


# Named fragments for --adsorbate. First entry is the binding atom and must sit at z=0:
# pymatgen anchors the lowest-z atom on the site, so ordering sets which end faces the surface.
ADSORBATE_FRAGMENTS: dict[str, tuple[tuple[str, ...], tuple[tuple[float, float, float], ...]]] = {
    "H": (("H",), ((0.0, 0.0, 0.0),)),
    "CO": (("C", "O"), ((0.0, 0.0, 0.0), (0.0, 0.0, 1.128))),  # C-down, gas-phase C-O = 1.128 A
    "H2O": (("O", "H", "H"), ((0.0, 0.0, 0.0), (0.0, 0.757, 0.5859), (0.0, -0.757, 0.5859))),
    "O2": (("O", "O"), ((0.0, 0.0, 0.0), (0.0, 0.0, 1.208))),  # end-on, gas-phase O-O = 1.208 A
    "O2-side": (("O", "O"), ((0.0, -0.604, 0.0), (0.0, 0.604, 0.0))), # Side-on: both atoms at z=0
    "O": (("O",), ((0.0, 0.0, 0.0),)),
}

@dataclass(frozen=True)
class RunConfig:
    reference: str = "MACE-mh1-omat"  
    candidate: str = "MACE-mh1-oc20"
    slab: SlabConfig = SlabConfig()
    relax: RelaxConfig = RelaxConfig()
    adsorbate: AdsorbateConfig = AdsorbateConfig()
    polymorph: str = "rutile-tio2"  