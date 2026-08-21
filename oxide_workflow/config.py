"""
Every chemistry and relaxation knob.

Defaults settings tuned for the stoichiometric rutile TiO2 (110) surface (pymatgen's
termination index 1). ``scripts/core/validate_materials.py`` re-checks this for any new material.

Every value below has a matching command-line flag. Edit this file to change the
default for everything; pass the flag to change one run. The flag always wins.
"""

from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class SlabConfig:
    miller_index: tuple[int, int, int] = (1, 1, 0)  
    min_slab_size: float = 12.0  # Å; ≈4 trilayers of rutile (110) 
    min_vacuum_size: float = 20.0  
    termination_index: int = 1  # standard rutile(110)
    lll_reduce: bool = True # Allows equivalent basis with shorter closer-to-perpendicular lattice vectors, easier to work with
    center_slab: bool = True # Centres slab on Z, meaning slabGenerator can easily find highest atom for surface
    supercell: tuple[int, int] = (4, 2)  # lateral replication (along a, b); dilutes defect images, decrease no. vacancies when multiplied
    freeze_bottom_fraction: float = 0.5  # fix the bottom fraction of slab thickness at bulk positions
    vacancy_block_radius: float | None = 1.3  # Å; same vacuum line-of-sight test as adsorbate placement. None = every symmetry-distinct O, not just surface ones

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
    # If set, check at this step whether the adsorbate is farther from the surface than it
    # was desorb_trend_window steps earlier -- a RECENT trend, not vs. the very start, so a
    # molecule that drifted out while reorienting and has already turned back in isn't
    # wrongly killed. Net outward drift *right now* means the site was never going to bind.
    # None = disabled (default; existing runs are unaffected).
    desorb_early_stop_step: int | None = None
    desorb_trend_window: int = 20
    # The reverse case: if max_steps runs out unconverged but the adsorbate is still
    # net-approaching the surface (genuinely still doing work, not just oscillating), give
    # it one extension of extend_steps more instead of calling it unconverged on a
    # trajectory that hadn't finished. False = disabled (default).
    extend_if_approaching: bool = False
    extend_steps: int = 100
    max_extensions: int = 1  # repeatable up to this many rounds, each requiring progress

# Named fragments for --adsorbate. First entry is the binding atom and must sit at z=0:
# pymatgen anchors the lowest-z atom on the site, so ordering sets which end faces the surface.
ADSORBATE_FRAGMENTS: dict[str, tuple[tuple[str, ...], tuple[tuple[float, float, float], ...]]] = {
    "H": (("H",), ((0.0, 0.0, 0.0),)),
    "CO": (("C", "O"), ((0.0, 0.0, 0.0), (0.0, 0.0, 1.128))),  # C-down, gas-phase C-O = 1.128 A
    "H2O": (("O", "H", "H"), ((0.0, 0.0, 0.0), (0.0, 0.757, 0.5859), (0.0, -0.757, 0.5859))),
    "O2": (("O", "O"), ((0.0, 0.0, 0.0), (0.0, 0.0, 1.208))),  # end-on, gas-phase O-O = 1.208 A
    "O2-side": (("O", "O"), ((0.0, -0.604, 0.0), (0.0, 0.604, 0.0))), # Side-on: both atoms at z=0
    "O": (("O",), ((0.0, 0.0, 0.0),)),
    "H2": (("H", "H"), ((0.0, 0.0, 0.0), (0.0, 0.0, 0.741))),  # gas-phase H-H = 0.741 A
    # O-down, tilted 20 deg off vertical: O anchors to the surface metal cation, H points
    # mostly away. O-H = 0.9573 A, same bond length as the H2O entry above. Only a
    # relaxation seed, not a constraint -- exact final tilt is whatever the optimizer finds,
    # and a 12-way orientation/standoff sweep on TiO2(110) confirmed every variant converges
    # to the same final bond length and energy regardless of seed tilt. The tilt still
    # matters for the *optimizer path*, though: pure-vertical (0 deg) and 40 deg both
    # produced an oscillatory approach that tripped the desorption early-stop safety check
    # even at 2x its default patience, while 20 deg and 60 deg converged cleanly and faster
    # (fewer steps) every time. 20 deg picked over 60 to stay closer to the "H points away"
    # literature convention (Comer et al. 2022) while still avoiding the bad basin.
    "OH": (("O", "H"), ((0.0, 0.0, 0.0), (0.0, 0.3274, 0.8995))),
    # linear O=C=O, gas-phase C=O = 1.16 A; first O anchors to the surface metal cation
    # (Chavez-Rocha et al., Molecules 2023, 28, 1776: CO2 approaches M1-OA first, then may
    # bend toward a surface oxygen as a second interaction).
    "CO2": (("O", "C", "O"), ((0.0, 0.0, 0.0), (0.0, 0.0, 1.16), (0.0, 0.0, 2.32))),
    # Vertical, terminal-CH3-down: one methyl group anchors (its 3 H's all sit near z=0, a
    # tripod on the surface) with the rest of the chain extending straight up. Chosen to
    # match GAME-Net-Ox's (van Hout et al. 2026, Digital Discovery 5, 407-414) own top_M5c
    # config with the largest C-backbone z-spread (~2.1-2.5 A vs ~0.6-1.4 A for every other
    # config, confirmed directly from their relaxed CONTCARs, not their labels) -- the one
    # DFT config in their dataset that is actually a standing/vertical propane rather than
    # flat-lying. Geometry from ASE's g2 reference set (Curtiss et al. 1997), rotated so the
    # central-to-anchor-terminal C-C bond is vertical.
    "Propane": (("H", "C", "C", "C", "H", "H", "H", "H", "H", "H", "H"),
                ((0.0000, 1.5057, 0.0000), (0.0000, 0.4884, 1.9257), (0.0000, 0.4884, 0.4012),
                 (0.0000, -0.9209, 2.5069), (-0.8769, 1.0344, 2.2911), (0.8769, 1.0344, 2.2911),
                 (0.0000, -0.9039, 3.6003), (0.8836, -0.0262, 0.0119), (-0.8836, -0.0262, 0.0119),
                 (-0.8836, -1.4770, 2.1796), (0.8836, -1.4770, 2.1796))),
    # Side-on through the C=C: both alkene carbons (and the rest of the sp2 vinyl framework)
    # sit flat at the same height (anchor = one C=C carbon, not an H), the methyl group
    # elevated above -- the classic Dewar-Chatt-Duncanson pi-complex geometry, and also what
    # GAME-Net-Ox's own best-binding top_M5c configs actually look like (C=C height
    # difference 0.01-0.19 A across all 3 oxides, confirmed from their relaxed CONTCARs).
    # Idealized sp2 geometry (C1=C2 1.318 A, C2-C3 1.501 A, /_C1C2C3 124.3 deg -- ASE's g2 set
    # has no propene entry). One methyl H geometrically must dip a hair below the vinyl
    # plane for an in-plane C-C bond (unavoidable for 3 tetrahedral substituents 120 deg
    # apart around an axis that itself has no z-component) -- clipped to z=0.02 rather than
    # left negative.
    "Propene": (("C", "C", "C", "H", "H", "H", "H", "H", "H"),
                ((0.0000, -1.3180, 0.0200), (0.0000, 0.0000, 0.0200), (1.2400, 0.8459, 0.0200),
                 (-0.9611, 0.5078, 0.0200), (0.9317, -0.7582, 0.0200), (-0.9317, -0.7582, 0.0200),
                 (0.9377, 0.6397, 0.0200), (1.4419, -0.0994, 0.5166), (0.4335, 1.3788, 0.5166))),
}

@dataclass(frozen=True)
class RunConfig:
    reference: str = "MACE-mh1-omat"  
    candidate: str = "MACE-mh1-oc20"
    slab: SlabConfig = SlabConfig()
    relax: RelaxConfig = RelaxConfig()
    adsorbate: AdsorbateConfig = AdsorbateConfig()
    polymorph: str = "mp-2657"  