"""
Configure chemistry knobs for slabs, relaxation, adsorbates
Currently configured towards literature advised termination 1 rutile TiO2 (110)
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

@dataclass(frozen=True)
class RelaxConfig:
    fmax: float = 0.05  # eV/Å
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

    # Where the adsorbate starts
    seed_standoff: float = 0.5  # Å added to the covalent bond length to not be too far or close
    min_normal_height: float = 0.5  # Å floor when the site is further from its atom sideways than the bond length, so there is no vertical solution
    min_clearance: float = 0.8  # reject a placement closer than this x the covalent bond length to any slab atom — the guard against spawning inside a surface atom
    symm_reduce: float = 0.01  # pymatgen: how close (fractional coords) before two sites merge

@dataclass(frozen=True)
class RunConfig:
    reference: str = "MACE-mh1-omat"  
    candidate: str = "UMA-oc22"
    slab: SlabConfig = SlabConfig()
    relax: RelaxConfig = RelaxConfig()
    adsorbate: AdsorbateConfig = AdsorbateConfig()
    polymorph: str = "mp-2657"  
