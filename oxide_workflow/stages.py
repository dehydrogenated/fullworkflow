"""Stage structure-building: cut a slab, and the decorate() for vacancies and adsorbates

Each stage's starting structure = previous stage's relaxed output + a fresh
modification, unrelaxed by construction. This module produces those *unrelaxed*
starting structures; relaxation is the backend's job
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from ase.data import atomic_numbers, covalent_radii
from pymatgen.analysis.adsorption import AdsorbateSiteFinder
from pymatgen.core import Molecule, Structure
from pymatgen.core.surface import SlabGenerator
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from .config import AdsorbateConfig, SlabConfig

# --- Universal tools: shared by the slab, vacancy, and adsorbate stages -------------------

# Computes the z height below which slab atoms count as frozen bulk; shared by the freeze
# itself and every site enumerator, so "frozen region" means the same thing everywhere.
def bottom_cutoff_z(
    structure: Structure, fraction: float, always_free: "set[int] | None" = None
) -> "float | None":
    if not 0.0 < fraction <= 1.0: # fraction = 0 means no freezing
        return None
    always_free = always_free or set() # adsorbates / atom index excluded from the slab
    slab_z = [s.coords[2] for i, s in enumerate(structure) if i not in always_free] # construct a list of all slab item heights, ignoring adsorbates 
    zmin, zmax = min(slab_z), max(slab_z)
    return zmin + fraction * (zmax - zmin)


# Freezes the bottom fraction of the slab at bulk positions in-place, mimicking bulk
# hardness while the top (free) face is left to relax.
def apply_bottom_freeze(
    structure: Structure, fraction: float, always_free: "set[int] | None" = None
) -> None:
    cutoff = bottom_cutoff_z(structure, fraction, always_free)
    if cutoff is None:
        return  # freezing disabled → leave every atom free
    always_free = always_free or set()
    flags = [] # list of each atoms free/frozen status
    for i, site in enumerate(structure):
        frozen = i not in always_free and site.coords[2] <= cutoff + 1e-6
        flags.append([not frozen] * 3)  # [F,F,F] = fixed, [T,T,T] = free
    structure.add_site_property("selective_dynamics", flags)  # pymatgen's POSCAR writer turns these True/False into T/F text later


_CN_CUTOFF = 2.6  # Coordination atom cutoff, Å; first cation-anion shell. Verified for rutile-family MO2s


# Computes each atom's (element, coordination deficit, coordination) — the data behind
# site_label and site ranking. Also shared by both the vacancy and adsorbate stages. etc. [0, [Ti,1,5]] = Ti5c
def _site_environments(substrate: Structure) -> "dict[int, tuple[str, int, int]]":
    coordination = {
        i: sum(1 for nb in shell if str(nb.specie) != str(substrate[i].specie))
        for i, shell in enumerate(substrate.get_all_neighbors(_CN_CUTOFF))
    }
    bulk: dict[str, int] = {}
    for i, c in coordination.items():
        element = str(substrate[i].specie)
        bulk[element] = max(bulk.get(element, 0), c)
    return {
        i: (str(substrate[i].specie), bulk[str(substrate[i].specie)] - c, c)
        for i, c in coordination.items()
    }


# Formats a surface atom's element + coordination number into the literature-style label (ex. Ti5c)
def site_label(element: str, coordination: int) -> str:
    return f"{element}{coordination}c"


# Returns indices of atoms with a clear line of sight to vacuum from above — exposed if no atom
# sits occlusion_dz higher within block_radius laterally. Depth only pre-filters the scan region for cost; 
# the occlusion test does the real work. Shared by the vacancy stage (which O counts as removable) 
# and the adsorbate stage where a fragment can land).
def exposed_surface_atoms(
    structure: Structure,
    depth: float,
    block_radius: float = 1.3,  # Å; verified stable 0.8-2.0 A across all 8 rutile-family materials
    occlusion_dz: float = 0.3,
) -> "set[int]":
    L = structure.lattice
    cart = structure.cart_coords
    frac = structure.frac_coords
    zmax = float(cart[:, 2].max())
    scan = [i for i in range(len(structure)) if cart[i][2] > zmax - depth]
    exposed: set[int] = set()
    for i in scan:
        zi = cart[i][2]
        occluded = False
        for j in range(len(structure)):
            if j == i or cart[j][2] <= zi + occlusion_dz:
                continue
            d = frac[i] - frac[j]
            d[:2] -= np.round(d[:2])  # min-image in-plane
            if np.linalg.norm((d @ L.matrix)[:2]) < block_radius:
                occluded = True
                break
        if not occluded:
            exposed.add(i)
    return exposed

# --- Slab ---------------------------------------------------------------------------------

# SlabGenerator's lattice reduction (lll_reduce=True) can mix a fraction of the long,
# vacuum-including c-vector into the in-plane a/b vectors to shorten them -- confirmed via
# SlabGenerator's un-reduced oriented_unit_cell (a_z=b_z=0 there), so it's a reduction
# artifact, not a real tilt of this tetragonal facet's flat-by-symmetry layers. Seen as
# b_z=-1.19 A on a real eSEN TiO2(110) slab vs. Orb-v2's ~-0.003 A on the same facet --
# small per-model differences in converged bulk lattice constants tip this from negligible
# to large.
#
# Whether it fires depends on min_vacuum_size, which has no business touching in-plane
# geometry: on mp-2657(110) the cut is clean at 5/13/25/30/50 A of vacuum and sheared at
# 8/16/20/35/40 (measured, not assumed), and 7 of 12 committed materials shear at the 20 A
# default. The cost is real -- the sheared cell compresses b from 6.5052 to 6.4784 A (0.41%)
# and, since every slab stage runs relax_cell=False, that strain is frozen for the whole
# relaxation. It also breaks the symmetry symm_reduce would otherwise use to merge
# equivalent sites, inflating TiO2(110) from 2 vacancy / 12 adsorbate sites to 3 / 19.
#
# An earlier fix rebuilt the cell with a_z=b_z=0 at fixed fractional coordinates. That only
# moves atoms along z, so it squared up the *box* while leaving the crystal inside it leaning
# by the same 0.09 A per 1 A of depth -- and made the defect harder to spot, because
# lattice.angles then reports a clean 90/90/90. pymatgen's Slab.get_orthogonal_c_slab() moves
# the atoms as well, so make_slab uses that; it is only reachable before Structure.from_sites()
# drops the Slab subclass, which is why the call order below matters.
def _require_orthogonal_c(slab: Structure, context: str, atol_deg: float = 1e-3) -> None:
    """Fail loudly if c isn't perpendicular to the surface plane.

    ``a`` and ``b`` span the surface and ``c`` is the vacuum direction, so alpha and beta
    must be 90 degrees or the slab sits tilted inside its own box. ``gamma`` is deliberately
    NOT checked: it is the surface cell's own shape -- legitimately 72.2 deg on WO2(110) and
    90.8 deg on LaCoO3(110) -- and forcing it to 90 would distort real crystals.
    """
    alpha, beta, gamma = slab.lattice.angles
    off = [f"{n}={v:.4f}" for n, v in (("alpha", alpha), ("beta", beta))
           if abs(v - 90.0) > atol_deg]
    if off:
        raise ValueError(
            f"{context}: slab c-vector is not perpendicular to the surface ({', '.join(off)}; "
            f"gamma={gamma:.4f} is the surface cell shape and is not checked). The slab is "
            "tilted inside its own box, which strains the in-plane lattice and breaks the "
            "symmetry used to merge equivalent sites."
        )


# Cuts the configured facet/termination from a relaxed bulk, replicates it to the configured
# supercell, and freezes the bottom fraction at bulk positions.
def make_slab(bulk: Structure, config: SlabConfig) -> Structure:
    gen = SlabGenerator(
        bulk,
        miller_index=config.miller_index,
        min_slab_size=config.min_slab_size,
        min_vacuum_size=config.min_vacuum_size,
        lll_reduce=config.lll_reduce,
        center_slab=config.center_slab,
    )
    slabs = gen.get_slabs()
    if not slabs:
        raise RuntimeError(f"no slabs generated for miller={config.miller_index}")
    if config.termination_index >= len(slabs):
        raise IndexError(
            f"termination_index {config.termination_index} out of range "
            f"({len(slabs)} terminations available)"
        )
    # get_orthogonal_c_slab() before from_sites(): it lives on pymatgen's Slab subclass,
    # which from_sites() discards. See the shear note above _require_orthogonal_c.
    slab = Structure.from_sites(slabs[config.termination_index].get_orthogonal_c_slab().sites)
    _require_orthogonal_c(slab, f"{config.miller_index} termination {config.termination_index}")
    nx, ny = config.supercell
    if (nx, ny) != (1, 1):
        slab.make_supercell([nx, ny, 1])
    apply_bottom_freeze(slab, config.freeze_bottom_fraction)
    return slab


# --- Vacancy -------------------------------------------------------------------------------

# One unrelaxed structure with a single oxygen removed plus which symmetry-distinct site 
@dataclass
class VacancyCandidate:
    structure: Structure
    site_id: dict  # {symmetry_class, frac_coord, site_index}


# Builds one vacancy candidate per symmetry-distinct surface oxygen site, by removing each
# in turn from its own copy of the slab.
def oxygen_vacancy_candidates(
    slab: Structure, symprec: float = 0.1, freeze_bottom_fraction: float = 0.0,
    block_radius: float | None = 1.3,
) -> list[VacancyCandidate]:
    sga = SpacegroupAnalyzer(slab, symprec=symprec)
    sym = sga.get_symmetrized_structure()
    wyckoffs = getattr(sym, "wyckoff_symbols", None)
    cutoff = bottom_cutoff_z(slab, freeze_bottom_fraction)
    z = slab.cart_coords[:, 2]
    exposed = (
        exposed_surface_atoms(slab, depth=float(z.max() - z.min()), block_radius=block_radius)
        if block_radius is not None else None
    )
    environments = _site_environments(slab)
    candidates: list[VacancyCandidate] = []

    # Iterate through different groups of atoms with equivalent geometry, scan for non-frozen exposed surface oxygens
    for group_idx, group in enumerate(sym.equivalent_indices):
        if str(slab[group[0]].specie) != "O":
            continue
        members = list(group)
        if cutoff is not None:
            members = [i for i in members if slab[i].coords[2] > cutoff + 1e-6]
            if not members:
                continue  # this O class lives only in the frozen region — skip it
        rep = max(members, key=lambda i: slab[i].coords[2]) # Topmost, not lowest-indexed: the adsorbate stage only ever sees the top face.
        if exposed is not None and rep not in exposed:
            continue  # not visible from vacuum -> not a surface vacancy
       
        vac = slab.copy()
        vac.remove_sites([rep])
        apply_bottom_freeze(vac, freeze_bottom_fraction)
        symmetry_class = wyckoffs[group_idx] if wyckoffs else f"group{group_idx}"
        element, _deficit, cn = environments[rep]
        candidates.append(
            VacancyCandidate(
                structure=vac,
                site_id={
                    "symmetry_class": symmetry_class,
                    "site_label": site_label(element, cn),  # e.g. "O2c" = bridging oxygen
                    "frac_coord": [float(x) for x in slab[rep].frac_coords],
                    "site_index": int(rep),
                },
            )
        )
    if not candidates:
        raise RuntimeError(
            "no oxygen exposed to vacuum on this slab — check the termination, or "
            "raise/clear SlabConfig.vacancy_block_radius to enumerate subsurface O too"
            if block_radius is not None
            else "no symmetry-distinct oxygen sites found on slab"
        )
    candidates.sort(key=lambda c: c.site_id["site_index"])
    return candidates


# --- Adsorbate placement: how high above a site to place the fragment ---------------------

# Computes sum of covalent radii as part of a guess for placement height above surface atom
def _covalent_height(surface_symbol: str, adsorbate_symbol: str) -> float:
    return float(
        covalent_radii[atomic_numbers[surface_symbol]]
        + covalent_radii[atomic_numbers[adsorbate_symbol]]
    )

# Returns height placed above surface atom for adsorbate. Depends on the chemistry and preset standoff.
def _target_distance(
    surface_symbol: str, adsorbate_symbol: str, config: AdsorbateConfig
) -> float:
    return _covalent_height(surface_symbol, adsorbate_symbol) + config.seed_standoff


# Takes a site's x,y + pool of exposed atom indices; returns (height, nearest exposed atom)
# for the adsorbate's binding atom. Height = tallest of every exposed atom's own required
# bond distance, so the placement clears the whole neighbourhood not just one reference atom.
# "Nearest" is tracked separately by pure lateral distance — on a bridge site (equidistant
# from both flanking atoms by construction) it can differ from the atom that set the height.
def _placement_z(
    structure: Structure,
    coord,
    pool: "set[int]",
    adsorbate_symbol: str,
    config: AdsorbateConfig,
) -> "tuple[float, int | None]":
    L = structure.lattice
    frac = structure.frac_coords
    cart = structure.cart_coords
    cf = L.get_fractional_coords(coord)  # site's x,y in fractional coords
    solved: "float | None" = None  # tallest required height found so far
    nearest_i, nearest_lat = None, 1e9  # closest exposed atom by lateral distance alone
    for i in pool:
        d = frac[i] - cf
        d[:2] -= np.round(d[:2])  # min-image in-plane
        lateral = float(np.linalg.norm((d @ L.matrix)[:2]))  # sideways-only distance to atom i
        if lateral < nearest_lat:
            nearest_i, nearest_lat = i, lateral  # track closest atom regardless of bonding
        target = _target_distance(str(structure[i].specie), adsorbate_symbol, config)  # this atom's own bond distance
        if lateral < target:  # only atoms close enough to actually constrain the height
            z = cart[i][2] + (target**2 - lateral**2) ** 0.5  # height that puts adsorbate exactly at target dist
            if solved is None or z > solved:
                solved = z  # keep the tallest (binding) requirement
    if solved is not None:
        return float(solved), nearest_i  # height solved by some atom's bond distance
    if nearest_i is None:
        return float(coord[2]), None  # no exposed atoms at all — nothing to reference
    return float(cart[nearest_i][2] + config.min_normal_height), nearest_i  # fallback: flat offset above nearest atom


# Takes placed (coord, anchor) pairs + a cap n; returns n of them chosen to be spread out
# and chemistry-aware, not just pymatgen's spatially-clustered enumeration order (which can
# sample the same environment 3x and miss the canonical site entirely — see site_label).
# Seeds one site per distinct environment first (most undercoordinated first, since that's
# where binding happens), then fills the rest by farthest-point spread. Deterministic.
def _spread_sites(placed: list, n: int | None, lattice, substrate: Structure,
                  environments: dict | None = None) -> list:
    if n is None or n >= len(placed):
        return list(placed)  # no cap needed, keep everything

    # Lateral (x,y) min-image distance between two placed coords; z is set by _placement_z.
    def _sep(a, b) -> float:
        f = lattice.get_fractional_coords(np.asarray(a) - np.asarray(b))
        f[2] = 0.0
        f -= np.round(f)
        return float(np.linalg.norm(f @ lattice.matrix))

    environments = environments if environments is not None else _site_environments(substrate)  # reuse if caller already has it

    # This placed site's (element, deficit, coordination) — the environment it's grouped by.
    def _env(i: int) -> "tuple[str, int, int]":
        site_atom = placed[i][1]
        return environments[site_atom] if site_atom is not None else ("", -1, -1)

    remaining = list(range(len(placed)))  # indices: coords are arrays, so `in`/`remove` break
    kept: list[int] = []
    # Seed: one representative per distinct environment, most undercoordinated (highest deficit) first.
    for env in sorted(dict.fromkeys(_env(i) for i in remaining), key=lambda e: -e[1]):
        if len(kept) == n:
            break  # budget already filled during seeding
        first = next(i for i in remaining if _env(i) == env)  # first placed site with this environment
        remaining.remove(first)
        kept.append(first)
    # Fill whatever budget is left with whichever remaining site is farthest from everything kept.
    while len(kept) < n and remaining:
        far = max(remaining, key=lambda i: min(_sep(placed[i][0], placed[k][0]) for k in kept))
        remaining.remove(far)
        kept.append(far)
    return [placed[i] for i in kept]  # back to (coord, anchor) pairs


# --- Adsorbate: candidates ------------------------------------------------------------

# One unrelaxed structure with a fragment placed at a single surface site, plus which site
# it landed on. Same .structure/.site_id contract as VacancyCandidate, for the shared funnel.
@dataclass
class AdsorbateCandidate:
    structure: Structure
    site_id: dict  # {symmetry_class(=position type), frac_coord, site_index}


# Takes a substrate + adsorbate config; returns one AdsorbateCandidate per distinct exposed
# site (ontop/bridge/hollow). Sites come from a Delaunay triangulation over exposed_surface_atoms
# (not pymatgen's default height window, which misses recessed cations like rutile's 5-fold Ti).
# Placed at every symmetry-reduced site of each type so the funnel's ranking finds the real
# binding site, not just whichever pymatgen lists first. Unrelaxed by construction.
def adsorbate_candidates(
    substrate: Structure, config: AdsorbateConfig, freeze_bottom_fraction: float = 0.0
) -> list[AdsorbateCandidate]:
    molecule = Molecule(list(config.species), [list(c) for c in config.coords])  # the fragment as a rigid molecule
    n_ads = len(molecule)  # atom count of the fragment
    ads_symbol = str(config.species[0])  # the binding atom (coords[0]); sets placement height
    exposed = exposed_surface_atoms(
        substrate, config.surface_depth, config.exposure_block_radius
    )  # which atoms a fragment could actually land on
    # pymatgen's assign_site_properties returns the slab untouched when surface_properties
    # is already present — that's the seam we use to substitute our own surface-atom set.
    props = ["surface" if i in exposed else "subsurface" for i in range(len(substrate))]
    asf = AdsorbateSiteFinder(substrate.copy(site_properties={"surface_properties": props}))  # pymatgen's site-finder, using our exposed set
    cutoff = bottom_cutoff_z(substrate, freeze_bottom_fraction)  # frozen/free boundary for this substrate
    L = substrate.lattice
    # Computed once here rather than per position type inside _spread_sites, and reused to
    # name each surviving site the way literature would (Ti5c, O2c, ...).
    environments = _site_environments(substrate)

    candidates: list[AdsorbateCandidate] = []  # accumulates the final results

    # Smallest adsorbate-slab distance as a fraction of covalent bond length; <1.0 means
    # inside bonding range, well below means inside an atom (unphysical, unrecoverable start).
    def _clearance(structure: Structure) -> float:
        frac = structure.frac_coords
        worst = 1e9  # smallest ratio found so far (start impossibly large)
        for i in range(len(structure) - n_ads, len(structure)):  # the adsorbate atoms (appended last)
            for j in range(len(structure) - n_ads):  # every slab atom
                d = frac[i] - frac[j]
                d -= np.round(d)  # full min-image
                dist = float(np.linalg.norm(d @ structure.lattice.matrix))  # real 3D distance, this pair
                worst = min(
                    worst,
                    dist
                    / _covalent_height(str(structure[j].specie), str(structure[i].specie)),  # ratio to expected bond length
                )
        return worst

    def _add(coord, symmetry_class: str, anchor: "int | None") -> None:
        if cutoff is not None and coord[2] <= cutoff + 1e-6:
            return  # top (free) surface only — a placement on the frozen face can't relax
        ads = asf.add_adsorbate(molecule, coord)  # unrelaxed placement
        # Rebuild plainly: drop AdsorbateSiteFinder's per-site bookkeeping properties
        # (surface_properties, bulk_wyckoff, …) — irrelevant to relaxation and noisy.
        clean = Structure(ads.lattice, ads.species, ads.frac_coords)
        if _clearance(clean) < config.min_clearance:
            return
        # Adsorbate atoms are appended last by add_adsorbate → keep them mobile.
        apply_bottom_freeze(
            clean,
            freeze_bottom_fraction,
            always_free=set(range(len(clean) - n_ads, len(clean))),
        )
        # The atom the site sits over, named the way a paper would name it. Exact for
        # ontop; for bridge/hollow it is the laterally nearest exposed atom, so read it as
        # "this site sits over a Ti5c", not as a full description of a two-atom bridge.
        label = ""
        if anchor is not None and anchor in environments:
            element, _deficit, cn = environments[anchor]
            label = site_label(element, cn)
        candidates.append(
            AdsorbateCandidate(
                structure=clean,
                site_id={
                    "symmetry_class": symmetry_class,
                    "site_label": label,
                    "frac_coord": [float(x) for x in L.get_fractional_coords(coord)],
                    "site_index": len(candidates),  # running index, unique per candidate
                },
            )
        )

    # Only pymatgen's x,y is used — hence no ``distance=`` here; _placement_z sets the height.
    sites = asf.find_adsorption_sites(
        symm_reduce=config.symm_reduce,
        positions=list(config.positions),
    )
    for ptype in config.positions:
        placed = []
        for coord in sites.get(ptype, []):
            z, anchor = _placement_z(substrate, coord, exposed, ads_symbol, config)
            placed.append((np.array([coord[0], coord[1], z]), anchor))
        # Cap each position type to a spread-out subset (all of them when uncapped).
        for coord, anchor in _spread_sites(placed, config.max_per_position, L, substrate,
                                           environments):
            _add(coord, ptype, anchor)

    if not candidates:
        raise RuntimeError("no adsorption sites found on substrate")
    candidates.sort(key=lambda c: c.site_id["site_index"])
    return candidates
