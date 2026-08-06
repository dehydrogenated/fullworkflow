"""Stage structure-building: cut a slab, and the decorate() vacancy interface (§4).

Each stage's starting structure = previous stage's relaxed output + a fresh
modification, unrelaxed by construction. This module produces those *unrelaxed*
starting structures; relaxation is the backend's job (design §3).

``decorate(substrate, modification)`` is the shared vacancy/adsorbate seam. Both the
vacancy modification (remove a representative O) and the adsorbate modification (place a
fragment at a heuristic height on the relaxed substrate) are implemented here, each
producing *unrelaxed* candidates keyed by an abstract site identity — symmetry class +
fractional coordinate, never line numbers.
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


def _covalent_height(surface_symbol: str, adsorbate_symbol: str) -> float:
    """Placement height above a surface atom = sum of covalent radii (Å), material-agnostic."""
    return float(
        covalent_radii[atomic_numbers[surface_symbol]]
        + covalent_radii[atomic_numbers[adsorbate_symbol]]
    )


def _target_distance(
    surface_symbol: str, adsorbate_symbol: str, config: AdsorbateConfig
) -> float:
    """Target adsorbate–surface *bond distance* (Å) to the coordinating atom.

    Covalent radii + ``seed_standoff`` — element-driven, so a sweep over many compositions
    needs no manual tuning. This is a *distance*, not a vertical height; the normal
    placement height is solved from it in ``_placement_z``.
    """
    return _covalent_height(surface_symbol, adsorbate_symbol) + config.seed_standoff


def _placement_z(
    structure: Structure,
    coord,
    pool: "set[int]",
    adsorbate_symbol: str,
    config: AdsorbateConfig,
) -> "tuple[float, int | None]":
    """(height Å, nearest exposed atom) for the adsorbate's binding atom over a site's x,y.

    The second value is the laterally closest exposed atom — what the site sits *over*.
    ``_spread_sites`` groups on its element so site selection can tell a Ti-topped site
    from an O-topped one. It is deliberately not the atom that set the height: Ti's larger
    covalent radius makes it the height constraint for most sites on rutile(110), including
    ones directly above a bridging O, so height would not discriminate.

    pymatgen returns each site at a flat plane-referenced height; only its x,y is kept and
    the height re-solved here, so placement is species- and lattice-aware. For every exposed
    atom, solve the z that would put the adsorbate exactly at its target bond distance
    (``sqrt(d² − lateral²)`` above it) and take the largest. The winning atom is then the
    binding constraint and every other one is at or beyond its own bond distance, so the
    placement clears the whole neighbourhood rather than a single reference atom.

    Solving against the *nearest* atom instead is ambiguous exactly where it matters: a
    bridge site is equidistant from both flanking atoms by construction, so when those sit
    at different heights (the bridging O vs the recessed Ti on rutile(110)) the tie-break
    moves the adsorbate by over an Å and can drop it inside the higher neighbour.

    When no exposed atom is within its bond distance laterally there is nothing to solve
    against — fall back to ``min_normal_height`` above the laterally closest one.
    """
    L = structure.lattice
    frac = structure.frac_coords
    cart = structure.cart_coords
    cf = L.get_fractional_coords(coord)
    solved: "float | None" = None
    nearest_i, nearest_lat = None, 1e9
    for i in pool:
        d = frac[i] - cf
        d[:2] -= np.round(d[:2])  # min-image in-plane
        lateral = float(np.linalg.norm((d @ L.matrix)[:2]))
        if lateral < nearest_lat:
            nearest_i, nearest_lat = i, lateral
        target = _target_distance(str(structure[i].specie), adsorbate_symbol, config)
        if lateral < target:
            z = cart[i][2] + (target**2 - lateral**2) ** 0.5
            if solved is None or z > solved:
                solved = z
    if solved is not None:
        return float(solved), nearest_i
    if nearest_i is None:
        return float(coord[2]), None
    return float(cart[nearest_i][2] + config.min_normal_height), nearest_i


def exposed_surface_atoms(
    structure: Structure,
    depth: float,
    block_radius: float,
    occlusion_dz: float = 0.3,
) -> "set[int]":
    """Indices of atoms exposed to the vacuum from the +z (top) side.

    An atom is *exposed* if no other atom sits above it (``occlusion_dz`` higher in z)
    within ``block_radius`` laterally (min-image xy). This is a geometric line-of-sight
    test, not a height window: it catches recessed-but-accessible cations (e.g. rutile
    5-fold Ti) that a top-slice would miss, while naturally excluding subsurface atoms
    (occluded by the layers above) and the bottom face (occluded by the whole slab).
    ``depth`` restricts the scan to the top region for cost; the occlusion test does the
    real work.
    """
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


def bottom_cutoff_z(
    structure: Structure, fraction: float, always_free: "set[int] | None" = None
) -> "float | None":
    """z below which slab atoms are frozen, or ``None`` when freezing is disabled.

    The cutoff is measured over the *slab* atoms only (``always_free`` — e.g. an adsorbate
    placed above the surface — is excluded from the z-range). Shared by the freeze itself
    and by the site enumerators, so "frozen region" means the same thing everywhere.
    """
    if not 0.0 < fraction <= 1.0:
        return None
    always_free = always_free or set()
    slab_z = [s.coords[2] for i, s in enumerate(structure) if i not in always_free]
    zmin, zmax = min(slab_z), max(slab_z)
    return zmin + fraction * (zmax - zmin)


def apply_bottom_freeze(
    structure: Structure, fraction: float, always_free: "set[int] | None" = None
) -> None:
    """Fix the bottom ``fraction`` of the slab's atomic thickness at bulk positions (§4).

    Standard DFT surface convention (F2B-style asymmetric slab): the lower part of the
    slab is held at bulk geometry to mimic bulk hardness while the top relaxes toward
    vacuum. Implemented as a pymatgen ``selective_dynamics`` site property — the POSCAR
    writer emits the flags and ASE reads them straight into ``FixAtoms`` in the worker,
    so no backend change is needed. In-place; sets ``selective_dynamics`` on every site.
    """
    cutoff = bottom_cutoff_z(structure, fraction, always_free)
    if cutoff is None:
        return  # freezing disabled / degenerate → leave every atom free (no property set)
    always_free = always_free or set()
    flags = []
    for i, site in enumerate(structure):
        frozen = i not in always_free and site.coords[2] <= cutoff + 1e-6
        flags.append([not frozen] * 3)  # [F,F,F] = fixed, [T,T,T] = free
    structure.add_site_property("selective_dynamics", flags)


def make_slab(bulk: Structure, config: SlabConfig) -> Structure:
    """Cut the pinned facet/termination from a relaxed bulk → unrelaxed slab (§4).

    Then replicate laterally to the configured supercell (coverage dilution for defects
    and adsorbates) and freeze the bottom fraction of the slab at bulk positions.
    """
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
    slab = Structure.from_sites(slabs[config.termination_index].sites)
    nx, ny = config.supercell
    if (nx, ny) != (1, 1):
        slab.make_supercell([nx, ny, 1])
    apply_bottom_freeze(slab, config.freeze_bottom_fraction)
    return slab


@dataclass
class VacancyCandidate:
    """One symmetry-distinct O-vacancy: the decorated (unrelaxed) structure + site id."""

    structure: Structure
    site_id: dict  # {symmetry_class, frac_coord, site_index}


def oxygen_vacancy_candidates(
    slab: Structure, symprec: float = 0.1, freeze_bottom_fraction: float = 0.0,
    max_sites: int | None = None,
) -> list[VacancyCandidate]:
    """decorate(slab, O-removal) → one candidate per symmetry-distinct O site (§4).

    Deterministic order (by originating site index) so that two models decorating the
    *same* substrate produce aligned candidate lists (seeded per-stage matching).
    ``freeze_bottom_fraction`` re-imposes the slab's bottom-layer freeze on each
    candidate so the relaxation keeps the F2B convention, and restricts enumeration to
    O sites on the free (top) surface — a vacancy inside the frozen region cannot relax,
    so it is not a meaningful candidate. Because the slab is top/bottom symmetric, a
    symmetry class can span both surfaces; such a class is kept, represented by its
    free-region member.
    """
    sga = SpacegroupAnalyzer(slab, symprec=symprec)
    sym = sga.get_symmetrized_structure()
    wyckoffs = getattr(sym, "wyckoff_symbols", None)
    cutoff = bottom_cutoff_z(slab, freeze_bottom_fraction)
    # Coordination of the O being removed, so a vacancy can be named the way literature
    # names it: on rutile(110) the classic defect is a missing bridging O2c, and an O3c
    # vacancy is a different (in-plane, much less favourable) thing entirely.
    environments = _site_environments(slab)

    candidates: list[VacancyCandidate] = []
    for group_idx, group in enumerate(sym.equivalent_indices):
        if str(slab[group[0]].specie) != "O":
            continue
        members = list(group)
        if cutoff is not None:
            members = [i for i in members if slab[i].coords[2] > cutoff + 1e-6]
            if not members:
                continue  # this O class lives only in the frozen region — skip it
        rep = members[0]  # free-region representative (lowest index → deterministic)
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
        raise RuntimeError("no symmetry-distinct oxygen sites found on slab")
    candidates.sort(key=lambda c: c.site_id["site_index"])
    # Smoke-test cap. Deliberately a plain head slice, not a spread like the adsorbate
    # sampler: vacancy classes are already symmetry-distinct, so there is no clustering to
    # correct for, and keeping the lowest site indices stays deterministic. It CAN drop the
    # true minimum-energy vacancy, so capped runs are for plumbing checks, not for numbers.
    return candidates[:max_sites] if max_sites else candidates


@dataclass
class AdsorbateCandidate:
    """One adsorption-site placement: the decorated (unrelaxed) structure + site id.

    Same ``.structure`` / ``.site_id`` contract as ``VacancyCandidate`` so both flow
    through the pipeline's shared funnel (the ``decorate`` seam).
    """

    structure: Structure
    site_id: dict  # {symmetry_class(=position type), frac_coord, site_index}


_CN_CUTOFF = 2.6  # Å; first cation-anion shell (rutile Ti-O is 1.95-2.00, second shell is >3)


def site_label(element: str, coordination: int) -> str:
    """Literature-style coordination label for a surface atom, e.g. ``"Ti5c"``.

    The standard surface-science shorthand: element symbol, number of counter-ion nearest
    neighbours, then "c" for coordinate. On rutile(110) the reactive sites are the
    five-coordinate cation ``Ti5c`` (bulk Ti is octahedral, 6-coordinate) and the bridging
    oxygen ``O2c`` (bulk O is 3-coordinate); ``Ti6c`` and ``O3c`` are bulk-like and inert.

    What counts as undercoordinated is a property of the structure, not of the element —
    in a tetrahedral oxide ``M4c`` would be the saturated one. That is why the pipeline
    ranks sites by *deficit* (see ``_site_environments``) and uses this label only for
    reporting: the label is what you compare against a paper, the deficit is what drives
    the sampling.
    """
    return f"{element}{coordination}c"


def _site_environments(substrate: Structure) -> "dict[int, tuple[str, int, int]]":
    """(element, coordination deficit, coordination) per atom — a surface site's identity.

    Deficit is counted against the most-coordinated atom of the same element in the cell,
    which is bulk-like by construction, so undercoordinated surface atoms score higher
    without hard-coding any polymorph's bulk coordination numbers. On rutile(110) this
    separates the reactive Ti5c and bridging O2c from the saturated Ti6c and O3c.

    The absolute coordination is carried alongside so the site can also be *named* the way
    literature names it (``site_label``). Deficit drives selection because it is
    material-agnostic; the absolute number is for the human reading the output.
    """
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


def _spread_sites(placed: list, n: int | None, lattice, substrate: Structure,
                  environments: dict | None = None) -> list:
    """Keep ``n`` of the ``(coord, site atom)`` pairs: one per environment, then spread.

    A plain ``[:n]`` slice takes pymatgen's enumeration order, which is spatially
    correlated: on a 4x2 rutile(110) cell the first three bridge sites land inside a
    ~1 A patch, so "3 bridge sites" is really one environment sampled three times.
    Farthest-point sampling fixes the clustering but is blind to chemistry, and that loses
    the site the run exists to measure: only 3 of 16 ontop sites on the rutile(110) vacancy
    substrate sit over a 5-fold Ti, so every cap below 6 dropped the canonical adsorption
    site while keeping four views of the same O-topped environment.

    So seed one site per distinct environment first, most coordinatively unsaturated first
    — that is where an adsorbate binds, and it puts Ti5c ahead of Ti6c even though pymatgen
    enumerates a Ti6c site first — then fill the remaining budget by farthest-point spread.

    Deterministic — same input order gives the same subset, so runs stay reproducible.
    """
    if n is None or n >= len(placed):
        return list(placed)

    def _sep(a, b) -> float:
        """Lateral (ab-plane) min-image distance; z is set later by _placement_z."""
        f = lattice.get_fractional_coords(np.asarray(a) - np.asarray(b))
        f[2] = 0.0
        f -= np.round(f)
        return float(np.linalg.norm(f @ lattice.matrix))

    environments = environments if environments is not None else _site_environments(substrate)

    def _env(i: int) -> "tuple[str, int, int]":
        site_atom = placed[i][1]
        return environments[site_atom] if site_atom is not None else ("", -1, -1)

    remaining = list(range(len(placed)))  # indices: coords are arrays, so `in`/`remove` break
    kept: list[int] = []
    for env in sorted(dict.fromkeys(_env(i) for i in remaining), key=lambda e: -e[1]):
        if len(kept) == n:
            break
        first = next(i for i in remaining if _env(i) == env)
        remaining.remove(first)
        kept.append(first)
    while len(kept) < n and remaining:
        far = max(remaining, key=lambda i: min(_sep(placed[i][0], placed[k][0]) for k in kept))
        remaining.remove(far)
        kept.append(far)
    return [placed[i] for i in kept]


def adsorbate_candidates(
    substrate: Structure, config: AdsorbateConfig, freeze_bottom_fraction: float = 0.0
) -> list[AdsorbateCandidate]:
    """decorate(substrate, adsorbate placement) → one candidate per distinct surface site.

    Sites come from a Delaunay triangulation over the atoms the vacuum can *see*
    (``exposed_surface_atoms``), not over pymatgen's default height window. That matters on
    a rumpled oxide: the window admits only the topmost row (the bridging O's on
    rutile(110)), so the 5-fold Ti sitting ~1 Å lower — the canonical adsorption site —
    never becomes a vertex, and every bridge/hollow is a midpoint of the top-O mesh alone.
    Feeding the exposed set in instead makes Ti ontop, Ti–O bridge and hollow sites all
    first-class candidates from one algorithm.

    The fragment is placed at *every* symmetry-reduced representative of each requested type
    — AdsorbML's "try several, keep the minimum" philosophy without its dependencies (§4) —
    so the funnel's energy ranking discovers the real binding site instead of betting the
    run on whichever site pymatgen happened to list first. Placements are unrelaxed by
    construction; relaxation is the backend's job. Enumerating on the *same* substrate
    yields the same ordered site list for both models, so seeded per-site matching aligns
    (as with vacancies).

    ``freeze_bottom_fraction`` re-imposes the slab's bottom-layer freeze after the plain
    rebuild (which drops site properties); the adsorbate atoms sit above the surface and
    are always left mobile. It also restricts placement to the free (top) surface — an
    adsorbate on the frozen face would sit against rigid atoms and cannot relax.
    """
    molecule = Molecule(list(config.species), [list(c) for c in config.coords])
    n_ads = len(molecule)
    ads_symbol = str(config.species[0])  # the binding atom (coords[0]); sets placement height
    exposed = exposed_surface_atoms(
        substrate, config.surface_depth, config.exposure_block_radius
    )
    # pymatgen's assign_site_properties returns the slab untouched when surface_properties
    # is already present — that's the seam we use to substitute our own surface-atom set.
    props = ["surface" if i in exposed else "subsurface" for i in range(len(substrate))]
    asf = AdsorbateSiteFinder(substrate.copy(site_properties={"surface_properties": props}))
    cutoff = bottom_cutoff_z(substrate, freeze_bottom_fraction)
    L = substrate.lattice
    # Computed once here rather than per position type inside _spread_sites, and reused to
    # name each surviving site the way literature would (Ti5c, O2c, ...).
    environments = _site_environments(substrate)

    candidates: list[AdsorbateCandidate] = []

    def _clearance(structure: Structure) -> float:
        """Smallest adsorbate–slab distance as a fraction of that pair's covalent bond
        length. Below 1.0 the adsorbate is inside bonding range; well below it is *inside*
        a surface atom, which is an unphysical start no relaxation can be trusted to fix."""
        frac = structure.frac_coords
        worst = 1e9
        for i in range(len(structure) - n_ads, len(structure)):
            for j in range(len(structure) - n_ads):
                d = frac[i] - frac[j]
                d -= np.round(d)  # full min-image
                dist = float(np.linalg.norm(d @ structure.lattice.matrix))
                worst = min(
                    worst,
                    dist
                    / _covalent_height(str(structure[j].specie), str(structure[i].specie)),
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
