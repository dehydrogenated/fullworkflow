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
from typing import Optional

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

    Explicit per-species control via ``config.site_distance_by_species`` (pin O vs metal,
    etc.); elements not listed fall back to a covalent-radii estimate + ``seed_standoff`` —
    the material-agnostic default, so a batch sweep over many compositions needs no manual
    tuning. This is a *distance*, not a vertical height; the normal placement height is
    solved from it in ``_normal_height``.
    """
    table = dict(config.site_distance_by_species)
    if surface_symbol in table:
        return float(table[surface_symbol])
    return _covalent_height(surface_symbol, adsorbate_symbol) + config.seed_standoff


def _normal_height(target_distance: float, lateral: float, floor: float) -> float:
    """Surface-normal height that puts the adsorbate at ``target_distance`` from an atom
    sitting ``lateral`` Å away in-plane: ``sqrt(d² − r²)``, floored.

    This is what makes placement lattice-aware rather than purely chemical: for an ontop
    site (``lateral`` ≈ 0) the height is the full bond length, but for a bridge/hollow site
    the adsorbate is already laterally displaced from its coordinating atom, so it sits
    *lower* to keep the same bond length — and how much lower is set by the atom spacing,
    i.e. the lattice. When the lateral offset exceeds the target (the adsorbate can't reach
    the atom vertically), clamp to ``floor``.
    """
    if lateral >= target_distance:
        return floor
    return float(max((target_distance**2 - lateral**2) ** 0.5, floor))


def _nearest_surface_atom(
    structure: Structure, coord, depth: float
) -> "tuple[int, float] | None":
    """(index, lateral distance Å) of the top-layer atom closest *laterally* to a site.

    Arm-1's geometric sites (ontop/bridge/hollow) come out at a flat plane-referenced
    height; to make placement species- and lattice-aware we identify the coordinating
    surface atom the site sits over (nearest in-plane) and re-reference the placement to
    that atom's z + a solved normal height. ``depth`` is the top-layer window: keep it thin
    (``surface_layer_tol``) so a bridge/hollow midpoint keys off the topmost flanking atoms
    (the bridging O's) rather than the recessed cation directly beneath it. For an ontop
    site this is the atom directly below (lateral ≈ 0); for bridge/hollow it is the nearest
    top-layer atom (lateral > 0).
    """
    L = structure.lattice
    cart = structure.cart_coords
    frac = structure.frac_coords
    zmax = float(cart[:, 2].max())
    cf = L.get_fractional_coords(coord)
    best_i, best_d = None, 1e9
    for i in range(len(structure)):
        if cart[i][2] <= zmax - depth:
            continue
        d = frac[i] - cf
        d[:2] -= np.round(d[:2])  # min-image in-plane
        lateral = float(np.linalg.norm((d @ L.matrix)[:2]))
        if lateral < best_d:
            best_i, best_d = i, lateral
    if best_i is None:
        return None
    return best_i, best_d


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
    slab: Structure, symprec: float = 0.1, freeze_bottom_fraction: float = 0.0
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
        candidates.append(
            VacancyCandidate(
                structure=vac,
                site_id={
                    "symmetry_class": symmetry_class,
                    "frac_coord": [float(x) for x in slab[rep].frac_coords],
                    "site_index": int(rep),
                },
            )
        )
    if not candidates:
        raise RuntimeError("no symmetry-distinct oxygen sites found on slab")
    candidates.sort(key=lambda c: c.site_id["site_index"])
    return candidates


@dataclass
class AdsorbateCandidate:
    """One adsorption-site placement: the decorated (unrelaxed) structure + site id.

    Same ``.structure`` / ``.site_id`` contract as ``VacancyCandidate`` so both flow
    through the pipeline's shared funnel (the ``decorate`` seam).
    """

    structure: Structure
    site_id: dict  # {symmetry_class(=position type), frac_coord, site_index}


def adsorbate_candidates(
    substrate: Structure, config: AdsorbateConfig, freeze_bottom_fraction: float = 0.0
) -> list[AdsorbateCandidate]:
    """decorate(substrate, adsorbate placement) → one candidate per distinct surface site.

    Enumerates ontop/bridge/hollow sites with pymatgen ``AdsorbateSiteFinder`` and places
    the (generic, config-specified) fragment at *every* symmetry-reduced representative of
    each requested type — AdsorbML's "try several, keep the minimum" philosophy without its
    dependencies (§4). This densified sampling (all reps, not just the first of each type)
    is what lets the funnel's energy ranking discover the real binding site instead of
    betting the run on whichever site pymatgen happened to list first. The placement is
    unrelaxed by construction; relaxation is the backend's job. Enumerating on the *same*
    substrate yields the same ordered site list for both models, so seeded per-site matching
    aligns (as with vacancies).

    ``freeze_bottom_fraction`` re-imposes the slab's bottom-layer freeze after the plain
    rebuild (which drops site properties); the adsorbate atoms sit above the surface and
    are always left mobile. It also restricts placement to the free (top) surface — an
    adsorbate on the frozen face would sit against rigid atoms and cannot relax.
    """
    molecule = Molecule(list(config.species), [list(c) for c in config.coords])
    n_ads = len(molecule)
    ads_symbol = str(config.species[0])  # the binding atom (coords[0]); sets seed height
    asf = AdsorbateSiteFinder(substrate)
    cutoff = bottom_cutoff_z(substrate, freeze_bottom_fraction)
    L = substrate.lattice

    candidates: list[AdsorbateCandidate] = []
    anchors: list[np.ndarray] = []  # cartesian anchor of each accepted placement (dedup key)

    def _is_duplicate(coord) -> bool:
        cf = L.get_fractional_coords(coord)
        for a in anchors:
            d = cf - L.get_fractional_coords(a)
            d -= np.round(d)  # full min-image (in-plane + normal)
            if np.linalg.norm(d @ L.matrix) < config.dedup_tol:
                return True
        return False

    def _add(coord, symmetry_class: str) -> None:
        if cutoff is not None and coord[2] <= cutoff + 1e-6:
            return  # top (free) surface only — a placement on the frozen face can't relax
        if _is_duplicate(coord):
            return  # arm-1/arm-2 (or within-arm) coincidence — keep the first-seen
        ads = asf.add_adsorbate(molecule, coord)  # unrelaxed placement
        # Rebuild plainly: drop AdsorbateSiteFinder's per-site bookkeeping properties
        # (surface_properties, bulk_wyckoff, …) — irrelevant to relaxation and noisy.
        clean = Structure(ads.lattice, ads.species, ads.frac_coords)
        # Adsorbate atoms are appended last by add_adsorbate → keep them mobile.
        apply_bottom_freeze(
            clean,
            freeze_bottom_fraction,
            always_free=set(range(len(clean) - n_ads, len(clean))),
        )
        candidates.append(
            AdsorbateCandidate(
                structure=clean,
                site_id={
                    "symmetry_class": symmetry_class,
                    "frac_coord": [float(x) for x in L.get_fractional_coords(coord)],
                    "site_index": len(candidates),  # running index, unique per candidate
                },
            )
        )
        anchors.append(np.asarray(coord, dtype=float))

    # Arm 1 — geometric (ontop/bridge/hollow), densified: every symmetry-reduced rep of each
    # type, in deterministic order (config.positions, then pymatgen's list order per type).
    # pymatgen returns each site at a flat plane-referenced height; we re-reference the
    # normal distance to the coordinating surface atom so it obeys the per-species standoff
    # (an H over an O starts lower than an H over a metal).
    sites = asf.find_adsorption_sites(
        distance=config.site_distance,
        symm_reduce=config.symm_reduce,
        positions=list(config.positions),
    )
    for ptype in config.positions:
        for coord in sites.get(ptype, []):
            near = _nearest_surface_atom(substrate, coord, config.surface_layer_tol)
            if near is not None:
                base, lateral = near
                specie = str(substrate[base].specie)
                d0 = _target_distance(specie, ads_symbol, config)
                h = _normal_height(d0, lateral, config.min_normal_height)
                z = substrate[base].coords[2] + h
                coord = np.array([coord[0], coord[1], z])
            _add(coord, ptype)

    # Arm 2 — seeding over symmetry-distinct *exposed* surface atoms. Recovers chemically-real
    # sites the height-window finder misses (e.g. rutile 5-fold Ti). Placed directly above the
    # atom at its per-species standoff (_species_offset); deduped against arm 1.
    if config.seed_surface_atoms:
        exposed = exposed_surface_atoms(
            substrate, config.surface_depth, config.exposure_block_radius
        )
        if exposed:
            sym = SpacegroupAnalyzer(substrate, symprec=0.1).get_symmetrized_structure()
            wyckoffs = getattr(sym, "wyckoff_symbols", None)
            for group_idx, group in enumerate(sym.equivalent_indices):
                members = sorted(i for i in group if i in exposed)
                if not members:
                    continue
                rep = members[0]  # lowest-index exposed representative → deterministic
                specie = str(substrate[rep].specie)
                wy = wyckoffs[group_idx] if wyckoffs else f"g{group_idx}"
                # Directly atop the atom → lateral 0 → normal height is the full target.
                height = _target_distance(specie, ads_symbol, config)
                coord = np.asarray(substrate[rep].coords, dtype=float) + np.array(
                    [0.0, 0.0, height]
                )
                _add(coord, f"atop_{specie}{wy}")

    if not candidates:
        raise RuntimeError("no adsorption sites found on substrate")
    candidates.sort(key=lambda c: c.site_id["site_index"])
    return candidates
