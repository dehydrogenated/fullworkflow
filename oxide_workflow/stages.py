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

from pymatgen.analysis.adsorption import AdsorbateSiteFinder
from pymatgen.core import Molecule, Structure
from pymatgen.core.surface import SlabGenerator
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from .config import AdsorbateConfig, SlabConfig


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
    """decorate(substrate, adsorbate placement) → one candidate per adsorption-site type.

    Enumerates ontop/bridge/hollow sites with pymatgen ``AdsorbateSiteFinder`` and places
    the (generic, config-specified) fragment at one symmetry-reduced representative of
    each requested type — AdsorbML's "try several, keep the minimum" philosophy without
    its dependencies (§4). The placement is unrelaxed by construction; relaxation is the
    backend's job. Enumerating on the *same* substrate yields the same ordered site list
    for both models, so seeded per-site matching aligns (as with vacancies).

    ``freeze_bottom_fraction`` re-imposes the slab's bottom-layer freeze after the plain
    rebuild (which drops site properties); the adsorbate atoms sit above the surface and
    are always left mobile. It also restricts placement to the free (top) surface — an
    adsorbate on the frozen face would sit against rigid atoms and cannot relax.
    """
    molecule = Molecule(list(config.species), [list(c) for c in config.coords])
    n_ads = len(molecule)
    asf = AdsorbateSiteFinder(substrate)
    sites = asf.find_adsorption_sites(
        distance=config.site_distance,
        symm_reduce=config.symm_reduce,
        positions=list(config.positions),
    )
    cutoff = bottom_cutoff_z(substrate, freeze_bottom_fraction)

    candidates: list[AdsorbateCandidate] = []
    for i, ptype in enumerate(config.positions):  # deterministic: ontop=0, bridge=1, hollow=2
        coords_list = sites.get(ptype, [])
        if cutoff is not None:
            coords_list = [c for c in coords_list if c[2] > cutoff + 1e-6]  # top surface only
        if not coords_list:
            continue  # a type with no sites on this surface — skip, don't fabricate one
        coord = coords_list[0]  # one representative of this adsorption-site type
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
                    "symmetry_class": ptype,
                    "frac_coord": [
                        float(x) for x in substrate.lattice.get_fractional_coords(coord)
                    ],
                    "site_index": i,
                },
            )
        )
    if not candidates:
        raise RuntimeError("no adsorption sites found on substrate")
    candidates.sort(key=lambda c: c.site_id["site_index"])
    return candidates
