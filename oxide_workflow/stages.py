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


def make_slab(bulk: Structure, config: SlabConfig) -> Structure:
    """Cut the pinned facet/termination from a relaxed bulk → unrelaxed slab (§4)."""
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
    return Structure.from_sites(slabs[config.termination_index].sites)


@dataclass
class VacancyCandidate:
    """One symmetry-distinct O-vacancy: the decorated (unrelaxed) structure + site id."""

    structure: Structure
    site_id: dict  # {symmetry_class, frac_coord, site_index}


def oxygen_vacancy_candidates(
    slab: Structure, symprec: float = 0.1
) -> list[VacancyCandidate]:
    """decorate(slab, O-removal) → one candidate per symmetry-distinct O site (§4).

    Deterministic order (by originating site index) so that two models decorating the
    *same* substrate produce aligned candidate lists (seeded per-stage matching).
    """
    sga = SpacegroupAnalyzer(slab, symprec=symprec)
    sym = sga.get_symmetrized_structure()
    wyckoffs = getattr(sym, "wyckoff_symbols", None)

    candidates: list[VacancyCandidate] = []
    for group_idx, group in enumerate(sym.equivalent_indices):
        rep = group[0]
        if str(slab[rep].specie) != "O":
            continue
        vac = slab.copy()
        vac.remove_sites([rep])
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
    substrate: Structure, config: AdsorbateConfig
) -> list[AdsorbateCandidate]:
    """decorate(substrate, adsorbate placement) → one candidate per adsorption-site type.

    Enumerates ontop/bridge/hollow sites with pymatgen ``AdsorbateSiteFinder`` and places
    the (generic, config-specified) fragment at one symmetry-reduced representative of
    each requested type — AdsorbML's "try several, keep the minimum" philosophy without
    its dependencies (§4). The placement is unrelaxed by construction; relaxation is the
    backend's job. Enumerating on the *same* substrate yields the same ordered site list
    for both models, so seeded per-site matching aligns (as with vacancies).
    """
    molecule = Molecule(list(config.species), [list(c) for c in config.coords])
    asf = AdsorbateSiteFinder(substrate)
    sites = asf.find_adsorption_sites(
        distance=config.site_distance,
        symm_reduce=config.symm_reduce,
        positions=list(config.positions),
    )

    candidates: list[AdsorbateCandidate] = []
    for i, ptype in enumerate(config.positions):  # deterministic: ontop=0, bridge=1, hollow=2
        coords_list = sites.get(ptype, [])
        if not coords_list:
            continue  # a type with no sites on this surface — skip, don't fabricate one
        coord = coords_list[0]  # one representative of this adsorption-site type
        ads = asf.add_adsorbate(molecule, coord)  # unrelaxed placement
        # Rebuild plainly: drop AdsorbateSiteFinder's per-site bookkeeping properties
        # (surface_properties, bulk_wyckoff, …) — irrelevant to relaxation and noisy.
        clean = Structure(ads.lattice, ads.species, ads.frac_coords)
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
