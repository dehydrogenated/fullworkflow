"""Stage structure-building: cut a slab, and the decorate() vacancy interface (§4).

Each stage's starting structure = previous stage's relaxed output + a fresh
modification, unrelaxed by construction. This module produces those *unrelaxed*
starting structures; relaxation is the backend's job (design §3).

``decorate(substrate, modification)`` is the shared vacancy/adsorbate seam. Here the
vacancy modification is implemented; adsorbate placement is the next slice (§8).
Site identity is stored as symmetry class + fractional coordinate — never line numbers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from pymatgen.core import Structure
from pymatgen.core.surface import SlabGenerator
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from .config import SlabConfig


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
