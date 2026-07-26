"""Step 5 (fast, model-free): slab cut + symmetry-distinct O-vacancy enumeration and
adsorbate-site placement (the two ``decorate`` modifications)."""

from __future__ import annotations

from oxide_workflow.config import AdsorbateConfig, SlabConfig
from oxide_workflow.stages import (
    adsorbate_candidates,
    make_slab,
    oxygen_vacancy_candidates,
)
from oxide_workflow.structures import rutile_tio2


def test_make_slab_110_is_stoichiometric_tio2():
    slab = make_slab(rutile_tio2(), SlabConfig())
    assert slab.composition.reduced_formula == "TiO2"
    # A vacuum gap exists: c is much larger than the in-plane vectors.
    a, b, c = slab.lattice.abc
    assert c > max(a, b) + 8.0


def test_vacancy_candidates_are_distinct_and_carry_site_identity():
    slab = make_slab(rutile_tio2(), SlabConfig())
    cands = oxygen_vacancy_candidates(slab)

    assert len(cands) >= 1
    for c in cands:
        # Exactly one O removed from the slab.
        assert len(c.structure) == len(slab) - 1
        # Site identity is symmetry class + fractional coordinate (never line numbers).
        assert set(c.site_id) == {"symmetry_class", "frac_coord", "site_index"}
        assert len(c.site_id["frac_coord"]) == 3
    # Deterministic ordering by site index (enables seeded per-stage site matching).
    idxs = [c.site_id["site_index"] for c in cands]
    assert idxs == sorted(idxs)


def test_adsorbate_candidates_carry_site_identity():
    slab = make_slab(rutile_tio2(), SlabConfig())
    cfg = AdsorbateConfig()  # single H, ontop/bridge/hollow
    cands = adsorbate_candidates(slab, cfg)

    n_ads = len(cfg.species)
    assert 1 <= len(cands) <= len(cfg.positions)
    for c in cands:
        # The fragment's atoms were added on top of the substrate (unrelaxed placement).
        assert len(c.structure) == len(slab) + n_ads
        # Site identity is symmetry class + fractional coordinate (never line numbers).
        assert set(c.site_id) == {"symmetry_class", "frac_coord", "site_index"}
        assert c.site_id["symmetry_class"] in cfg.positions
        assert len(c.site_id["frac_coord"]) == 3
    # Deterministic ordering by site index (enables seeded per-site matching).
    idxs = [c.site_id["site_index"] for c in cands]
    assert idxs == sorted(idxs)


def test_termination_index_out_of_range_raises():
    import pytest

    with pytest.raises(IndexError):
        make_slab(rutile_tio2(), SlabConfig(termination_index=99))
