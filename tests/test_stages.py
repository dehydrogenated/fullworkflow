"""Step 5 (fast, model-free): slab cut + symmetry-distinct O-vacancy enumeration and
adsorbate-site placement (the two ``decorate`` modifications)."""

from __future__ import annotations

import re

import numpy as np
import pytest
from pymatgen.core.adsorption import AdsorbateSiteFinder

from oxide_workflow.config import AdsorbateConfig, SlabConfig
from oxide_workflow.stages import (
    _site_environments,
    adsorbate_candidates,
    exposed_surface_atoms,
    make_slab,
    oxygen_vacancy_candidates,
    site_label,
)
from oxide_workflow.structures import get_structure


def test_make_slab_110_is_stoichiometric_tio2():
    slab = make_slab(get_structure("mp-2657"), SlabConfig())
    assert slab.composition.reduced_formula == "TiO2"
    # A vacuum gap exists: c is much larger than the in-plane vectors.
    a, b, c = slab.lattice.abc
    assert c > max(a, b) + 8.0


def test_vacancy_candidates_are_distinct_and_carry_site_identity():
    slab = make_slab(get_structure("mp-2657"), SlabConfig())
    cands = oxygen_vacancy_candidates(slab)

    assert len(cands) >= 1
    for c in cands:
        # Exactly one O removed from the slab.
        assert len(c.structure) == len(slab) - 1
        # Site identity is symmetry class + fractional coordinate (never line numbers).
        # both identities: the geometric class AND the chemical coordination label
        assert set(c.site_id) == {"symmetry_class", "site_label", "frac_coord", "site_index"}
        assert re.fullmatch(r"[A-Z][a-z]?\d+c", c.site_id["site_label"]), c.site_id
        assert len(c.site_id["frac_coord"]) == 3
    # Deterministic ordering by site index (enables seeded per-stage site matching).
    idxs = [c.site_id["site_index"] for c in cands]
    assert idxs == sorted(idxs)


def test_vacancy_enumeration_is_limited_to_the_surface():
    """Default keeps the two surface O shells; clearing block_radius exposes subsurface O."""
    slab = make_slab(get_structure("mp-2657"), SlabConfig())

    surface = oxygen_vacancy_candidates(slab, block_radius=1.3)
    z = slab.cart_coords[:, 2]
    exposed = exposed_surface_atoms(slab, depth=float(z.max() - z.min()), block_radius=1.3)
    assert all(c.site_id["site_index"] in exposed for c in surface)
    # The bridging O2c and the in-plane O3c — the funnel needs both to have a ranking.
    assert {c.site_id["site_label"] for c in surface} == {"O2c", "O3c"}

    everything = oxygen_vacancy_candidates(slab, block_radius=None)
    assert len(everything) > len(surface)  # subsurface classes are the difference


def test_vacancy_sites_come_from_the_top_face_at_any_freeze_fraction():
    """The adsorbate only ever lands on the top face, so the vacancy must be cut there.

    Representatives were previously chosen by lowest site index, which runs bottom-up: at
    freeze=0 that put every vacancy on the *bottom* surface.
    """
    slab = make_slab(get_structure("mp-2657"), SlabConfig())
    z_top = max(s.coords[2] for s in slab)

    for freeze in (0.5, 0.3, 0.0):
        cands = oxygen_vacancy_candidates(slab, freeze_bottom_fraction=freeze)
        assert cands
        for c in cands:
            depth = z_top - slab[c.site_id["site_index"]].coords[2]
            assert depth <= 1.8, (freeze, c.site_id, depth)


def test_adsorbate_candidates_carry_site_identity():
    slab = make_slab(get_structure("mp-2657"), SlabConfig())
    cfg = AdsorbateConfig()  # single H, ontop/bridge/hollow
    cands = adsorbate_candidates(slab, cfg)

    n_ads = len(cfg.species)
    # Densified sampling: every symmetry-reduced rep of each requested type, so the count
    # is at least one per available type (typically several more), not capped at one each.
    assert len(cands) >= 1
    for c in cands:
        # The fragment's atoms were added on top of the substrate (unrelaxed placement).
        assert len(c.structure) == len(slab) + n_ads
        # Site identity is symmetry class + fractional coordinate (never line numbers).
        # both identities: the geometric class AND the chemical coordination label
        assert set(c.site_id) == {"symmetry_class", "site_label", "frac_coord", "site_index"}
        assert re.fullmatch(r"[A-Z][a-z]?\d+c", c.site_id["site_label"]), c.site_id
        # Arm 1 uses a geometric type; arm 2 uses an ``atop_<element><wyckoff>`` label.
        sc = c.site_id["symmetry_class"]
        assert sc in cfg.positions or sc.startswith("atop_")
        assert len(c.site_id["frac_coord"]) == 3
    # Deterministic ordering by site index (enables seeded per-site matching).
    idxs = [c.site_id["site_index"] for c in cands]
    assert idxs == sorted(idxs)


def test_termination_index_out_of_range_raises():
    with pytest.raises(IndexError):
        make_slab(get_structure("mp-2657"), SlabConfig(termination_index=99))


# --- the triangulation vertex set -------------------------------------------------
#
# ``exposed_surface_atoms`` is the only thing standing between the slab and pymatgen's
# Delaunay mesh: whatever it returns becomes the vertex set, and the mesh is built on
# the *2D projection* of those vertices (pymatgen drops the out-of-plane coordinate),
# so a vertex set spanning several Å in z is triangulated as if it were coplanar. These
# tests pin the set down on a fixture where the right answer is known by hand.


BLOCK_RADIUS_RANGE = np.round(np.arange(1.0, 2.001, 0.05), 3)


@pytest.fixture(scope="module")
def pristine_and_vacancy():
    """(pristine slab, bridging-O vacancy slab, removed site index) — no network, no model.

    The default 4x2 rutile(110) cut, and the O2c candidate the vacancy funnel actually
    produces, so the fixture exercises the production path rather than a hand-built defect.
    """
    cfg = SlabConfig()
    slab = make_slab(get_structure("mp-2657"), cfg)
    cands = oxygen_vacancy_candidates(
        slab,
        freeze_bottom_fraction=cfg.freeze_bottom_fraction,
        block_radius=cfg.vacancy_block_radius,
    )
    o2c = next(c for c in cands if c.site_id["site_label"] == "O2c")
    return slab, o2c.structure, o2c.site_id["site_index"]


def _lateral_clearance(structure, occlusion_dz=0.3):
    """Per-atom distance to the nearest atom that sits above it — the quantity
    ``block_radius`` thresholds. ``exposed`` is exactly ``{i: clearance[i] >= block_radius}``,
    so the clearance spectrum says everything about the parameter's sensitivity."""
    L, cart, frac = structure.lattice, structure.cart_coords, structure.frac_coords
    clearance = {}
    for i in range(len(structure)):
        best = np.inf
        for j in range(len(structure)):
            if j == i or cart[j][2] <= cart[i][2] + occlusion_dz:
                continue
            d = frac[i] - frac[j]
            d[:2] -= np.round(d[:2])  # min-image in-plane
            best = min(best, float(np.linalg.norm((d @ L.matrix)[:2])))
        clearance[i] = best
    return clearance


def _lateral(structure, i, j):
    d = structure.frac_coords[i] - structure.frac_coords[j]
    d[:2] -= np.round(d[:2])
    return float(np.linalg.norm((d @ structure.lattice.matrix)[:2]))


def test_vacancy_exposes_exactly_the_oxygen_under_the_bridging_site(pristine_and_vacancy):
    """Removing the bridging O opens a line of sight straight down, and to nothing else.

    The vacancy's whole point is that an adsorbate can now reach into the trough, so the
    sub-bridging O must join the vertex set; anything *else* joining it would mean the
    occlusion test is leaking subsurface atoms in whenever a defect perturbs the geometry.
    """
    slab, vac, removed = pristine_and_vacancy
    cfg = AdsorbateConfig()

    before = exposed_surface_atoms(slab, cfg.surface_depth, cfg.exposure_block_radius)
    after = exposed_surface_atoms(vac, cfg.surface_depth, cfg.exposure_block_radius)
    # remove_sites shifts every index above the removed one down by one.
    after_in_slab = {i if i < removed else i + 1 for i in after}

    assert before - after_in_slab == {removed}, "only the removed O should leave the set"
    gained = after_in_slab - before
    assert len(gained) == 1, sorted(gained)

    (new,) = gained
    element, _deficit, cn = _site_environments(slab)[new]
    assert site_label(element, cn) == "O3c"  # second O shell, not a cation, not a third party
    # Directly beneath the removed bridging O: the vacancy floor, ~2.5 A down.
    assert _lateral(slab, new, removed) < 0.05
    dz = slab[removed].coords[2] - slab[new].coords[2]
    assert 2.3 < dz < 2.8, dz
    # ...and it is only reachable because surface_depth scans that far down.
    assert cfg.surface_depth > dz, (cfg.surface_depth, dz)


def test_block_radius_is_not_stable_over_one_to_two_angstroms(pristine_and_vacancy):
    """The flagged set has a breakpoint inside [1.0, 2.0] Å, so the value is chemistry.

    On rutile(110) the clearance spectrum is quantised — {0.00, 1.48, 2.47, 3.25, inf} Å —
    so the set is constant on [1.0, 1.47] and constant on [1.48, 2.0], with exactly one
    flip between them. What flips is the six-fold Ti sitting 1.48 Å laterally from (and
    1.27 Å below) the bridging O it is bonded to: below the breakpoint it is a
    triangulation vertex and an ontop candidate, above it, it is not. That is a decision
    about whether a bulk-coordinated, roofed cation is adsorption-accessible, and the
    default 1.3 answers "yes". Not a tolerance — a claim, and one worth defending.
    """
    slab, vac, _removed = pristine_and_vacancy
    cfg = AdsorbateConfig()

    for structure in (slab, vac):
        sets = {
            br: frozenset(exposed_surface_atoms(structure, cfg.surface_depth, float(br)))
            for br in BLOCK_RADIUS_RANGE
        }
        assert len(set(sets.values())) > 1, "no breakpoint — this test's premise is stale"

    low = exposed_surface_atoms(slab, cfg.surface_depth, 1.0)
    default = exposed_surface_atoms(slab, cfg.surface_depth, cfg.exposure_block_radius)
    high = exposed_surface_atoms(slab, cfg.surface_depth, 2.0)
    assert low == default  # the default sits on the lower plateau, not on a knife edge
    assert high < default

    environments = _site_environments(slab)
    flipped = {site_label(*(environments[i][0], environments[i][2])) for i in default - high}
    assert flipped == {"Ti6c"}, flipped
    assert len(default - high) == 8


def test_block_radius_cannot_separate_the_vacancy_floor_from_the_roofed_cation(
    pristine_and_vacancy,
):
    """Raising block_radius to drop the Ti6c false positives also drops the vacancy floor.

    Both sit 1.479 Å laterally from the same bridging-Ti row — the sub-bridging O because
    the O directly above it is gone, the Ti6c because the bridging O straddles it. They are
    degenerate under a single lateral radius, so no value of this one knob keeps the site
    the defect exists to create while rejecting the roofed cation. Any fix has to look at
    something other than lateral distance (e.g. solid angle to the vacuum).
    """
    _slab, vac, _removed = pristine_and_vacancy
    cfg = AdsorbateConfig()

    clearance = _lateral_clearance(vac)
    environments = _site_environments(vac)
    exposed = exposed_surface_atoms(vac, cfg.surface_depth, cfg.exposure_block_radius)

    z_top = max(s.coords[2] for s in vac)
    floor = [i for i in exposed if z_top - vac[i].coords[2] > 2.0]
    assert len(floor) == 1, floor  # the vacancy floor O, and nothing else that deep
    roofed = [
        i
        for i in exposed
        if site_label(environments[i][0], environments[i][2]) == "Ti6c"
    ]
    assert roofed

    assert clearance[floor[0]] == pytest.approx(1.479, abs=1e-2)
    for i in roofed:
        assert clearance[i] == pytest.approx(clearance[floor[0]], abs=1e-3)


def test_every_exposed_atom_becomes_a_triangulation_vertex(pristine_and_vacancy):
    """The exposed set reaches pymatgen intact — before *and* after the defect.

    ``assign_site_properties`` returns the slab untouched when ``surface_properties`` is
    already set, which is the seam ``adsorbate_candidates`` uses. If a pymatgen upgrade
    ever removed that short-circuit, the height window would silently take over and the
    5-fold Ti would stop being a vertex; this is the tripwire.

    It also records what the vertex set *is*: not a plane. The mesh is triangulated on the
    2D projection, so on the vacancy substrate atoms spanning ~2.5 Å in z are meshed as
    coplanar, and some bridge/hollow midpoints therefore span a top-layer atom and the
    vacancy floor. ``_placement_z`` re-solves the height so nothing is placed inside the
    slab, but the *site list* still contains those pairs.
    """
    for structure in pristine_and_vacancy[:2]:
        cfg = AdsorbateConfig()
        exposed = exposed_surface_atoms(
            structure, cfg.surface_depth, cfg.exposure_block_radius
        )
        props = [
            "surface" if i in exposed else "subsurface" for i in range(len(structure))
        ]
        asf = AdsorbateSiteFinder(
            structure.copy(site_properties={"surface_properties": props})
        )

        # Same atoms, not merely the same count: nothing is dropped or substituted.
        assert len(asf.surface_sites) == len(exposed)
        got = sorted(tuple(np.round(s.frac_coords, 6)) for s in asf.surface_sites)
        want = sorted(tuple(np.round(structure[i].frac_coords, 6)) for i in exposed)
        assert got == want
        # Every vertex is replicated across pymatgen's 5x5 extended mesh.
        assert len(asf.get_extended_surface_mesh()) == 25 * len(exposed)

    _slab, vac, _removed = pristine_and_vacancy
    cfg = AdsorbateConfig()
    exposed = exposed_surface_atoms(vac, cfg.surface_depth, cfg.exposure_block_radius)
    z = [vac[i].coords[2] for i in exposed]
    assert max(z) - min(z) > 2.0  # projected flat by the Delaunay, in spite of this
