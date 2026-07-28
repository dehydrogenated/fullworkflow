"""placement_quality_flags: flag relaxations that did no real work.

The motivating failure: an adsorbate placed too far from the surface starts already
below the force threshold and "converges" without ever moving. These checks must catch
that, while NOT flagging a relaxation that did real work but landed on the wrong site
(e.g. H migrating onto a Ti) — a distinct problem.
"""

from __future__ import annotations

from oxide_workflow.checks import placement_quality_flags

FMAX = 0.05  # the pipeline's default relax fmax target (eV/Å)


def test_healthy_relaxation_has_no_flags():
    # Real work: forces well above threshold at the start, many steps, adsorbate moved.
    flags = placement_quality_flags(
        start_fmax=0.9, fmax_target=FMAX, nsteps=45, adsorbate_max_disp=0.8
    )
    assert flags == []


def test_non_interacting_placement_is_flagged_by_start_fmax():
    # bridge/hollow case: H placed 3+ Å out, start_fmax already below target.
    flags = placement_quality_flags(
        start_fmax=0.030, fmax_target=FMAX, nsteps=8, adsorbate_max_disp=0.002
    )
    assert any("start_fmax" in f for f in flags)
    assert any("barely moved" in f for f in flags)


def test_ontop_frozen_caught_by_displacement_even_when_forces_above_target():
    # ontop case: start_fmax 0.11 > target (so NOT caught by force), but H never moved.
    flags = placement_quality_flags(
        start_fmax=0.111, fmax_target=FMAX, nsteps=9, adsorbate_max_disp=0.007
    )
    assert flags  # must be flagged
    assert all("start_fmax" not in f for f in flags)  # not via the force signal
    assert any("barely moved" in f for f in flags)


def test_correct_placement_with_large_start_fmax_is_NOT_flagged():
    # H placed right on a bridging O (hydroxyl at ~0.97 A): the adsorbate barely moves, but
    # start_fmax is large because the *surface* had real forces to relax. Small displacement
    # + large starting force = placed-on-the-answer, not a frozen non-interacting placement.
    flags = placement_quality_flags(
        start_fmax=5.0, fmax_target=FMAX, nsteps=44, adsorbate_max_disp=0.025
    )
    assert flags == []


def test_wrong_site_migration_is_NOT_flagged():
    # H fell onto a Ti (wrong site) but that IS real work: large displacement, many steps,
    # meaningful starting force. This is a sampling problem, not a no-work problem.
    flags = placement_quality_flags(
        start_fmax=0.4, fmax_target=FMAX, nsteps=159, adsorbate_max_disp=1.59
    )
    assert flags == []


def test_placed_on_binding_site_is_flagged_by_start_distance():
    # H spawned at ~the O–H bond length (1.0 vs covalent 0.97): on the answer, no room to
    # relax down. Large start_fmax + small displacement do NOT suppress this — it's a
    # distinct placement-quality signal (cf. site0/site6 in the validation run).
    flags = placement_quality_flags(
        start_fmax=5.0,
        fmax_target=FMAX,
        nsteps=44,
        adsorbate_max_disp=0.025,
        start_ads_distance=1.0,
        ads_bond_length=0.97,
    )
    assert any("placed on binding site" in f for f in flags)


def test_lateral_wiggle_still_flagged_when_placed_on_site():
    # site6 shape: adsorbate moved 0.314 Å (clears the barely-moved threshold) but was
    # spawned at the bond length, so the on-site signal must still fire.
    flags = placement_quality_flags(
        start_fmax=4.1,
        fmax_target=FMAX,
        nsteps=37,
        adsorbate_max_disp=0.314,
        start_ads_distance=1.0,
        ads_bond_length=0.97,
    )
    assert any("placed on binding site" in f for f in flags)


def test_adsorbate_with_approach_room_not_flagged_on_site():
    # Spawned well outside the bond length (room to relax down): no on-site flag.
    flags = placement_quality_flags(
        start_fmax=0.9,
        fmax_target=FMAX,
        nsteps=45,
        adsorbate_max_disp=0.8,
        start_ads_distance=1.6,
        ads_bond_length=0.97,
    )
    assert all("placed on binding site" not in f for f in flags)


def test_short_relaxation_flag():
    flags = placement_quality_flags(
        start_fmax=0.2, fmax_target=FMAX, nsteps=3, adsorbate_max_disp=0.3
    )
    assert any("short relaxation" in f for f in flags)


def test_non_adsorbate_stage_only_uses_start_fmax():
    # Non-adsorbate stages pass nsteps=None / adsorbate_max_disp=None: only the universal
    # "input already stationary" signal applies.
    assert placement_quality_flags(start_fmax=0.2, fmax_target=FMAX) == []
    flags = placement_quality_flags(start_fmax=0.01, fmax_target=FMAX)
    assert len(flags) == 1 and "start_fmax" in flags[0]
