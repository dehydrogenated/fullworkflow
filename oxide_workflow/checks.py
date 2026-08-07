"""Post-relaxation sanity flags — quality gates on the enumerate→relax funnel.

A relaxation can "succeed" (converge to fmax) while having done no physical work. This
is exactly the adsorbate failure mode we hit: a fragment placed too far from the surface
starts already below the force threshold, so the optimizer stops with the adsorbate still
sitting at its (non-interacting) placement instead of relaxing into a real binding
configuration.

These are pure, data-only checks over signals the relaxation already records
(``start_fmax``, ``nsteps``, and — for adsorbates — how far the adsorbate atoms actually
moved from placement, how close they were spawned to the surface, and where the binding
atom ended up). They *warn*; they never mutate a result. Crucially they separate distinct
pathologies:

- **did no work (placed too far)** — adsorbate spawned outside bonding range, starts below
  the force threshold, and never moves (caught by ``start_fmax`` / displacement);
- **did no informative work (placed too close)** — adsorbate spawned essentially *at* its
  bond length, so it's on the answer with no room to relax down into the well (caught by
  ``start_ads_distance``);
- **did work but moved the wrong way (desorbed)** — the adsorbate moved substantially, but
  away from the surface rather than toward it, ending well past its bond length. A large
  displacement alone reads as "real interaction"; only the *final* distance catches a site
  that genuinely doesn't bind (caught by ``end_ads_distance``);
- **did work but found the wrong site** — e.g. H falling onto a Ti instead of forming
  the bridging-O hydroxyl. That is a placement/sampling problem, not a no-work problem,
  and is deliberately *not* flagged here (a large adsorbate displacement clears it).
"""

from __future__ import annotations

from typing import Optional


def placement_quality_flags(
    *,
    start_fmax: Optional[float],
    fmax_target: float,
    nsteps: Optional[int] = None,
    adsorbate_max_disp: Optional[float] = None,
    start_ads_distance: Optional[float] = None,
    end_ads_distance: Optional[float] = None,
    ads_bond_length: Optional[float] = None,
    min_steps: int = 5,
    disp_tol: float = 0.1,
    substantial_fmax: float = 0.5,
    on_site_tol: float = 1.1,
    desorb_tol: float = 2.0,
) -> list[str]:
    """Return warning strings for a relaxation that likely did no real work.

    An empty list means the relaxation looks fine. Pass only the signals that are
    meaningful for the stage: ``adsorbate_max_disp``/``start_ads_distance`` are
    adsorbate-specific, while ``start_fmax`` is universally informative (any stage whose
    input is already stationary did nothing).

    - ``start_fmax <= fmax_target``: the stage input was already at/under the force
      threshold, so relaxation had nothing to do. For an adsorbate this means the
      placement never interacted with the surface (the frozen-adsorbate failure).
    - ``nsteps < min_steps``: suspiciously short trajectory (coarse backup signal).
    - ``adsorbate_max_disp < disp_tol``: the adsorbate atoms barely moved from where
      they were placed — evidence of a non-interacting placement, *unless* the input
      forces were large (``start_fmax >= substantial_fmax``). A small displacement with
      large starting forces means the adsorbate was placed essentially on its bonding site
      and the real relaxation work happened in the substrate (e.g. H dropped right onto a
      bridging O at ~0.97 Å) — a correct result, not a frozen one. Only when the starting
      forces are *also* small is a tiny displacement genuine no-work evidence.
    - ``start_ads_distance <= on_site_tol * ads_bond_length``: the *opposite* extreme to a
      non-interacting placement — the adsorbate was spawned essentially at (or inside) its
      bond length to the nearest surface atom, i.e. placed on the answer with no room to
      relax *down* into the well. The trajectory then can't distinguish a genuine binding
      approach from a pre-placed guess, so the site's energy may be right but uninformative.
      Flagged independently of the force/displacement signals (a placed-on-answer adsorbate
      can still show lateral wiggle or large substrate forces).
    - ``end_ads_distance >= desorb_tol * ads_bond_length``: the adsorbate ended up well past
      its bond length — it may have moved a lot (clearing the ``adsorbate_max_disp`` check
      above), but away from the surface rather than toward it. A large displacement alone
      reads as real interaction; this is the check that catches a site that genuinely
      doesn't bind. Independent of ``start_ads_distance``/``start_fmax`` because a placement
      that started close and with strong forces can still desorb.
    """
    flags: list[str] = []
    if start_fmax is not None and start_fmax <= fmax_target:
        flags.append(
            f"start_fmax {start_fmax:.4f} <= fmax_target {fmax_target}: "
            "input already stationary; relaxation did no work"
        )
    if nsteps is not None and nsteps < min_steps:
        flags.append(f"short relaxation: {nsteps} step(s) < {min_steps}")
    did_substantial_work = start_fmax is not None and start_fmax >= substantial_fmax
    if (
        adsorbate_max_disp is not None
        and adsorbate_max_disp < disp_tol
        and not did_substantial_work
    ):
        flags.append(
            f"adsorbate barely moved: max displacement {adsorbate_max_disp:.3f} A "
            f"< {disp_tol} A (likely non-interacting placement)"
        )
    if (
        start_ads_distance is not None
        and ads_bond_length is not None
        and start_ads_distance <= ads_bond_length * on_site_tol
    ):
        flags.append(
            f"adsorbate placed on binding site: start distance {start_ads_distance:.3f} A "
            f"<= {on_site_tol:g}x bond length {ads_bond_length:.3f} A "
            "(placed on the answer; no room to relax into the well — uninformative trajectory)"
        )
    if (
        end_ads_distance is not None
        and ads_bond_length is not None
        and end_ads_distance >= ads_bond_length * desorb_tol
    ):
        flags.append(
            f"adsorbate desorbed: end distance {end_ads_distance:.3f} A "
            f">= {desorb_tol:g}x bond length {ads_bond_length:.3f} A "
            "(moved away from the surface; site does not bind — not a converged adsorption energy)"
        )
    return flags
