"""Oxide surface reactivity workflow — MLIP benchmarking and adsorption-energy screening.

What it does
------------
Relaxes a four-stage chain with a machine-learned interatomic potential:

    bulk -> slab -> oxygen vacancy -> adsorbate

Each stage takes the previous stage's *relaxed* geometry, applies one modification, and
relaxes again. The last two stages are screenings: every symmetry-distinct site is relaxed
and the lowest-energy one is carried forward. The headline output is ``E_ads``, the
adsorption energy at the best site.

A ``reference`` model produces the ground-truth chain once; each ``candidate`` model is
then benchmarked against it under one or both protocols — ``full_pipeline`` (candidate
builds every stage from its own relaxed output; realistic accumulated error) and ``seeded``
(each stage from the reference's; intrinsic per-stage error, for comparison against DFT).

How the modules fit together
----------------------------
``pipeline``    orchestration. Asks ``structures`` for a bulk cell, ``stages`` for each
                stage's unrelaxed geometry, and ``backends`` to relax it; compares models
                with ``diverge``; writes everything through ``records``.
``config``      every chemistry and relaxation knob, in one place. Each has a CLI flag.
``structures``  identifier (mp-id, alias, or file path) -> bulk cell, resolved offline.
``stages``      the geometry work: cutting slabs, enumerating vacancies, placing
                adsorbates. Most of the scientific judgment lives here.
``backends``    the one interface every model sits behind. Launches the model's own conda
                env on ``worker_relax``, so mutually incompatible envs coexist and the
                orchestrator never imports a model.
``energetics``  E_ads and the gas-phase reference state.
``diverge``     candidate-vs-reference geometry and energy comparison.
``checks``      post-relaxation quality flags (placements that did no real work).
``records``     the divergence table and the browsable run tree on disk.

Start reading at ``config``, then ``stages``, then ``pipeline``.
"""

__version__ = "0.4.0"
