"""Pipeline orchestration — the glue around tested parts.

Running it
==========

    python -m oxide_workflow.pipeline --material mp-825 --adsorbate O2 --max-sites 6 \\
        --outdir runs/rutile_O2/mp-825

One material per invocation. Add ``--help`` for the full flag list.

Where the settings come from
----------------------------
Every knob lives in ``config.py`` as a frozen dataclass (``SlabConfig``, ``RelaxConfig``,
``AdsorbateConfig``, all held by ``RunConfig``). The CLI builds a default ``RunConfig``
and then applies ``dataclasses.replace`` for each flag you passed — see the ``__main__``
block at the bottom of this file, which is the whole override mechanism.

So there are exactly two ways to change a run, and the flag always wins:

    edit config.py   ->  new default for every run from now on
    pass a flag      ->  applies to this run only

Nothing else needs editing. If a flag exists for a setting, prefer it; editing the file
is for changing what "default" means.

The flags
---------
material and models
    --material      mp-id or a path to your own CIF/POSCAR
                    (default: RunConfig.polymorph, "mp-2657")
    --candidates    model(s) to benchmark, space-separated (default: RunConfig.candidate)
    --protocol      full_pipeline | seeded | both (default: full_pipeline)

slab geometry -> SlabConfig
    --miller        facet, "110" or "1,1,-1" (default 110)
    --termination   which termination of that facet (default 1)
    --thickness     min slab thickness in A (default 12) — the layer-count knob
    --vacuum        min vacuum gap in A (default 12)
    --supercell     lateral replication "nx,ny" (default 4,2 = 192 atoms; 1,1 = 24)
    --freeze        fraction of the slab held at bulk positions (default 0.5)

screening
    --adsorbate         H | CO | H2O | O2 | O2-side | O (default: AdsorbateConfig pins H)
    --max-sites         adsorbate sites kept per position type (default: all)
    --vacancy-block-radius  Å; how close a taller neighbor blocks vacuum line-of-sight
                        (default 1.3, same test as adsorbate placement; 0 = no limit)

relaxation -> RelaxConfig
    --fmax          force convergence, eV/A (default 0.02)
    --max-steps     optimizer step cap (default 300)

    --outdir        where the run tree goes (default runs/latest)

Worked examples
---------------
    # thicker slab, 80% frozen, on a different facet
    python -m oxide_workflow.pipeline --miller 101 --thickness 20 --freeze 0.8

    # two candidates against the reference, both protocols
    python -m oxide_workflow.pipeline --candidates MACE-mh1-oc20 UMA-oc22 --protocol both

    # fast smoke check that a material runs at all (numbers NOT usable)
    python -m oxide_workflow.pipeline --material mp-856 --supercell 1,1 \\
        --max-sites 1 --fmax 0.2 --max-steps 20

Reading the result
------------------
    python scripts/report.py runs/rutile_O2/mp-825     # one run, per-stage detail
    python scripts/report.py runs/rutile_O2            # a directory of runs, compared

Related entry points: ``scripts/run_stage.py`` runs a single stage (and can start from a
structure file, skipping everything before it); ``scripts/run_family.py`` wraps
``run_batch`` for unattended multi-material sweeps with cost estimates and resume.

What it does
============

Runs the pseudo-reference plumbing test: the reference backend generates the full
relaxed chain (bulk → slab → vacancy → adsorbate) stage by stage; the candidate backend
then runs the same chain under one or both protocols (selected via ``protocols=`` /
``--protocol``; default ``full_pipeline`` alone):

- **seeded** (per-stage): each stage built from the *reference's* relaxed previous
  stage → intrinsic per-stage error. The protocol for stage-by-stage comparison against
  a DFT ground truth.
- **full-pipeline**: each stage built from the *candidate's own* relaxed previous
  stage → realistic accumulated error. What you get screening with the model alone.

Everything expensive is written to disk immediately. Relaxed geometries land in a
browsable, VESTA-friendly tree (one folder per model, split by stage; each leaf carries
``POSCAR``/``CONTCAR``/``trajectory.xyz`` + a lightweight ``OUTCAR`` of per-step
energies; ``header.json`` timing rollups at the aggregate levels). The run-root analysis
tables — a long-format ``divergence.jsonl``, a ``candidates.jsonl`` recording every
vacancy/adsorbate candidate's energy per model (the ranking-fidelity by-product),
and ``summary.json`` — stay at the run root.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
from ase.data import atomic_numbers, covalent_radii

from .backends import Backend, RelaxResult, get_backend, relax
from .checks import placement_quality_flags
from .config import ADSORBATE_FRAGMENTS, RunConfig
from .diverge import diverge
from .energetics import (
    GAS_CACHE_DIR,
    adsorption_energy,
    gas_reference_energy,
    OXYGEN_REFERENCE,
    oxygen_chemical_potential,
    vacancy_formation_energy,
)
from .records import (
    DivergenceRecord,
    OUTCAR_NAME,
    append_divergence,
    format_outcar,
    leaf_dir,
    read_relaxation,
    relax_subfolder_name,
    stage_dir,
    write_header,
    write_ranking,
    write_relaxation,
)
from .records import _sanitize
from .stages import adsorbate_candidates, make_slab, oxygen_vacancy_candidates
from .structures import get_structure, structure_provenance


@dataclass
class StageOutput:
    """A model's relaxed structure at one stage, plus energy and tree/timing bookkeeping."""

    structure: object  # pymatgen Structure
    energy: float
    start_fmax: float
    e_ads: float | None = None  # adsorbate stage only; None elsewhere
    e_vac: float | None = None  # vacancy stage only; None elsewhere
    site_id: dict | None = None
    # --- tree + rollup bookkeeping (populated by _relax_record) ---
    elapsed_s: float = 0.0
    model: str = ""
    protocol: str = ""
    stage: str = ""
    leaf: Path | None = None
    header: dict | None = None
    opt_log: str = ""
    canonical: bool = False
    geometry_source: str = ""
    flags: list[str] = field(default_factory=list)


def _provenance(cfg: RunConfig) -> dict:
    """What ``cfg.polymorph`` resolved to (formula, spacegroup, source).

    Best-effort: a run must never die at the final write-out because a provenance lookup
    failed, so an unresolvable identifier just yields empty labels.
    """
    try:
        return structure_provenance(cfg.polymorph)
    except Exception:  # noqa: BLE001
        return {}


def _facet(cfg: RunConfig, stage: str) -> str:
    return "".join(map(str, cfg.slab.miller_index)) if stage != "bulk" else ""


def _adsorbate_max_disp(initial, final, n_ads: int) -> float | None:
    """Max displacement (Å, min-image) of the last ``n_ads`` atoms, initial → final.

    The adsorbate fragment is appended last by ``adsorbate_candidates`` and relaxation
    preserves atom order, so the trailing ``n_ads`` sites are the adsorbate. Returns
    ``None`` if the atom counts don't line up (nothing to compare)."""
    if n_ads <= 0 or len(initial) != len(final) or len(final) < n_ads:
        return None
    idx = list(range(len(final) - n_ads, len(final)))
    frac = final.frac_coords[idx] - initial.frac_coords[idx]
    frac -= np.round(frac)  # minimum image under PBC
    cart = frac @ final.lattice.matrix
    return float(np.linalg.norm(cart, axis=1).max())


def _adsorbate_anchor_distance(structure, n_ads: int):
    """(distance Å, covalent bond length Å) of the adsorbate binding atom to its nearest
    surface atom, in the given structure — call on the initial structure for the
    "placed too close" check, or the final structure for the "desorbed" check.

    The binding atom is the first adsorbate atom (``coords[0]`` in the fragment), appended
    at index ``len - n_ads``. Compares its distance to the nearest substrate atom against
    the covalent bond length for that element pair. Returns ``(None, None)`` if there's
    nothing to compare."""
    if n_ads <= 0 or len(structure) <= n_ads:
        return None, None
    ads_i = len(structure) - n_ads
    L = structure.lattice
    best_j, best_d = None, 1e9
    for j in range(len(structure) - n_ads):
        d = structure.frac_coords[ads_i] - structure.frac_coords[j]
        d -= np.round(d)  # minimum image under PBC
        dist = float(np.linalg.norm(d @ L.matrix))
        if dist < best_d:
            best_j, best_d = j, dist
    if best_j is None:
        return None, None
    ads_sym = str(structure[ads_i].specie)
    surf_sym = str(structure[best_j].specie)
    bond = float(
        covalent_radii[atomic_numbers[surf_sym]] + covalent_radii[atomic_numbers[ads_sym]]
    )
    return best_d, bond


def _relax_header(
    res: RelaxResult,
    backend: Backend,
    cfg: RunConfig,
    *,
    stage: str,
    protocol: str,
    facet: str,
    site_id: dict | None,
    canonical: bool,
    geometry_source: str,
    adsorbate_max_disp: float | None,
    e_ads: float | None,
    e_vac: float | None,
    flags: list[str],
) -> dict:
    """Leaf-scope header dict: everything the OUTCAR summary block reports."""
    site = relax_subfolder_name(stage, site_id) or "-" if site_id else "-"
    return {
        "model": backend.name,
        "stage": stage,
        "protocol": protocol,
        "facet": facet,
        "composition": res.structure.composition.reduced_formula,
        "site": site,
        # What the site IS chemically (Ti5c, O2c, ...), as opposed to where its folder is.
        # This is the form to compare against literature.
        "site_label": (site_id or {}).get("site_label") or None,
        "site_id": site_id,
        "canonical": canonical,
        "optimizer": cfg.relax.optimizer,
        "fmax_target": cfg.relax.fmax,
        # Not rendered into the OUTCAR block; carried so a resumed leaf can tell whether it
        # stopped because it converged or because it ran out of the *then-current* budget.
        "max_steps": cfg.relax.max_steps,
        "converged": res.converged,
        "nsteps": res.nsteps,
        "n_frames": res.meta.get("n_frames"),
        "elapsed_s": res.meta.get("elapsed_s"),
        "start_fmax": res.start_fmax,
        "energy": res.energy,
        "e_ads": e_ads,
        "e_vac": e_vac,
        "fmax": res.fmax,
        "geometry_source": geometry_source,
        "adsorbate_max_disp": adsorbate_max_disp,
        "flags": flags,
    }


def _same_input(a, b, tol: float = 1e-3) -> bool:
    """Are two structures the same stage input, to POSCAR round-trip precision?

    Periodic-aware (an atom at frac 0.999 and one at -0.001 are the same atom) and tolerant
    of the six decimals POSCAR keeps, which is ~1e-5 Å — far inside ``tol``.
    """
    if len(a) != len(b):
        return False
    if [s.species_string for s in a] != [s.species_string for s in b]:
        return False
    if not np.allclose(a.lattice.matrix, b.lattice.matrix, atol=tol):
        return False
    d = a.frac_coords - b.frac_coords
    d -= np.round(d)  # nearest periodic image, so wrap-around is not a difference
    return bool(np.abs(d @ a.lattice.matrix).max() < tol)


def _resume(leaf: Path, structure, cfg: RunConfig) -> dict | None:
    """A completed leaf that answers *this* question, or ``None`` to relax it again.

    Sweep-level resume: a walltime kill or a Ctrl-C leaves a run part-way through its
    hundreds of relaxations, and re-entering it should cost only the ones that never
    finished. MLIP relaxation is deterministic, so a cached leaf is not an approximation of
    the answer — it *is* the answer, and reusing it is free.

    That only holds while the question is unchanged, so a leaf is reused only if all three
    agree with the current run:

    1. it is complete (``read_relaxation`` returns something at all);
    2. its recorded stage input matches the structure about to be relaxed — this is the
       guard that matters, because it catches a changed supercell, facet, or adsorbate, and
       any upstream geometry that shifted, without needing to enumerate those cases;
    3. the relaxation settings match, and it did not stop early against a step cap that has
       since been raised.

    Any mismatch re-relaxes and overwrites, so a rerun into an existing directory converges
    on results consistent with the current config rather than silently mixing settings.
    """
    cached = read_relaxation(leaf)
    if cached is None:
        return None
    h = cached["header"]
    if h.get("fmax_target") != cfg.relax.fmax or h.get("optimizer") != cfg.relax.optimizer:
        return None
    # Unconverged against a cap that has since been raised: the extra budget is exactly
    # what this leaf was short of, so redo it rather than inheriting a truncated answer.
    if not h.get("converged") and cfg.relax.max_steps > (h.get("max_steps") or 0):
        return None
    if not _same_input(cached["initial"], structure):
        return None
    return cached


def _relax_record(
    structure,
    backend: Backend,
    *,
    stage: str,
    protocol: str,
    geometry_source: str,
    cfg: RunConfig,
    outdir: Path,
    relax_cell: bool,
    site_id: dict | None = None,
    canonical: bool = False,
    e_ads_reference: tuple[float, float] | None = None,
    e_vac_reference: tuple[float, float] | None = None,
) -> StageOutput:
    """Relax one structure and write its leaf folder (POSCAR/CONTCAR/trajectory/OUTCAR).

    Returns a StageOutput carrying the relaxed geometry plus the leaf path / header /
    optimizer log so a multi-candidate funnel can flip ``canonical`` on the winner later.

    ``e_ads_reference`` is the ``(bare substrate, isolated fragment)`` energy pair for the
    adsorbate stage, and ``e_vac_reference`` the ``(pristine slab, mu_O)`` pair for the
    vacancy stage — both from *this* backend. Supplying one records the corresponding
    derived energy alongside the total. Omit them on stages where they are undefined.

    Every relaxation in the pipeline passes through here, so this is also where resume
    lives: a leaf already on disk that answers the same question is loaded instead of
    recomputed (see ``_resume``).
    """
    sdir = stage_dir(outdir, backend.name, protocol, stage)
    leaf = leaf_dir(sdir, stage, site_id)

    def _derived(energy: float) -> tuple[float | None, float | None]:
        """``(E_ads, E_vac)``, each None on the stage where it is undefined."""
        return (
            adsorption_energy(energy, *e_ads_reference) if e_ads_reference is not None else None,
            vacancy_formation_energy(energy, *e_vac_reference) if e_vac_reference is not None else None,
        )

    cached = _resume(leaf, structure, cfg)
    if cached is not None:
        # Re-derive E_ads/E_vac rather than trusting the cached numbers: their reference
        # terms come from this run's own chain, so deriving them here keeps a resumed leaf
        # on exactly the same footing as a freshly relaxed one.
        header = dict(cached["header"])
        header["e_ads"], header["e_vac"] = _derived(header["energy"])
        header["canonical"] = canonical
        opt_log = cached["opt_log"]
        write_relaxation(
            leaf, initial=cached["initial"], final=cached["final"],
            trajectory_src=None, header=header, opt_log=opt_log,
        )
        return StageOutput(
            structure=cached["final"],
            energy=header["energy"],
            start_fmax=header["start_fmax"],
            e_ads=header["e_ads"],
            e_vac=header["e_vac"],
            site_id=site_id,
            elapsed_s=header.get("elapsed_s") or 0.0,
            model=backend.name,
            protocol=protocol,
            stage=stage,
            leaf=leaf,
            header=header,
            opt_log=opt_log,
            canonical=canonical,
            geometry_source=geometry_source,
            flags=header.get("flags") or [],
        )

    res = relax(
        structure,
        backend,
        relax_cell=relax_cell,
        fmax=cfg.relax.fmax,
        max_steps=cfg.relax.max_steps,
        optimizer=cfg.relax.optimizer,
    )
    # Post-relaxation sanity flags. The adsorbate displacement is the direct evidence of a
    # non-interacting placement (adsorbate frozen at its start); start_fmax is meaningful
    # for any stage. Only compute the adsorbate move on the adsorbate stage.
    ads_max_disp = (
        _adsorbate_max_disp(structure, res.structure, len(cfg.adsorbate.species))
        if stage == "adsorbate"
        else None
    )
    start_ads_distance, ads_bond_length = (
        _adsorbate_anchor_distance(structure, len(cfg.adsorbate.species))
        if stage == "adsorbate"
        else (None, None)
    )
    # Same measurement, on the relaxed structure — reuses the placement-time bond length
    # as the reference unit for both checks (start too close / end too far).
    end_ads_distance, _ = (
        _adsorbate_anchor_distance(res.structure, len(cfg.adsorbate.species))
        if stage == "adsorbate"
        else (None, None)
    )
    flags = placement_quality_flags(
        start_fmax=res.start_fmax,
        fmax_target=cfg.relax.fmax,
        nsteps=res.nsteps if stage == "adsorbate" else None,
        adsorbate_max_disp=ads_max_disp,
        start_ads_distance=start_ads_distance,
        end_ads_distance=end_ads_distance,
        ads_bond_length=ads_bond_length,
    )
    e_ads, e_vac = _derived(res.energy)
    header = _relax_header(
        res, backend, cfg,
        stage=stage, protocol=protocol, facet=_facet(cfg, stage),
        site_id=site_id, canonical=canonical, geometry_source=geometry_source,
        adsorbate_max_disp=ads_max_disp, e_ads=e_ads, e_vac=e_vac, flags=flags,
    )
    opt_log = res.meta.get("opt_log", "")
    write_relaxation(
        leaf,
        initial=structure,
        final=res.structure,
        trajectory_src=res.meta.get("trajectory"),
        header=header,
        opt_log=opt_log,
    )
    # Propagate the geometry as it landed on disk, not the in-memory original. A resumed
    # run can only ever see the disk copy, so reading it back here is what makes the two
    # chains bit-identical — and downstream stage builders are sensitive enough to their
    # input object that "numerically equal" is not enough (see _resume).
    final = read_relaxation(leaf)["final"]
    return StageOutput(
        structure=final,
        energy=res.energy,
        start_fmax=res.start_fmax,
        e_ads=e_ads,
        e_vac=e_vac,
        site_id=site_id,
        elapsed_s=res.meta.get("elapsed_s") or 0.0,
        model=backend.name,
        protocol=protocol,
        stage=stage,
        leaf=leaf,
        header=header,
        opt_log=opt_log,
        canonical=canonical,
        geometry_source=geometry_source,
        flags=flags,
    )


def _record_candidate_energy(
    path: Path, *, stage: str, model: str, protocol: str, site_id: dict, energy: float,
    e_ads: float | None = None, e_vac: float | None = None,
) -> None:
    with path.open("a") as f:
        f.write(
            json.dumps(
                {
                    "stage": stage,
                    "model": model,
                    "protocol": protocol,
                    "symmetry_class": site_id["symmetry_class"],
                    "site_label": site_id.get("site_label", ""),
                    "site_index": site_id["site_index"],
                    "energy": energy,
                    # Each is None on the stage where it is undefined: E_ads on vacancy
                    # rows, E_vac on adsorbate rows.
                    "e_ads": e_ads,
                    "e_vac": e_vac,
                }
            )
            + "\n"
        )


def _ranking_rows(ranked: list[StageOutput], stage: str) -> list[dict]:
    """Ranked StageOutputs -> ``rankings.csv`` rows (``ranked`` is sorted, lowest first)."""
    emin = ranked[0].energy
    return [
        {
            "rank": i,
            "folder": relax_subfolder_name(stage, o.site_id),
            "site_index": o.site_id["site_index"],
            "symmetry_class": o.site_id["symmetry_class"],
            "site_label": o.site_id.get("site_label", ""),
            "energy_eV": round(o.energy, 6),
            "dE_from_min_eV": round(o.energy - emin, 6),
            "e_ads_eV": None if o.e_ads is None else round(o.e_ads, 6),
            "e_vac_eV": None if o.e_vac is None else round(o.e_vac, 6),
            "converged": o.header.get("converged"),
            "nsteps": o.header.get("nsteps"),
            "flags": "; ".join(o.flags),
        }
        for i, o in enumerate(ranked, start=1)
    ]


def _run_funnel(
    candidates,
    backend: Backend,
    *,
    stage: str,
    protocol: str,
    geometry_source: str,
    cfg: RunConfig,
    outdir: Path,
    candidates_table: Path,
    e_ads_reference: tuple[float, float] | None = None,
    e_vac_reference: tuple[float, float] | None = None,
) -> dict[int, StageOutput]:
    """Relax ALL decorate() candidates for one stage → record each. Returns per-site outputs.

    Stage-agnostic: ``candidates`` is any iterable of objects exposing ``.structure`` and
    ``.site_id`` (vacancy or adsorbate). Keyed by originating ``site_index`` so callers can
    select the min OR a matched site. This is the shared funnel behind the ``decorate``
    seam — the only per-stage difference lives in the candidate generator. After all sites
    relax, the min-energy one is flagged ``canonical`` (its OUTCAR rewritten) and a
    stage-scope ``header.json`` is written.

    ``e_ads_reference`` (adsorbate stage) and ``e_vac_reference`` (vacancy stage) turn on
    the derived-energy columns; see ``_relax_record``. Site selection is unaffected by
    either — both differ from the total energy by a constant within one funnel, so ranking
    by any of the three picks the same winner.
    """
    outputs: dict[int, StageOutput] = {}
    for cand in candidates:
        si = cand.site_id["site_index"]
        out = _relax_record(
            cand.structure,
            backend,
            stage=stage,
            protocol=protocol,
            geometry_source=geometry_source,
            cfg=cfg,
            outdir=outdir,
            relax_cell=False,
            site_id=cand.site_id,
            e_ads_reference=e_ads_reference,
            e_vac_reference=e_vac_reference,
        )
        _record_candidate_energy(
            candidates_table,
            stage=stage,
            model=backend.name,
            protocol=protocol,
            site_id=cand.site_id,
            energy=out.energy,
            e_ads=out.e_ads,
            e_vac=out.e_vac,
        )
        outputs[si] = out

    win = min(outputs, key=lambda si: outputs[si].energy)  # min-energy candidate
    winner = outputs[win]
    winner.canonical = True
    winner.header["canonical"] = True
    (winner.leaf / OUTCAR_NAME).write_text(format_outcar(winner.header, winner.opt_log))

    sdir = stage_dir(outdir, backend.name, protocol, stage)
    ranked = sorted(outputs.values(), key=lambda o: o.energy)
    write_ranking(sdir, _ranking_rows(ranked, stage))
    # How decisively this funnel chose. Below the model's own accuracy (tens of meV) the
    # top sites are degenerate within error and the ordering carries no information.
    margin = round(ranked[1].energy - ranked[0].energy, 6) if len(ranked) > 1 else None

    write_header(
        sdir,
        {
            "stage": stage,
            "protocol": protocol,
            "geometry_source": geometry_source,
            "elapsed_s": round(sum(o.elapsed_s for o in outputs.values()), 3),
            "n_relaxations": len(outputs),
            "canonical": relax_subfolder_name(stage, winner.site_id),
            # The headline number for the adsorbate stage: the winning site's binding
            # energy. None on the vacancy stage.
            "min_e_ads": None if winner.e_ads is None else round(winner.e_ads, 6),
            # Lowest vacancy formation energy = the most easily formed vacancy.
            "min_e_vac": None if winner.e_vac is None else round(winner.e_vac, 6),
            "oxygen_reference": OXYGEN_REFERENCE if winner.e_vac is not None else None,
            "margin_to_runner_up_eV": margin,
        },
    )
    return outputs


def _emit(
    table: Path,
    ref_out: StageOutput,
    cand_out: StageOutput,
    *,
    stage: str,
    model: str,
    protocol: str,
    cfg: RunConfig,
) -> DivergenceRecord:
    rec = diverge(
        ref_out.structure,
        cand_out.structure,
        stage=stage,
        model=model,
        e_ref=ref_out.energy,
        e_cand=cand_out.energy,
        start_fmax=cand_out.start_fmax,
        protocol=protocol,
        polymorph=cfg.polymorph,
        facet=_facet(cfg, stage),
        flags=cand_out.flags,
    )
    append_divergence(table, rec)
    return rec


# --- Timing / header rollups over the whole run tree -------------------------------------


def _group_key(o: StageOutput) -> str:
    """Rollup bucket for one relaxation within its model's subtree.

    ``bulk`` sits at the model root (candidate's shared warm start); reference stages have
    no protocol level; candidate seeded/full stages nest as ``<protocol>/<stage>``.
    """
    if o.stage == "bulk":
        return "bulk"
    if o.protocol == "reference":
        return o.stage
    return f"{o.protocol}/{o.stage}"


def _canonical_pointer(outs: list[StageOutput]):
    """The canonical member of a group: subfolder name (multi-candidate) or True (single)."""
    for o in outs:
        if o.canonical:
            return relax_subfolder_name(o.stage, o.site_id) or True
    return None


def _rollup(outs: list[StageOutput]) -> dict:
    return {
        "elapsed_s": round(sum(o.elapsed_s for o in outs), 3),
        "n_relaxations": len(outs),
        "canonical": _canonical_pointer(outs),
    }


def _write_tree_headers(outdir: Path, cfg: RunConfig, all_outs: list[StageOutput]) -> float:
    """Write model/protocol/run ``header.json`` rollups. Returns total elapsed (s)."""
    by_model: dict[str, list[StageOutput]] = defaultdict(list)
    for o in all_outs:
        by_model[o.model].append(o)

    elapsed_by_model: dict[str, dict] = {}
    for model, mouts in by_model.items():
        groups: dict[str, list[StageOutput]] = defaultdict(list)
        for o in mouts:
            groups[_group_key(o)].append(o)
        group_rollups = {g: _rollup(gs) for g, gs in groups.items()}
        total = round(sum(o.elapsed_s for o in mouts), 3)

        write_header(outdir / model, {
            "model": model,
            "total_elapsed_s": total,
            "groups": group_rollups,
        })

        # Protocol-scope headers (candidate only): seeded / full_pipeline.
        protocols = sorted({o.protocol for o in mouts if o.stage != "bulk" and o.protocol != "reference"})
        for proto in protocols:
            stages: dict[str, list[StageOutput]] = defaultdict(list)
            for o in mouts:
                if o.protocol == proto and o.stage != "bulk":
                    stages[o.stage].append(o)
            write_header(outdir / model / proto, {
                "protocol": proto,
                "total_elapsed_s": round(sum(o.elapsed_s for ss in stages.values() for o in ss), 3),
                "stages": {st: _rollup(ss) for st, ss in stages.items()},
            })

        elapsed_by_model[model] = {
            "total_s": total,
            "by_stage": {g: r["elapsed_s"] for g, r in group_rollups.items()},
        }

    total_elapsed = round(sum(o.elapsed_s for o in all_outs), 3)
    candidates = [m for m in by_model if m != cfg.reference]  # every non-reference model
    provenance = _provenance(cfg)
    write_header(outdir, {
        "run_id": Path(outdir).name,
        "reference": cfg.reference,
        "candidates": candidates,
        "models": [cfg.reference, *candidates],
        "facet": "".join(map(str, cfg.slab.miller_index)),
        "polymorph": cfg.polymorph,
        "formula": provenance.get("formula", ""),
        "spacegroup": provenance.get("spacegroup", ""),
        "total_elapsed_s": total_elapsed,
        "elapsed_by_model": elapsed_by_model,
    })
    return total_elapsed


@dataclass
class ReferenceChain:
    """The reference model's relaxed ground truth — computed once, reused by every candidate."""

    bulk: StageOutput
    slab: StageOutput
    vacancy: StageOutput  # canonical (min-energy) vacancy
    adsorbate: StageOutput  # canonical (min-energy) adsorbate placement
    vac: dict[int, StageOutput]
    ads: dict[int, StageOutput]
    canonical_site: int
    ads_canonical: int
    bulk_input: object  # the warm-start bulk cell (shared with candidate bulk)
    outputs: list[StageOutput]  # every reference relaxation, for timing/header rollups


@dataclass
class CandidateChain:
    """One candidate model benchmarked against a shared ReferenceChain."""

    outputs: list[StageOutput]
    # protocol -> {"vacancy": site_index, "adsorbate": site_index}: the min-energy site each
    # protocol's own funnel picked. Under ``seeded`` these are the candidate's own picks even
    # though the divergence rows compare the reference-matched site — the two can disagree,
    # and that disagreement is the ranking-fidelity signal.
    sites: dict[str, dict[str, int]] = field(default_factory=dict)


def _run_reference_chain(
    ref: Backend, cfg: RunConfig, outdir: Path, bulk_input, cand_table: Path,
    gas_cache: str | Path = GAS_CACHE_DIR,
) -> ReferenceChain:
    """Relax the reference chain bulk → slab → vacancy → adsorbate (the ground truth)."""
    outs: list[StageOutput] = []

    ref_bulk = _relax_record(
        bulk_input, ref, stage="bulk", protocol="reference",
        geometry_source="db", cfg=cfg, outdir=outdir, relax_cell=True, canonical=True,
    )
    outs.append(ref_bulk)

    ref_slab = _relax_record(
        make_slab(ref_bulk.structure, cfg.slab), ref, stage="slab", protocol="reference",
        geometry_source="cut_from_relaxed_bulk", cfg=cfg, outdir=outdir, relax_cell=False, canonical=True,
    )
    outs.append(ref_slab)

    # E_vac reference state: this model's own relaxed pristine slab (already in hand —
    # it is the very slab the vacancies are cut from) plus the oxygen chemical potential.
    mu_o = oxygen_chemical_potential(ref, cfg, relax, cachedir=gas_cache)

    ref_vac = _run_funnel(
        oxygen_vacancy_candidates(ref_slab.structure, freeze_bottom_fraction=cfg.slab.freeze_bottom_fraction, block_radius=cfg.slab.vacancy_block_radius), ref, stage="vacancy",
        protocol="reference", geometry_source="cut_from_relaxed_slab", cfg=cfg,
        outdir=outdir, candidates_table=cand_table,
        e_vac_reference=(ref_slab.energy, mu_o),
    )
    outs.extend(ref_vac.values())
    canonical_site = min(ref_vac, key=lambda si: ref_vac[si].energy)  # min-energy vacancy

    # E_ads reference state, both terms from the reference backend: the bare substrate is
    # the relaxed vacancy slab the fragment is about to be placed on (already relaxed by
    # the funnel above, so it costs nothing), and the gas term is the isolated fragment.
    ref_e_ads = (
        ref_vac[canonical_site].energy,
        gas_reference_energy(ref, cfg, relax, cachedir=gas_cache),
    )

    ref_ads = _run_funnel(
        adsorbate_candidates(ref_vac[canonical_site].structure, cfg.adsorbate, freeze_bottom_fraction=cfg.slab.freeze_bottom_fraction), ref, stage="adsorbate",
        protocol="reference", geometry_source="placed_on_relaxed_vacancy", cfg=cfg,
        outdir=outdir, candidates_table=cand_table, e_ads_reference=ref_e_ads,
    )
    outs.extend(ref_ads.values())
    ads_canonical = min(ref_ads, key=lambda si: ref_ads[si].energy)  # min-energy placement

    return ReferenceChain(
        bulk=ref_bulk, slab=ref_slab, vacancy=ref_vac[canonical_site], adsorbate=ref_ads[ads_canonical],
        vac=ref_vac, ads=ref_ads, canonical_site=canonical_site, ads_canonical=ads_canonical,
        bulk_input=bulk_input, outputs=outs,
    )


# The two benchmarking protocols, in run order. They differ only in *which* relaxed
# geometry feeds each stage:
#
# - ``seeded``        - each stage built from the REFERENCE's relaxed previous stage.
#                       Isolates the intrinsic per-stage error, because the candidate never
#                       inherits its own upstream mistakes. This is the protocol for
#                       stage-by-stage comparison against a DFT ground truth.
# - ``full_pipeline`` - each stage built from the CANDIDATE's OWN relaxed previous stage.
#                       The realistic accumulated error: what you actually get screening
#                       with this model alone, with no reference to lean on.
PROTOCOLS = ("seeded", "full_pipeline")

_GEOMETRY_SOURCES = {
    "seeded": {
        "slab": "cut_from_reference_bulk",
        "vacancy": "reference_slab",
        "adsorbate": "placed_on_reference_vacancy",
    },
    "full_pipeline": {
        "slab": "cut_from_candidate_bulk",
        "vacancy": "candidate_slab",
        "adsorbate": "placed_on_candidate_vacancy",
    },
}


def _run_protocol_chain(
    cand: Backend, rc: ReferenceChain, cfg: RunConfig, outdir: Path,
    div_table: Path, cand_table: Path, *, protocol: str, cand_bulk: StageOutput,
    e_gas: float, mu_o: float,
) -> tuple[list[StageOutput], dict[str, int]]:
    """Run ONE protocol's slab -> vacancy -> adsorbate chain for one candidate.

    The protocols are structurally identical; they differ in exactly two ways, both
    parameterized here rather than by duplicating the chain:

    1. **Which substrate** each stage is built on - the reference's relaxed output
       (``seeded``) or the candidate's own (``full_pipeline``).
    2. **Which site the divergence row compares.** ``seeded`` compares the candidate's
       relaxation of the *same* site the reference picked, because per-stage attribution
       only means something when both sides are the same physical configuration.
       ``full_pipeline`` compares the candidate's *own* min-energy pick against the
       reference's, so a ranking disagreement shows up as a large divergence - which is
       exactly what that protocol exists to measure.

    Returns ``(relaxation outputs, {"vacancy": site_index, "adsorbate": site_index})``,
    where the site indices are always this protocol's own min-energy picks.
    """
    seeded = protocol == "seeded"
    src = _GEOMETRY_SOURCES[protocol]
    frozen = cfg.slab.freeze_bottom_fraction
    outs: list[StageOutput] = []

    # ---- slab ------------------------------------------------------------------------
    slab_substrate = rc.bulk.structure if seeded else cand_bulk.structure
    c_slab = _relax_record(
        make_slab(slab_substrate, cfg.slab), cand, stage="slab", protocol=protocol,
        geometry_source=src["slab"], cfg=cfg, outdir=outdir, relax_cell=False, canonical=True,
    )
    outs.append(c_slab)
    _emit(div_table, rc.slab, c_slab, stage="slab", model=cand.name, protocol=protocol, cfg=cfg)

    # ---- E_vac reference state (pristine slab, from THIS backend) --------------------
    # Same same-calculator rule as E_ads: the pristine term must be this model's energy
    # for the exact slab the vacancies were cut from.
    #
    # full_pipeline cuts them from its own relaxed slab, so c_slab.energy is already the
    # right number and costs nothing. seeded cuts them from the REFERENCE's relaxed slab,
    # which this model has never evaluated — c_slab above is its relaxation of a slab cut
    # from the reference's *bulk*, a different geometry — so relax the reference slab once
    # here. One slab relaxation against one per site in the funnel below.
    vac_substrate = rc.slab.structure if seeded else c_slab.structure
    if seeded:
        pristine = _relax_record(
            vac_substrate, cand, stage="pristine_slab", protocol=protocol,
            geometry_source="reference_slab", cfg=cfg, outdir=outdir,
            relax_cell=False, canonical=True,
        )
        outs.append(pristine)
        e_pristine = pristine.energy
    else:
        e_pristine = c_slab.energy

    # ---- vacancy funnel --------------------------------------------------------------
    c_vac = _run_funnel(
        oxygen_vacancy_candidates(vac_substrate, freeze_bottom_fraction=frozen, block_radius=cfg.slab.vacancy_block_radius), cand,
        stage="vacancy", protocol=protocol, geometry_source=src["vacancy"], cfg=cfg,
        outdir=outdir, candidates_table=cand_table,
        e_vac_reference=(e_pristine, mu_o),
    )
    outs.extend(c_vac.values())
    own_vac = min(c_vac, key=lambda si: c_vac[si].energy)  # this protocol's own min
    emit_vac = rc.canonical_site if seeded else own_vac
    if emit_vac in c_vac:  # a seeded match can be absent if enumeration diverged
        _emit(div_table, rc.vacancy, c_vac[emit_vac], stage="vacancy",
              model=cand.name, protocol=protocol, cfg=cfg)

    # ---- E_ads reference state (both terms from THIS backend) ------------------------
    # E_ads only cancels the model's per-atom energy offset if all three terms come from
    # the same calculator, so the bare-substrate term must be *this* model's energy for
    # the exact substrate the fragment sits on.
    #
    # full_pipeline places on this model's own relaxed vacancy, so that energy is already
    # in hand and the reference state is free. seeded places on the REFERENCE model's
    # relaxed vacancy — a geometry this model has never evaluated — so relax it once here.
    # That is one extra slab relaxation per candidate, against one per site in the funnel
    # below; using the reference's own energy instead would mix calculators and leave the
    # ~tens-of-eV offset uncancelled, swamping the binding energy entirely.
    ads_substrate = rc.vacancy.structure if seeded else c_vac[own_vac].structure
    if seeded:
        bare = _relax_record(
            ads_substrate, cand, stage="bare_substrate", protocol=protocol,
            geometry_source="reference_vacancy", cfg=cfg, outdir=outdir,
            relax_cell=False, canonical=True,
        )
        outs.append(bare)
        e_substrate = bare.energy
    else:
        e_substrate = c_vac[own_vac].energy

    # ---- adsorbate funnel ------------------------------------------------------------
    c_ads = _run_funnel(
        adsorbate_candidates(ads_substrate, cfg.adsorbate, freeze_bottom_fraction=frozen), cand,
        stage="adsorbate", protocol=protocol, geometry_source=src["adsorbate"], cfg=cfg,
        outdir=outdir, candidates_table=cand_table,
        e_ads_reference=(e_substrate, e_gas),
    )
    outs.extend(c_ads.values())
    own_ads = min(c_ads, key=lambda si: c_ads[si].energy)
    emit_ads = rc.ads_canonical if seeded else own_ads
    if emit_ads in c_ads:
        _emit(div_table, rc.adsorbate, c_ads[emit_ads], stage="adsorbate",
              model=cand.name, protocol=protocol, cfg=cfg)

    return outs, {"vacancy": own_vac, "adsorbate": own_ads}


def _run_candidate_chain(
    cand: Backend, rc: ReferenceChain, cfg: RunConfig, outdir: Path,
    div_table: Path, cand_table: Path, protocols: tuple[str, ...] = PROTOCOLS,
    gas_cache: str | Path = GAS_CACHE_DIR,
) -> CandidateChain:
    """Run one candidate's requested protocol chains against the shared reference.

    Everything is keyed off ``cand`` (its name is the tree folder + divergence ``model``),
    so N candidates coexist under one run dir with one shared reference subtree. The bulk
    relaxation is shared by every protocol - they all warm-start from the same database
    cell - so it runs once no matter how many protocols are requested.
    """
    outs: list[StageOutput] = []

    # ---- Candidate: bulk (shared warm start; identical under every protocol) ----------
    # Labelled with the first requested protocol purely so the divergence row has one; the
    # tree puts it at <model>/bulk with no protocol level either way.
    bulk_protocol = protocols[0]
    cand_bulk = _relax_record(
        rc.bulk_input, cand, stage="bulk", protocol=bulk_protocol,
        geometry_source="db", cfg=cfg, outdir=outdir, relax_cell=True, canonical=True,
    )
    outs.append(cand_bulk)
    _emit(div_table, rc.bulk, cand_bulk, stage="bulk", model=cand.name,
          protocol=bulk_protocol, cfg=cfg)

    # This model's own isolated-fragment energy — shared by every protocol, and cached
    # across runs, so it is computed at most once per (model, fragment, fmax).
    e_gas = gas_reference_energy(cand, cfg, relax, cachedir=gas_cache)
    mu_o = oxygen_chemical_potential(cand, cfg, relax, cachedir=gas_cache)

    sites: dict[str, dict[str, int]] = {}
    for protocol in protocols:
        chain_outs, chain_sites = _run_protocol_chain(
            cand, rc, cfg, outdir, div_table, cand_table,
            protocol=protocol, cand_bulk=cand_bulk, e_gas=e_gas, mu_o=mu_o,
        )
        outs.extend(chain_outs)
        sites[protocol] = chain_sites

    return CandidateChain(outputs=outs, sites=sites)


def _fresh_tables(outdir: Path) -> tuple[Path, Path]:
    """Create the run dir and truncate the two run-root analysis tables."""
    outdir.mkdir(parents=True, exist_ok=True)
    div_table = outdir / "divergence.jsonl"
    cand_table = outdir / "candidates.jsonl"
    for p in (div_table, cand_table):
        if p.exists():
            p.unlink()
    return div_table, cand_table


def _fragment_label(acfg) -> str:
    """The ``ADSORBATE_FRAGMENTS`` key for this fragment, else a counted formula.

    Reverse lookup rather than a stored field, so the label is exactly what you would pass
    back to ``--adsorbate``. The fallback is hand-rolled because pymatgen's
    ``reduced_formula`` special-cases diatomic gases — it renders a single H atom as "H2",
    which would silently mislabel every atomic-adsorbate run.
    """
    want = (tuple(acfg.species), tuple(tuple(c) for c in acfg.coords))
    for name, (species, coords) in ADSORBATE_FRAGMENTS.items():
        if want == (tuple(species), tuple(tuple(c) for c in coords)):
            return name
    counts = Counter(acfg.species)
    return "".join(s if n == 1 else f"{s}{n}" for s, n in counts.items())


def _min_metric(outs: list[StageOutput], stage: str, attr: str) -> dict[str, float]:
    """protocol -> the lowest value of ``attr`` found on ``stage``.

    The two headline results of a run: the minimum adsorption energy over all sampled
    adsorption sites, and the minimum vacancy formation energy over all vacancy sites
    (the most easily formed vacancy).
    """
    best: dict[str, float] = {}
    for o in outs:
        if o.stage != stage:
            continue
        v = getattr(o, attr)
        if v is None:
            continue
        if o.protocol not in best or v < best[o.protocol]:
            best[o.protocol] = round(v, 6)
    return best


def _check_protocols(protocols) -> tuple[str, ...]:
    """Normalize a protocol selection to run order, rejecting unknown names."""
    requested = tuple(protocols)
    unknown = [p for p in requested if p not in PROTOCOLS]
    if unknown:
        raise ValueError(f"unknown protocol(s) {unknown}; choose from {list(PROTOCOLS)}")
    if not requested:
        raise ValueError(f"need at least one protocol; choose from {list(PROTOCOLS)}")
    return tuple(p for p in PROTOCOLS if p in requested)  # canonical order, deduplicated


def run(
    cfg: RunConfig | None = None,
    outdir: str | Path = "runs/latest",
    protocols: tuple[str, ...] = ("full_pipeline",),
    gas_cache: str | Path = GAS_CACHE_DIR,
) -> dict:
    """Benchmark a single candidate against the reference (thin wrapper over ``run_sweep``)."""
    cfg = cfg or RunConfig()
    return run_sweep(
        [cfg.candidate], cfg=cfg, outdir=outdir, protocols=protocols, gas_cache=gas_cache
    )


def run_sweep(
    candidates: list[str] | tuple[str, ...] | None = None,
    cfg: RunConfig | None = None,
    outdir: str | Path = "runs/latest",
    protocols: tuple[str, ...] = ("full_pipeline",),
    gas_cache: str | Path = GAS_CACHE_DIR,
) -> dict:
    """Benchmark many candidates against ONE shared reference chain.

    The reference model (``cfg.reference``) relaxes the full bulk→slab→vacancy→adsorbate
    chain exactly once; every candidate in ``candidates`` (default: all registered
    heads/tasks except the reference) is then run against that shared ground truth. Each
    candidate lands in its own ``<outdir>/<candidate>/`` subtree beside the single
    reference subtree, and its divergence rows share the run-root ``divergence.jsonl``
    keyed by model — so the reference is never recomputed per candidate.

    ``protocols`` selects which chains each candidate runs (see ``PROTOCOLS``). The
    default is ``full_pipeline`` alone — the realistic accumulated-error number, and half
    the cost of running both. Pass ``PROTOCOLS`` for per-stage attribution as well; the
    reference chain is shared either way, so adding ``seeded`` only costs candidate time.
    """
    from .backends import ALL_CANDIDATES

    cfg = cfg or RunConfig()
    protocols = _check_protocols(protocols)
    candidates = list(candidates if candidates is not None else ALL_CANDIDATES)
    outdir = Path(outdir)
    div_table, cand_table = _fresh_tables(outdir)

    ref = get_backend(cfg.reference)
    bulk_input = get_structure(cfg.polymorph)  # mp-id / alias → warm-start bulk cell

    rc = _run_reference_chain(ref, cfg, outdir, bulk_input, cand_table, gas_cache)
    all_outs: list[StageOutput] = list(rc.outputs)

    per_candidate: dict[str, dict] = {}
    for name in candidates:
        cc = _run_candidate_chain(
            get_backend(name), rc, cfg, outdir, div_table, cand_table, protocols, gas_cache
        )
        all_outs.extend(cc.outputs)
        per_candidate[name] = {
            "sites": cc.sites,
            "min_e_ads": _min_metric(cc.outputs, "adsorbate", "e_ads"),
            "min_e_vac": _min_metric(cc.outputs, "vacancy", "e_vac"),
        }

    # ---- Header rollups over the whole tree (timing, canonical pointers) --------------
    total_elapsed_s = _write_tree_headers(outdir, cfg, all_outs)

    provenance = _provenance(cfg)
    summary = {
        "reference": cfg.reference,
        "candidates": candidates,
        "protocols": list(protocols),
        "facet": "".join(map(str, cfg.slab.miller_index)),
        "polymorph": cfg.polymorph,
        # What the identifier actually resolved to. Recorded so a run is self-describing
        # no matter what its directory is called — folder names are for humans, this is
        # the ground truth. Formula alone is not a unique identity (rutile and anatase are
        # both TiO2), which is why the mp-id stays alongside it.
        "formula": provenance.get("formula", ""),
        "spacegroup": provenance.get("spacegroup", ""),
        "reference_canonical_vacancy_site": rc.canonical_site,
        "reference_canonical_adsorbate_site": rc.ads_canonical,
        "adsorbate": _fragment_label(cfg.adsorbate),
        # The headline result: lowest E_ads over all sampled sites (eV, negative = bound).
        "reference_min_e_ads": _min_metric(rc.outputs, "adsorbate", "e_ads").get("reference"),
        # Lowest oxygen vacancy formation energy = the most easily formed vacancy.
        "reference_min_e_vac": _min_metric(rc.outputs, "vacancy", "e_vac").get("reference"),
        "oxygen_reference": OXYGEN_REFERENCE,
        "n_vacancy_sites": len(rc.vac),
        "n_adsorbate_sites": len(rc.ads),
        "per_candidate": per_candidate,
        "total_elapsed_s": total_elapsed_s,
        "divergence_table": str(div_table),
        "candidates_table": str(cand_table),
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def _completed_in(outdir: Path) -> dict[str, dict]:
    """Every material with a finished ``summary.json`` under ``outdir``, keyed by identifier.

    Read back from disk rather than tracked in memory so the index reflects the whole
    batch directory, not just the current invocation.
    """
    found: dict[str, dict] = {}
    if not outdir.is_dir():
        return found
    for sub in sorted(outdir.iterdir()):
        path = sub / "summary.json"
        if not (sub.is_dir() and path.exists()):
            continue
        try:
            s = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue  # a run killed mid-write; the material simply reads as not done
        found[s.get("polymorph") or sub.name] = s
    return found


def _write_batch_index(
    outdir: Path, cfg: RunConfig, protocols, requested, failures: dict[str, str]
) -> dict:
    """Write ``batch.json`` describing every material in the directory.

    Deliberately a directory survey rather than a record of this invocation. Running one
    material at a time into the same ``--outdir`` is a normal way to work through a sweep —
    it keeps each run short and lets results accumulate — and an index built only from the
    current call would erase the record of every earlier one.

    Failures carry over the same way: a material that failed before and is not now complete
    stays listed, and one that has since succeeded drops off.
    """
    done = _completed_in(outdir)
    previous = {}
    index_path = outdir / "batch.json"
    if index_path.exists():
        try:
            previous = json.loads(index_path.read_text()).get("failed", {})
        except json.JSONDecodeError:
            previous = {}
    still_failing = {m: e for m, e in {**previous, **failures}.items() if m not in done}

    index = {
        "reference": cfg.reference,
        "candidate": cfg.candidate,
        "protocols": list(_check_protocols(protocols)),
        "adsorbate": _fragment_label(cfg.adsorbate),
        "requested_this_run": list(requested),
        "completed": sorted(done),
        "failed": still_failing,
        "total_elapsed_s": round(sum(s.get("total_elapsed_s", 0) for s in done.values()), 3),
        "min_e_ads": {m: s.get("reference_min_e_ads") for m, s in sorted(done.items())},
        "min_e_vac": {m: s.get("reference_min_e_vac") for m, s in sorted(done.items())},
        "runs": {m: _sanitize(m) for m in sorted(done)},
    }
    index_path.write_text(json.dumps(index, indent=2))
    return index


def run_batch(
    materials: list[str] | tuple[str, ...],
    cfg: RunConfig | None = None,
    outdir: str | Path = "runs/latest",
    protocols: tuple[str, ...] = ("full_pipeline",),
    gas_cache: str | Path = GAS_CACHE_DIR,
    resume: bool = True,
    continue_on_error: bool = True,
) -> dict[str, dict]:
    """Run the full benchmark for several materials, one browsable subtree each.

    ``materials`` is a list of identifiers (mp-ids / aliases) resolvable by
    ``structures.get_structure`` — the future-run capability to outline a few chemistries
    and sweep them in one call. Each material relaxes with the same ``cfg`` (only its
    ``polymorph`` label swapped) into ``<outdir>/<identifier>/``; a ``batch.json`` index at
    the batch root points at every per-material run and totals the wall time.

    Returns ``{identifier: summary}``. A single-element list reproduces ``run`` exactly,
    just nested one level deeper — so the current single-rutile default is unchanged when
    called as ``run(...)`` directly.

    A family sweep runs for days, so it is restartable by default: ``resume`` skips any
    material that already has a ``summary.json``, and ``continue_on_error`` records a
    failing material in ``batch.json`` and moves on instead of discarding the sweep. Rerun
    the same command to retry only what failed.
    """
    cfg = cfg or RunConfig()
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    summaries: dict[str, dict] = {}
    failures: dict[str, str] = {}
    for material in materials:
        sub = outdir / _sanitize(material)
        done = sub / "summary.json"
        if resume and done.exists():
            # A finished material is expensive (hours) and immutable — never redo it.
            summaries[material] = json.loads(done.read_text())
            print(f"[batch] {material}: already complete, skipping", flush=True)
            continue
        try:
            # gas_cache is deliberately NOT per-material: the isolated fragment is the same
            # reference state for every material, so one cached number serves the batch.
            summaries[material] = run(
                replace(cfg, polymorph=material), outdir=sub, protocols=protocols,
                gas_cache=gas_cache,
            )
        except Exception as exc:  # noqa: BLE001
            # One bad material must not cost the whole sweep. Record why and continue; the
            # batch index lists failures so a rerun (resume=True) retries only those.
            if not continue_on_error:
                raise
            failures[material] = f"{type(exc).__name__}: {exc}"
            print(f"[batch] {material} FAILED: {failures[material]}", flush=True)

    _write_batch_index(outdir, cfg, protocols, materials, failures)
    return summaries


def _cli_miller(text: str) -> tuple[int, int, int]:
    """Parse a Miller index: "110" or "1,1,0" or "1,1,-1" (commas needed for negatives)."""
    parts = text.split(",") if "," in text else list(text)
    idx = tuple(int(p) for p in parts)
    if len(idx) != 3:
        raise ValueError(f"miller index needs 3 components, got {text!r}")
    return idx


if __name__ == "__main__":
    import argparse
    import pprint

    parser = argparse.ArgumentParser(
        description="Run the MLIP divergence benchmark (bulk -> slab -> vacancy -> adsorbate).",
    )
    parser.add_argument(
        "--material",
        help="mp-id (e.g. mp-2657) or a path to a CIF/POSCAR. Defaults to RunConfig.polymorph.",
    )
    parser.add_argument(
        "--miller",
        help='facet as "110" or "1,1,-1". Usually needs setting when --material changes.',
    )
    parser.add_argument(
        "--adsorbate",
        choices=sorted(ADSORBATE_FRAGMENTS),
        help="adsorbate fragment (default: whatever AdsorbateConfig pins)",
    )
    parser.add_argument(
        "--max-sites",
        type=int,
        help="cap adsorbate sites kept per position type (ontop/bridge/hollow). "
             "Sampling cap for smoke tests; omit to enumerate all.",
    )
    _slab = RunConfig().slab
    parser.add_argument(
        "--termination", type=int,
        help=f"which termination of the facet (default {_slab.termination_index}). "
             "Run scripts/validate_materials.py to see what each index actually is.",
    )
    parser.add_argument(
        "--thickness", type=float, metavar="A",
        help=f"min slab thickness in A (default {_slab.min_slab_size}) — the layer-count knob",
    )
    parser.add_argument(
        "--vacuum", type=float, metavar="A",
        help=f"min vacuum gap in A (default {_slab.min_vacuum_size})",
    )
    parser.add_argument(
        "--freeze", type=float, metavar="FRAC",
        help=f"fraction of slab thickness held at bulk positions, 0-1 "
             f"(default {_slab.freeze_bottom_fraction})",
    )
    parser.add_argument(
        "--supercell", metavar="NX,NY",
        help=f"lateral slab replication (default {_slab.supercell[0]},{_slab.supercell[1]}). "
             "The dominant cost knob: 4,2 is 192 atoms, 1,1 is 24.",
    )
    parser.add_argument(
        "--seed-standoff", type=float, metavar="A",
        help=f"A added to the covalent bond length when placing the adsorbate "
             f"(default {RunConfig().adsorbate.seed_standoff}). Too small starts the "
             "fragment inside bonding range; too large and it never interacts.",
    )
    parser.add_argument(
        "--vacancy-block-radius", type=float, metavar="A",
        help=f"how close a taller neighbor can sit before it blocks an O's line of sight "
             f"to vacuum (default {_slab.vacancy_block_radius}; same test as adsorbate "
             f"placement). 0 = no limit, enumerating subsurface O too.",
    )
    parser.add_argument(
        "--fmax", type=float,
        help=f"force convergence, eV/A (default {RunConfig().relax.fmax}). Loosen to ~0.1 "
             "for a smoke test; it is the biggest single speed lever.",
    )
    parser.add_argument(
        "--max-steps", type=int,
        help=f"optimizer step cap per relaxation (default {RunConfig().relax.max_steps})",
    )
    parser.add_argument("--outdir", default="runs/latest", help="run directory")
    parser.add_argument(
        "--candidates",
        nargs="+",
        help="candidate models to benchmark (default: the single RunConfig.candidate)",
    )
    parser.add_argument(
        "--protocol",
        choices=("full_pipeline", "seeded", "both"),
        default="full_pipeline",
        help="which candidate chain(s) to run. full_pipeline (default): each stage from "
             "the candidate's own relaxed previous stage - realistic accumulated error. "
             "seeded: each stage from the reference's - intrinsic per-stage error, for "
             "stage-by-stage comparison against a DFT ground truth. both: run each, at "
             "roughly double the candidate cost (the reference chain is shared).",
    )
    args = parser.parse_args()

    protocols = PROTOCOLS if args.protocol == "both" else (args.protocol,)

    cfg = RunConfig()
    if args.material:
        cfg = replace(cfg, polymorph=args.material)
    if args.miller:
        cfg = replace(cfg, slab=replace(cfg.slab, miller_index=_cli_miller(args.miller)))
    if args.adsorbate:
        species, coords = ADSORBATE_FRAGMENTS[args.adsorbate]
        cfg = replace(cfg, adsorbate=replace(cfg.adsorbate, species=species, coords=coords))
    if args.max_sites:
        cfg = replace(cfg, adsorbate=replace(cfg.adsorbate, max_per_position=args.max_sites))
    if args.seed_standoff is not None:
        cfg = replace(cfg, adsorbate=replace(cfg.adsorbate, seed_standoff=args.seed_standoff))
    if args.vacancy_block_radius is not None:
        # 0 clears the limit; argparse can't express "None means unset" for a float flag.
        cfg = replace(cfg, slab=replace(
            cfg.slab, vacancy_block_radius=args.vacancy_block_radius or None))
    slab_over = {}
    if args.termination is not None:
        slab_over["termination_index"] = args.termination
    if args.thickness is not None:
        slab_over["min_slab_size"] = args.thickness
    if args.vacuum is not None:
        slab_over["min_vacuum_size"] = args.vacuum
    if args.freeze is not None:
        slab_over["freeze_bottom_fraction"] = args.freeze
    if args.supercell:
        nx, ny = (int(v) for v in args.supercell.split(","))
        slab_over["supercell"] = (nx, ny)
    if slab_over:
        cfg = replace(cfg, slab=replace(cfg.slab, **slab_over))
    if args.fmax is not None:
        cfg = replace(cfg, relax=replace(cfg.relax, fmax=args.fmax))
    if args.max_steps is not None:
        cfg = replace(cfg, relax=replace(cfg.relax, max_steps=args.max_steps))

    # No --candidates keeps the old single-candidate behaviour of `python -m ...pipeline`.
    if args.candidates:
        pprint.pprint(run_sweep(args.candidates, cfg=cfg, outdir=args.outdir, protocols=protocols))
    else:
        pprint.pprint(run(cfg, outdir=args.outdir, protocols=protocols))
