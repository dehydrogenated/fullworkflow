"""Pipeline orchestration — the glue around tested parts (design §7 step 5).

Runs the pseudo-reference plumbing test: the reference backend generates the full
relaxed chain (bulk → slab → vacancy → adsorbate) stage by stage; the candidate backend
then runs the same chain in two modes:

- **seeded** (per-stage): each stage built from the *reference's* relaxed previous
  stage → intrinsic per-stage error.
- **full-pipeline**: each stage built from the *candidate's own* relaxed previous
  stage → realistic accumulated error.

Everything expensive is written to disk immediately. Relaxed geometries land in a
browsable, VESTA-friendly tree (one folder per model, split by stage; each leaf carries
``POSCAR``/``CONTCAR``/``trajectory.xyz`` + a lightweight ``OUTCAR`` of per-step
energies; ``header.json`` timing rollups at the aggregate levels). The run-root analysis
tables — a long-format ``divergence.jsonl``, a ``candidates.jsonl`` recording every
vacancy/adsorbate candidate's energy per model (the ranking-fidelity by-product, §2),
and ``summary.json`` — stay at the run root.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
from ase.data import atomic_numbers, covalent_radii

from .backends import Backend, RelaxResult, get_backend, relax
from .checks import placement_quality_flags
from .config import ADSORBATE_FRAGMENTS, RunConfig
from .diverge import diverge
from .records import (
    DivergenceRecord,
    OUTCAR_NAME,
    append_divergence,
    format_outcar,
    leaf_dir,
    relax_subfolder_name,
    stage_dir,
    write_header,
    write_relaxation,
)
from .records import _sanitize
from .stages import adsorbate_candidates, make_slab, oxygen_vacancy_candidates
from .structures import get_structure


@dataclass
class StageOutput:
    """A model's relaxed structure at one stage, plus energy and tree/timing bookkeeping."""

    structure: object  # pymatgen Structure
    energy: float
    start_fmax: float
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


def _adsorbate_start_distance(initial, n_ads: int):
    """(distance Å, covalent bond length Å) of the adsorbate binding atom to its nearest
    surface atom *at placement*.

    The binding atom is the first adsorbate atom (``coords[0]`` in the fragment), appended
    at index ``len - n_ads``. Compares its placement distance to the nearest substrate atom
    against the covalent bond length for that element pair — the reference for "spawned on
    the answer" (too close) detection. Returns ``(None, None)`` if there's nothing to
    compare."""
    if n_ads <= 0 or len(initial) <= n_ads:
        return None, None
    ads_i = len(initial) - n_ads
    L = initial.lattice
    best_j, best_d = None, 1e9
    for j in range(len(initial) - n_ads):
        d = initial.frac_coords[ads_i] - initial.frac_coords[j]
        d -= np.round(d)  # minimum image under PBC
        dist = float(np.linalg.norm(d @ L.matrix))
        if dist < best_d:
            best_j, best_d = j, dist
    if best_j is None:
        return None, None
    ads_sym = str(initial[ads_i].specie)
    surf_sym = str(initial[best_j].specie)
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
    flags: list[str],
) -> dict:
    """Leaf-scope header dict: everything the OUTCAR summary block reports (design §5)."""
    site = relax_subfolder_name(stage, site_id) or "-" if site_id else "-"
    return {
        "model": backend.name,
        "stage": stage,
        "protocol": protocol,
        "facet": facet,
        "composition": res.structure.composition.reduced_formula,
        "site": site,
        "site_id": site_id,
        "canonical": canonical,
        "optimizer": cfg.relax.optimizer,
        "fmax_target": cfg.relax.fmax,
        "converged": res.converged,
        "nsteps": res.nsteps,
        "n_frames": res.meta.get("n_frames"),
        "elapsed_s": res.meta.get("elapsed_s"),
        "start_fmax": res.start_fmax,
        "energy": res.energy,
        "fmax": res.fmax,
        "geometry_source": geometry_source,
        "adsorbate_max_disp": adsorbate_max_disp,
        "flags": flags,
    }


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
) -> StageOutput:
    """Relax one structure and write its leaf folder (POSCAR/CONTCAR/trajectory/OUTCAR).

    Returns a StageOutput carrying the relaxed geometry plus the leaf path / header /
    optimizer log so a multi-candidate funnel can flip ``canonical`` on the winner later.
    """
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
        _adsorbate_start_distance(structure, len(cfg.adsorbate.species))
        if stage == "adsorbate"
        else (None, None)
    )
    flags = placement_quality_flags(
        start_fmax=res.start_fmax,
        fmax_target=cfg.relax.fmax,
        nsteps=res.nsteps if stage == "adsorbate" else None,
        adsorbate_max_disp=ads_max_disp,
        start_ads_distance=start_ads_distance,
        ads_bond_length=ads_bond_length,
    )
    sdir = stage_dir(outdir, backend.name, protocol, stage)
    leaf = leaf_dir(sdir, stage, site_id)
    header = _relax_header(
        res, backend, cfg,
        stage=stage, protocol=protocol, facet=_facet(cfg, stage),
        site_id=site_id, canonical=canonical, geometry_source=geometry_source,
        adsorbate_max_disp=ads_max_disp, flags=flags,
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
    return StageOutput(
        structure=res.structure,
        energy=res.energy,
        start_fmax=res.start_fmax,
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
    path: Path, *, stage: str, model: str, protocol: str, site_id: dict, energy: float
) -> None:
    with path.open("a") as f:
        f.write(
            json.dumps(
                {
                    "stage": stage,
                    "model": model,
                    "protocol": protocol,
                    "symmetry_class": site_id["symmetry_class"],
                    "site_index": site_id["site_index"],
                    "energy": energy,
                }
            )
            + "\n"
        )


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
) -> dict[int, StageOutput]:
    """Relax ALL decorate() candidates for one stage → record each. Returns per-site outputs.

    Stage-agnostic: ``candidates`` is any iterable of objects exposing ``.structure`` and
    ``.site_id`` (vacancy or adsorbate). Keyed by originating ``site_index`` so callers can
    select the min OR a matched site. This is the shared funnel behind the ``decorate``
    seam — the only per-stage difference lives in the candidate generator. After all sites
    relax, the min-energy one is flagged ``canonical`` (its OUTCAR rewritten) and a
    stage-scope ``header.json`` is written.
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
        )
        _record_candidate_energy(
            candidates_table,
            stage=stage,
            model=backend.name,
            protocol=protocol,
            site_id=cand.site_id,
            energy=out.energy,
        )
        outputs[si] = out

    win = min(outputs, key=lambda si: outputs[si].energy)  # min-energy candidate
    winner = outputs[win]
    winner.canonical = True
    winner.header["canonical"] = True
    (winner.leaf / OUTCAR_NAME).write_text(format_outcar(winner.header, winner.opt_log))

    write_header(
        stage_dir(outdir, backend.name, protocol, stage),
        {
            "stage": stage,
            "protocol": protocol,
            "geometry_source": geometry_source,
            "elapsed_s": round(sum(o.elapsed_s for o in outputs.values()), 3),
            "n_relaxations": len(outputs),
            "canonical": relax_subfolder_name(stage, winner.site_id),
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
    write_header(outdir, {
        "run_id": Path(outdir).name,
        "reference": cfg.reference,
        "candidates": candidates,
        "models": [cfg.reference, *candidates],
        "facet": "".join(map(str, cfg.slab.miller_index)),
        "polymorph": cfg.polymorph,
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
    fullpipeline_vacancy_site: int
    fullpipeline_adsorbate_site: int


def _run_reference_chain(
    ref: Backend, cfg: RunConfig, outdir: Path, bulk_input, cand_table: Path
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

    ref_vac = _run_funnel(
        oxygen_vacancy_candidates(ref_slab.structure, freeze_bottom_fraction=cfg.slab.freeze_bottom_fraction), ref, stage="vacancy",
        protocol="reference", geometry_source="cut_from_relaxed_slab", cfg=cfg,
        outdir=outdir, candidates_table=cand_table,
    )
    outs.extend(ref_vac.values())
    canonical_site = min(ref_vac, key=lambda si: ref_vac[si].energy)  # min-energy vacancy

    ref_ads = _run_funnel(
        adsorbate_candidates(ref_vac[canonical_site].structure, cfg.adsorbate, freeze_bottom_fraction=cfg.slab.freeze_bottom_fraction), ref, stage="adsorbate",
        protocol="reference", geometry_source="placed_on_relaxed_vacancy", cfg=cfg,
        outdir=outdir, candidates_table=cand_table,
    )
    outs.extend(ref_ads.values())
    ads_canonical = min(ref_ads, key=lambda si: ref_ads[si].energy)  # min-energy placement

    return ReferenceChain(
        bulk=ref_bulk, slab=ref_slab, vacancy=ref_vac[canonical_site], adsorbate=ref_ads[ads_canonical],
        vac=ref_vac, ads=ref_ads, canonical_site=canonical_site, ads_canonical=ads_canonical,
        bulk_input=bulk_input, outputs=outs,
    )


def _run_candidate_chain(
    cand: Backend, rc: ReferenceChain, cfg: RunConfig, outdir: Path,
    div_table: Path, cand_table: Path,
) -> CandidateChain:
    """Run one candidate's seeded + full-pipeline chains against the shared reference.

    Everything is keyed off ``cand`` (its name is the tree folder + divergence ``model``),
    so N candidates coexist under one run dir with one shared reference subtree.
    """
    outs: list[StageOutput] = []

    # ---- Candidate: bulk (shared warm start; seeded == full-pipeline here) ------------
    cand_bulk = _relax_record(
        rc.bulk_input, cand, stage="bulk", protocol="seeded",
        geometry_source="db", cfg=cfg, outdir=outdir, relax_cell=True, canonical=True,
    )
    outs.append(cand_bulk)
    _emit(div_table, rc.bulk, cand_bulk, stage="bulk", model=cand.name, protocol="seeded", cfg=cfg)

    # ---- Candidate: SEEDED (each stage from the reference's relaxed previous stage) ---
    cs_slab = _relax_record(
        make_slab(rc.bulk.structure, cfg.slab), cand, stage="slab", protocol="seeded",
        geometry_source="cut_from_reference_bulk", cfg=cfg, outdir=outdir, relax_cell=False, canonical=True,
    )
    outs.append(cs_slab)
    _emit(div_table, rc.slab, cs_slab, stage="slab", model=cand.name, protocol="seeded", cfg=cfg)

    cs_vac = _run_funnel(
        oxygen_vacancy_candidates(rc.slab.structure, freeze_bottom_fraction=cfg.slab.freeze_bottom_fraction), cand, stage="vacancy",
        protocol="seeded", geometry_source="reference_slab", cfg=cfg, outdir=outdir,
        candidates_table=cand_table,
    )
    outs.extend(cs_vac.values())
    # Per-stage attribution: compare the candidate's relaxation of the SAME site.
    if rc.canonical_site in cs_vac:
        _emit(div_table, rc.vacancy, cs_vac[rc.canonical_site], stage="vacancy",
              model=cand.name, protocol="seeded", cfg=cfg)

    # Adsorbate seeded: placed on the *reference's* relaxed vacancy substrate.
    cs_ads = _run_funnel(
        adsorbate_candidates(rc.vacancy.structure, cfg.adsorbate, freeze_bottom_fraction=cfg.slab.freeze_bottom_fraction), cand, stage="adsorbate",
        protocol="seeded", geometry_source="placed_on_reference_vacancy", cfg=cfg,
        outdir=outdir, candidates_table=cand_table,
    )
    outs.extend(cs_ads.values())
    if rc.ads_canonical in cs_ads:
        _emit(div_table, rc.adsorbate, cs_ads[rc.ads_canonical], stage="adsorbate",
              model=cand.name, protocol="seeded", cfg=cfg)

    # ---- Candidate: FULL-PIPELINE (each stage from candidate's own previous stage) ----
    cf_slab = _relax_record(
        make_slab(cand_bulk.structure, cfg.slab), cand, stage="slab", protocol="full_pipeline",
        geometry_source="cut_from_candidate_bulk", cfg=cfg, outdir=outdir, relax_cell=False, canonical=True,
    )
    outs.append(cf_slab)
    _emit(div_table, rc.slab, cf_slab, stage="slab", model=cand.name, protocol="full_pipeline", cfg=cfg)

    cf_vac = _run_funnel(
        oxygen_vacancy_candidates(cf_slab.structure, freeze_bottom_fraction=cfg.slab.freeze_bottom_fraction), cand, stage="vacancy",
        protocol="full_pipeline", geometry_source="candidate_slab", cfg=cfg, outdir=outdir,
        candidates_table=cand_table,
    )
    outs.extend(cf_vac.values())
    cf_site = min(cf_vac, key=lambda si: cf_vac[si].energy)  # candidate's own min
    _emit(div_table, rc.vacancy, cf_vac[cf_site], stage="vacancy", model=cand.name,
          protocol="full_pipeline", cfg=cfg)

    # Adsorbate full-pipeline: placed on the *candidate's own* min-energy vacancy substrate.
    cf_ads = _run_funnel(
        adsorbate_candidates(cf_vac[cf_site].structure, cfg.adsorbate, freeze_bottom_fraction=cfg.slab.freeze_bottom_fraction), cand,
        stage="adsorbate", protocol="full_pipeline",
        geometry_source="placed_on_candidate_vacancy", cfg=cfg, outdir=outdir,
        candidates_table=cand_table,
    )
    outs.extend(cf_ads.values())
    cf_ads_site = min(cf_ads, key=lambda si: cf_ads[si].energy)  # candidate's own min
    _emit(div_table, rc.adsorbate, cf_ads[cf_ads_site], stage="adsorbate",
          model=cand.name, protocol="full_pipeline", cfg=cfg)

    return CandidateChain(
        outputs=outs, fullpipeline_vacancy_site=cf_site, fullpipeline_adsorbate_site=cf_ads_site,
    )


def _fresh_tables(outdir: Path) -> tuple[Path, Path]:
    """Create the run dir and truncate the two run-root analysis tables."""
    outdir.mkdir(parents=True, exist_ok=True)
    div_table = outdir / "divergence.jsonl"
    cand_table = outdir / "candidates.jsonl"
    for p in (div_table, cand_table):
        if p.exists():
            p.unlink()
    return div_table, cand_table


def run(cfg: RunConfig | None = None, outdir: str | Path = "runs/latest") -> dict:
    """Benchmark a single candidate against the reference (thin wrapper over ``run_sweep``)."""
    cfg = cfg or RunConfig()
    return run_sweep([cfg.candidate], cfg=cfg, outdir=outdir)


def run_sweep(
    candidates: list[str] | tuple[str, ...] | None = None,
    cfg: RunConfig | None = None,
    outdir: str | Path = "runs/latest",
) -> dict:
    """Benchmark many candidates against ONE shared reference chain.

    The reference model (``cfg.reference``) relaxes the full bulk→slab→vacancy→adsorbate
    chain exactly once; every candidate in ``candidates`` (default: all registered
    heads/tasks except the reference) is then run against that shared ground truth. Each
    candidate lands in its own ``<outdir>/<candidate>/`` subtree beside the single
    reference subtree, and its divergence rows share the run-root ``divergence.jsonl``
    keyed by model — so the reference is never recomputed per candidate.
    """
    from .backends import ALL_CANDIDATES

    cfg = cfg or RunConfig()
    candidates = list(candidates if candidates is not None else ALL_CANDIDATES)
    outdir = Path(outdir)
    div_table, cand_table = _fresh_tables(outdir)

    ref = get_backend(cfg.reference)
    bulk_input = get_structure(cfg.polymorph)  # mp-id / alias → warm-start bulk cell

    rc = _run_reference_chain(ref, cfg, outdir, bulk_input, cand_table)
    all_outs: list[StageOutput] = list(rc.outputs)

    per_candidate: dict[str, dict] = {}
    for name in candidates:
        cc = _run_candidate_chain(get_backend(name), rc, cfg, outdir, div_table, cand_table)
        all_outs.extend(cc.outputs)
        per_candidate[name] = {
            "fullpipeline_vacancy_site": cc.fullpipeline_vacancy_site,
            "fullpipeline_adsorbate_site": cc.fullpipeline_adsorbate_site,
        }

    # ---- Header rollups over the whole tree (timing, canonical pointers) --------------
    total_elapsed_s = _write_tree_headers(outdir, cfg, all_outs)

    summary = {
        "reference": cfg.reference,
        "candidates": candidates,
        "facet": "".join(map(str, cfg.slab.miller_index)),
        "polymorph": cfg.polymorph,
        "reference_canonical_vacancy_site": rc.canonical_site,
        "reference_canonical_adsorbate_site": rc.ads_canonical,
        "n_vacancy_sites": len(rc.vac),
        "n_adsorbate_sites": len(rc.ads),
        "per_candidate": per_candidate,
        "total_elapsed_s": total_elapsed_s,
        "divergence_table": str(div_table),
        "candidates_table": str(cand_table),
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def run_batch(
    materials: list[str] | tuple[str, ...],
    cfg: RunConfig | None = None,
    outdir: str | Path = "runs/latest",
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
    """
    cfg = cfg or RunConfig()
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    summaries: dict[str, dict] = {}
    for material in materials:
        sub = outdir / _sanitize(material)
        summaries[material] = run(replace(cfg, polymorph=material), outdir=sub)

    index = {
        "materials": list(materials),
        "reference": cfg.reference,
        "candidate": cfg.candidate,
        "total_elapsed_s": round(sum(s["total_elapsed_s"] for s in summaries.values()), 3),
        "runs": {m: _sanitize(m) for m in materials},
    }
    (outdir / "batch.json").write_text(json.dumps(index, indent=2))
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
        help="mp-id (e.g. mp-2657), path to a CIF/POSCAR, or a registered alias. "
             "Defaults to RunConfig.polymorph.",
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
    parser.add_argument("--outdir", default="runs/latest", help="run directory")
    parser.add_argument(
        "--candidates",
        nargs="+",
        help="candidate models to benchmark (default: the single RunConfig.candidate)",
    )
    args = parser.parse_args()

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

    # No --candidates keeps the old single-candidate behaviour of `python -m ...pipeline`.
    if args.candidates:
        pprint.pprint(run_sweep(args.candidates, cfg=cfg, outdir=args.outdir))
    else:
        pprint.pprint(run(cfg, outdir=args.outdir))
