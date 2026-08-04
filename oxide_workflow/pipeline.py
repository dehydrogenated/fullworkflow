"""Pipeline orchestration — the glue around tested parts (design §7 step 5).

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
vacancy/adsorbate candidate's energy per model (the ranking-fidelity by-product, §2),
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
from .energetics import GAS_CACHE_DIR, adsorption_energy, gas_reference_energy
from .records import (
    DivergenceRecord,
    OUTCAR_NAME,
    append_divergence,
    format_outcar,
    leaf_dir,
    relax_subfolder_name,
    stage_dir,
    write_header,
    write_ranking,
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
    e_ads: float | None = None  # adsorbate stage only; None elsewhere
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
    e_ads: float | None,
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
        "e_ads": e_ads,
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
    e_ads_reference: tuple[float, float] | None = None,
) -> StageOutput:
    """Relax one structure and write its leaf folder (POSCAR/CONTCAR/trajectory/OUTCAR).

    Returns a StageOutput carrying the relaxed geometry plus the leaf path / header /
    optimizer log so a multi-candidate funnel can flip ``canonical`` on the winner later.

    ``e_ads_reference`` is the ``(bare substrate, isolated fragment)`` energy pair for the
    adsorbate stage, both from *this* backend; supplying it records E_ads alongside the
    total energy. Omit it on every other stage, where E_ads is undefined.
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
    e_ads = (
        adsorption_energy(res.energy, *e_ads_reference)
        if e_ads_reference is not None
        else None
    )
    sdir = stage_dir(outdir, backend.name, protocol, stage)
    leaf = leaf_dir(sdir, stage, site_id)
    header = _relax_header(
        res, backend, cfg,
        stage=stage, protocol=protocol, facet=_facet(cfg, stage),
        site_id=site_id, canonical=canonical, geometry_source=geometry_source,
        adsorbate_max_disp=ads_max_disp, e_ads=e_ads, flags=flags,
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
        e_ads=e_ads,
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
    e_ads: float | None = None,
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
                    # None on the vacancy stage, where E_ads is undefined.
                    "e_ads": e_ads,
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
            "energy_eV": round(o.energy, 6),
            "dE_from_min_eV": round(o.energy - emin, 6),
            "e_ads_eV": None if o.e_ads is None else round(o.e_ads, 6),
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
) -> dict[int, StageOutput]:
    """Relax ALL decorate() candidates for one stage → record each. Returns per-site outputs.

    Stage-agnostic: ``candidates`` is any iterable of objects exposing ``.structure`` and
    ``.site_id`` (vacancy or adsorbate). Keyed by originating ``site_index`` so callers can
    select the min OR a matched site. This is the shared funnel behind the ``decorate``
    seam — the only per-stage difference lives in the candidate generator. After all sites
    relax, the min-energy one is flagged ``canonical`` (its OUTCAR rewritten) and a
    stage-scope ``header.json`` is written.

    ``e_ads_reference`` (adsorbate stage only) turns on E_ads recording; see
    ``_relax_record``. Site selection is unaffected by it — E_ads differs from the total
    energy by a constant within one funnel, so ranking by either picks the same winner.
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
        )
        _record_candidate_energy(
            candidates_table,
            stage=stage,
            model=backend.name,
            protocol=protocol,
            site_id=cand.site_id,
            energy=out.energy,
            e_ads=out.e_ads,
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

    ref_vac = _run_funnel(
        oxygen_vacancy_candidates(ref_slab.structure, freeze_bottom_fraction=cfg.slab.freeze_bottom_fraction, max_sites=cfg.slab.max_vacancy_sites), ref, stage="vacancy",
        protocol="reference", geometry_source="cut_from_relaxed_slab", cfg=cfg,
        outdir=outdir, candidates_table=cand_table,
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
    e_gas: float,
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

    # ---- vacancy funnel --------------------------------------------------------------
    vac_substrate = rc.slab.structure if seeded else c_slab.structure
    c_vac = _run_funnel(
        oxygen_vacancy_candidates(vac_substrate, freeze_bottom_fraction=frozen, max_sites=cfg.slab.max_vacancy_sites), cand,
        stage="vacancy", protocol=protocol, geometry_source=src["vacancy"], cfg=cfg,
        outdir=outdir, candidates_table=cand_table,
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

    sites: dict[str, dict[str, int]] = {}
    for protocol in protocols:
        chain_outs, chain_sites = _run_protocol_chain(
            cand, rc, cfg, outdir, div_table, cand_table,
            protocol=protocol, cand_bulk=cand_bulk, e_gas=e_gas,
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


def _min_e_ads(outs: list[StageOutput]) -> dict[str, float]:
    """protocol -> the lowest (most strongly bound) E_ads found on the adsorbate stage.

    The headline result of a run: the minimum adsorption energy over all sampled sites.
    """
    best: dict[str, float] = {}
    for o in outs:
        if o.stage != "adsorbate" or o.e_ads is None:
            continue
        if o.protocol not in best or o.e_ads < best[o.protocol]:
            best[o.protocol] = round(o.e_ads, 6)
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
        per_candidate[name] = {"sites": cc.sites, "min_e_ads": _min_e_ads(cc.outputs)}

    # ---- Header rollups over the whole tree (timing, canonical pointers) --------------
    total_elapsed_s = _write_tree_headers(outdir, cfg, all_outs)

    summary = {
        "reference": cfg.reference,
        "candidates": candidates,
        "protocols": list(protocols),
        "facet": "".join(map(str, cfg.slab.miller_index)),
        "polymorph": cfg.polymorph,
        "reference_canonical_vacancy_site": rc.canonical_site,
        "reference_canonical_adsorbate_site": rc.ads_canonical,
        "adsorbate": _fragment_label(cfg.adsorbate),
        # The headline result: lowest E_ads over all sampled sites (eV, negative = bound).
        "reference_min_e_ads": _min_e_ads(rc.outputs).get("reference"),
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

    index = {
        "materials": list(materials),
        "reference": cfg.reference,
        "candidate": cfg.candidate,
        "protocols": list(_check_protocols(protocols)),
        "adsorbate": _fragment_label(cfg.adsorbate),
        "completed": sorted(summaries),
        "failed": failures,
        "total_elapsed_s": round(sum(s["total_elapsed_s"] for s in summaries.values()), 3),
        "runs": {m: _sanitize(m) for m in summaries},
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
        "--max-vacancy-sites", type=int,
        help="cap the vacancy funnel (default: all symmetry-distinct O sites). Smoke-test "
             "knob — keeps the first N by site index, so it can drop the true minimum.",
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
    if args.max_vacancy_sites:
        cfg = replace(cfg, slab=replace(cfg.slab, max_vacancy_sites=args.max_vacancy_sites))
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
