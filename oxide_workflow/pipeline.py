"""Pipeline orchestration — the glue around tested parts (design §7 step 5).

Runs the pseudo-reference plumbing test: the reference backend generates the full
relaxed chain (bulk → slab → vacancy → adsorbate) stage by stage; the candidate backend
then runs the same chain in two modes:

- **seeded** (per-stage): each stage built from the *reference's* relaxed previous
  stage → intrinsic per-stage error.
- **full-pipeline**: each stage built from the *candidate's own* relaxed previous
  stage → realistic accumulated error.

Everything expensive is written to disk immediately: one StructureRecord per relaxed
geometry, a long-format ``divergence.jsonl``, and a ``candidates.jsonl`` recording
every vacancy/adsorbate candidate's energy per model (the ranking-fidelity by-product, §2).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .backends import Backend, RelaxResult, get_backend, relax
from .config import RunConfig
from .diverge import diverge
from .records import DivergenceRecord, StructureRecord, append_divergence
from .stages import adsorbate_candidates, make_slab, oxygen_vacancy_candidates
from .structures import rutile_tio2


@dataclass
class StageOutput:
    """A model's canonical (selected) relaxed structure at one stage, plus its energy."""

    structure: object  # pymatgen Structure
    energy: float
    start_fmax: float
    site_id: dict | None = None


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
    tag: str = "",
) -> RelaxResult:
    """Relax one structure, persist a StructureRecord, return the RelaxResult."""
    res = relax(
        structure,
        backend,
        relax_cell=relax_cell,
        fmax=cfg.relax.fmax,
        max_steps=cfg.relax.max_steps,
        optimizer=cfg.relax.optimizer,
    )
    rec = StructureRecord(
        structure=res.structure,
        stage=stage,
        model=backend.name,
        polymorph=cfg.polymorph,
        facet="".join(map(str, cfg.slab.miller_index)) if stage != "bulk" else "",
        geometry_source=geometry_source,
        protocol=protocol,
        energy=res.energy,
        fmax=res.fmax,
        site_id=site_id,
        record_id="_".join(p for p in [stage, protocol, backend.name, tag] if p),
    )
    rec.save(outdir / "structures")
    return res


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
    seam — the only per-stage difference lives in the candidate generator.
    """
    outputs: dict[int, StageOutput] = {}
    for cand in candidates:
        si = cand.site_id["site_index"]
        res = _relax_record(
            cand.structure,
            backend,
            stage=stage,
            protocol=protocol,
            geometry_source=geometry_source,
            cfg=cfg,
            outdir=outdir,
            relax_cell=False,
            site_id=cand.site_id,
            tag=f"site{si}",
        )
        _record_candidate_energy(
            candidates_table,
            stage=stage,
            model=backend.name,
            protocol=protocol,
            site_id=cand.site_id,
            energy=res.energy,
        )
        outputs[si] = StageOutput(res.structure, res.energy, res.start_fmax, cand.site_id)
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
        facet="".join(map(str, cfg.slab.miller_index)) if stage != "bulk" else "",
    )
    append_divergence(table, rec)
    return rec


def run(cfg: RunConfig | None = None, outdir: str | Path = "runs/latest") -> dict:
    cfg = cfg or RunConfig()
    outdir = Path(outdir)
    (outdir / "structures").mkdir(parents=True, exist_ok=True)
    div_table = outdir / "divergence.jsonl"
    cand_table = outdir / "candidates.jsonl"
    for p in (div_table, cand_table):
        if p.exists():
            p.unlink()

    ref = get_backend(cfg.reference)
    cand = get_backend(cfg.candidate)
    bulk_input = rutile_tio2()

    # ---- Reference chain (generates the relaxed ground truth, stage by stage) --------
    r_bulk = _relax_record(
        bulk_input, ref, stage="bulk", protocol="reference",
        geometry_source="db", cfg=cfg, outdir=outdir, relax_cell=True,
    )
    ref_bulk = StageOutput(r_bulk.structure, r_bulk.energy, r_bulk.start_fmax)

    r_slab = _relax_record(
        make_slab(ref_bulk.structure, cfg.slab), ref, stage="slab", protocol="reference",
        geometry_source="cut_from_relaxed_bulk", cfg=cfg, outdir=outdir, relax_cell=False,
    )
    ref_slab = StageOutput(r_slab.structure, r_slab.energy, r_slab.start_fmax)

    ref_vac = _run_funnel(
        oxygen_vacancy_candidates(ref_slab.structure), ref, stage="vacancy",
        protocol="reference", geometry_source="cut_from_relaxed_slab", cfg=cfg,
        outdir=outdir, candidates_table=cand_table,
    )
    canonical_site = min(ref_vac, key=lambda si: ref_vac[si].energy)  # min-energy vacancy
    ref_vacancy = ref_vac[canonical_site]

    ref_ads = _run_funnel(
        adsorbate_candidates(ref_vacancy.structure, cfg.adsorbate), ref, stage="adsorbate",
        protocol="reference", geometry_source="placed_on_relaxed_vacancy", cfg=cfg,
        outdir=outdir, candidates_table=cand_table,
    )
    ads_canonical = min(ref_ads, key=lambda si: ref_ads[si].energy)  # min-energy placement
    ref_adsorbate = ref_ads[ads_canonical]

    # ---- Candidate: bulk (shared warm start; seeded == full-pipeline here) ------------
    c_bulk = _relax_record(
        bulk_input, cand, stage="bulk", protocol="seeded",
        geometry_source="db", cfg=cfg, outdir=outdir, relax_cell=True,
    )
    cand_bulk = StageOutput(c_bulk.structure, c_bulk.energy, c_bulk.start_fmax)
    _emit(div_table, ref_bulk, cand_bulk, stage="bulk", model=cand.name,
          protocol="seeded", cfg=cfg)

    # ---- Candidate: SEEDED (each stage from the reference's relaxed previous stage) ---
    cs_slab = _relax_record(
        make_slab(ref_bulk.structure, cfg.slab), cand, stage="slab", protocol="seeded",
        geometry_source="cut_from_reference_bulk", cfg=cfg, outdir=outdir, relax_cell=False,
    )
    _emit(div_table, ref_slab, StageOutput(cs_slab.structure, cs_slab.energy, cs_slab.start_fmax),
          stage="slab", model=cand.name, protocol="seeded", cfg=cfg)

    cs_vac = _run_funnel(
        oxygen_vacancy_candidates(ref_slab.structure), cand, stage="vacancy",
        protocol="seeded", geometry_source="reference_slab", cfg=cfg, outdir=outdir,
        candidates_table=cand_table,
    )
    # Per-stage attribution: compare the candidate's relaxation of the SAME site.
    if canonical_site in cs_vac:
        _emit(div_table, ref_vacancy, cs_vac[canonical_site], stage="vacancy",
              model=cand.name, protocol="seeded", cfg=cfg)

    # Adsorbate seeded: placed on the *reference's* relaxed vacancy substrate.
    cs_ads = _run_funnel(
        adsorbate_candidates(ref_vacancy.structure, cfg.adsorbate), cand, stage="adsorbate",
        protocol="seeded", geometry_source="placed_on_reference_vacancy", cfg=cfg,
        outdir=outdir, candidates_table=cand_table,
    )
    if ads_canonical in cs_ads:
        _emit(div_table, ref_adsorbate, cs_ads[ads_canonical], stage="adsorbate",
              model=cand.name, protocol="seeded", cfg=cfg)

    # ---- Candidate: FULL-PIPELINE (each stage from candidate's own previous stage) ----
    cf_slab = _relax_record(
        make_slab(cand_bulk.structure, cfg.slab), cand, stage="slab", protocol="full_pipeline",
        geometry_source="cut_from_candidate_bulk", cfg=cfg, outdir=outdir, relax_cell=False,
    )
    cand_slab_full = StageOutput(cf_slab.structure, cf_slab.energy, cf_slab.start_fmax)
    _emit(div_table, ref_slab, cand_slab_full, stage="slab", model=cand.name,
          protocol="full_pipeline", cfg=cfg)

    cf_vac = _run_funnel(
        oxygen_vacancy_candidates(cand_slab_full.structure), cand, stage="vacancy",
        protocol="full_pipeline", geometry_source="candidate_slab", cfg=cfg, outdir=outdir,
        candidates_table=cand_table,
    )
    cf_site = min(cf_vac, key=lambda si: cf_vac[si].energy)  # candidate's own min
    _emit(div_table, ref_vacancy, cf_vac[cf_site], stage="vacancy", model=cand.name,
          protocol="full_pipeline", cfg=cfg)

    # Adsorbate full-pipeline: placed on the *candidate's own* min-energy vacancy substrate.
    cf_ads = _run_funnel(
        adsorbate_candidates(cf_vac[cf_site].structure, cfg.adsorbate), cand,
        stage="adsorbate", protocol="full_pipeline",
        geometry_source="placed_on_candidate_vacancy", cfg=cfg, outdir=outdir,
        candidates_table=cand_table,
    )
    cf_ads_site = min(cf_ads, key=lambda si: cf_ads[si].energy)  # candidate's own min
    _emit(div_table, ref_adsorbate, cf_ads[cf_ads_site], stage="adsorbate",
          model=cand.name, protocol="full_pipeline", cfg=cfg)

    summary = {
        "reference": cfg.reference,
        "candidate": cfg.candidate,
        "facet": "".join(map(str, cfg.slab.miller_index)),
        "reference_canonical_vacancy_site": canonical_site,
        "candidate_fullpipeline_vacancy_site": cf_site,
        "n_vacancy_sites": len(ref_vac),
        "reference_canonical_adsorbate_site": ads_canonical,
        "candidate_fullpipeline_adsorbate_site": cf_ads_site,
        "n_adsorbate_sites": len(ref_ads),
        "divergence_table": str(div_table),
        "candidates_table": str(cand_table),
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    import pprint

    pprint.pprint(run())
