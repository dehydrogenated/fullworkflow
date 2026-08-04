"""Writer + tree-layout coverage — pure pymatgen, no conda/GPU env required.

Two layers:
- Unit tests for the hierarchical writers in ``records.py`` (path rules, leaf files,
  OUTCAR composition, graceful degradation).
- A fake-backend integration test that runs the whole ``pipeline.run`` with ``relax``
  monkeypatched to an identity relaxation, asserting the full browsable tree, per-level
  ``header.json`` rollups, per-candidate subfolders, and exactly one ``canonical`` per
  stage — all without a model environment.
"""

from __future__ import annotations

import numpy as np
import pytest
from pymatgen.core import Lattice, Structure

from oxide_workflow.records import (
    format_outcar,
    leaf_dir,
    relax_subfolder_name,
    stage_dir,
    write_relaxation,
)


def _cubic(scale: float = 3.0) -> Structure:
    """A tiny 2-atom cubic cell (cheap stand-in for a relaxed geometry)."""
    return Structure(Lattice.cubic(scale), ["Ti", "O"], [[0, 0, 0], [0.5, 0.5, 0.5]])


_SAMPLE_LOG = (
    "      Step     Time          Energy          fmax\n"
    "FIRE:    0 12:00:00     -10.000000        0.900\n"
    "FIRE:    1 12:00:01     -10.500000        0.400\n"
    "FIRE:    2 12:00:02     -10.900000        0.030\n"
)


def _header(**over) -> dict:
    base = {
        "model": "UMA-s", "stage": "vacancy", "protocol": "seeded", "facet": "110",
        "composition": "TiO2", "site": "site4_Ti2c", "canonical": True,
        "optimizer": "FIRE", "fmax_target": 0.05, "converged": True, "nsteps": 2,
        "n_frames": 3, "elapsed_s": 12.3, "start_fmax": 0.9, "energy": -10.9, "fmax": 0.03,
    }
    base.update(over)
    return base


# ---- path helpers -----------------------------------------------------------------------


def test_stage_dir_path_rule(tmp_path):
    # reference → <model>/<stage> (no protocol level)
    assert stage_dir(tmp_path, "MACE-OMAT24", "reference", "slab") == tmp_path / "MACE-OMAT24" / "slab"
    # candidate bulk → <model>/bulk (shared warm start, no protocol level)
    assert stage_dir(tmp_path, "UMA-s", "seeded", "bulk") == tmp_path / "UMA-s" / "bulk"
    # candidate non-bulk → <model>/<protocol>/<stage>
    assert stage_dir(tmp_path, "UMA-s", "seeded", "vacancy") == tmp_path / "UMA-s" / "seeded" / "vacancy"
    assert stage_dir(tmp_path, "UMA-s", "full_pipeline", "adsorbate") == tmp_path / "UMA-s" / "full_pipeline" / "adsorbate"


def test_leaf_subfolder_names(tmp_path):
    sdir = tmp_path / "UMA-s" / "seeded" / "vacancy"
    vac_site = {"symmetry_class": "Ti 2c", "site_index": 4}
    assert relax_subfolder_name("vacancy", vac_site) == "site4_Ti2c"  # spaces stripped
    assert leaf_dir(sdir, "vacancy", vac_site) == sdir / "site4_Ti2c"
    # adsorbate → site-indexed name (densified sampling: type alone isn't unique)
    ads_site = {"symmetry_class": "ontop", "site_index": 0}
    assert relax_subfolder_name("adsorbate", ads_site) == "site0_ontop"
    # single-relax stages → no subfolder (files live in the stage dir)
    assert relax_subfolder_name("bulk", None) == ""
    assert leaf_dir(sdir, "slab", None) == sdir


# ---- write_relaxation leaf --------------------------------------------------------------


def test_write_relaxation_produces_leaf_files(tmp_path):
    initial, final = _cubic(3.0), _cubic(3.05)
    traj = tmp_path / "src.xyz"
    traj.write_text('2\nLattice="3 0 0 0 3 0 0 0 3"\nTi 0 0 0\nO 1.5 1.5 1.5\n')

    dest = write_relaxation(
        tmp_path / "leaf", initial=initial, final=final,
        trajectory_src=traj, header=_header(), opt_log=_SAMPLE_LOG,
    )

    assert (dest / "POSCAR").exists()
    assert (dest / "CONTCAR").exists()
    assert (dest / "trajectory.xyz").exists()
    assert (dest / "OUTCAR").exists()
    # leaf carries NO header.json — the OUTCAR is its summary (duplication eliminated)
    assert not (dest / "header.json").exists()

    # CONTCAR round-trips to the FINAL structure (the priority artifact)
    back = Structure.from_file(dest / "CONTCAR")
    assert np.allclose(back.lattice.abc, final.lattice.abc)

    outcar = (dest / "OUTCAR").read_text()
    assert "canonical: True" in outcar
    assert "final_energy (eV): -10.9" in outcar
    assert "nsteps: 2" in outcar
    # per-step rows surfaced from the optimizer log
    assert "FIRE:    2" in outcar and "-10.900000" in outcar


def test_write_relaxation_graceful_without_trajectory(tmp_path):
    dest = write_relaxation(
        tmp_path / "leaf", initial=_cubic(), final=_cubic(3.1),
        trajectory_src=None, header=_header(canonical=False), opt_log="",
    )
    assert (dest / "POSCAR").exists() and (dest / "CONTCAR").exists()
    assert not (dest / "trajectory.xyz").exists()  # graceful skip, no crash
    outcar = (dest / "OUTCAR").read_text()
    assert "canonical: False" in outcar
    # empty opt_log → header block only, no per-step section
    assert "per-step" not in outcar


def test_format_outcar_omits_missing_keys():
    text = format_outcar({"stage": "bulk", "energy": -5.0, "canonical": True}, opt_log="")
    assert "stage: bulk" in text
    assert "final_energy (eV): -5.0" in text
    assert "nsteps" not in text  # absent key not rendered


# ---- fake-backend full-pipeline integration (no conda) ----------------------------------


def _fake_relax(structure, backend, relax_cell=False, **overrides):
    """Identity relaxation: returns the input geometry with canned energetics + journey meta."""
    from oxide_workflow.backends import RelaxResult

    # Deterministic, slightly site-dependent energy so min-energy selection has a winner.
    energy = -float(len(structure)) - float(structure.frac_coords.sum()) * 1e-3
    return RelaxResult(
        structure=structure,
        energy=energy,
        fmax=0.02,
        start_fmax=0.9,
        nsteps=3,
        converged=True,
        meta={
            "backend": backend.name,
            "elapsed_s": 1.5,
            "trajectory": str(_FAKE_TRAJ),
            "n_frames": 2,
            "opt_log": _SAMPLE_LOG,
        },
    )


_FAKE_TRAJ = None  # set by the test to a real temp file


def _canonical_count(stage_directory) -> int:
    """Count leaf OUTCARs flagged canonical under a stage directory (recursively)."""
    from pathlib import Path

    n = 0
    for outcar in Path(stage_directory).rglob("OUTCAR"):
        if "canonical: True" in outcar.read_text():
            n += 1
    return n


def test_pipeline_builds_full_tree_with_fake_backend(tmp_path, monkeypatch):
    from oxide_workflow import pipeline
    from oxide_workflow.config import RelaxConfig, RunConfig, SlabConfig

    global _FAKE_TRAJ
    _FAKE_TRAJ = tmp_path / "traj_src.xyz"
    _FAKE_TRAJ.write_text(
        '2\nLattice="3 0 0 0 3 0 0 0 3" Properties=species:S:1:pos:R:3\nH 0 0 0\nH 1.5 1.5 1.5\n'
        '2\nLattice="3 0 0 0 3 0 0 0 3" Properties=species:S:1:pos:R:3\nH 0.1 0 0\nH 1.5 1.5 1.5\n'
    )
    monkeypatch.setattr(pipeline, "relax", _fake_relax)

    # Small slab keeps the pymatgen symmetry/adsorption work fast; relax is faked anyway.
    cfg = RunConfig(slab=SlabConfig(supercell=(1, 1)), relax=RelaxConfig(max_steps=1))
    outdir = tmp_path / "run1"
    # Both protocols: this test pins the *full* tree layout, protocol level included.
    summary = pipeline.run(
        cfg, outdir=outdir, protocols=pipeline.PROTOCOLS, gas_cache=tmp_path / "gas"
    )

    # --- top-level tree ---
    assert (outdir / "header.json").exists()
    for name in ("divergence.jsonl", "candidates.jsonl", "summary.json"):
        assert (outdir / name).exists()  # root analysis tables unchanged

    ref = outdir / cfg.reference
    uma = outdir / cfg.candidate
    assert (ref / "header.json").exists() and (uma / "header.json").exists()

    # reference: no protocol level; single-relax stages hold files directly
    for stage in ("bulk", "slab", "vacancy", "adsorbate"):
        assert (ref / stage).is_dir()
    assert (ref / "bulk" / "OUTCAR").exists() and (ref / "bulk" / "CONTCAR").exists()
    assert (ref / "slab" / "trajectory.xyz").exists()

    # candidate: shared bulk at model root; seeded + full_pipeline protocol subtrees
    assert (uma / "bulk" / "OUTCAR").exists()
    for proto in ("seeded", "full_pipeline"):
        assert (uma / proto / "header.json").exists()
        assert (uma / proto / "slab" / "OUTCAR").exists()
        assert (uma / proto / "vacancy" / "header.json").exists()
        assert (uma / proto / "adsorbate" / "header.json").exists()

    # multi-candidate stages: per-site / per-position subfolders
    vac_sites = [d.name for d in (ref / "vacancy").iterdir() if d.is_dir()]
    assert vac_sites and all(s.startswith("site") for s in vac_sites)
    ads_positions = {d.name for d in (ref / "adsorbate").iterdir() if d.is_dir()}
    # Densified geometric arm (ontop/bridge/hollow) + covalent-radii atom-seeding arm
    # (atop_<element><wyckoff>); every leaf is site-indexed for uniqueness.
    assert ads_positions and all(
        d.startswith("site")
        and (
            d.split("_", 1)[1] in {"ontop", "bridge", "hollow"}
            or d.split("_", 1)[1].startswith("atop")
        )
        for d in ads_positions
    )

    # exactly one canonical per stage (single-relax stages count their sole OUTCAR)
    assert _canonical_count(ref / "bulk") == 1
    assert _canonical_count(ref / "slab") == 1
    assert _canonical_count(ref / "vacancy") == 1
    assert _canonical_count(ref / "adsorbate") == 1
    assert _canonical_count(uma / "seeded" / "vacancy") == 1
    assert _canonical_count(uma / "full_pipeline" / "adsorbate") == 1

    # run header timing rollup populated for both models
    import json

    run_header = json.loads((outdir / "header.json").read_text())
    assert set(run_header["elapsed_by_model"]) == {cfg.reference, cfg.candidate}
    assert run_header["total_elapsed_s"] > 0
    assert summary["total_elapsed_s"] > 0


# ---- material resolution + batch runner (no conda) --------------------------------------


def test_get_structure_resolves_alias_and_rejects_unknown():
    from oxide_workflow.structures import get_structure, structure_provenance

    # the built-in alias resolves offline, and says so
    assert get_structure("rutile-tio2").composition.reduced_formula == "TiO2"
    assert structure_provenance("rutile-tio2")["source"] == "builtin"
    # an unregistered, non-mp identifier fails loudly (names the known set)
    with pytest.raises(KeyError):
        get_structure("not-a-material")


def test_get_structure_loads_a_local_cif(tmp_path):
    """The 'upload a CIF' route: point at a file, get a conventional ordered cell."""
    from oxide_workflow.structures import get_structure, rutile_tio2, structure_provenance

    cif = tmp_path / "my_structure.cif"
    rutile_tio2().to(filename=str(cif))

    struct = get_structure(str(cif))
    assert struct.composition.reduced_formula == "TiO2"
    prov = structure_provenance(str(cif))
    assert prov["source"] == "file"
    assert prov["spacegroup"] == "P4_2/mnm"


def test_get_structure_rejects_a_disordered_cif(tmp_path):
    """Partial occupancies fail at resolution, not three stages later in the backend."""
    from pymatgen.core import Lattice, Structure

    from oxide_workflow.structures import get_structure

    disordered = Structure(
        Lattice.cubic(4.0), [{"Ti": 0.9, "Nb": 0.1}, {"O": 1.0}], [[0, 0, 0], [0.5, 0.5, 0.5]]
    )
    cif = tmp_path / "disordered.cif"
    disordered.to(filename=str(cif))

    with pytest.raises(ValueError, match="disordered"):
        get_structure(str(cif))


def test_run_batch_builds_per_material_subtrees(tmp_path, monkeypatch):
    from oxide_workflow import pipeline
    from oxide_workflow.config import RelaxConfig, RunConfig, SlabConfig
    from oxide_workflow.structures import register_structure, rutile_tio2

    global _FAKE_TRAJ
    _FAKE_TRAJ = tmp_path / "traj_src.xyz"
    _FAKE_TRAJ.write_text(
        '2\nLattice="3 0 0 0 3 0 0 0 3" Properties=species:S:1:pos:R:3\nH 0 0 0\nH 1.5 1.5 1.5\n'
    )
    monkeypatch.setattr(pipeline, "relax", _fake_relax)
    # a second "material" — same cell, distinct identifier — proves the sweep, no MP fetch
    register_structure("mp-fake-2", rutile_tio2)

    cfg = RunConfig(slab=SlabConfig(supercell=(1, 1)), relax=RelaxConfig(max_steps=1))
    outdir = tmp_path / "batch"
    materials = ["rutile-tio2", "mp-fake-2"]
    summaries = pipeline.run_batch(
        materials, cfg, outdir=outdir, gas_cache=tmp_path / "gas"
    )

    # one browsable subtree per material, each a complete run
    assert set(summaries) == set(materials)
    for m in materials:
        sub = outdir / m
        assert (sub / "header.json").exists()
        assert (sub / cfg.reference / "bulk" / "OUTCAR").exists()
        assert summaries[m]["total_elapsed_s"] > 0

    # batch index totals the sweep and points at every per-material run
    import json

    index = json.loads((outdir / "batch.json").read_text())
    assert index["materials"] == materials
    assert index["total_elapsed_s"] == pytest.approx(
        sum(summaries[m]["total_elapsed_s"] for m in materials)
    )
    assert set(index["runs"]) == set(materials)


# ---- shared-reference candidate sweep (no conda) ----------------------------------------


def test_run_sweep_shares_one_reference_across_candidates(tmp_path, monkeypatch):
    import json

    from oxide_workflow import pipeline
    from oxide_workflow.config import RelaxConfig, RunConfig, SlabConfig

    global _FAKE_TRAJ
    _FAKE_TRAJ = tmp_path / "traj_src.xyz"
    _FAKE_TRAJ.write_text(
        '2\nLattice="3 0 0 0 3 0 0 0 3" Properties=species:S:1:pos:R:3\nH 0 0 0\nH 1.5 1.5 1.5\n'
    )
    monkeypatch.setattr(pipeline, "relax", _fake_relax)

    cfg = RunConfig(slab=SlabConfig(supercell=(1, 1)), relax=RelaxConfig(max_steps=1))
    outdir = tmp_path / "sweep"
    # a MACE head + a UMA task as candidates against the single mh1-omat reference
    candidates = ["UMA-oc22", "MACE-mh1-mp"]
    summary = pipeline.run_sweep(
        candidates, cfg, outdir=outdir, protocols=pipeline.PROTOCOLS,
        gas_cache=tmp_path / "gas",
    )

    # ONE reference subtree (stages directly, no protocol level) — shared, not recomputed
    ref = outdir / cfg.reference
    assert (ref / "bulk" / "OUTCAR").exists()
    assert (ref / "slab" / "OUTCAR").exists()
    assert not (ref / "seeded").exists()  # reference has no protocol subtrees

    # each candidate gets its own full subtree (bulk + both protocols)
    for name in candidates:
        sub = outdir / name
        assert (sub / "bulk" / "OUTCAR").exists()
        for proto in ("seeded", "full_pipeline"):
            assert (sub / proto / "slab" / "OUTCAR").exists()
            assert (sub / proto / "vacancy" / "header.json").exists()

    # one shared divergence table, keyed by model — carries rows for BOTH candidates only
    div_models = {
        json.loads(ln)["model"]
        for ln in (outdir / "divergence.jsonl").read_text().splitlines() if ln.strip()
    }
    assert div_models == set(candidates)  # reference is the baseline, never its own row

    # run header + summary enumerate the reference once and every candidate
    run_header = json.loads((outdir / "header.json").read_text())
    assert run_header["reference"] == cfg.reference
    assert set(run_header["candidates"]) == set(candidates)
    assert run_header["models"][0] == cfg.reference
    assert set(summary["per_candidate"]) == set(candidates)
    assert set(run_header["elapsed_by_model"]) == {cfg.reference, *candidates}


@pytest.mark.parametrize("protocol", ["full_pipeline", "seeded"])
def test_protocol_flag_runs_only_the_requested_chain(tmp_path, monkeypatch, protocol):
    """A single-protocol run builds only that protocol's subtree and divergence rows.

    The reference chain is unconditional (it is the baseline both protocols compare to),
    so restricting the protocol must shrink the *candidate* subtree and nothing else.
    """
    import json

    from oxide_workflow import pipeline
    from oxide_workflow.config import RelaxConfig, RunConfig, SlabConfig

    global _FAKE_TRAJ
    _FAKE_TRAJ = tmp_path / "traj_src.xyz"
    _FAKE_TRAJ.write_text(
        '2\nLattice="3 0 0 0 3 0 0 0 3" Properties=species:S:1:pos:R:3\nH 0 0 0\nH 1.5 1.5 1.5\n'
    )
    monkeypatch.setattr(pipeline, "relax", _fake_relax)

    cfg = RunConfig(slab=SlabConfig(supercell=(1, 1)), relax=RelaxConfig(max_steps=1))
    outdir = tmp_path / protocol
    summary = pipeline.run_sweep(
        ["UMA-oc22"], cfg, outdir=outdir, protocols=(protocol,), gas_cache=tmp_path / "gas"
    )

    other = next(p for p in pipeline.PROTOCOLS if p != protocol)
    sub = outdir / "UMA-oc22"
    assert (sub / "bulk" / "OUTCAR").exists()  # shared warm start, always present
    assert (sub / protocol / "slab" / "OUTCAR").exists()
    assert not (sub / other).exists()  # the unrequested chain never ran

    # The reference subtree is unaffected by the protocol choice.
    assert (outdir / cfg.reference / "adsorbate").is_dir()

    rows = [
        json.loads(ln)
        for ln in (outdir / "divergence.jsonl").read_text().splitlines() if ln.strip()
    ]
    assert {r["protocol"] for r in rows} == {protocol}
    assert summary["protocols"] == [protocol]
    assert set(summary["per_candidate"]["UMA-oc22"]["sites"]) == {protocol}


def test_unknown_protocol_is_rejected():
    from oxide_workflow import pipeline

    with pytest.raises(ValueError, match="unknown protocol"):
        pipeline._check_protocols(("full_pipeline", "typo"))
    with pytest.raises(ValueError, match="at least one protocol"):
        pipeline._check_protocols(())


# ---- adsorption energy ------------------------------------------------------------------


def test_adsorption_energy_arithmetic_and_missing_terms():
    from oxide_workflow.energetics import adsorption_energy

    # E_ads = E(slab+ads) - E(bare slab) - E(gas); negative = bound.
    assert adsorption_energy(-100.0, -95.0, -1.5) == pytest.approx(-3.5)
    # Any missing term yields None rather than a silently wrong number.
    assert adsorption_energy(None, -95.0, -1.5) is None
    assert adsorption_energy(-100.0, None, -1.5) is None
    assert adsorption_energy(-100.0, -95.0, None) is None


def test_fragment_key_separates_same_formula_different_geometry():
    from oxide_workflow.energetics import fragment_key

    o2 = fragment_key(("O", "O"), ((0.0, 0.0, 0.0), (0.0, 0.0, 1.208)))
    far = fragment_key(("O", "O"), ((0.0, 0.0, 0.0), (0.0, 0.0, 6.0)))
    assert o2 != far  # different reference states must not share a cache entry
    assert o2.startswith("OO_")


def test_gas_reference_energy_is_cached_and_keyed_by_fmax(tmp_path, monkeypatch):
    """The gas reference is computed once per (model, fragment, fmax) and reused."""
    from dataclasses import replace

    from oxide_workflow.backends import get_backend
    from oxide_workflow.config import RelaxConfig, RunConfig
    from oxide_workflow.energetics import gas_reference_energy

    global _FAKE_TRAJ
    _FAKE_TRAJ = tmp_path / "traj_src.xyz"
    _FAKE_TRAJ.write_text(
        '2\nLattice="3 0 0 0 3 0 0 0 3" Properties=species:S:1:pos:R:3\nH 0 0 0\nH 1.5 1.5 1.5\n'
    )
    calls = []

    def counting_relax(structure, backend, **kw):
        calls.append(backend.name)
        return _fake_relax(structure, backend, **kw)

    cfg = RunConfig(relax=RelaxConfig(fmax=0.02))
    backend = get_backend("MACE-mh1-omat")

    first = gas_reference_energy(backend, cfg, counting_relax, cachedir=tmp_path)
    second = gas_reference_energy(backend, cfg, counting_relax, cachedir=tmp_path)
    assert first == second
    assert len(calls) == 1  # second call served from disk, no relaxation

    # A tighter fmax is a different reference state — it must NOT reuse the loose one.
    tighter = replace(cfg, relax=RelaxConfig(fmax=0.001))
    gas_reference_energy(backend, tighter, counting_relax, cachedir=tmp_path)
    assert len(calls) == 2


def test_e_ads_recorded_per_site_and_uses_one_calculator(tmp_path, monkeypatch):
    """Every adsorbate row carries E_ads; vacancy rows carry none.

    Also pins the same-calculator rule: each model's E_ads must be built from its own
    substrate and gas energies, so the fake backend's per-model energies produce a value
    consistent with that model's own recorded totals.
    """
    import json

    from oxide_workflow import pipeline
    from oxide_workflow.config import RelaxConfig, RunConfig, SlabConfig

    global _FAKE_TRAJ
    _FAKE_TRAJ = tmp_path / "traj_src.xyz"
    _FAKE_TRAJ.write_text(
        '2\nLattice="3 0 0 0 3 0 0 0 3" Properties=species:S:1:pos:R:3\nH 0 0 0\nH 1.5 1.5 1.5\n'
    )
    monkeypatch.setattr(pipeline, "relax", _fake_relax)

    cfg = RunConfig(slab=SlabConfig(supercell=(1, 1)), relax=RelaxConfig(max_steps=1))
    outdir = tmp_path / "eads"
    summary = pipeline.run(cfg, outdir=outdir, gas_cache=tmp_path / "gas")

    rows = [
        json.loads(ln)
        for ln in (outdir / "candidates.jsonl").read_text().splitlines() if ln.strip()
    ]
    ads = [r for r in rows if r["stage"] == "adsorbate"]
    vac = [r for r in rows if r["stage"] == "vacancy"]
    assert ads and all(r["e_ads"] is not None for r in ads)
    assert vac and all(r["e_ads"] is None for r in vac)  # undefined on the vacancy stage

    # The headline number equals the minimum over the reference's own adsorbate rows.
    ref_ads = [r for r in ads if r["protocol"] == "reference"]
    assert summary["reference_min_e_ads"] == pytest.approx(
        min(r["e_ads"] for r in ref_ads), abs=1e-6
    )
    assert summary["adsorbate"] == "".join(cfg.adsorbate.species)

    # E_ads reaches the leaf OUTCAR too.
    outcars = list((outdir / cfg.reference / "adsorbate").rglob("OUTCAR"))
    assert outcars and any("E_ads (eV):" in p.read_text() for p in outcars)


def test_stage_rankings_file(tmp_path, monkeypatch):
    """Each funnel stage gets a rankings.csv ordering its sites, best first."""
    import csv
    import json

    from oxide_workflow import pipeline
    from oxide_workflow.config import RelaxConfig, RunConfig, SlabConfig

    global _FAKE_TRAJ
    _FAKE_TRAJ = tmp_path / "traj_src.xyz"
    _FAKE_TRAJ.write_text(
        '2\nLattice="3 0 0 0 3 0 0 0 3" Properties=species:S:1:pos:R:3\nH 0 0 0\nH 1.5 1.5 1.5\n'
    )
    monkeypatch.setattr(pipeline, "relax", _fake_relax)

    cfg = RunConfig(slab=SlabConfig(supercell=(1, 1)), relax=RelaxConfig(max_steps=1))
    outdir = tmp_path / "rank"
    pipeline.run(cfg, outdir=outdir, gas_cache=tmp_path / "gas")

    for stage in ("vacancy", "adsorbate"):
        sdir = outdir / cfg.reference / stage
        rows = list(csv.DictReader((sdir / "rankings.csv").open()))
        header = json.loads((sdir / "header.json").read_text())

        assert len(rows) == header["n_relaxations"]  # every relaxed site is listed
        energies = [float(r["energy_eV"]) for r in rows]
        assert energies == sorted(energies)  # most stable first
        assert [int(r["rank"]) for r in rows] == list(range(1, len(rows) + 1))
        assert float(rows[0]["dE_from_min_eV"]) == 0.0

        # rank 1 is the canonical winner, and its folder really exists
        assert rows[0]["folder"] == header["canonical"]
        assert (sdir / rows[0]["folder"] / "CONTCAR").exists()

        # the recorded margin is the rank1 -> rank2 gap
        assert header["margin_to_runner_up_eV"] == pytest.approx(
            energies[1] - energies[0], abs=1e-6
        )

        # E_ads is present on the adsorbate stage and blank on the vacancy stage
        if stage == "adsorbate":
            assert all(r["e_ads_eV"] for r in rows)
            assert header["min_e_ads"] == pytest.approx(float(rows[0]["e_ads_eV"]), abs=1e-6)
        else:
            assert all(r["e_ads_eV"] == "" for r in rows)
            assert header["min_e_ads"] is None
