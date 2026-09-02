"""CO2 adsorption retest: does bending appear once placement stops biasing it away?

v1 (co2_adsorption_benchmark.py) placed CO2 perfectly vertically with a standoff so close
that every winning site was flagged "no room to relax into the well" -- near an initial
guess, not a genuinely explored minimum.

v2 (this file, first version) fixed the symmetry (N_AZIMUTH azimuthal tilts, see
azimuthal_orientations) and widened the standoff to 1.2 A past the covalent sum. Result:
228/228 valid attempts, zero failures, zero bending anywhere -- but convergence was
suspiciously fast (8-36 steps), suggesting CO2 was still falling straight into a shallow
nearby minimum rather than genuinely exploring. 1.2 A was still a guess, not literature's
number.

v3 (current): SEED_STANDOFF pushed to ~5 A total (literature's actual number -- see the
constant's comment), which needs three more things to be safe and correctly interpreted:

  1. Repeatable extension (max_extensions in AdsorbateConfig/worker_relax.py): one 100-step
     top-up isn't necessarily enough to finish a genuinely long approach, so this repeats up
     to MAX_EXTENSIONS times, each round requiring continued progress.
  2. Anchor-specific desorb/extend checks (anchor_surface_distance in worker_relax.py,
     replacing the old whole-fragment-minimum distance): critical fix -- the old check would
     have flagged a legitimate reorientation (the far O swinging toward a different surface
     atom while the anchor stays bonded) as desorption, killing exactly the trajectories
     this retest exists to let happen. A genuine float-off moves the anchor; a reorientation
     doesn't.
  3. Trajectory persistence (explicit workdir per relaxation, organized under
     outdir/<oxide>/<model>/adsorbate/site<N>_<type>/<orientation>/): v2 called relax()
     without a workdir, so every structure/trajectory went to an ephemeral tempdir and was
     lost. This time every attempt's POSCAR/CONTCAR/trajectory.xyz/OUTCAR survives on disk.

Sites: the original 3 ontop candidates (undercoordinated metal, O2c, 6-fold metal) had a
10/10 record across the v1 sweep -- the undercoordinated metal bound every time, the other
two desorbed every time. undercoordinated_metal_sites() below spends the orientation budget
on that record instead of a blind search: ontop PLUS the nearest bridge and hollow sites
anchored at the same undercoordinated-metal atom (stages.py labels bridge/hollow candidates
by nearest exposed neighbor, so this is a direct site_label match, not a new distance
calculation), PLUS the ontop bridging-oxygen (O2c) site kept explicitly since it's the
literal "Os" of the paper's secondary C-Os bending interaction and deserves a fair attempt
under the corrected placement rather than being assumed dead on the v1/v2 result. O3c
(in-plane oxygen, never in the paper's mechanism, never tested even in v1) stays excluded.
4 sites total.

Every relaxation is wrapped in its own try/except (both per-orientation and, one level up,
per (model, oxide) pair) so a single worker crash on an unattended overnight run costs that
one attempt, not the rest of the sweep -- see results_retest.jsonl's failed/error fields and
scripts/studies/molecular/co2_retest_report.py.

Bulk and slab relaxations are resumed for free from an existing run at the same --outdir
(pipeline._relax_record's own resume-from-disk logic) -- only the adsorbate stage repeats,
since standoff/orientation only affect adsorbate placement. Point --outdir at a copy of the
original run's directory to skip re-relaxing bulk/slab on Sockeye.

    python scripts/studies/molecular/co2_adsorption_retest.py runs/co2_ads_benchmark

Writes runs/co2_ads_benchmark/results_retest.jsonl: EVERY (model, oxide, site, orientation)
attempt, not just winners, so the full picture is inspectable afterward.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import replace
from pathlib import Path

import numpy as np

from oxide_workflow import pipeline
from oxide_workflow.backends import get_backend, relax
from oxide_workflow.config import ADSORBATE_FRAGMENTS, RunConfig
from oxide_workflow.energetics import adsorption_energy, gas_reference_energy
from oxide_workflow.pipeline import _adsorbate_anchor_distance
from oxide_workflow.stages import adsorbate_candidates, make_slab
from oxide_workflow.structures import get_structure

MODELS = ["UMA-omat", "MACE-mh1-omat"]
DESORB_TOL = 2.0  # matches checks.py's placement_quality_flags default

KCAL_TO_EV = 0.0433641
OXIDES = {
    "TiO2": {"mp_id": "mp-2657", "lit_e_ads_kcal": -30.450, "lit_angle": 129.2},
    "GeO2": {"mp_id": "mp-470", "lit_e_ads_kcal": -19.540, "lit_angle": 142.9},
    "SnO2": {"mp_id": "mp-856", "lit_e_ads_kcal": -17.113, "lit_angle": 136.3},
    "IrO2": {"mp_id": "mp-2723", "lit_e_ads_kcal": -4.671, "lit_angle": 178.6},
    "PbO2": {"mp_id": "mp-20725", "lit_e_ads_kcal": -9.663, "lit_angle": 173.9},
}
for _o in OXIDES.values():
    _o["lit_e_ads_eV"] = _o["lit_e_ads_kcal"] * KCAL_TO_EV

ALL_OXIDES = tuple(OXIDES)  # for the --oxides flag's default/validation

POLAR_TILT_DEG = 25.0
N_AZIMUTH = 5
# Fast convergence (8-36 steps) at the previous 1.2 A standoff meant CO2 was falling
# straight into a shallow, nearby minimum rather than genuinely exploring -- pushed out to
# match literature's actual 5 A starting distance instead of guessing at an intermediate
# value again (Ti-O covalent sum ~2.26 A + 2.8 A standoff =~ 5 A total).
SEED_STANDOFF = 2.8
DESORB_CHECK_STEP = 100
DESORB_TREND_WINDOW = 20  # compare distance at 100 against distance at 80, not against t=0
EXTEND_STEPS = 100
MAX_EXTENSIONS = 3  # repeatable: up to 300 + 3*100 = 600 steps for a genuinely long approach
TRIVIAL_START_TOL = 1.0  # start_fmax within this multiple of fmax -> flag as no real work


def azimuthal_orientations(structure, anchor_idx: int, n_ads: int):
    """[(label, Structure), ...]: vertical baseline + N_AZIMUTH evenly-spaced tilts.

    Rigid rotation of the fragment about its anchor atom (index ``anchor_idx``, unmoved):
    every other fragment atom is placed at its own original anchor-distance along the same
    (theta, phi) direction, which keeps the linear CO2 unit's bond lengths and colinearity
    intact at the new starting orientation.
    """
    anchor = structure[anchor_idx].coords.copy()
    radii = [
        float(np.linalg.norm(structure[anchor_idx + k].coords - anchor))
        for k in range(1, n_ads)
    ]
    variants = [("vertical", structure.copy())]
    theta = math.radians(POLAR_TILT_DEG)
    for i in range(N_AZIMUTH):
        phi = math.radians(i * 360.0 / N_AZIMUTH)
        direction = np.array([
            math.sin(theta) * math.cos(phi),
            math.sin(theta) * math.sin(phi),
            math.cos(theta),
        ])
        tilted = structure.copy()
        for k, r in zip(range(1, n_ads), radii):
            tilted[anchor_idx + k].coords = anchor + r * direction
        variants.append((f"tilt{i}_az{i * 360 // N_AZIMUTH}", tilted))
    return variants


def _coordination_of(label: str) -> int | None:
    m = re.match(r"^[A-Za-z]+(\d+)c$", label or "")
    return int(m.group(1)) if m else None


def undercoordinated_metal_sites(candidates):
    """4 candidates: ontop/bridge/hollow anchored at the single most undercoordinated metal
    atom, PLUS the ontop bridging-oxygen (O2c) site -- stages.py labels bridge/hollow sites
    by their nearest exposed neighbor, so "site_label == the metal's label" is exactly "this
    bridge/hollow sits next to that atom".

    The undercoordinated-metal ontop site won 10/10 of the original sweep's (model, oxide)
    pairs outright; O2c and the 6-fold metal both desorbed in all 10 -- but that was under
    the too-close, symmetric-start placement this retest exists to fix, so O2c (the literal
    "Os" of the paper's secondary C-Os bending interaction) still gets a fair fresh attempt
    here rather than being assumed dead. O3c (in-plane oxygen, never in the paper's
    mechanism and never tested even in the original sweep) is deliberately left out.
    """
    by_type: dict[str, list] = {"ontop": [], "bridge": [], "hollow": []}
    for c in candidates:
        by_type.setdefault(c.site_id["symmetry_class"], []).append(c)

    metal_ontop = [
        c for c in by_type["ontop"]
        if c.site_id["site_label"] and not c.site_id["site_label"].startswith("O")
    ]
    if not metal_ontop:
        raise RuntimeError("no metal ontop site found among candidates")
    target = min(metal_ontop, key=lambda c: _coordination_of(c.site_id["site_label"]) or 99)
    label = target.site_id["site_label"]

    picked = [target]
    for ptype in ("bridge", "hollow"):
        match = next((c for c in by_type[ptype] if c.site_id["site_label"] == label), None)
        if match is not None:
            picked.append(match)
        else:
            print(f"    no {ptype} site anchored at {label} found among candidates -- skipping")

    o2c = next((c for c in by_type["ontop"] if c.site_id["site_label"] == "O2c"), None)
    if o2c is not None:
        picked.append(o2c)
    else:
        print("    no ontop O2c site found among candidates -- skipping")
    return picked, label


def run_one(model: str, oxide: str, outdir: Path, cfg: RunConfig) -> list[dict]:
    backend = get_backend(model)
    mp_id = OXIDES[oxide]["mp_id"]
    odir = outdir / oxide

    def relax_record(structure, stage, source_desc, relax_cell=False):
        out = pipeline._relax_record(
            structure, backend, stage=stage, protocol="reference",
            geometry_source=source_desc, cfg=cfg, outdir=odir,
            relax_cell=relax_cell, canonical=True,
        )
        print(f"    {stage:14s} E={out.energy:.4f} eV  {out.header['nsteps']} steps  "
              f"{out.elapsed_s:.0f}s", flush=True)
        return out

    print(f"  [{model} / {oxide}]")
    start = get_structure(mp_id)
    bulk = relax_record(start, "bulk", "db", relax_cell=True).structure

    slab_in = make_slab(bulk, cfg.slab)
    slab_out = relax_record(slab_in, "slab", "cut_from_relaxed_bulk")
    pristine_energy = slab_out.energy
    pristine_structure = slab_out.structure

    co2_species, co2_coords = ADSORBATE_FRAGMENTS["CO2"]
    n_ads = len(co2_species)
    e_gas = gas_reference_energy(
        backend, cfg, pipeline.relax, species=co2_species, coords=co2_coords,
    )

    all_candidates = adsorbate_candidates(
        pristine_structure, cfg.adsorbate, freeze_bottom_fraction=cfg.slab.freeze_bottom_fraction,
    )
    candidates, metal_label = undercoordinated_metal_sites(all_candidates)
    print(f"    {len(candidates)} site(s) (ontop/bridge/hollow near {metal_label}) x "
          f"{1 + N_AZIMUTH} orientation(s) = {len(candidates) * (1 + N_AZIMUTH)} relaxations",
          flush=True)

    rows = []
    for cand in candidates:
        anchor_idx = len(cand.structure) - n_ads
        for label, oriented in azimuthal_orientations(cand.structure, anchor_idx, n_ads):
            site_dir = (
                odir / model / "adsorbate"
                / f"site{cand.site_id['site_index']}_{cand.site_id['symmetry_class']}"
                / label
            )
            try:
                res = relax(
                    oriented, backend, workdir=site_dir,
                    fmax=cfg.relax.fmax, max_steps=cfg.relax.max_steps, optimizer=cfg.relax.optimizer,
                    desorb_check_n_ads=n_ads, desorb_check_step=DESORB_CHECK_STEP,
                    desorb_trend_window=DESORB_TREND_WINDOW,
                    extend_if_approaching=True, extend_steps=EXTEND_STEPS,
                    max_extensions=MAX_EXTENSIONS,
                )
            except Exception as e:
                row = {
                    "model": model, "oxide": oxide, "site_index": cand.site_id["site_index"],
                    "symmetry_class": cand.site_id["symmetry_class"], "orientation": label,
                    "failed": True, "error": str(e)[:500],
                    "e_ads_eV": None, "oco_angle_deg": None,
                    "adsorbed": None, "desorbing": None,
                    "desorbing_early_stop": None, "desorbing_final_geometry": None,
                    "trivial_start": None, "start_fmax": None,
                    "end_anchor_distance_A": None, "anchor_bond_length_A": None,
                    "extended": None, "extensions_used": None, "converged": None, "nsteps": None,
                }
                rows.append(row)
                print(f"    site{row['site_index']} {label:14s} RELAX FAILED: "
                      f"{row['error'][:200]}", flush=True)
                continue
            e_ads = adsorption_energy(res.energy, pristine_energy, e_gas)
            angle = res.structure.get_angle(anchor_idx, anchor_idx + 1, anchor_idx + n_ads - 1)
            early_stopped = bool(res.meta.get("early_stopped_desorbing"))
            # The in-loop check above only fires at ONE specific step -- a slow drift that
            # crosses "farther than start" only afterward, or that settles into a low-force
            # but still-far final position, would otherwise sail through as "converged"
            # unflagged. This is the same final-geometry check pipeline.py uses everywhere
            # else (_adsorbate_anchor_distance + checks.py's desorb_tol convention), applied
            # here explicitly since this script bypasses pipeline._relax_record.
            end_dist, bond_len = _adsorbate_anchor_distance(res.structure, n_ads)
            desorbed_final = (
                end_dist is not None and bond_len is not None
                and end_dist >= DESORB_TOL * bond_len
            )
            desorbing = early_stopped or desorbed_final
            extended = bool(res.meta.get("extended"))
            # Fast-convergence pathology (checks.py's "input already stationary; relaxation
            # did no work"): if the starting force was already at/near fmax, whatever nsteps
            # says "converged" tells you nothing -- there was nothing to relax in the first
            # place. Independent of desorbing/adsorbed; can co-occur with either.
            trivial_start = res.start_fmax <= TRIVIAL_START_TOL * cfg.relax.fmax
            adsorbed = (not desorbing) and bool(res.converged)
            row = {
                "model": model, "oxide": oxide, "site_index": cand.site_id["site_index"],
                "symmetry_class": cand.site_id["symmetry_class"], "orientation": label,
                "failed": False, "error": None,
                "e_ads_eV": e_ads, "oco_angle_deg": angle,
                "adsorbed": adsorbed, "desorbing": desorbing,
                "desorbing_early_stop": early_stopped, "desorbing_final_geometry": desorbed_final,
                "trivial_start": trivial_start, "start_fmax": res.start_fmax,
                "end_anchor_distance_A": end_dist, "anchor_bond_length_A": bond_len,
                "extended": extended, "extensions_used": res.meta.get("extensions_used", 0),
                "converged": res.converged, "nsteps": res.nsteps,
            }
            rows.append(row)
            if desorbing:
                status = "DESORBING(early)" if early_stopped else "DESORBING(final)"
            elif extended:
                status = "EXTENDED"
            else:
                status = "OK"
            trivial_tag = " TRIVIAL_START" if trivial_start else ""
            print(f"    site{row['site_index']} {label:14s} E_ads={e_ads:+.4f} eV  "
                  f"angle={angle:6.1f}  {status:16s}"
                  f"nsteps={res.nsteps}{trivial_tag}", flush=True)
    return rows


def main(outdir: Path, fmax: float, oxides: list[str] = None, models: list[str] = None) -> None:
    oxides = oxides or list(OXIDES)
    models = models or MODELS
    co2_species, co2_coords = ADSORBATE_FRAGMENTS["CO2"]
    cfg = RunConfig()
    cfg = replace(cfg, adsorbate=replace(
        cfg.adsorbate, species=co2_species, coords=co2_coords,
        # Enumerate all three position types uncapped; undercoordinated_metal_sites() then
        # keeps only the ontop/bridge/hollow trio anchored at the undercoordinated metal,
        # so nothing here is spent relaxing sites with no binding track record.
        positions=("ontop", "bridge", "hollow"), max_per_position=None,
        seed_standoff=SEED_STANDOFF, desorb_early_stop_step=DESORB_CHECK_STEP,
        desorb_trend_window=DESORB_TREND_WINDOW,
        extend_if_approaching=True, extend_steps=EXTEND_STEPS, max_extensions=MAX_EXTENSIONS,
    ))
    cfg = replace(cfg, relax=replace(cfg.relax, fmax=fmax))
    outdir.mkdir(parents=True, exist_ok=True)
    results_path = outdir / "results_retest.jsonl"

    # One (model, oxide) pair failing before it even reaches the per-orientation loop
    # (bulk/slab relax, gas reference, site-finding) must not cost the other 9 pairs on an
    # unattended overnight run -- same pattern as sockeye_oc22.sh's model sweep. Individual
    # relax() failures inside the orientation loop are already caught in run_one() itself;
    # this is the outer net for everything before that.
    pair_failures: list[tuple[str, str, str]] = []
    for model in models:
        for oxide in oxides:
            try:
                rows = run_one(model, oxide, outdir, cfg)
            except Exception as e:
                print(f"  [{model} / {oxide}] PAIR FAILED before orientation loop: {e}",
                      flush=True)
                pair_failures.append((model, oxide, str(e)[:500]))
                rows = [{
                    "model": model, "oxide": oxide, "site_index": None, "symmetry_class": None,
                    "orientation": None, "failed": True, "error": f"pair-level: {e}"[:500],
                    "e_ads_eV": None, "oco_angle_deg": None,
                    "adsorbed": None, "desorbing": None,
                    "desorbing_early_stop": None, "desorbing_final_geometry": None,
                    "trivial_start": None, "start_fmax": None,
                    "end_anchor_distance_A": None, "anchor_bond_length_A": None,
                    "extended": None, "extensions_used": None, "converged": None, "nsteps": None,
                }]
            with results_path.open("a") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")

    print(f"\nwrote {results_path}")
    if pair_failures:
        print(f"\n{len(pair_failures)} (model, oxide) pair(s) failed before relaxing any site:")
        for model, oxide, err in pair_failures:
            print(f"  {model:16s}{oxide:8s}{err[:150]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("outdir", type=Path)
    ap.add_argument("--fmax", type=float, default=0.02)
    ap.add_argument("--oxides", nargs="+", choices=ALL_OXIDES, default=list(ALL_OXIDES),
                     help=f"which oxide(s) to run (default: all {len(ALL_OXIDES)})")
    ap.add_argument("--models", nargs="+", choices=MODELS, default=MODELS,
                     help="which model(s) to run (default: both)")
    a = ap.parse_args()
    main(a.outdir, a.fmax, a.oxides, a.models)
