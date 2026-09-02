"""H2O dissociation probe: does a well-oriented starting guess find the dissociated
state (Ti5c-OH + O2c-H) on its own, without hand-building a separate endpoint or
scanning a reaction coordinate?

Places molecular H2O on the most undercoordinated surface Ti, oriented so one O-H bond
points at the nearest bridging O2c -- giving the proton the shortest possible path to a
second surface bond, rather than a blind vertical/tilted guess (contrast the CO2 track's
azimuthal_orientations(), which sweeps blindly because there's no obvious "target atom"
for CO2's secondary interaction). One full, unconstrained relaxation per model: if the
model's own PES wants to transfer that H to O2c, this orientation gives it the best shot.

Then: take whatever the relaxation settles into (dissociated, or still molecular but
doubly-bonded), nudge the two fragments a small distance further apart on their own
sites, and re-relax. If the nudged copy relaxes back to about the same energy/geometry,
that's evidence of a genuine local minimum, not an artifact of where the starting
orientation happened to land.

Kowalski, Meyer & Marx, PRB 79, 115410 (2009) is the eventual literature target (Table VI
+ NEB barrier ~0.14-0.16 eV) for TiO2 specifically, but this script doesn't compare
against it yet -- it only asks the prior, cheaper question: can each model find the
dissociated basin at all.

Covers TiO2/RuO2/IrO2 (OXIDES) -- Gonzalez et al. 2019's three materials. All 4
orientations were compared once already; MODES_TO_RUN narrows the default relax loop to
just "bisector" (the one that adsorbed rather than repulsed) plus a DIAGONAL_NUDGE rigid
translate on top of it, since rotation about a fixed O can't close a multi-A gap to O2c
on its own -- see DIAGONAL_NUDGE's own comment. Re-widen MODES_TO_RUN to retest the others
per-oxide if bisector doesn't transfer cleanly to Ru/Ir.

    python scripts/studies/molecular/h2o_dissociation_probe.py runs/h2o_dissociation_probe --oxides TiO2
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import replace
from pathlib import Path

import numpy as np
from pymatgen.core import Structure

from oxide_workflow import pipeline
from oxide_workflow.backends import get_backend, relax
from oxide_workflow.config import ADSORBATE_FRAGMENTS, RunConfig
from oxide_workflow.energetics import adsorption_energy, gas_reference_energy
from oxide_workflow.stages import (
    _site_environments,
    adsorbate_candidates,
    exposed_surface_atoms,
    make_slab,
    site_label,
)
from oxide_workflow.structures import get_structure

# TiO2/RuO2/IrO2 -- Gonzalez et al. 2019 (ACS Omega 4, 2989-2999)'s three materials.
# find_ti_and_o2c_anchors()/orient_toward() are written generically (lowest-coordination
# exposed metal + its nearest exposed O2c) despite the Ti-suffixed names below -- those
# names predate this becoming a 3-material sweep and are kept as-is rather than renamed
# across every call site for a purely cosmetic change.
OXIDES = {"TiO2": "mp-2657", "RuO2": "mp-825", "IrO2": "mp-2723"}
MODELS = ["MACE-mh1-omat", "UMA-oc22", "SevenNet-omni-omat24"]
FMAX = 0.02
OH_BOND_MAX = 1.3  # A; beyond this an O-H has dissociated, not stretched (oc22_diverge.py precedent)
# A; target O-H bond length to snap the transferring H onto O2c for the nudge step --
# matches ADSORBATE_FRAGMENTS["H2O"]'s own O-H length (norm of (0, 0.757, 0.5859)).
NUDGE_TARGET_BOND = 0.957
# +1 A past the config default (0.2) so the molecule starts with real room to reorient
# during relaxation instead of nearly on top of its own answer, and reads clearly as a
# separate starting frame rather than a placement glued to the surface.
SEED_STANDOFF = 1.2
# Rigid translate of the whole placed molecule (O and both H) toward O2c, on top of
# orient_toward()'s rotation-only seed. Rotation alone can't close the gap: O is pinned
# directly above Ti by the site-finder, and a real O-H bond is only ~0.957 A, so even a
# perfectly-aimed H reaches barely a third of the way to an O2c that starts several A off
# (confirmed empirically -- nudge_apart's own docstring records the bisector orientation
# leaving the approaching H at 1.95 A after a *full* relaxation, down from ~2.9-3.6 A at
# the start; most of that gap was a distance problem, not an orientation problem). 1.0 A
# closes a meaningful fraction of that without placing O implausibly off its M5c anchor
# or biasing the H already into a formed bond -- a starting guess, not the answer.
DIAGONAL_NUDGE = 1.0
DESORB_CHECK_STEP = 100
DESORB_TREND_WINDOW = 20
EXTEND_STEPS = 100
MAX_EXTENSIONS = 3


def _coordination_of(label: str) -> "int | None":
    m = re.match(r"^[A-Za-z]+(\d+)c$", label or "")
    return int(m.group(1)) if m else None


def min_image_vector(lattice, cart_a: np.ndarray, cart_b: np.ndarray) -> np.ndarray:
    """The Cartesian displacement cart_a - cart_b through whichever periodic image makes
    it shortest -- a raw ``cart_a - cart_b`` can point almost the full cell width and
    backwards from the true short-way direction whenever the two points are close only
    through the periodic boundary (e.g. one near x=0, the other near x=cell_a)."""
    frac_diff = lattice.get_fractional_coords(cart_a - cart_b)
    frac_diff -= np.round(frac_diff)
    return lattice.get_cartesian_coords(frac_diff)


def undercoordinated_metal_site(candidates):
    metal_ontop = [
        c for c in candidates
        if c.site_id["symmetry_class"] == "ontop"
        and c.site_id["site_label"] and not c.site_id["site_label"].startswith("O")
    ]
    if not metal_ontop:
        raise RuntimeError("no metal ontop site found")
    return min(metal_ontop, key=lambda c: _coordination_of(c.site_id["site_label"]) or 99)


def find_ti_and_o2c_anchors(pristine: Structure) -> "tuple[int, int]":
    """(ti_idx, o2c_idx): the most undercoordinated exposed Ti, and its nearest exposed
    bridging O2c -- both substrate atom indices into ``pristine``, found independently of
    adsorbate_candidates so the target direction is known before any fragment is placed."""
    z = pristine.cart_coords[:, 2]
    exposed = exposed_surface_atoms(pristine, depth=float(z.max() - z.min()))
    environments = _site_environments(pristine)

    ti_candidates = [i for i in exposed if environments[i][0] != "O"]
    if not ti_candidates:
        raise RuntimeError("no exposed metal atom found")
    ti_idx = min(ti_candidates, key=lambda i: environments[i][2])  # lowest coordination number

    o2c_candidates = [
        i for i in exposed
        if environments[i][0] == "O" and site_label(*environments[i][::2]) == "O2c"
    ]
    if not o2c_candidates:
        raise RuntimeError("no exposed O2c site found")
    o2c_idx = min(o2c_candidates, key=lambda i: pristine.get_distance(ti_idx, i))
    return ti_idx, o2c_idx


ORIENTATION_MODES = ("vertical", "vertical_flipped", "bisector", "single_h_ti_facing")
# Which of ORIENTATION_MODES run_one() actually relaxes. All 4 were compared once already
# (see git history / prior run notes): single_h_ti_facing repulsed instead of adsorbing --
# its pre-bend mirror step starts both H's pointed down into the surface/metal region
# instead of preserving the template's lone-pair-down pose, a likely steric cause -- while
# bisector adsorbed (H...O2c closed to 1.95 A after a full relaxation, per nudge_apart's
# docstring). Narrowed to bisector only so a placement-tuning pass (DIAGONAL_NUDGE) doesn't
# re-pay the cost of relaxing all 4 modes for every model.
MODES_TO_RUN = ("bisector",)


def _rotation_aligning(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Rotation matrix mapping unit vector ``a`` onto unit vector ``b`` (Rodrigues'
    formula: rotate about axis = cross(a, b) by angle = arccos(dot(a, b))). Degenerate
    when the cross product vanishes -- already aligned (identity) or exactly antiparallel
    (180 deg about any axis perpendicular to ``a``, picked via an arbitrary second
    reference vector since cross(a, a) gives no axis to rotate about)."""
    cos_angle = float(np.clip(np.dot(a, b), -1.0, 1.0))
    axis = np.cross(a, b)
    axis_norm = np.linalg.norm(axis)

    if axis_norm >= 1e-8:
        axis = axis / axis_norm
        angle = math.acos(cos_angle)
        K = np.array([
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ])
        return np.eye(3) + math.sin(angle) * K + (1.0 - cos_angle) * (K @ K)
    if cos_angle > 0:
        return np.eye(3)
    perp = np.array([1.0, 0.0, 0.0])
    axis = np.cross(a, perp)
    axis = axis / np.linalg.norm(axis)
    K = np.array([
        [0.0, -axis[2], axis[1]],
        [axis[2], 0.0, -axis[0]],
        [-axis[1], axis[0], 0.0],
    ])
    return np.eye(3) + 2.0 * (K @ K)  # Rodrigues at theta=pi: sin=0, 1-cos=2


def orient_toward(
    structure: Structure, anchor_idx: int, n_ads: int, target_dir: np.ndarray,
    mode: str = "bisector",
) -> Structure:
    """Reorient the placed fragment about its anchor atom (O) so it reaches toward
    ``target_dir`` (a unit Cartesian vector, anchor -> target O2c). Preserves every H's
    own O-H bond length (rotation only, no stretching). Returns a new Structure;
    ``structure`` is untouched.

    ``structure[anchor_idx]`` is the O atom (already placed at its real site);
    ``structure[anchor_idx+1 .. anchor_idx+n_ads-1]`` are the two H atoms.

    Several starting guesses, since it's not obvious a priori which one a given model's
    PES prefers:

    - ``"bisector"``: rigidly rotate the *whole* molecule so its default "up" direction
      (the H-O-H bisector, which points straight away from the surface in
      ADSORBATE_FRAGMENTS' H2O template -- both H's have positive z) points at O2c
      instead. Both H's swing toward the target together, symmetric about the new axis.
    - ``"single_h_ti_facing"``: leave the molecule's default lone-pair-down pose mirrored
      through O first (z negated, so both H's start pointing back down toward Ti instead
      of away from it -- the wedge opens toward the surface), then bend only whichever
      single H sits closest in angle to ``target_dir`` onto it, leaving the other H at
      its mirrored position. Picks the H needing the smaller rotation, purely to keep the
      starting guess as close to the mirrored placement as possible.
    - ``"vertical"``: no rotation at all -- the plain bent H2O template (both H's at
      their default (0, +-0.757, 0.5859) position), straight up over the anchor. The O
      is directly above the Ti in every mode (only the H's ever move), but this is the
      untilted baseline for a plain visual sanity check.
    - ``"vertical_flipped"``: the same plain bent template rotated 180 deg about the
      y-axis through the O anchor (local x is always 0 in the template, so this is
      exactly z -> -z for both H's) -- both H's now point down toward the surface
      instead of up away from it. O stays exactly on top of Ti; only the H's flip.
    """
    if mode not in ORIENTATION_MODES:
        raise ValueError(f"mode must be one of {ORIENTATION_MODES}, got {mode!r}")
    target_dir = np.asarray(target_dir, dtype=float)
    target_dir = target_dir / np.linalg.norm(target_dir)
    anchor = structure[anchor_idx].coords
    oriented = structure.copy()

    if mode == "vertical":
        return oriented

    if mode == "vertical_flipped":
        for k in range(anchor_idx + 1, anchor_idx + n_ads):
            relative = structure[k].coords - anchor
            oriented[k].coords = anchor + relative * np.array([-1.0, 1.0, -1.0])
        return oriented

    if mode == "bisector":
        rotation = _rotation_aligning(np.array([0.0, 0.0, 1.0]), target_dir)
        for k in range(anchor_idx + 1, anchor_idx + n_ads):
            relative = structure[k].coords - anchor
            oriented[k].coords = anchor + rotation @ relative
        return oriented

    # single_h / single_h_ti_facing: bend only the closer H onto target_dir, leave the
    # other H at its (possibly mirrored) template position.
    h_indices = list(range(anchor_idx + 1, anchor_idx + n_ads))
    relatives = {k: structure[k].coords - anchor for k in h_indices}
    if mode == "single_h_ti_facing":
        relatives = {k: rel * np.array([1.0, 1.0, -1.0]) for k, rel in relatives.items()}

    angle_to_target = {
        k: math.acos(float(np.clip(np.dot(rel / np.linalg.norm(rel), target_dir), -1.0, 1.0)))
        for k, rel in relatives.items()
    }
    bend_idx = min(h_indices, key=lambda k: angle_to_target[k])
    bond_length = float(np.linalg.norm(relatives[bend_idx]))
    oriented[bend_idx].coords = anchor + bond_length * target_dir
    for k, rel in relatives.items():
        if k != bend_idx:
            oriented[k].coords = anchor + rel
    return oriented


def translate_toward(
    structure: Structure, anchor_idx: int, n_ads: int, target_dir: np.ndarray, distance: float,
) -> Structure:
    """Rigidly translate the whole fragment (O and every H) by ``distance`` along
    ``target_dir``. Unlike ``orient_toward``, which only rotates the H's about a fixed O,
    this moves the O itself -- see DIAGONAL_NUDGE's own comment for why that's the piece
    rotation alone can't fix. Every internal O-H bond length and H-O-H angle is preserved
    (a pure translation of every atom by the same vector); only the rigid body's position
    relative to the surface changes."""
    translated = structure.copy()
    for k in range(anchor_idx, anchor_idx + n_ads):
        translated[k].coords = structure[k].coords + distance * target_dir
    return translated


def is_dissociated(structure: Structure, h_idx: int, o_water_idx: int, o2c_idx: int) -> bool:
    """True once the oriented H has left the water oxygen's bonding range and entered the
    bridging O2c's -- i.e. a real O-H bond broke and a new one formed, not just a stretch
    (OH_BOND_MAX matches oc22_diverge.py's own dissociated-vs-stretched cutoff).

    Wrapped in bool(): Structure.get_distance() returns a numpy float, so the comparisons
    below are numpy.bool_, not Python's built-in bool -- json.dumps() rejects numpy.bool_
    (it checks isinstance(x, bool) specifically), so a caller writing this straight into a
    results row crashes on write. Confirmed via an actual TypeError on Sockeye, not assumed."""
    return bool(
        structure.get_distance(h_idx, o_water_idx) > OH_BOND_MAX
        and structure.get_distance(h_idx, o2c_idx) < OH_BOND_MAX
    )


def nudge_apart(
    structure: Structure, h_idx: int, o2c_idx: int, target_bond_length: float = NUDGE_TARGET_BOND,
) -> Structure:
    """Snap ``h_idx`` onto O2c at a real O-H bonding distance (``target_bond_length``,
    the H2O template's own O-H length by default -- see NUDGE_TARGET_BOND) instead of a
    small symmetric separation. A first relaxation only closed part of the gap (e.g. the
    1.95 A the bisector run left it at, down from ~2.9-3.6 A at the start, but still far
    from a bonded 0.96-ish A) -- rather than nudge a further fixed increment and hope,
    finish the approach directly along H's own current direction to O2c, so the second
    relaxation starts from "the transfer basically happened" instead of "a bit closer
    than before." The water O and the other H are untouched: forcing H fully onto O2c
    already stretches its old O-H bond on its own (real evidence for
    ``is_dissociated()``, not a separate push needed).

    Tests whether the settled configuration is a real local minimum (the nudged copy
    relaxes back to ~the same energy/geometry) or just resting where the first
    relaxation happened to stop."""
    nudged = structure.copy()
    toward_o2c = structure[h_idx].coords - structure[o2c_idx].coords
    toward_o2c = toward_o2c / np.linalg.norm(toward_o2c)
    nudged[h_idx].coords = structure[o2c_idx].coords + target_bond_length * toward_o2c
    return nudged


def run_one(
    model: str, oxide: str, outdir: Path, cfg: RunConfig, dump_frame: bool = False,
) -> "dict | None":
    backend = get_backend(model)
    odir = outdir / oxide / model

    def relax_record(structure, stage, source_desc, relax_cell=False):
        out = pipeline._relax_record(
            structure, backend, stage=stage, protocol="reference",
            geometry_source=source_desc, cfg=cfg, outdir=odir,
            relax_cell=relax_cell, canonical=True,
        )
        print(f"    {stage:14s} E={out.energy:.4f} eV  {out.header['nsteps']} steps  "
              f"{out.elapsed_s:.0f}s", flush=True)
        return out

    print(f"[{oxide} / {model}]", flush=True)
    start = get_structure(OXIDES[oxide])
    bulk = relax_record(start, "bulk", "db", relax_cell=True).structure
    slab_in = make_slab(bulk, cfg.slab)
    slab_out = relax_record(slab_in, "slab", "cut_from_relaxed_bulk")
    pristine_energy = slab_out.energy
    pristine_structure = slab_out.structure

    ti_idx, o2c_idx = find_ti_and_o2c_anchors(pristine_structure)
    print(f"    Ti anchor idx={ti_idx}, target O2c idx={o2c_idx}, "
          f"distance={pristine_structure.get_distance(ti_idx, o2c_idx):.3f} A", flush=True)

    h2o_species, h2o_coords = ADSORBATE_FRAGMENTS["H2O"]
    n_ads = len(h2o_species)
    e_gas = gas_reference_energy(
        backend, cfg, pipeline.relax, species=h2o_species, coords=h2o_coords,
    )

    candidates = adsorbate_candidates(
        pristine_structure,
        replace(cfg.adsorbate, species=h2o_species, coords=h2o_coords, positions=("ontop",),
                seed_standoff=SEED_STANDOFF),
        freeze_bottom_fraction=cfg.slab.freeze_bottom_fraction,
    )
    cand = undercoordinated_metal_site(candidates)
    anchor_idx = len(cand.structure) - n_ads  # O
    h_near_idx = anchor_idx + 1

    target_dir = min_image_vector(
        pristine_structure.lattice, pristine_structure[o2c_idx].coords, pristine_structure[ti_idx].coords,
    )
    target_dir = target_dir / np.linalg.norm(target_dir)
    # Ti->O2c is not lateral -- see h2o_adsorption_benchmark.py's identical comment.
    # Translating along the full 3D target_dir pushes the seed higher with every nudge
    # instead of sliding it sideways; orient_toward still gets the full 3D vector (the H
    # really should tilt up toward O2c), only the rigid translate is restricted to lateral.
    lateral_dir = np.array([target_dir[0], target_dir[1], 0.0])
    lateral_norm = np.linalg.norm(lateral_dir)
    lateral_dir = lateral_dir / lateral_norm if lateral_norm > 1e-8 else target_dir

    rows = []
    for mode in MODES_TO_RUN:
        oriented = orient_toward(cand.structure, anchor_idx, n_ads, target_dir, mode=mode)
        oriented = translate_toward(oriented, anchor_idx, n_ads, lateral_dir, DIAGONAL_NUDGE)
        start_o_ti = oriented.get_distance(anchor_idx, ti_idx)
        start_h_o2c = oriented.get_distance(h_near_idx, o2c_idx)
        print(f"    [{mode}] after {DIAGONAL_NUDGE} A nudge: O-Ti={start_o_ti:.3f} A, "
              f"H...O2c={start_h_o2c:.3f} A (pristine Ti-O2c was "
              f"{pristine_structure.get_distance(ti_idx, o2c_idx):.3f} A)", flush=True)

        if dump_frame:
            frame_path = odir / "adsorbate" / mode / "starting_frame.vasp"
            frame_path.parent.mkdir(parents=True, exist_ok=True)
            oriented.to(filename=str(frame_path), fmt="poscar")
            print(f"    [{mode}] wrote {frame_path} -- unrelaxed, no adsorbate "
                  f"relaxation run", flush=True)
            continue

        res = relax(
            oriented, backend, workdir=odir / "adsorbate" / mode,
            fmax=cfg.relax.fmax, max_steps=cfg.relax.max_steps, optimizer=cfg.relax.optimizer,
            desorb_check_n_ads=n_ads, desorb_check_step=DESORB_CHECK_STEP,
            desorb_trend_window=DESORB_TREND_WINDOW,
            extend_if_approaching=True, extend_steps=EXTEND_STEPS, max_extensions=MAX_EXTENSIONS,
        )
        e_ads = adsorption_energy(res.energy, pristine_energy, e_gas)
        dissociated = is_dissociated(res.structure, h_near_idx, anchor_idx, o2c_idx)
        print(f"    [{mode}] E_ads={e_ads:+.4f} eV  dissociated={dissociated}  "
              f"converged={res.converged}  nsteps={res.nsteps}", flush=True)

        nudged = nudge_apart(res.structure, h_near_idx, o2c_idx)
        res_nudged = relax(
            nudged, backend, workdir=odir / "adsorbate" / mode / "nudged",
            fmax=cfg.relax.fmax, max_steps=cfg.relax.max_steps, optimizer=cfg.relax.optimizer,
            desorb_check_n_ads=n_ads, desorb_check_step=DESORB_CHECK_STEP,
            desorb_trend_window=DESORB_TREND_WINDOW,
            extend_if_approaching=True, extend_steps=EXTEND_STEPS, max_extensions=MAX_EXTENSIONS,
        )
        e_ads_nudged = adsorption_energy(res_nudged.energy, pristine_energy, e_gas)
        dissociated_nudged = is_dissociated(res_nudged.structure, h_near_idx, anchor_idx, o2c_idx)
        print(f"    [{mode}] nudged E_ads={e_ads_nudged:+.4f} eV  dissociated={dissociated_nudged}  "
              f"converged={res_nudged.converged}  nsteps={res_nudged.nsteps}  "
              f"dE_from_nudge={e_ads_nudged - e_ads:+.4f} eV", flush=True)

        rows.append({
            "model": model, "oxide": oxide, "orientation": mode, "ti_idx": ti_idx, "o2c_idx": o2c_idx,
            "e_ads_eV": e_ads, "dissociated": dissociated,
            "converged": res.converged, "nsteps": res.nsteps,
            "e_ads_nudged_eV": e_ads_nudged, "dissociated_nudged": dissociated_nudged,
            "converged_nudged": res_nudged.converged, "nsteps_nudged": res_nudged.nsteps,
            "d_e_nudge_eV": e_ads_nudged - e_ads,
        })

    return rows or None


def main(outdir: Path, oxides: list[str], models: list[str], dump_frame: bool) -> None:
    cfg = RunConfig()
    cfg = replace(cfg, relax=replace(cfg.relax, fmax=FMAX))
    outdir.mkdir(parents=True, exist_ok=True)
    results_path = outdir / "results.jsonl"
    # Per-pair try/except so one bad (oxide, model) -- OOM, a model's own loader error,
    # anything -- doesn't silently kill every pair queued after it with zero record of
    # why (the earlier failure mode: an uncaught exception here took the whole script
    # down mid-sweep and results.jsonl just stopped, with nothing in it saying where or
    # why). Matches o_adsorption_benchmark.py's pair_failures pattern.
    pair_failures: list[str] = []
    for oxide in oxides:
        for model in models:
            tag = f"{oxide}/{model}"
            try:
                rows = run_one(model, oxide, outdir, cfg, dump_frame=dump_frame)
            except Exception as e:
                print(f"  [{tag}] PAIR FAILED: {e}", flush=True)
                pair_failures.append(f"{tag}: {str(e)[:300]}")
                rows = [{
                    "model": model, "oxide": oxide, "orientation": None, "failed": True,
                    "error": f"{e}"[:2000], "e_ads_eV": None, "dissociated": None,
                    "converged": None, "nsteps": None,
                }]
            if not rows:
                continue
            with results_path.open("a") as f:
                for row in rows:
                    f.write(json.dumps(row) + "\n")
    if not dump_frame:
        print(f"\nwrote {results_path}")
    if pair_failures:
        print(f"\n{len(pair_failures)} pair(s) failed:")
        for p in pair_failures:
            print(f"  {p}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("outdir", type=Path)
    ap.add_argument("--oxides", nargs="+", default=list(OXIDES), choices=list(OXIDES))
    ap.add_argument("--models", nargs="+", default=MODELS, choices=MODELS)
    ap.add_argument(
        "--dump-frame", action="store_true",
        help="write the oriented, unrelaxed starting structure and stop -- no adsorbate "
             "relaxation, for a visual check (e.g. in OVITO) before committing to a run.",
    )
    a = ap.parse_args()
    main(a.outdir, a.oxides, a.models, a.dump_frame)
