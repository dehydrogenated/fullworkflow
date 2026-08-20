#!/usr/bin/env python
"""Relaxation worker — runs *inside a model's conda env*.

Import-light on purpose: depends only on ASE + the model package. It must NOT import
pymatgen or ``oxide_workflow`` — those live in the orchestrator env. All I/O is on disk
(POSCAR in, POSCAR out, JSON result), which is the subprocess-isolation boundary made
physical.

Invocation:  ``python worker_relax.py <job.json>``
The job spec and all referenced files live in the same directory as ``job.json``.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from ase.io import read, write
from ase.optimize import BFGS, FIRE

OPTIMIZERS = {"FIRE": FIRE, "BFGS": BFGS}

# Pass in a Backend spec to build an ASE calculator
def build_calculator(spec: dict):
    loader = spec["loader"]
    if loader == "mace":
        from mace.calculators import mace_mp

        kwargs = dict(
            model=spec["model_path"],
            device=spec.get("device", "cpu"),
            default_dtype=spec.get("dtype", "float64"),
        )
        # Multi-head models (mace-mh-1) need the exact head string, or MACE silently
        # falls back to the last head and returns duplicate numbers.
        if spec.get("head"):
            kwargs["head"] = spec["head"]
        return mace_mp(**kwargs)
    if loader == "fairchem":
        from fairchem.core import FAIRChemCalculator, pretrained_mlip

        pu = pretrained_mlip.load_predict_unit(
            spec["model_path"], device=spec.get("device", "cpu")
        )
        return FAIRChemCalculator(pu, task_name=spec.get("task", "omat"))
    if loader == "chgnet":
        from chgnet.model import CHGNet
        from chgnet.model.dynamics import CHGNetCalculator

        # model_path doubles as the CHGNet release tag (e.g. "0.3.0"); weights ship
        # inside the package, no local checkpoint file or network needed.
        model = CHGNet.load(
            model_name=spec["model_path"], use_device=spec.get("device", "cpu"),
            verbose=False,
        )
        return CHGNetCalculator(model=model, use_device=spec.get("device", "cpu"))
    if loader == "orb":
        from orb_models.forcefield import pretrained
        from orb_models.forcefield.calculator import ORBCalculator

        # model_path doubles as the pretrained-checkpoint function name (e.g.
        # "orb_v2"); weights download once from Orbital Materials' S3 bucket and
        # are cached locally by cached_path after that -- unless checkpoint_path
        # points at an already-staged local file (needed offline, e.g. a Sockeye
        # compute node with no outbound network), in which case load from that
        # directly instead of ever touching the network.
        # compile=False: orb_v2()'s default lets torch.compile kick in, which JITs CPU
        # kernels via a C++ compiler (g++) -- fine on a dev machine with Xcode CLI tools,
        # but Sockeye compute nodes don't have one in PATH and die with InvalidCxxCompiler.
        # Disabling compile avoids needing a C++ toolchain at all rather than chasing down
        # whether/how to make one available on a compute node.
        kwargs = {"device": spec.get("device", "cpu"), "compile": False}
        ckpt = spec.get("checkpoint_path")
        if ckpt and Path(ckpt).exists():
            kwargs["weights_path"] = ckpt
        orbff = getattr(pretrained, spec["model_path"])(**kwargs)
        return ORBCalculator(orbff, device=spec.get("device", "cpu"))
    if loader == "sevenn":
        from sevenn.calculator import SevenNetCalculator

        # task doubles as the SevenNet "modal" selector (e.g. "oc22", "omat24") --
        # same multi-modality-checkpoint pattern as fairchem's task_name.
        return SevenNetCalculator(
            model=spec["model_path"], modal=spec.get("task", "mpa"),
            device=spec.get("device", "cpu"), enable_cueq=False, enable_flash=False,
        )
    if loader == "esen":
        from fairchem.core.common.relaxation.ase_utils import OCPCalculator

        # Pre-restructuring fairchem API (fairchem-core==1.10.0 in the "esen" env, not the
        # "fairchem" env's 2.x) -- the only one that reads this checkpoint's legacy format.
        # OCPCalculator is a plain ASE Calculator (populates self.results directly), so
        # nothing else about this worker's relax loop needs to know it's a different API.
        return OCPCalculator(
            checkpoint_path=spec["model_path"], cpu=(spec.get("device", "cpu") != "cuda"),
        )
    raise ValueError(f"unknown loader: {loader!r}")


def max_force(atoms) -> float:
    f = atoms.get_forces()
    return float(np.sqrt((f ** 2).sum(axis=1)).max())


def anchor_surface_distance(atoms, n_ads: int) -> float:
    """Distance from the adsorbate's ANCHOR atom (index -n_ads, its first/binding atom --
    see config.py's ADSORBATE_FRAGMENTS convention: "first entry is the binding atom") to
    its nearest surface neighbor. Adsorbate atoms are always appended last (see stages.py).

    Deliberately the anchor alone, not the whole fragment's minimum distance: tracking
    every adsorbate atom would also catch a legitimate reorientation (the far end of a
    molecule swinging toward a different surface atom while still bonded) as if it were
    desorption. A genuine float-off moves the anchor -- the actual bond -- away from the
    surface; a reorientation moves the OTHER atoms around a still-bonded anchor.
    """
    anchor_idx = len(atoms) - n_ads
    d = atoms.get_all_distances(mic=True)
    return float(d[anchor_idx, :anchor_idx].min())


def main(jobfile: str) -> None:
    workdir = Path(jobfile).parent
    spec = json.loads(Path(jobfile).read_text())

    atoms = read(workdir / spec["input_poscar"], format="vasp")
    atoms.calc = build_calculator(spec)

    start_fmax = max_force(atoms)  # force at the stage input, before relaxing

    fmax = float(spec["fmax"])
    max_steps = int(spec["max_steps"])
    optimizer = OPTIMIZERS[spec["optimizer"]]

    target = atoms
    if spec.get("relax_cell", False):
        from ase.filters import FrechetCellFilter

        target = FrechetCellFilter(atoms)

    # Extended-xyz trajectory of the *real* atoms (not the FrechetCellFilter target) so
    # each frame carries evolving positions and the cell (Lattice=) for VESTA/OVITO.
    traj = workdir / spec.get("trajectory_xyz", "trajectory.xyz")
    if traj.exists():
        traj.unlink()
    write(traj, atoms, format="extxyz", append=True)  # frame 0 = initial (calc already primed)

    # Optional early-stop: at desorb_check_step, compare the ANCHOR atom's CURRENT distance
    # to the surface against its distance desorb_trend_window steps earlier -- a RECENT
    # trend, not "vs. the very start". A molecule can legitimately drift outward for a while
    # while reorienting and then turn back in; comparing only against t=0 would still read
    # as "farther than start" and wrongly kill a trajectory that has already turned around.
    # Comparing against the recent window instead asks "is it moving away RIGHT NOW", which
    # is what actually distinguishes an in-progress reorientation from real desorption. Uses
    # anchor_surface_distance, not the whole fragment, so the far atoms swinging around
    # during that same reorientation don't contaminate the signal either (see its docstring).
    n_ads = spec.get("desorb_check_n_ads")
    check_step = spec.get("desorb_check_step")
    trend_window = int(spec.get("desorb_trend_window", 20))
    trend_ref_step = max(check_step - trend_window, 0) if check_step and n_ads else None
    d_trend_ref = None
    early_stopped_desorbing = False

    # The reverse case: max_steps runs out unconverged, but the adsorbate hasn't floated
    # off -- checked against the anchor's STARTING distance (fixed at t=0), not a rolling
    # per-window baseline: an adsorbate that already reached the surface can legitimately
    # stop monotonically closing the gap while still doing real work (e.g. an H shimmying
    # laterally toward a bridge O, distance from ITS anchor barely changing round to round)
    # -- a rolling window reads that as "stopped closing" and cuts the extension short even
    # though the trajectory is nowhere near desorbing. Comparing against the fixed start
    # instead asks "is it still on/near the surface it landed on", the same basis the
    # terminal desorption check below uses (end_dist > start_dist), so a trajectory only
    # loses its extension budget once it's actually trending toward that same failure mode.
    # Repeatable up to max_extensions rounds.
    extend_if_approaching = bool(spec.get("extend_if_approaching", False))
    extend_steps = int(spec.get("extend_steps", 100))
    max_extensions = int(spec.get("max_extensions", 1))
    d_start = anchor_surface_distance(atoms, n_ads) if extend_if_approaching and n_ads else None
    extensions_used = 0

    opt = optimizer(target, logfile=str(workdir / "opt.log"))
    opt.attach(lambda: write(traj, atoms, format="extxyz", append=True), interval=1)
    t0 = time.time()
    if n_ads and (check_step or extend_if_approaching):
        for _converged in opt.irun(fmax=fmax, steps=max_steps):
            if trend_ref_step is not None and opt.nsteps == trend_ref_step:
                d_trend_ref = anchor_surface_distance(atoms, n_ads)
            if check_step and opt.nsteps == check_step:
                d_now = anchor_surface_distance(atoms, n_ads)
                if d_trend_ref is not None and d_now > d_trend_ref:
                    early_stopped_desorbing = True
                    break

        while (
            not early_stopped_desorbing
            and extend_if_approaching
            and d_start is not None
            and max_force(atoms) > fmax
            and extensions_used < max_extensions
        ):
            d_now = anchor_surface_distance(atoms, n_ads)
            if not (d_now < d_start):  # at or past where it started -- treat as done extending
                break
            for _converged in opt.irun(fmax=fmax, steps=extend_steps):
                pass
            extensions_used += 1
    else:
        opt.run(fmax=fmax, steps=max_steps)
    elapsed = time.time() - t0
    extended = extensions_used > 0

    n_frames = sum(1 for line in traj.read_text().splitlines() if line.strip().isdigit())

    final_fmax = max_force(atoms)  # physical atomic forces, not filter-generalized
    result = {
        "energy": float(atoms.get_potential_energy()),
        "start_fmax": start_fmax,
        "fmax": final_fmax,
        "nsteps": int(opt.get_number_of_steps()),
        "converged": bool(final_fmax <= fmax),
        "elapsed_s": elapsed,
        "loader": spec["loader"],
        "model_path": spec.get("model_path"),
        "relax_cell": bool(spec.get("relax_cell", False)),
        "trajectory": traj.name,
        "n_frames": n_frames,
        "early_stopped_desorbing": early_stopped_desorbing,
        "extended": extended,
        "extensions_used": extensions_used,
    }
    write(workdir / spec["output_poscar"], atoms, format="vasp")
    (workdir / spec["result_json"]).write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main(sys.argv[1])
