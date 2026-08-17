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
        # are cached locally by cached_path after that.
        orbff = getattr(pretrained, spec["model_path"])(device=spec.get("device", "cpu"))
        return ORBCalculator(orbff, device=spec.get("device", "cpu"))
    raise ValueError(f"unknown loader: {loader!r}")


def max_force(atoms) -> float:
    f = atoms.get_forces()
    return float(np.sqrt((f ** 2).sum(axis=1)).max())


def min_adsorbate_surface_distance(atoms, n_ads: int) -> float:
    """Shortest distance between any of the last ``n_ads`` (adsorbate) atoms and any atom
    before them (the surface). Adsorbate atoms are always appended last (see stages.py)."""
    d = atoms.get_all_distances(mic=True)
    return float(d[-n_ads:, :-n_ads].min())


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

    # Optional early-stop: at desorb_check_step, compare the adsorbate's distance to the
    # surface against its starting distance. Net outward drift by then means the site was
    # never going to bind (see checks.py's post-hoc "adsorbate desorbed" flag, which this
    # complements by catching the same failure mode *during* relaxation instead of after
    # burning the full step budget on it).
    n_ads = spec.get("desorb_check_n_ads")
    check_step = spec.get("desorb_check_step")
    early_stopped_desorbing = False
    d_start = min_adsorbate_surface_distance(atoms, n_ads) if n_ads and check_step else None

    # The reverse case: max_steps runs out unconverged, but the adsorbate is still
    # net-approaching the surface (real work still happening, not oscillation) -- checked
    # by comparing distance at (max_steps - extend_steps) against distance at max_steps.
    extend_if_approaching = bool(spec.get("extend_if_approaching", False))
    extend_steps = int(spec.get("extend_steps", 100))
    extend_window_step = max(max_steps - extend_steps, 0) if extend_if_approaching and n_ads else None
    d_before_extend_window = None
    extended = False

    opt = optimizer(target, logfile=str(workdir / "opt.log"))
    opt.attach(lambda: write(traj, atoms, format="extxyz", append=True), interval=1)
    t0 = time.time()
    if n_ads and (check_step or extend_if_approaching):
        for _converged in opt.irun(fmax=fmax, steps=max_steps):
            if check_step and opt.nsteps == check_step:
                d_now = min_adsorbate_surface_distance(atoms, n_ads)
                if d_now > d_start:
                    early_stopped_desorbing = True
                    break
            if extend_window_step is not None and opt.nsteps == extend_window_step:
                d_before_extend_window = min_adsorbate_surface_distance(atoms, n_ads)

        if (
            not early_stopped_desorbing
            and d_before_extend_window is not None
            and max_force(atoms) > fmax
        ):
            d_now = min_adsorbate_surface_distance(atoms, n_ads)
            if d_now < d_before_extend_window:  # still net-closing the gap -- give it more
                extended = True
                for _converged in opt.irun(fmax=fmax, steps=extend_steps):
                    pass
    else:
        opt.run(fmax=fmax, steps=max_steps)
    elapsed = time.time() - t0

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
    }
    write(workdir / spec["output_poscar"], atoms, format="vasp")
    (workdir / spec["result_json"]).write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main(sys.argv[1])
