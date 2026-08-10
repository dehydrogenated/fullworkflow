"""
Backends — the one interface every model sits behind

``relax(structure, backend) -> RelaxResult`` (relaxed structure, energy, trajectory
metadata). Models live in mutually incompatible conda envs, so the orchestrator never
imports a model: it writes a POSCAR + job spec to disk, launches the model env's
interpreter on ``worker_relax.py``, and reads the result back from disk. This
subprocess isolation *is* the backend abstraction in physical form.

Backends carry a capability declaration (``can_relax``, ``is_async``,
``training_labels``); DFT will enter later as an async backend behind this same
interface (deferred, §8).
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from pymatgen.core import Structure

WORKER = Path(__file__).parent / "worker_relax.py"

# The two machine-specific paths. Defaults are the local Mac layout, so nothing changes
# here; set the env vars on a cluster (Sockeye) where conda and the checkpoints live
# elsewhere. Env vars rather than a config field because they must be set *before* the
# module is imported, and because a scheduler passes them into a job for free.
CONDA_BASE = os.environ.get("OXW_CONDA_BASE", "/opt/anaconda3")
MODEL_DIR = Path(os.environ.get("OXW_MODEL_DIR", Path.home() / "Desktop/mace_test/models"))


@dataclass
class Backend:
    """A model behind the relax interface, plus how to launch its isolated worker."""

    name: str
    env: str  # conda env name the worker runs in
    loader: str  # worker dispatch key: "mace" | "fairchem" | ...
    model_path: str
    device: str = "cpu"
    dtype: str = "float64"
    task: str = "omat"  # fairchem/UMA task head; ignored by other loaders
    head: Optional[str] = None  # MACE multi-head selector (e.g. "omat_pbe"); None = model default
    fmax: float = 0.03  # eV/Å convergence
    max_steps: int = 500
    optimizer: str = "FIRE"
    # capability declaration (design §6)
    can_relax: bool = True
    is_async: bool = False
    training_labels: tuple[str, ...] = ()
    python: Optional[str] = None  # override interpreter path; else derived from env

    def interpreter(self) -> str:
        return self.python or f"{CONDA_BASE}/envs/{self.env}/bin/python"


@dataclass
class RelaxResult:
    structure: Structure
    energy: float
    fmax: float  # final max atomic force (eV/Å)
    start_fmax: float  # force at the stage input before relaxing (design §5)
    nsteps: int
    converged: bool
    meta: dict = field(default_factory=dict)  # trajectory metadata


def relax(
    structure: Structure,
    backend: Backend,
    workdir: str | Path | None = None,
    relax_cell: bool = False,
    **overrides,
) -> RelaxResult:
    """Relax ``structure`` with ``backend`` in its isolated env; read result from disk.

    ``relax_cell=True`` for bulk (cell + positions); ``False`` for slabs/defects/
    adsorbates where the cell is pinned. ``overrides`` may set ``fmax``, ``max_steps``,
    ``optimizer`` per call.
    """
    if not backend.can_relax:
        raise ValueError(f"backend {backend.name!r} declares can_relax=False")

    work = Path(workdir) if workdir else Path(tempfile.mkdtemp(prefix="relax_"))
    work.mkdir(parents=True, exist_ok=True)

    in_poscar, out_poscar, result_json = "input.vasp", "relaxed.vasp", "result.json"
    structure.to(filename=str(work / in_poscar), fmt="poscar")

    spec = {
        "input_poscar": in_poscar,
        "output_poscar": out_poscar,
        "result_json": result_json,
        "loader": backend.loader,
        "model_path": backend.model_path,
        "device": backend.device,
        "dtype": backend.dtype,
        "task": backend.task,
        "head": backend.head,
        "fmax": overrides.get("fmax", backend.fmax),
        "max_steps": overrides.get("max_steps", backend.max_steps),
        "optimizer": overrides.get("optimizer", backend.optimizer),
        "relax_cell": relax_cell,
        "trajectory_xyz": "trajectory.xyz",
    }
    jobfile = work / "job.json"
    jobfile.write_text(json.dumps(spec, indent=2))

    proc = subprocess.run(
        [backend.interpreter(), str(WORKER), str(jobfile)],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"relax worker failed (backend={backend.name}, env={backend.env}, "
            f"rc={proc.returncode}).\n--- stderr ---\n{proc.stderr[-4000:]}"
        )

    result = json.loads((work / result_json).read_text())
    relaxed = Structure.from_file(work / out_poscar)
    traj_path = work / result.get("trajectory", "trajectory.xyz")
    opt_log = work / "opt.log"
    return RelaxResult(
        structure=relaxed,
        energy=result["energy"],
        fmax=result["fmax"],
        start_fmax=result["start_fmax"],
        nsteps=result["nsteps"],
        converged=result["converged"],
        meta={
            "backend": backend.name,
            "env": backend.env,
            "loader": backend.loader,
            "elapsed_s": result["elapsed_s"],
            "relax_cell": relax_cell,
            "workdir": str(work),
            # journey artifacts surfaced for the hierarchical writer (design: OUTCAR + xyz)
            "trajectory": str(traj_path) if traj_path.exists() else None,
            "n_frames": result.get("n_frames"),
            "opt_log": opt_log.read_text() if opt_log.exists() else "",
        },
    )


# --- Registry: prototype backends (models cached locally; see design §7) -------------
#
# Two multi-output checkpoints, each exposing several heads/tasks. Every head/task is
# registered as its own Backend (same file, different selector) so any can be picked as
# reference or candidate and lands in its own tree folder.
#
#   MACE  mace-mh-1.model   → heads via mace_mp(head=...)   (env: mace-clean)
#   UMA   uma-s-1p2.pt      → tasks via task_name=...       (env: fairchem)
#
# The reference is mace-mh-1's OMat24 (PBE) head — the updated OMat24 that benchmarked
# more accurately than the standalone cached MACE-OMAT24 it replaces.

MACE_MH1 = str(MODEL_DIR / "mace-mh-1.model")
UMA_CKPT = str(MODEL_DIR / "uma-s-1p2.pt")


def _mace(name: str, head: str, labels: tuple[str, ...]) -> Backend:
    return Backend(
        name=name, env="mace-clean", loader="mace", model_path=MACE_MH1,
        head=head, training_labels=labels,
    )


def _uma(name: str, task: str, labels: tuple[str, ...]) -> Backend:
    return Backend(
        name=name, env="fairchem", loader="fairchem", model_path=UMA_CKPT,
        task=task, training_labels=labels,
    )


REGISTRY: dict[str, Backend] = {
    # ---- MACE mace-mh-1 heads (mace_test.py: the model's real heads) --------------------
    # Reference: the OMat24 PBE head (bulk-mat, mixed-Hamiltonian) — the updated OMat24.
    "MACE-mh1-omat": _mace("MACE-mh1-omat", "omat_pbe", ("OMat24", "PBE")),
    "MACE-mh1-mp": _mace("MACE-mh1-mp", "mp_pbe_refit_add", ("MPtrj", "PBE")),
    "MACE-mh1-oc20": _mace("MACE-mh1-oc20", "oc20_usemppbe", ("OC20", "PBE-surf")),
    "MACE-mh1-matpes": _mace("MACE-mh1-matpes", "matpes_r2scan", ("MatPES", "r2SCAN")),
    "MACE-mh1-spice": _mace("MACE-mh1-spice", "spice_wB97M", ("SPICE", "wB97M")),
    "MACE-mh1-omol": _mace("MACE-mh1-omol", "omol", ("OMol", "molec")),
    # ---- UMA uma-s-1p2 tasks (uma_test.py: FAIRChem task heads) -------------------------
    "UMA-oc22": _uma("UMA-oc22", "oc22", ("OC22", "oxide-cat")),   # prime oxide-cat candidate
    "UMA-oc20": _uma("UMA-oc20", "oc20", ("OC20", "cat")),
    "UMA-oc25": _uma("UMA-oc25", "oc25", ("OC25", "e-cat")),
    "UMA-omat": _uma("UMA-omat", "omat", ("OMat24", "bulk-mat")),
    "UMA-omol": _uma("UMA-omol", "omol", ("OMol", "molec")),
    "UMA-odac": _uma("UMA-odac", "odac", ("ODAC", "MOF")),
    "UMA-omc": _uma("UMA-omc", "omc", ("OMC", "mol-cryst")),
}

# Convenience groupings for candidate sweeps (every head/task, minus the reference).
MACE_HEADS = tuple(n for n in REGISTRY if n.startswith("MACE-mh1-"))
UMA_TASKS = tuple(n for n in REGISTRY if n.startswith("UMA-"))
ALL_CANDIDATES = tuple(n for n in (*MACE_HEADS, *UMA_TASKS) if n != "MACE-mh1-omat")


def get_backend(name: str) -> Backend:
    return REGISTRY[name]
