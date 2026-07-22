"""Backends — the one interface every model sits behind (design §6).

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
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from pymatgen.core import Structure

WORKER = Path(__file__).parent / "worker_relax.py"
CONDA_BASE = "/opt/anaconda3"


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
    fmax: float = 0.05  # eV/Å convergence
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
        "fmax": overrides.get("fmax", backend.fmax),
        "max_steps": overrides.get("max_steps", backend.max_steps),
        "optimizer": overrides.get("optimizer", backend.optimizer),
        "relax_cell": relax_cell,
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
        },
    )


# --- Registry: prototype backends (models cached locally; see design §7) -------------

REGISTRY: dict[str, Backend] = {
    "MACE-OMAT24": Backend(
        name="MACE-OMAT24",
        env="mace-clean",
        loader="mace",
        model_path=str(Path.home() / ".cache/mace/maceomat0mediummodel"),
        training_labels=("OMat24",),
    ),
    "UMA-s": Backend(
        name="UMA-s",
        env="fairchem",
        loader="fairchem",
        model_path=str(Path.home() / "Desktop/mace_test/models/uma-s-1p2.pt"),
        task="omat",
        training_labels=("OMat24", "OC20", "OMol", "ODAC", "OMC"),
    ),
}


def get_backend(name: str) -> Backend:
    return REGISTRY[name]
