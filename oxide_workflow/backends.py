"""
Backends — the one interface every model sits behind

``relax(structure, backend) -> RelaxResult`` (relaxed structure, energy, trajectory
metadata). Models live in mutually incompatible conda envs, so the orchestrator never
imports a model: it writes a POSCAR + job spec to disk, launches the model env's
interpreter on ``worker_relax.py``, and reads the result back from disk. Will be used to
oass RelaxResult to records.py to be saved.
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

# Sockeye's job scheduler sets these for its own conda/checkpoint locations; default to the local Mac layout otherwise.
CONDA_BASE = os.environ.get("OXW_CONDA_BASE", "/opt/anaconda3")
MODEL_DIR = Path(os.environ.get("OXW_MODEL_DIR", Path.home() / "Desktop/mace_test/models"))

# Sockeye's job script sets this to "cuda" when SLURM granted a GPU; defaults to "cpu" for local Mac runs.
DEVICE = os.environ.get("OXW_DEVICE", "cpu")

# A model behind the relax interface, plus how to launch its isolated worker.
@dataclass
class Backend:
    name: str
    env: str  # conda env name the worker runs in
    loader: str  # worker dispatch key: "mace" | "fairchem" | ...
    model_path: str
    device: str = DEVICE  # module-level default; see OXW_DEVICE above
    dtype: str = "float64" # change to float32 if simulation time is too long
    task: str = "omat"  # fairchem/UMA task head; ignored by other loaders
    head: Optional[str] = None  # MACE multi-head selector (e.g. "omat_pbe"); None = model default
   
    # Capability declaration
    can_relax: bool = True
    training_labels: tuple[str, ...] = ()
    python: Optional[str] = None  # override interpreter path; else derived from env

    # Returns which python do I launch to run this model
    def interpreter(self) -> str:
        return self.python or f"{CONDA_BASE}/envs/{self.env}/bin/python"


@dataclass
class RelaxResult:
    structure: Structure
    energy: float
    fmax: float  # final max atomic force (eV/Å)
    start_fmax: float  # force at the stage input before relaxing
    nsteps: int
    converged: bool
    meta: dict = field(default_factory=dict)  # trajectory metadata, each new relax gets empty dictionary

# Relaxes structure in backend's isolated conda env via worker_relax.py, returning the
# result read back from disk. fmax/max_steps/optimizer are required so every caller states
# its own convergence target, instead of silently inheriting a RelaxConfig or Backend default.
def relax(
    structure: Structure,
    backend: Backend,
    *,
    fmax: float,
    max_steps: int,
    optimizer: str,
    workdir: str | Path | None = None,
    relax_cell: bool = False, # True relaxes both lattice and atomic positions, should only be true for bulk relaxation
    desorb_check_n_ads: int | None = None,  # last N atoms are the mobile adsorbate
    desorb_check_step: int | None = None,  # step at which to test for net outward drift
    desorb_trend_window: int = 20,  # compare against this many steps earlier, not vs. start
    extend_if_approaching: bool = False,  # give max_steps one extension if still closing in
    extend_steps: int = 100,
    max_extensions: int = 1,  # repeatable up to this many rounds, each requiring progress
) -> RelaxResult:

    if not backend.can_relax:
        raise ValueError(f"backend {backend.name!r} declares can_relax=False")

    work = Path(workdir) if workdir else Path(tempfile.mkdtemp(prefix="relax_"))
    work.mkdir(parents=True, exist_ok=True)

    in_poscar, out_poscar, result_json = "input.vasp", "relaxed.vasp", "result.json"
    structure.to(filename=str(work / in_poscar), fmt="poscar")

    # Job order - everything the worker needs to run the relaxation
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
        "fmax": fmax,
        "max_steps": max_steps,
        "optimizer": optimizer,
        "relax_cell": relax_cell,
        "trajectory_xyz": "trajectory.xyz",
        "desorb_check_n_ads": desorb_check_n_ads,
        "desorb_check_step": desorb_check_step,
        "desorb_trend_window": desorb_trend_window,
        "extend_if_approaching": extend_if_approaching,
        "extend_steps": extend_steps,
        "max_extensions": max_extensions,
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
            # journey artifacts surfaced for the hierarchical writer: OUTCAR + xyz
            "trajectory": str(traj_path) if traj_path.exists() else None,
            "n_frames": result.get("n_frames"),
            "opt_log": opt_log.read_text() if opt_log.exists() else "",
            "early_stopped_desorbing": result.get("early_stopped_desorbing", False),
            "extended": result.get("extended", False),
            "extensions_used": result.get("extensions_used", 0),
        },
    )


# --- Registry: prototype backends (models cached locally) ----------------------------
#
# MACE and UMA are two multi-output checkpoints, each exposing several heads/tasks. Every
# head/task is registered as its own Backend (same file, different selector) so any can be
# picked as reference or candidate and lands in its own tree folder. CHGNet and Orb are
# single-checkpoint models with no head/task selector — model_path doubles as a release
# tag/function name instead of a literal file path (see _chgnet/_orb docstrings).
#
#   MACE    mace-mh-1.model   → heads via mace_mp(head=...)        (env: mace-clean)
#   UMA     uma-s-1p2.pt      → tasks via task_name=...            (env: fairchem)
#   CHGNet  bundled in pip package, selected by release tag        (env: chgnet)
#   Orb     fetched+cached on first use, selected by ckpt function (env: orb)
#
# The reference is mace-mh-1's OMat24 (PBE) head — the updated OMat24 that benchmarked
# more accurately than the standalone cached MACE-OMAT24 it replaces.

MACE_MH1 = str(MODEL_DIR / "mace-mh-1.model")
UMA_CKPT = str(MODEL_DIR / "uma-s-1p2.pt")
# Gated on HuggingFace (facebook/UMA, requires accepting the FAIR Chemistry License) --
# not auto-fetchable. Download uma-m-1p1.pt by hand and place it in MODEL_DIR; nothing
# else needed, _uma_m() below points at it the same way _uma() points at UMA_CKPT.
# Heads-up: facebookresearch/fairchem#2095 reports uma-m-1p1 predictor construction
# stalling indefinitely on some setups (uma-s loads fine) -- unresolved as of this
# writing, so a hang on first load is a known issue, not necessarily a config mistake.
UMA_M_CKPT = str(MODEL_DIR / "uma-m-1p1.pt")


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


def _uma_m(name: str, task: str, labels: tuple[str, ...]) -> Backend:
    return Backend(
        name=name, env="fairchem", loader="fairchem", model_path=UMA_M_CKPT,
        task=task, training_labels=labels,
    )


def _chgnet(name: str, model_name: str, labels: tuple[str, ...]) -> Backend:
    # model_path is the CHGNet release tag, not a file path — weights ship inside
    # the pip package, so there is nothing in MODEL_DIR to point at.
    return Backend(
        name=name, env="chgnet", loader="chgnet", model_path=model_name,
        training_labels=labels,
    )


def _orb(name: str, checkpoint: str, labels: tuple[str, ...]) -> Backend:
    # model_path is the pretrained-checkpoint function name in orb_models.forcefield.pretrained
    # (e.g. "orb_v2"), not a file path — weights are fetched once and cached by cached_path.
    return Backend(
        name=name, env="orb", loader="orb", model_path=checkpoint,
        training_labels=labels,
    )


SEVENNET_OMNI_CKPT = str(MODEL_DIR / "sevennet-omni.pth")


def _sevennet(name: str, modal: str, labels: tuple[str, ...]) -> Backend:
    # Like MACE/UMA, SevenNet-Omni is one multi-modality checkpoint; ``task`` doubles as
    # the modal selector here (same field fairchem's task_name reuses).
    return Backend(
        name=name, env="sevenn", loader="sevenn", model_path=SEVENNET_OMNI_CKPT,
        task=modal, training_labels=labels,
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
    # ---- UMA-M (uma-m-1p1, gated download -- see UMA_M_CKPT above) ----------------------
    "UMA-M-oc22": _uma_m("UMA-M-oc22", "oc22", ("OC22", "oxide-cat")),
    "UMA-M-omat": _uma_m("UMA-M-omat", "omat", ("OMat24", "bulk-mat")),
    # ---- CHGNet (env: chgnet) ------------------------------------------------------------
    "CHGNet-0.3.0": _chgnet("CHGNet-0.3.0", "0.3.0", ("MPtrj", "PBE/PBE+U")),
    # ---- Orb (env: orb) -------------------------------------------------------------------
    "Orb-v2": _orb("Orb-v2", "orb_v2", ("MPtrj+Alexandria", "PBE")),
    # ---- SevenNet-Omni (env: sevenn) -- one checkpoint, modal = training-dataset selector,
    # same pattern as MACE-mh1's heads / UMA's tasks. Modalities available in this
    # checkpoint: omat24, mpa, omol25_low, omol25_high, matpes_pbe, matpes_r2scan,
    # mp_r2scan, oc20, oc22, spice, qcml, odac23, pet_mad -- only the ones matching our
    # existing MACE/UMA comparison axes are registered below; add more the same way.
    "SevenNet-omni-oc22": _sevennet("SevenNet-omni-oc22", "oc22", ("OC22", "oxide-cat")),
    "SevenNet-omni-oc20": _sevennet("SevenNet-omni-oc20", "oc20", ("OC20", "cat")),
    "SevenNet-omni-omat24": _sevennet("SevenNet-omni-omat24", "omat24", ("OMat24", "bulk-mat")),
    "SevenNet-omni-mpa": _sevennet("SevenNet-omni-mpa", "mpa", ("MPtrj+Alexandria", "PBE")),
}

# Convenience groupings for candidate sweeps (every head/task/checkpoint, minus the reference).
MACE_HEADS = tuple(n for n in REGISTRY if n.startswith("MACE-mh1-"))
UMA_TASKS = tuple(n for n in REGISTRY if n.startswith("UMA-"))
CHGNET_MODELS = tuple(n for n in REGISTRY if n.startswith("CHGNet-"))
ORB_MODELS = tuple(n for n in REGISTRY if n.startswith("Orb-"))
ALL_CANDIDATES = tuple(
    n for n in (*MACE_HEADS, *UMA_TASKS, *CHGNET_MODELS, *ORB_MODELS)
    if n != "MACE-mh1-omat"
)


def get_backend(name: str) -> Backend:
    return REGISTRY[name]
