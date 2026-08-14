"""Records — the on-disk currency of the pipeline.

Everything expensive is written to disk the moment it exists, so a run that dies part-way
still leaves usable results. Two kinds of output live here:

- ``DivergenceRecord``: one long-format row of the divergence table (candidate vs
  reference at one stage). Appended to a JSONL table.
- **The run tree**: the browsable per-relaxation folders written by ``write_relaxation``,
  the aggregate ``header.json`` rollups, and the per-stage ``rankings.csv``.

Structures are pymatgen ``Structure`` objects. POSCAR is the shared geometry format that
model-env workers (ASE-only, no pymatgen) also read and write, so it is the handoff format
across the subprocess-isolation boundary.
"""

from __future__ import annotations

import csv
import json
import shutil
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Optional

from pymatgen.core import Structure


@dataclass
class DivergenceRecord:
    """One long-format row of the divergence table.

    The displacement triple (``mean_displacement``, ``rmsd``, ``max_displacement``)
    comes from a single per-atom displacement vector computed after StructureMatcher
    alignment under PBC: ``rmsd ≈ mean`` ⇒ uniform drift; ``rmsd ≫ mean`` ⇒ localized
    failure; ``max_disp_atom`` / ``max_disp_species`` name the culprit.

    ``active_site_dBO`` and ``symmetry_match`` are the optional chemistry-aware layer,
    metadata-gated (left ``None`` when not applicable).
    """

    composition: str
    stage: str
    model: str
    polymorph: str = ""
    facet: str = ""
    termination: str = ""
    geometry_source: str = ""
    protocol: str = ""  # seeded | full_pipeline
    start_fmax_at_ref_geom: Optional[float] = None
    mean_displacement: Optional[float] = None
    rmsd: Optional[float] = None
    max_displacement: Optional[float] = None
    max_disp_atom: Optional[int] = None  # index of worst atom
    max_disp_species: Optional[str] = None  # element of worst atom (its identity)
    energy_error: Optional[float] = None  # candidate energy at its own min vs reference
    active_site_dBO: Optional[float] = None
    symmetry_match: Optional[bool] = None
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @classmethod
    def from_dict(cls, d: dict) -> "DivergenceRecord":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


def append_divergence(path: str | Path, record: DivergenceRecord) -> Path:
    """Append one divergence row to a long-format JSONL table (written immediately)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(record.to_dict()) + "\n")
    return path


def read_divergence_table(path: str | Path) -> list[DivergenceRecord]:
    """Read a JSONL divergence table back into records."""
    lines = Path(path).read_text().splitlines()
    return [DivergenceRecord.from_dict(json.loads(ln)) for ln in lines if ln.strip()]


# --- Hierarchical run tree (browsable, VESTA-friendly layout; design: reorganize runs/) ---
#
# One partitioned tree replaces the old flat ``structures/`` dump. Shared dimensions
# (model, protocol, stage, site) are encoded in the *path*, so lower levels never repeat
# them. Aggregate directories carry a ``header.json`` (timing rollups); each leaf
# relaxation folder carries a lightweight text ``OUTCAR`` (summary block + per-step
# energies) plus ``POSCAR`` (initial) / ``CONTCAR`` (final) / ``trajectory.xyz``.

HEADER_NAME = "header.json"
RANKING_NAME = "rankings.csv"
OUTCAR_NAME = "OUTCAR"
POSCAR_NAME = "POSCAR"
CONTCAR_NAME = "CONTCAR"
TRAJECTORY_NAME = "trajectory.xyz"
# Machine-readable twin of the OUTCAR summary block, and the resume marker. OUTCAR is for
# reading; this is for re-entering a run that was killed part-way (see ``read_relaxation``).
RELAX_NAME = "relax.json"

# Stages with exactly one relaxation → their files live directly in the stage dir, with no
# per-site subfolder. ``bare_substrate`` and ``pristine_slab`` are the E_ads and E_vac reference
# states that the seeded protocol has to relax for itself (full_pipeline gets both free
# from its own chain); run_stage.py relaxes them too when it enters mid-chain from a file.
_SINGLE_RELAX_STAGES = ("bulk", "slab", "bare_substrate", "pristine_slab")


def _sanitize(text: str) -> str:
    """Filesystem-safe token: strip path separators and whitespace."""
    return "".join(c for c in str(text).replace("/", "").replace("\\", "") if not c.isspace())


def stage_dir(run_dir: str | Path, model: str, protocol: str, stage: str) -> Path:
    """Directory for one stage of one model+protocol chain (the path rule).

    - ``protocol == "reference"`` → ``<run>/<model>/<stage>`` (no protocol level)
    - ``stage == "bulk"``          → ``<run>/<model>/bulk`` (candidate's shared warm start)
    - otherwise                    → ``<run>/<model>/<protocol>/<stage>``
    """
    base = Path(run_dir) / model
    if protocol == "reference":
        return base / stage
    if stage == "bulk":
        return base / "bulk"
    return base / protocol / stage


def relax_subfolder_name(stage: str, site_id: Optional[dict]) -> str:
    """Leaf subfolder name for a candidate, or ``""`` for single-relaxation stages.

    - vacancy   → ``site<site_index>_<sanitized symmetry_class>``
    - adsorbate → ``site<site_index>_<position type>`` (densified sampling places several
      sites of the same type — e.g. multiple distinct ``ontop`` — so the type alone is no
      longer unique; the site index disambiguates them)
    - bulk / slab / bare_substrate / pristine_slab → ``""`` (files in the stage dir)
    """
    if stage in _SINGLE_RELAX_STAGES or not site_id:
        return ""
    sym = site_id.get("symmetry_class", "")
    if stage in ("vacancy", "adsorbate"):
        return f"site{site_id.get('site_index')}_{_sanitize(sym)}"
    return _sanitize(sym)


def leaf_dir(stage_directory: str | Path, stage: str, site_id: Optional[dict]) -> Path:
    """Leaf relaxation folder inside ``stage_directory`` (adds a per-candidate subfolder)."""
    sub = relax_subfolder_name(stage, site_id)
    stage_directory = Path(stage_directory)
    return stage_directory / sub if sub else stage_directory


# Columns of the per-stage ``rankings.csv``, in order. ``folder`` is the leaf subfolder
# name, so a row points straight at the geometry that produced it.
RANKING_COLUMNS = (
    "rank",
    "folder",
    "site_index",
    "symmetry_class",
    "site_label",
    "energy_eV",
    "dE_from_min_eV",
    "e_ads_eV",
    "e_vac_eV",
    "converged",
    "nsteps",
    "flags",
)


def write_ranking(directory: str | Path, rows: list[dict]) -> Path:
    """Write a funnel stage's site ranking as CSV, most stable (lowest energy) first.

    The stage directory already holds one leaf folder per candidate, but nothing that
    *orders* them — you had to reconstruct the ranking from the run-root
    ``candidates.jsonl``. This puts the answer next to the geometries it ranks.

    ``dE_from_min_eV`` is the column to read before believing any site selection: when the
    gap to rank 2 is smaller than the model's own accuracy (tens of meV), the sites are
    degenerate within error and the ordering is noise, not a result.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / RANKING_NAME
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RANKING_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return path


def write_header(directory: str | Path, header: dict) -> Path:
    """Write an aggregate-level ``header.json`` (run/model/protocol/stage scope)."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / HEADER_NAME
    path.write_text(json.dumps(header, indent=2))
    return path


# Header keys rendered (in order) into the OUTCAR summary block, when present.
_OUTCAR_LINES = (
    ("model", "model"),
    ("stage", "stage"),
    ("protocol", "protocol"),
    ("facet", "facet"),
    ("composition", "composition"),
    ("site", "site"),
    ("site_label", "chemical_site"),  # coordination label, e.g. Ti5c / O2c
    ("canonical", "canonical"),
    ("optimizer", "optimizer"),
    ("fmax_target", "fmax_target (eV/A)"),
    ("converged", "converged"),
    ("nsteps", "nsteps"),
    ("n_frames", "n_frames"),
    ("elapsed_s", "elapsed_s"),
    ("start_fmax", "start_fmax (eV/A)"),
    ("energy", "final_energy (eV)"),
    ("e_ads", "E_ads (eV)"),  # adsorbate stage only; negative = bound
    ("e_vac", "E_vac (eV)"),  # vacancy stage only; lower = easier to form
    ("fmax", "final_fmax (eV/A)"),
    ("adsorbate_max_disp", "adsorbate_max_disp (A)"),
    ("flags", "flags"),
)


def format_outcar(header: dict, opt_log: str = "") -> str:
    """Compose the lightweight text ``OUTCAR``: summary block + per-step energy table.

    Positions are intentionally omitted (they live in POSCAR/CONTCAR/trajectory.xyz),
    so this stays tiny versus a real VASP OUTCAR. ``opt_log`` is ASE's optimizer log
    (per-step ``step / time / energy / fmax``); an empty log yields the block only.
    """
    out = ["# lightweight OUTCAR (energies only; geometry in POSCAR/CONTCAR/trajectory.xyz)"]
    for key, label in _OUTCAR_LINES:
        if key in header and header[key] is not None:
            value = header[key]
            if isinstance(value, (list, tuple)):
                if not value:
                    continue  # empty flag list → nothing to report
                value = "; ".join(str(v) for v in value)
            out.append(f"{label}: {value}")
    if opt_log.strip():
        out.append("#")
        out.append("# per-step (from ASE optimizer log; fmax is optimizer-generalized for cell relaxations):")
        out.append(opt_log.rstrip("\n"))
    return "\n".join(out) + "\n"


def write_relaxation(
    dest: str | Path,
    *,
    initial: Structure,
    final: Structure,
    trajectory_src: Optional[str | Path],
    header: dict,
    opt_log: str = "",
) -> Path:
    """Write one leaf relaxation folder: OUTCAR + POSCAR + CONTCAR + trajectory.xyz.

    - ``POSCAR``  = the (unrelaxed) stage input; ``CONTCAR`` = the relaxed output (priority).
    - ``trajectory.xyz`` copied from ``trajectory_src`` when present (graceful skip if the
      worker produced none — e.g. an ASE-less backend).
    - ``OUTCAR`` composed from ``header`` + ``opt_log`` (per-step energies).
    - ``relax.json`` = the same header as data, written LAST so its presence means every
      other file in the leaf is already complete. A job killed mid-write leaves a leaf
      without it, and ``read_relaxation`` then declines to resume that leaf.

    Returns the leaf directory.
    """
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    initial.to(filename=str(dest / POSCAR_NAME), fmt="poscar")
    final.to(filename=str(dest / CONTCAR_NAME), fmt="poscar")
    if trajectory_src is not None and Path(trajectory_src).exists():
        shutil.copyfile(trajectory_src, dest / TRAJECTORY_NAME)
    (dest / OUTCAR_NAME).write_text(format_outcar(header, opt_log))
    (dest / RELAX_NAME).write_text(json.dumps({"header": header, "opt_log": opt_log}, indent=2))
    return dest


def read_relaxation(dest: str | Path) -> Optional[dict]:
    """Read back a completed leaf, or ``None`` if it is absent/partial/unreadable.

    The counterpart to ``write_relaxation``: returns ``{"header", "opt_log", "initial",
    "final"}``, where ``initial`` is the stage input the leaf was produced from and
    ``final`` the relaxed geometry. Callers use ``initial`` to decide whether the cached
    result answers the question they are about to ask (see ``pipeline._resume``).

    Anything unexpected — missing marker, truncated JSON, unparseable POSCAR — returns
    ``None`` rather than raising, because the only cost of declining to resume is redoing
    a relaxation, while resuming something malformed would corrupt the run silently.
    """
    dest = Path(dest)
    try:
        payload = json.loads((dest / RELAX_NAME).read_text())
        return {
            "header": payload["header"],
            "opt_log": payload.get("opt_log", ""),
            "initial": Structure.from_file(dest / POSCAR_NAME),
            "final": Structure.from_file(dest / CONTCAR_NAME),
        }
    except (OSError, ValueError, KeyError):
        return None
