"""Records — the on-disk currency of the pipeline (design §2 step 2, §5).

Two dataclass records, each serialized to disk immediately:

- ``StructureRecord``: a structure at a stage boundary (first-class input) plus its
  provenance. Serialized as JSON metadata + a POSCAR of the geometry.
- ``DivergenceRecord``: one long-format row of the divergence table. The schema in
  design §5 is canonical. Serialized as JSON (appended to a JSONL table).

Structures are pymatgen ``Structure`` objects. POSCAR is the shared geometry format
that model-env workers (ASE-only, no pymatgen) also read and write, so it is the
handoff format across the subprocess-isolation boundary.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Optional

from pymatgen.core import Structure


def _plain_fields(obj: Any) -> dict:
    """Dataclass -> dict of its scalar/metadata fields, excluding ``structure``."""
    out = {}
    for f in fields(obj):
        if f.name == "structure":
            continue
        out[f.name] = getattr(obj, f.name)
    return out


@dataclass
class StructureRecord:
    """A structure at a stage boundary, with provenance (design §4, §5 head columns).

    ``geometry_source`` and ``protocol`` capture *how* this geometry came to be, which
    is what makes seeded vs full-pipeline attribution possible downstream:

    - ``geometry_source``: e.g. ``db`` (warm start), ``seeded`` (built from another
      model's relaxed previous stage), ``relaxed`` (this model's own relaxed output).
    - ``protocol``: ``reference`` | ``seeded`` | ``full_pipeline`` | ``basin`` | ``input``.

    ``site_id`` stores a modification's identity as symmetry class + fractional
    coordinate (design §4) — never file line numbers.
    """

    structure: Structure
    stage: str  # bulk | slab | vacancy | adsorbate | assembly
    model: str  # backend/model that produced this geometry ("" if unrelaxed input)
    composition: str = ""  # reduced formula; auto-filled from structure if blank
    polymorph: str = ""  # polymorph label / source mp-id (e.g. "rutile", "mp-2657")
    facet: str = ""  # e.g. "110"; "" for bulk
    termination: str = ""  # "" for bulk
    geometry_source: str = ""
    protocol: str = ""
    energy: Optional[float] = None  # total energy after relax (eV); None if unrelaxed
    fmax: Optional[float] = None  # max force at this geometry (eV/Å)
    site_id: Optional[dict] = None  # {symmetry_class, frac_coord} for the modification
    meta: dict = field(default_factory=dict)
    record_id: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.composition and self.structure is not None:
            self.composition = self.structure.composition.reduced_formula

    def default_id(self) -> str:
        parts = [self.composition, self.stage, self.model or "unrelaxed", self.protocol]
        return "_".join(p for p in parts if p) or "structure"

    def to_metadata(self) -> dict:
        return _plain_fields(self)

    def save(self, directory: str | Path) -> Path:
        """Write ``<record_id>.json`` (metadata) + ``<record_id>.vasp`` (POSCAR).

        Returns the path to the JSON file. Assigns ``record_id`` if not already set.
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        if self.record_id is None:
            self.record_id = self.default_id()
        poscar_path = directory / f"{self.record_id}.vasp"
        self.structure.to(filename=str(poscar_path), fmt="poscar")
        meta = self.to_metadata()
        meta["poscar"] = poscar_path.name
        json_path = directory / f"{self.record_id}.json"
        json_path.write_text(json.dumps(meta, indent=2))
        return json_path

    @classmethod
    def load(cls, json_path: str | Path) -> "StructureRecord":
        json_path = Path(json_path)
        meta = json.loads(json_path.read_text())
        poscar_name = meta.pop("poscar")
        structure = Structure.from_file(json_path.parent / poscar_name)
        return cls(structure=structure, **meta)


@dataclass
class DivergenceRecord:
    """One long-format row of the divergence table (design §5, canonical schema).

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
OUTCAR_NAME = "OUTCAR"
POSCAR_NAME = "POSCAR"
CONTCAR_NAME = "CONTCAR"
TRAJECTORY_NAME = "trajectory.xyz"

_SINGLE_RELAX_STAGES = ("bulk", "slab")  # one relaxation → files live directly in the stage dir


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
    - bulk/slab → ``""`` (files live directly in the stage dir)
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
    ("canonical", "canonical"),
    ("optimizer", "optimizer"),
    ("fmax_target", "fmax_target (eV/A)"),
    ("converged", "converged"),
    ("nsteps", "nsteps"),
    ("n_frames", "n_frames"),
    ("elapsed_s", "elapsed_s"),
    ("start_fmax", "start_fmax (eV/A)"),
    ("energy", "final_energy (eV)"),
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

    Returns the leaf directory.
    """
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    initial.to(filename=str(dest / POSCAR_NAME), fmt="poscar")
    final.to(filename=str(dest / CONTCAR_NAME), fmt="poscar")
    if trajectory_src is not None and Path(trajectory_src).exists():
        shutil.copyfile(trajectory_src, dest / TRAJECTORY_NAME)
    (dest / OUTCAR_NAME).write_text(format_outcar(header, opt_log))
    return dest
