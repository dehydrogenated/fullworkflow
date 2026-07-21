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
