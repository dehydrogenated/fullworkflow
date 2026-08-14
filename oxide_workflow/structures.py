"""Resolves a material identifier (mp-id) or a file path to a bulk Structure.

Saved materials are committed as data/structures/<id>_<formula>/<id>.cif+.json, so runs
work offline (e.g. on Sockeye).
"""

from __future__ import annotations
import json
from datetime import datetime, timezone
from pathlib import Path
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer


# Saved cells live here and are meant to be committed for offline computation
STRUCTURE_DIR = Path(__file__).resolve().parent.parent / "data" / "structures"


# Returns a dictionary with formula, n_sites, spacegroup
def _describe(structure: Structure) -> dict:
    # Provenance fields we can derive from the cell itself.
    return {
        "formula": structure.composition.reduced_formula,
        "n_sites": len(structure), # structures behaves like a list of sites, len gives no. atoms
        "spacegroup": SpacegroupAnalyzer(structure).get_space_group_symbol(), # ex. P4_2/MNM
    }


# Reject partial occupancies — MLIPs need one species per site.
def _require_ordered(structure: Structure, source: str) -> None:
    if not structure.is_ordered:
        raise ValueError(
            f"{source} is disordered (partial site occupancies); the relaxation chain needs "
            "an ordered cell. Pick an ordered entry, or build a supercell approximant first."
        )


# Converts cell into a standardized cell with a fixed coordinate system before cleaving for accurate indices
def _conventional(structure: Structure) -> Structure:
    return SpacegroupAnalyzer(structure).get_conventional_standard_structure()


# Where a saved material's folder is: STRUCTURE_DIR/<id>_<formula>/. Use glob to match id and folder.
def _structure_dir_for(identifier: str) -> Path | None:
    if not STRUCTURE_DIR.is_dir():
        return None
    matches = sorted(STRUCTURE_DIR.glob(f"{identifier}_*"))
    return matches[0] if matches else None


# Used to normalize a cell and save the ID cif and json in STRUCTURE_DIR/<id>_<formula>/.
def add_material(structure: Structure, identifier: str, provenance: dict) -> Path:
    _require_ordered(structure, identifier) # check if there are partial occupancies
    structure = _conventional(structure) # Apply conventions for pymatgen
    description = _describe(structure)
    folder = STRUCTURE_DIR / f"{identifier}_{description['formula']}" # Can now sort my formula
    cif_path, meta_path = folder / f"{identifier}.cif", folder / f"{identifier}.json"
    folder.mkdir(parents=True, exist_ok=True)
    structure.to(filename=str(cif_path))
    meta_path.write_text(
        json.dumps(
            {
                "identifier": identifier,
                **provenance,
                "added_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                **description,
            },
            indent=2,
        )
    )
    return cif_path


# Scans STRUCTURE_DIR for every saved material's CIF and returns their identifiers
def available() -> list[str]:
    return sorted(p.stem for p in STRUCTURE_DIR.glob("*/*.cif")) if STRUCTURE_DIR.is_dir() else []


# Shared lookup behind get_structure()/structure_provenance() for structure and provenance
def _resolve(identifier: str) -> tuple[Structure, dict]:

    # One-off: run directly on a file, nothing committed or reusable by name.
    path = Path(identifier)
    if path.suffix and path.is_file():
        structure = Structure.from_file(str(path))
        _require_ordered(structure, str(path))
        structure = _conventional(structure)
        return structure, {
            "identifier": identifier,
            "source": "file",
            "path": str(path.resolve()),
            **_describe(structure),
        }

    # Saved: whatever's already committed to STRUCTURE_DIR, fetched or hand-added.
    folder = _structure_dir_for(identifier)
    if folder is None:
        raise KeyError(
            f"unknown material {identifier!r}. Available: {available()}. To add one, run "
            f"`python scripts/fetch_structure.py {identifier}` on a networked machine "
            f"(needs MP_API_KEY) and commit the pair it writes to {STRUCTURE_DIR}."
        )
    cif_path, meta_path = folder / f"{identifier}.cif", folder / f"{identifier}.json"

    structure = Structure.from_file(str(cif_path))
    provenance = (
        json.loads(meta_path.read_text())
        if meta_path.exists()
        else {"identifier": identifier, "source": "unrecorded", **_describe(structure)}
    )
    return structure, provenance


# The structure-only half of _resolve() — what callers building or relaxing a cell want.
def get_structure(identifier: str) -> Structure:
    return _resolve(identifier)[0]


# The provenance-only half of _resolve() — for stamping into run records, not computation.
def structure_provenance(identifier: str) -> dict:
    return _resolve(identifier)[1]
