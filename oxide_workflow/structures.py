"""
Specify which bulk to start with
Resolves a built-in alias or a saved identifier to a warm-start bulk cell
"""

from __future__ import annotations
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from pymatgen.core import Lattice, Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

# Saved cells live here and are meant to be committed. Two reasons: Sockeye compute
# nodes have no outbound network, and MP re-relaxes entries between database releases —
# a committed cell keeps a rerun of the same benchmark reproducible.
STRUCTURE_DIR = Path(__file__).resolve().parent.parent / "data" / "structures"


def rutile_tio2() -> Structure:
    """Canonical rutile TiO2 (space group P4_2/mnm), experimental lattice.

    Offline fallback prototype. Pass "mp-2657" for MP's own relaxed cell — in practice it
    agrees to ~0.1% in a and ~0.3% in volume, so the two warm starts are interchangeable.
    Re-relaxed by every backend anyway, so these constants are not load-bearing.
    """
    a, c, u = 4.5937, 2.9587, 0.3050
    lattice = Lattice.tetragonal(a, c)
    species = ["Ti", "Ti", "O", "O", "O", "O"]
    coords = [
        [0.0, 0.0, 0.0],
        [0.5, 0.5, 0.5],
        [u, u, 0.0],
        [-u, -u, 0.0],
        [0.5 + u, 0.5 - u, 0.5],
        [0.5 - u, 0.5 + u, 0.5],
    ]
    return Structure(lattice, species, coords)


# --- Material registry: identifier -> bulk Structure factory -----------------------------
# Built-in aliases only. Everything else is read from STRUCTURE_DIR, populated by
# scripts/fetch_structure.py.

STRUCTURE_REGISTRY: dict[str, Callable[[], Structure]] = {
    "rutile-tio2": rutile_tio2,  # hand-written experimental cell, no network needed
}


def register_structure(identifier: str, factory: Callable[[], Structure]) -> None:
    """Register a material identifier -> Structure factory, for callers adding batch members."""
    STRUCTURE_REGISTRY[identifier] = factory


def _describe(structure: Structure) -> dict:
    """Provenance fields we can derive from the cell itself."""
    return {
        "formula": structure.composition.reduced_formula,
        "n_sites": len(structure),
        "spacegroup": SpacegroupAnalyzer(structure).get_space_group_symbol(),
    }


def _require_ordered(structure: Structure, source: str) -> None:
    """Reject partial occupancies — MLIPs need one species per site.

    Real ICSD/COD CIFs are often disordered (e.g. Ti0.9Nb0.1). Failing when the material is
    added beats SlabGenerator or the backend failing three stages later.
    """
    if not structure.is_ordered:
        raise ValueError(
            f"{source} is disordered (partial site occupancies); the relaxation chain needs "
            "an ordered cell. Pick an ordered entry, or build a supercell approximant first."
        )


def _conventional(structure: Structure) -> Structure:
    """Conventional standard cell. Fix the coordinate system before cleaving.

    SlabConfig.miller_index is defined against the conventional setting, so a primitive
    input would cut a different facet than the one the run is labelled with.
    """
    return SpacegroupAnalyzer(structure).get_conventional_standard_structure()


def _structure_paths(identifier: str) -> tuple[Path, Path]:
    stem = identifier.replace("/", "_").replace("\\", "_")
    return STRUCTURE_DIR / f"{stem}.cif", STRUCTURE_DIR / f"{stem}.json"

# used to fetch and save MP-IDs
def add_material(structure: Structure, identifier: str, provenance: dict) -> Path:
    """Normalize a cell and save it as <identifier>.cif + .json in STRUCTURE_DIR.

    The only writer into STRUCTURE_DIR. Enforcing the guards here — rather than on each
    read — is what lets get_structure() trust the folder: everything in it is ordered and
    in the conventional setting by construction, so a saved cell and a freshly fetched one
    cut the same facet.
    """
    _require_ordered(structure, identifier) # check if there are partial occupancies
    structure = _conventional(structure) # Apply conventions for pymatgen

    cif_path, meta_path = _structure_paths(identifier)
    STRUCTURE_DIR.mkdir(parents=True, exist_ok=True)
    structure.to(filename=str(cif_path))
    meta_path.write_text(
        json.dumps(
            {
                "identifier": identifier,
                **provenance,
                "added_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                **_describe(structure),
            },
            indent=2,
        )
    )
    return cif_path


def available() -> list[str]:
    """Identifiers that resolve offline right now — aliases plus whatever has been saved."""
    saved = sorted(p.stem for p in STRUCTURE_DIR.glob("*.cif")) if STRUCTURE_DIR.is_dir() else []
    return sorted(set(STRUCTURE_REGISTRY) | set(saved))


def _resolve(identifier: str) -> tuple[Structure, dict]:
    """Identifier -> (structure, provenance). Never hits the network.

    Resolution order: built-in alias, then a path on disk, then a saved cell in
    ``STRUCTURE_DIR``. The path branch is what lets you point a run straight at a CIF or
    POSCAR you built yourself, without going through Materials Project at all.
    """
    if identifier in STRUCTURE_REGISTRY:
        structure = STRUCTURE_REGISTRY[identifier]()
        return structure, {"identifier": identifier, "source": "builtin", **_describe(structure)}

    # A path to a structure file: normalized through the same guards as a saved cell, so a
    # hand-supplied CIF cuts the same facet a fetched one would.
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

    cif_path, meta_path = _structure_paths(identifier)
    if not cif_path.exists():
        raise KeyError(
            f"unknown material {identifier!r}. Available: {available()}. To add one, run "
            f"`python scripts/fetch_structure.py {identifier}` on a networked machine "
            f"(needs MP_API_KEY) and commit the pair it writes to {STRUCTURE_DIR}."
        )

    structure = Structure.from_file(str(cif_path))
    provenance = (
        json.loads(meta_path.read_text())
        if meta_path.exists()
        else {"identifier": identifier, "source": "unrecorded", **_describe(structure)}
    )
    return structure, provenance


def get_structure(identifier: str) -> Structure:
    """Resolve a built-in alias or a saved identifier to a bulk cell. Gets rid of provenance dict."""
    return _resolve(identifier)[0]


def structure_provenance(identifier: str) -> dict:
    """What the identifier actually resolved to — for stamping into run records."""
    return _resolve(identifier)[1]
