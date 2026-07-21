"""Step 2 verification: records round-trip through disk (JSON + POSCAR)."""

from __future__ import annotations

from pymatgen.core import Lattice, Structure

from oxide_workflow.records import (
    DivergenceRecord,
    StructureRecord,
    append_divergence,
    read_divergence_table,
)


def rutile_tio2() -> Structure:
    """Canonical rutile TiO2 (P4_2/mnm) — the prototype material, material-agnostic code."""
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


def test_structure_record_roundtrip(tmp_path):
    struct = rutile_tio2()
    rec = StructureRecord(
        structure=struct,
        stage="bulk",
        model="MACE-OMAT24",
        polymorph="mp-2657",
        geometry_source="db",
        protocol="reference",
        energy=-123.456,
        fmax=0.01,
    )
    assert rec.composition == "TiO2"  # auto-filled in __post_init__

    json_path = rec.save(tmp_path)
    assert json_path.exists()
    assert (tmp_path / f"{rec.record_id}.vasp").exists()

    loaded = StructureRecord.load(json_path)
    assert loaded.composition == "TiO2"
    assert loaded.stage == "bulk"
    assert loaded.model == "MACE-OMAT24"
    assert loaded.polymorph == "mp-2657"
    assert loaded.protocol == "reference"
    assert loaded.energy == -123.456
    assert loaded.structure.composition.reduced_formula == "TiO2"
    assert len(loaded.structure) == len(struct)


def test_divergence_table_append_and_read(tmp_path):
    table = tmp_path / "divergence.jsonl"
    r1 = DivergenceRecord(
        composition="TiO2",
        stage="slab",
        model="UMA",
        protocol="seeded",
        rmsd=0.12,
        mean_displacement=0.10,
        max_displacement=0.30,
        max_disp_atom=4,
        max_disp_species="O",
        energy_error=0.05,
    )
    r2 = DivergenceRecord(
        composition="TiO2", stage="vacancy", model="UMA", protocol="full_pipeline", rmsd=0.4
    )
    append_divergence(table, r1)
    append_divergence(table, r2)

    rows = read_divergence_table(table)
    assert len(rows) == 2
    assert rows[0].stage == "slab" and rows[0].max_disp_species == "O"
    assert rows[1].protocol == "full_pipeline"
    # Optional chemistry-aware fields stay gated off by default.
    assert rows[0].active_site_dBO is None and rows[0].symmetry_match is None
