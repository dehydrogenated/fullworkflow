"""Divergence table round-trips through disk as JSONL."""

from __future__ import annotations

from oxide_workflow.records import (
    DivergenceRecord,
    append_divergence,
    read_divergence_table,
)


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
