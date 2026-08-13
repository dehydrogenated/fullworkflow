"""Step 4 verification: diverge() on a structure vs a rattled copy of itself."""

from __future__ import annotations


from oxide_workflow.diverge import diverge
from oxide_workflow.records import DivergenceRecord, append_divergence, read_divergence_table
from oxide_workflow.structures import get_structure


def test_identical_structures_have_zero_divergence():
    ref = get_structure("mp-2657")
    rec = diverge(ref, ref.copy(), stage="bulk", model="self")
    assert rec.meta["matched"] is True
    assert rec.mean_displacement < 1e-6
    assert rec.rmsd < 1e-6
    assert rec.max_displacement < 1e-6


def test_localized_rattle_is_attributed_to_the_culprit_atom():
    ref = get_structure("mp-2657")
    cand = ref.copy()

    # Move exactly one O atom a clear amount; leave the rest in place -> localized.
    # Oblique displacement (not axis-aligned): rutile's P4_2/mnm symmetry maps this O
    # site onto another one under a pure x- or z-only shift, so StructureMatcher's
    # best-fit search can tie between "this atom moved" and "the cell got relabeled by
    # that symmetry op" and mislabel the culprit. A displacement with all three
    # components breaks the tie unambiguously — and is more representative of real
    # relaxation noise, which never aligns with a crystal symmetry axis anyway.
    o_indices = [i for i, s in enumerate(ref) if str(s.specie) == "O"]
    culprit = o_indices[0]
    cand.translate_sites(culprit, [0.15, 0.1, 0.05], frac_coords=False, to_unit_cell=False)

    rec = diverge(
        ref, cand, stage="bulk", model="candidate", e_ref=-53.0, e_cand=-52.4,
        start_fmax=0.11, protocol="seeded",
    )

    assert rec.meta["matched"] is True
    # Localized signature: rmsd noticeably exceeds the mean.
    assert rec.rmsd > rec.mean_displacement
    # The worst atom is the O we moved, correctly identified.
    assert rec.max_disp_species == "O"
    assert rec.max_disp_atom == culprit
    assert rec.max_displacement > 0.1
    # Ordering invariants of the displacement triple.
    assert rec.mean_displacement <= rec.rmsd <= rec.max_displacement + 1e-9
    # Energy error carried through with correct sign (cand - ref).
    assert abs(rec.energy_error - 0.6) < 1e-9
    assert rec.start_fmax_at_ref_geom == 0.11
    assert rec.protocol == "seeded"


def test_uniform_drift_signature():
    ref = get_structure("mp-2657")
    cand = ref.copy()
    # Rigidly shift every atom by the same vector -> uniform drift: rmsd ~= mean.
    cand.translate_sites(range(len(cand)), [0.05, 0.05, 0.05], frac_coords=False)
    rec = diverge(ref, cand, stage="bulk", model="candidate")
    assert rec.meta["matched"] is True
    # After alignment a pure rigid shift is largely removed, but crucially rmsd ~= mean.
    assert abs(rec.rmsd - rec.mean_displacement) < 1e-6


def test_divergence_record_serializes(tmp_path):
    ref = get_structure("mp-2657")
    cand = ref.copy()
    cand.translate_sites(2, [0.1, 0.0, 0.0], frac_coords=False)
    rec = diverge(ref, cand, stage="bulk", model="candidate")
    assert isinstance(rec, DivergenceRecord)
    table = tmp_path / "div.jsonl"
    append_divergence(table, rec)
    (back,) = read_divergence_table(table)
    assert back.max_disp_species == rec.max_disp_species
