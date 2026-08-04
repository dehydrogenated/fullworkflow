"""Fetch the rutile-structure oxide family into data/structures/ in one go.

    MP_API_KEY=... python scripts/fetch_rutiles.py            # the curated set
    MP_API_KEY=... python scripts/fetch_rutiles.py --discover # list the family, fetch nothing
    MP_API_KEY=... python scripts/fetch_rutiles.py --formulas TiO2 SnO2

Materials Project ids are opaque integers, so this resolves them by *querying* rather than
hardcoding: every entry is found by formula **plus spacegroup 136 (P4_2/mnm)**, which is
the rutile structure itself. A hardcoded id that drifted (or that someone mistyped) would
silently fetch the wrong material and there would be nothing in the output to catch it.

Run once on a networked machine and commit what it writes; the pipeline never needs the
network or the key again. Get a key from materialsproject.org (profile -> API) and set it
in your shell — do not paste it into this file, which is committed.
"""

from __future__ import annotations

import argparse
import os

from oxide_workflow.structures import add_material

RUTILE_SPACEGROUP = 136  # P4_2/mnm — the rutile structure type

# The well-behaved set: true rutiles, non-magnetic, no metal-insulator transition. Only the
# cation changes, so the family is a clean "chemistry only" axis for a sweep.
#
# Several are NOT the ground-state polymorph of their composition (stishovite is the
# high-pressure SiO2; rutile GeO2 competes with a quartz-like form), which is why nothing
# here filters on stability — the query pins the *structure* and takes the best entry that
# has it.
WELL_BEHAVED = {
    "TiO2": "rutile — the reference case, and what every knob here was tuned on",
    "SnO2": "cassiterite — the closest analogue to TiO2, gas-sensing workhorse",
    "GeO2": "argutite — rutile form; a quartz-like polymorph is more stable",
    "RuO2": "metallic, the OER benchmark catalyst",
    "IrO2": "metallic, the other OER benchmark",
    "PbO2": "plattnerite (beta-PbO2)",
    "SiO2": "stishovite — high-pressure rutile SiO2, far off-hull by construction",
}

# Deliberately excluded from the default set. All are rutile-*derived* but break the clean
# axis: the first three are magnetic or strongly correlated (these MLIPs carry no spin
# state, so errors will be large and systematic), and several are distorted away from
# P4_2/mnm at room temperature, so "same structure, different chemistry" no longer holds.
HARD_CASES = {
    "CrO2": "ferromagnetic half-metal",
    "MnO2": "pyrolusite — antiferromagnetic",
    "VO2": "metal-insulator transition; monoclinic below ~340 K",
    "NbO2": "distorted rutile at room temperature",
}


def _best_entry(mpr, formula: str):
    """Lowest-energy MP entry with this formula *in the rutile spacegroup*.

    Not filtered on stability: several rutiles are metastable polymorphs (stishovite is
    the obvious one), so requiring on-hull entries would silently drop them. Among entries
    that genuinely have the rutile structure, the lowest hull distance is the right pick.
    """
    docs = mpr.materials.summary.search(
        formula=formula,
        spacegroup_number=RUTILE_SPACEGROUP,
        fields=["material_id", "formula_pretty", "energy_above_hull", "symmetry",
                "is_stable", "theoretical", "nsites"],
    )
    if not docs:
        return None
    return min(docs, key=lambda d: d.energy_above_hull if d.energy_above_hull is not None else 9e9)


def discover(mpr) -> None:
    """Print every binary oxide MP has in the rutile spacegroup — the family, unfiltered."""
    docs = mpr.materials.summary.search(
        spacegroup_number=RUTILE_SPACEGROUP,
        num_elements=2,
        elements=["O"],
        fields=["material_id", "formula_pretty", "energy_above_hull", "is_stable", "theoretical"],
    )
    print(f"\n{len(docs)} binary oxide entries in spacegroup {RUTILE_SPACEGROUP} (P4_2/mnm)\n")
    print(f"{'formula':10s}{'mp-id':14s}{'E_hull eV/at':>13s}{'stable':>8s}{'theoretical':>13s}")
    for d in sorted(docs, key=lambda d: (d.formula_pretty, d.energy_above_hull or 9e9)):
        hull = d.energy_above_hull
        print(f"{d.formula_pretty:10s}{str(d.material_id):14s}"
              f"{(f'{hull:.4f}' if hull is not None else '-'):>13s}"
              f"{str(d.is_stable):>8s}{str(d.theoretical):>13s}")
    print("\n(nothing fetched — rerun without --discover to write the curated set)")


def main(formulas: list[str], do_discover: bool, include_hard: bool) -> None:
    api_key = os.environ.get("MP_API_KEY")
    if not api_key:
        raise SystemExit(
            "MP_API_KEY is not set. Get a key from materialsproject.org (profile -> API) "
            "and export it in your shell:\n"
            "    export MP_API_KEY=your_key\n"
            "then re-run. Never hardcode it — this file is committed."
        )

    from mp_api.client import MPRester

    wanted = dict(WELL_BEHAVED)
    if include_hard:
        wanted.update(HARD_CASES)
    if formulas:
        wanted = {f: wanted.get(f, "") for f in formulas}

    with MPRester(api_key) as mpr:
        if do_discover:
            discover(mpr)
            return

        db_version = mpr.get_database_version()
        print(f"MP database {db_version}\nfetching {len(wanted)} rutile(s)\n")
        ok, missing = [], []
        for formula, note in wanted.items():
            doc = _best_entry(mpr, formula)
            if doc is None:
                missing.append(formula)
                print(f"  {formula:8s} NOT FOUND in spacegroup {RUTILE_SPACEGROUP}")
                continue
            mp_id = str(doc.material_id)
            structure = mpr.get_structure_by_material_id(mp_id, conventional_unit_cell=True)
            path = add_material(
                structure, mp_id,
                {
                    "source": "materials-project",
                    "query": f"formula={formula} spacegroup_number={RUTILE_SPACEGROUP}",
                    "energy_above_hull": doc.energy_above_hull,
                    "is_stable": doc.is_stable,
                    "theoretical": doc.theoretical,
                    "family": "rutile",
                    "note": note,
                    "mp_database_version": db_version,
                },
            )
            hull = doc.energy_above_hull
            ok.append((formula, mp_id))
            print(f"  {formula:8s} {mp_id:12s} E_hull={hull:.4f} eV/at -> {path.name}")

    print(f"\nwrote {len(ok)} structure(s) to data/structures/ — commit the .cif/.json pairs")
    if missing:
        print(f"not found: {', '.join(missing)}")
    if ok:
        print("\nnext: python scripts/validate_materials.py")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--formulas", nargs="+", help="override the curated set, e.g. TiO2 SnO2")
    ap.add_argument("--discover", action="store_true",
                    help="list every binary oxide in the rutile spacegroup; fetch nothing")
    ap.add_argument("--include-hard", action="store_true",
                    help=f"also fetch the magnetic/distorted cases: {', '.join(HARD_CASES)}")
    a = ap.parse_args()
    main(a.formulas or [], a.discover, a.include_hard)
