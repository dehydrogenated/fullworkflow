"""Fetch a Materials Project entry into data/structures/ so runs can resolve it offline.

One-time bootstrap per material: run this on a networked machine, commit the .cif/.json
pair it writes, then point RunConfig.polymorph at the same id. MP_API_KEY is only needed
here — the pipeline itself never reads it.

    MP_API_KEY=... python scripts/core/fetch_structure.py mp-2657
"""

from __future__ import annotations

import argparse
import os

from oxide_workflow.structures import add_material


def _calc_type(mpr, mp_id: str) -> str:
    """Which functional produced the cell MP just handed back.

    ``origins`` says which task the structure came from; ``calc_types`` maps that task to
    its functional (e.g. "GGA+U Structure Optimization" vs "r2SCAN Structure Optimization").
    Worth recording: it decides whether the warm start is experiment-like or not — plain
    GGA overestimates oxide lattice constants by ~1%, r2SCAN lands much closer.

    Never fatal; a schema change downgrades this to "unknown" rather than killing a fetch.
    """
    try:
        doc = mpr.materials.search(material_ids=[mp_id], fields=["calc_types", "origins"])[0]
        calc_types = {str(k): str(v) for k, v in (doc.calc_types or {}).items()}
        for origin in doc.origins or []:
            if getattr(origin, "name", None) == "structure":
                return calc_types.get(str(origin.task_id), "unknown")
        return "; ".join(sorted(set(calc_types.values()))) or "unknown"
    except Exception as exc:  # noqa: BLE001 - provenance is best-effort, the cell is not
        return f"unknown ({type(exc).__name__})"


def main(mp_id: str) -> None:
    api_key = os.environ.get("MP_API_KEY")
    if not api_key:
        raise SystemExit(
            "MP_API_KEY is not set. Get a key from materialsproject.org (profile -> API), "
            "then re-run as `MP_API_KEY=your_key python scripts/core/fetch_structure.py "
            f"{mp_id}`. Do not hardcode it — this file is committed."
        )

    from mp_api.client import MPRester  # lazy: only fetching needs the optional mp extra

    with MPRester(api_key) as mpr:
        # conventional_unit_cell=True is required — the API default is the PRIMITIVE cell.
        structure = mpr.get_structure_by_material_id(mp_id, conventional_unit_cell=True)
        provenance = {
            "source": "materials-project",
            "calc_type": _calc_type(mpr, mp_id),  # the functional behind the warm start
            "mp_database_version": mpr.get_database_version(),
        }

    path = add_material(structure, mp_id, provenance)
    print(f"wrote {path} (+ .json) — commit both, then set RunConfig.polymorph = {mp_id!r}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mp_id", help="Materials Project id, e.g. mp-2657")
    main(ap.parse_args().mp_id)
