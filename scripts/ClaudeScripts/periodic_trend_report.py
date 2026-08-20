#!/usr/bin/env python
"""Builds the JSON data blob behind the periodic-trend artifact (ΔE_O / ΔE_OH vs. group
number, one line per model x row x facet, styled after Comer et al. 2022 Fig. 3) --
scripts/slurm/sockeye_mo2_sweep_{3d,4d,5d}.slurm's own OXIDES arrays are the group-number
mapping, since each list is already ordered group 4->14 within its row.

Reads every results.jsonl-shaped file passed on the command line, dedupes by keeping the
LAST row per (model, oxide, facet, adsorbate) -- the same append-mode-overlap rule used
throughout this session (medium test CPU/GPU overlap, extension-fix retest, and now
whatever backfill rows land alongside the original sweep output). Literature values come
along for free: every row already carries lit_e_ads_eV, Comer's own DFT+U number for that
exact combo, so no separate literature file needs joining in.

Usage:
    python periodic_trend_report.py OUT.json FILE1.jsonl [FILE2.jsonl ...]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROWS: dict[str, list[str]] = {
    "3d": ["TiO2", "VO2", "CrO2", "MnO2", "FeO2", "CoO2", "NiO2", "CuO2", "ZnO2", "GaO2", "GeO2"],
    "4d": ["ZrO2", "NbO2", "MoO2", "TcO2", "RuO2", "RhO2", "PdO2", "AgO2", "CdO2", "InO2", "SnO2"],
    "5d": ["HfO2", "TaO2", "WO2", "ReO2", "OsO2", "IrO2", "PtO2", "AuO2", "HgO2", "TlO2", "PbO2"],
}
GROUPS = list(range(4, 15))  # 4..14, position i in each ROWS list is group GROUPS[i]
OXIDE_TO_ROW_GROUP = {
    oxide: (row, GROUPS[i]) for row, oxides in ROWS.items() for i, oxide in enumerate(oxides)
}


def load_rows(paths: list[Path]) -> dict[tuple, dict]:
    """One dict entry per (model, oxide, facet, adsorbate), last-row-wins on duplicates."""
    latest: dict[tuple, dict] = {}
    for path in paths:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                key = (r.get("model"), r.get("oxide"), r.get("facet"), r.get("adsorbate"))
                latest[key] = r  # later files/lines overwrite earlier ones
    return latest


def build_series(latest: dict[tuple, dict]) -> dict:
    """{adsorbate: {model: {row: {facet: {group: value_or_None}}}}} for both e_ads_eV
    (raw) and lit_e_ads_eV, plus a parallel 'converged'/'failed' map so the chart can grey
    out or flag points that aren't trustworthy yet (still running, didn't converge, etc.)."""
    models = sorted({k[0] for k in latest if k[0]})
    out = {
        "generated_from": [],
        "models": models,
        "adsorbates": {},
    }
    for adsorbate in ("O", "OH"):
        out["adsorbates"][adsorbate] = {"models": {}, "literature": {}}
        # literature is model-independent -- pull it once from whichever row has it
        lit_by_row_facet_group: dict = {row: {"110": {}, "100": {}} for row in ROWS}
        for model in models:
            model_data = {row: {"110": {}, "100": {}} for row in ROWS}
            for oxide, (row, group) in OXIDE_TO_ROW_GROUP.items():
                for facet in ("110", "100"):
                    key = (model, oxide, facet, adsorbate)
                    r = latest.get(key)
                    if r is None:
                        continue
                    if lit_by_row_facet_group[row][facet].get(group) is None and r.get("lit_e_ads_eV") is not None:
                        lit_by_row_facet_group[row][facet][group] = r["lit_e_ads_eV"]
                    if r.get("failed") or r.get("e_ads_eV") is None:
                        continue
                    model_data[row][facet][group] = {
                        "value": r["e_ads_eV"],
                        "converged": bool(r.get("converged")),
                        "desorbing": bool(r.get("desorbing")),
                    }
            out["adsorbates"][adsorbate]["models"][model] = model_data
        out["adsorbates"][adsorbate]["literature"] = lit_by_row_facet_group
    return out


def main(out_path: Path, in_paths: list[Path]) -> None:
    latest = load_rows(in_paths)
    data = build_series(latest)
    data["generated_from"] = [str(p) for p in in_paths]
    data["row_count"] = len(latest)
    out_path.write_text(json.dumps(data, indent=None, separators=(",", ":")))
    print(f"{len(latest)} unique (model,oxide,facet,adsorbate) rows -> {out_path}")
    print(f"models: {data['models']}")


if __name__ == "__main__":
    out = Path(sys.argv[1])
    ins = [Path(p) for p in sys.argv[2:]]
    main(out, ins)
