"""Full flag/status report for a co2_adsorption_retest.py run.

Reads EVERY row of results_retest.jsonl (not just winners) and reports:
  1. Operational health: how many of the expected relaxations ran, how many failed outright
     (worker crash/exception) vs. desorbed vs. extended vs. converged cleanly vs. neither
     converged nor flagged either way (worth knowing about on its own).
  2. The actual science question: did any site/orientation show real bending (O-C-O
     meaningfully off 180 degrees), broken down by model/oxide/site type.

    python scripts/co2_retest_report.py runs/co2_ads_benchmark
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

BENT_THRESHOLD_DEG = 175.0  # below this counts as "showing real bending", not just noise


def load(path: Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"no results file at {path}")
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def status_of(row: dict) -> str:
    if row.get("failed"):
        return "FAILED"
    if row.get("desorbing"):
        return "desorbing (early-stop)" if row.get("desorbing_early_stop") else "desorbing (final geometry)"
    if row.get("converged") is False:
        return "unconverged (not flagged either way)"
    if row.get("trivial_start"):
        return "adsorbed (trivial start -- no real work done)"
    if row.get("extended"):
        return "adsorbed (extended)"
    return "adsorbed"


def main(rundir: Path) -> None:
    rows = load(rundir / "results_retest.jsonl")
    n = len(rows)
    print(f"{n} relaxation attempt(s) recorded in {rundir / 'results_retest.jsonl'}\n")

    # --- overall status breakdown --------------------------------------------------------
    statuses = Counter(status_of(r) for r in rows)
    print(f"{'-'*60}\nOVERALL STATUS\n{'-'*60}")
    for status, count in statuses.most_common():
        print(f"  {status:38s}{count:>5d}")

    # --- failures, with the actual error message ------------------------------------------
    failures = [r for r in rows if r.get("failed")]
    if failures:
        print(f"\n{'-'*60}\nFAILURES ({len(failures)})\n{'-'*60}")
        for r in failures:
            where = f"{r['model']}/{r['oxide']}"
            site = f"site{r['site_index']} {r.get('orientation') or ''}".strip()
            print(f"  {where:28s}{site:20s}{(r.get('error') or '')[:120]}")

    # --- per (model, oxide) pair table -----------------------------------------------------
    print(f"\n{'-'*60}\nPER (model, oxide) BREAKDOWN\n{'-'*60}")
    by_pair: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        by_pair[(r["model"], r["oxide"])].append(r)
    print(f"{'model':16s}{'oxide':8s}{'n':>4s}{'ads':>5s}{'desorb':>8s}{'d_early':>8s}"
          f"{'trivial':>8s}{'extend':>8s}{'unconv':>8s}{'fail':>6s}{'min angle':>11s}")
    for (model, oxide), rs in sorted(by_pair.items()):
        ads = sum(1 for r in rs if r.get("adsorbed"))
        desorb = sum(1 for r in rs if r.get("desorbing"))
        d_early = sum(1 for r in rs if r.get("desorbing_early_stop"))
        trivial = sum(1 for r in rs if r.get("trivial_start"))
        extend = sum(1 for r in rs if r.get("extended"))
        unconv = sum(1 for r in rs if r.get("converged") is False and not r.get("desorbing"))
        fail = sum(1 for r in rs if r.get("failed"))
        angles = [r["oco_angle_deg"] for r in rs if r.get("oco_angle_deg") is not None]
        min_angle = f"{min(angles):.1f}" if angles else "-"
        print(f"{model:16s}{oxide:8s}{len(rs):>4d}{ads:>5d}{desorb:>8d}{d_early:>8d}"
              f"{trivial:>8d}{extend:>8d}{unconv:>8d}{fail:>6d}{min_angle:>11s}")

    # --- the actual science: did bending show up anywhere? --------------------------------
    print(f"\n{'-'*60}\nBENDING (O-C-O < {BENT_THRESHOLD_DEG} deg)\n{'-'*60}")
    bent = [r for r in rows if r.get("oco_angle_deg") is not None
            and r["oco_angle_deg"] < BENT_THRESHOLD_DEG]
    if not bent:
        print("  none -- every converged/extended site is still ~linear")
    else:
        print(f"  {len(bent)} site/orientation attempt(s) show real bending:\n")
        print(f"  {'model':16s}{'oxide':8s}{'site':6s}{'type':8s}{'orientation':16s}"
              f"{'angle':>8s}{'E_ads':>10s}")
        for r in sorted(bent, key=lambda r: r["oco_angle_deg"]):
            print(f"  {r['model']:16s}{r['oxide']:8s}{'site'+str(r['site_index']):6s}"
                  f"{r['symmetry_class']:8s}{r['orientation']:16s}"
                  f"{r['oco_angle_deg']:>8.1f}{r['e_ads_eV']:>10.3f}")

    # --- best (most bound, non-desorbing, non-failed) result per (model, oxide) -----------
    print(f"\n{'-'*60}\nBEST RESULT PER (model, oxide)\n{'-'*60}")
    print(f"{'model':16s}{'oxide':8s}{'E_ads':>10s}{'angle':>8s}{'site type':>12s}")
    for (model, oxide), rs in sorted(by_pair.items()):
        candidates = [r for r in rs if not r.get("failed") and not r.get("desorbing")
                      and r.get("e_ads_eV") is not None]
        if not candidates:
            print(f"{model:16s}{oxide:8s}{'no binding site found':>32s}")
            continue
        best = min(candidates, key=lambda r: r["e_ads_eV"])
        print(f"{model:16s}{oxide:8s}{best['e_ads_eV']:>10.3f}{best['oco_angle_deg']:>8.1f}"
              f"{best['symmetry_class']:>12s}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("rundir", type=Path)
    main(ap.parse_args().rundir)
