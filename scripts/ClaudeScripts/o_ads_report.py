"""Full flag/status report for o_adsorption_benchmark.py, plus trend-fidelity vs. literature.

Absolute PBE-vs-MLIP agreement is not expected; the more meaningful check is whether each
model reproduces Zhao & Kulik's periodic trend across the 7 oxides (which metals bind O
most/least favorably), so this reports both operational health and per-model trend
correlation, not just a comparison table.

    python scripts/o_ads_report.py runs/o_ads_benchmark
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


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


def spearman(xs: list[float], ys: list[float]) -> float | None:
    """Rank correlation, no scipy dependency. None if fewer than 2 points or no variance."""
    n = len(xs)
    if n < 2:
        return None

    def ranks(vals: list[float]) -> list[float]:
        order = sorted(range(len(vals)), key=lambda i: vals[i])
        r = [0.0] * len(vals)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and vals[order[j + 1]] == vals[order[i]]:
                j += 1
            avg_rank = (i + j) / 2.0 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg_rank
            i = j + 1
        return r

    rx, ry = ranks(xs), ranks(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    vx = sum((a - mx) ** 2 for a in rx)
    vy = sum((b - my) ** 2 for b in ry)
    if vx == 0 or vy == 0:
        return None
    return cov / (vx * vy) ** 0.5


def main(rundir: Path) -> None:
    rows = load(rundir / "results.jsonl")
    n = len(rows)
    print(f"{n} relaxation attempt(s) recorded in {rundir / 'results.jsonl'}\n")

    print(f"{'-'*60}\nOVERALL STATUS\n{'-'*60}")
    statuses: dict[str, int] = defaultdict(int)
    for r in rows:
        statuses[status_of(r)] += 1
    for status, count in sorted(statuses.items(), key=lambda kv: -kv[1]):
        print(f"  {status:38s}{count:>5d}")

    failures = [r for r in rows if r.get("failed")]
    if failures:
        print(f"\n{'-'*60}\nFAILURES ({len(failures)})\n{'-'*60}")
        for r in failures:
            print(f"  {r['model']:16s}{r['oxide']:8s}{(r.get('error') or '')[:130]}")

    print(f"\n{'-'*60}\nPER (model, oxide) COMPARISON\n{'-'*60}")
    print(f"{'model':16s}{'oxide':8s}{'E_ads':>9s}{'lit E_ads':>11s}{'diff':>8s}"
          f"{'bond A':>9s}{'lit bond':>10s}{'status':>28s}")
    by_model: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_model[r["model"]].append(r)
        lit_e = r.get("lit_e_ads_eV")
        lit_e_s = f"{lit_e:.3f}" if lit_e is not None else "--"
        lit_bond = r.get("lit_bond_A")
        lit_bond_s = f"{lit_bond:.3f}" if lit_bond is not None else "--"
        if r.get("e_ads_eV") is not None and lit_e is not None:
            diff = r["e_ads_eV"] - lit_e
            bond = r.get("bond_A")
            bond_s = f"{bond:.3f}" if bond is not None else "-"
            print(f"{r['model']:16s}{r['oxide']:8s}{r['e_ads_eV']:>9.3f}{lit_e_s:>11s}"
                  f"{diff:>8.3f}{bond_s:>9s}{lit_bond_s:>10s}{status_of(r):>28s}")
        else:
            print(f"{r['model']:16s}{r['oxide']:8s}{'--':>9s}{lit_e_s:>11s}"
                  f"{'--':>8s}{'--':>9s}{lit_bond_s:>10s}{status_of(r):>28s}")

    print(f"\n{'-'*60}\nPER-MODEL TREND FIDELITY (vs. Zhao & Kulik PBE/PW)\n{'-'*60}")
    print(f"{'model':16s}{'n':>4s}{'MAE E_ads':>11s}{'MAE bond':>10s}{'rank corr':>11s}")
    for model, rs in sorted(by_model.items()):
        valid = [r for r in rs if r.get("e_ads_eV") is not None]
        if not valid:
            print(f"{model:16s}{'0':>4s}{'--':>11s}{'--':>10s}{'--':>11s}")
            continue
        mae_e = sum(abs(r["e_ads_eV"] - r["lit_e_ads_eV"]) for r in valid) / len(valid)
        bonded = [r for r in valid if r.get("bond_A") is not None]
        mae_b = (sum(abs(r["bond_A"] - r["lit_bond_A"]) for r in bonded) / len(bonded)) if bonded else None
        rho = spearman([r["e_ads_eV"] for r in valid], [r["lit_e_ads_eV"] for r in valid])
        mae_b_s = f"{mae_b:.3f}" if mae_b is not None else "--"
        rho_s = f"{rho:+.3f}" if rho is not None else "--"
        print(f"{model:16s}{len(valid):>4d}{mae_e:>11.3f}{mae_b_s:>10s}{rho_s:>11s}")
    print("\n(rank corr near +1.0 = reproduces which metals bind O most/least favorably;")
    print(" near 0 = no relationship to the literature trend; negative = trend inverted)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("rundir", type=Path)
    main(ap.parse_args().rundir)
