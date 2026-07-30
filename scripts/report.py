"""Summarize and plot a finished (or in-progress) run directory.

Reads the three run-root tables — ``summary.json``, ``divergence.jsonl``,
``candidates.jsonl`` — and reports the two questions the benchmark exists to answer:

1. **Geometry/energy divergence** per stage and protocol.
2. **Ranking fidelity** — does the candidate pick the *same* vacancy and adsorbate site as
   the reference? A model can reproduce the reference's geometry closely and still rank
   sites differently, which is the failure mode that matters for screening.

    python scripts/report.py runs/COtest_july29
    python scripts/report.py runs/H_oc20_0729 --no-plot

Safe to run mid-run: both tables are appended as stages finish, so a partial run reports
whatever has landed so far.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

STAGES = ("bulk", "slab", "vacancy", "adsorbate")


def _load(rundir: Path):
    summary_path = rundir / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}

    def jsonl(name):
        p = rundir / name
        return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []

    return summary, jsonl("divergence.jsonl"), jsonl("candidates.jsonl")


def _num(v, width=10, prec=4):
    return f"{v:{width}.{prec}f}" if isinstance(v, (int, float)) else f"{'-':>{width}}"


def print_header(rundir: Path, summary: dict, div: list) -> None:
    print(f"\n{'=' * 72}\nRUN  {rundir}\n{'=' * 72}")
    if not summary:
        print("(no summary.json — run still in progress; reporting partial tables)\n")
        return
    mins = summary.get("total_elapsed_s", 0) / 60
    print(f"material    {summary.get('polymorph')}   facet {summary.get('facet')}")
    print(f"reference   {summary.get('reference')}")
    print(f"candidates  {', '.join(summary.get('candidates', []))}")
    print(f"wall time   {mins:.1f} min")
    print(f"sites       {summary.get('n_vacancy_sites')} vacancy, "
          f"{summary.get('n_adsorbate_sites')} adsorbate\n")


def print_divergence(div: list) -> None:
    """Per-stage geometry + energy divergence, split by protocol."""
    print(f"{'-' * 72}\nDIVERGENCE  (candidate relative to reference)\n{'-' * 72}")
    if not div:
        print("(no divergence rows yet)\n")
        return
    print(f"{'model':18s}{'protocol':15s}{'stage':11s}"
          f"{'rmsd A':>10s}{'maxdisp A':>11s}{'dE eV':>11s}  flags")
    for r in sorted(div, key=lambda r: (r["model"], r["protocol"], STAGES.index(r["stage"]))):
        flags = ",".join(r.get("meta", {}).get("flags", []))
        if not r.get("meta", {}).get("matched", True):
            flags = (flags + " " if flags else "") + "UNMATCHED(different site)"
        print(f"{r['model']:18s}{r['protocol']:15s}{r['stage']:11s}"
              f"{_num(r['rmsd'])}{_num(r['max_displacement'], 11)}"
              f"{_num(r['energy_error'], 11, 3)}  {flags}")
    print()


def _winners(cands: list) -> dict:
    """(stage, model, protocol) -> (winning site_index, its energy, n sites considered).

    The winner is the minimum-energy site, which is exactly what the pipeline's funnel
    selects — so this reconstructs each chain's choice from the recorded table.
    """
    groups: dict[tuple, list] = defaultdict(list)
    for c in cands:
        groups[(c["stage"], c["model"], c["protocol"])].append(c)
    out = {}
    for key, rows in groups.items():
        best = min(rows, key=lambda r: r["energy"])
        out[key] = (best["site_index"], best["energy"], len(rows))
    return out


def print_sites(summary: dict, cands: list) -> None:
    """Best vacancy / adsorbate placement per chain, and whether the rankings agree."""
    print(f"{'-' * 72}\nBEST SITES  (min-energy placement each chain selected)\n{'-' * 72}")
    if not cands:
        print("(no candidate rows yet)\n")
        return

    wins = _winners(cands)
    reference = summary.get("reference")

    for stage in ("vacancy", "adsorbate"):
        keys = sorted(k for k in wins if k[0] == stage)
        if not keys:
            continue
        ref_key = next((k for k in keys if k[2] == "reference"), None)
        ref_site = wins[ref_key][0] if ref_key else None
        print(f"\n[{stage}]")
        print(f"  {'model':18s}{'protocol':15s}{'best site':>10s}{'n sites':>9s}"
              f"{'energy eV':>13s}  vs reference")
        for k in keys:
            site, energy, n = wins[k]
            if k[2] == "reference":
                verdict = "(reference)"
            elif ref_site is None:
                verdict = ""
            else:
                verdict = "SAME site" if site == ref_site else f"DIFFERENT (ref picked {ref_site})"
            print(f"  {k[1]:18s}{k[2]:15s}{site:>10d}{n:>9d}{energy:>13.4f}  {verdict}")

        # Ranking margin: how decisively the reference preferred its winner. A margin
        # smaller than the models' mutual energy error means the ranking is not resolved.
        if ref_key:
            ref_rows = sorted(
                (c for c in cands if c["stage"] == stage and c["protocol"] == "reference"),
                key=lambda r: r["energy"],
            )
            if len(ref_rows) > 1:
                margin = ref_rows[1]["energy"] - ref_rows[0]["energy"]
                print(f"  reference margin over runner-up: {margin:.3f} eV "
                      f"(site {ref_rows[1]['site_index']})")
    print()


def plot(rundir: Path, summary: dict, div: list, cands: list) -> Path | None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    site_stages = [s for s in ("vacancy", "adsorbate") if any(c["stage"] == s for c in cands)]
    n_panels = 1 + len(site_stages)
    fig, axes = plt.subplots(1, n_panels, figsize=(6.5 * n_panels, 4.5))
    axes = list(axes) if n_panels > 1 else [axes]

    # --- panel 1: geometric divergence by stage, one bar group per protocol -------------
    ax = axes[0]
    protocols = sorted({r["protocol"] for r in div})
    stages = [s for s in STAGES if any(r["stage"] == s for r in div)]
    width = 0.8 / max(len(protocols), 1)
    for i, proto in enumerate(protocols):
        xs, vals, missing = [], [], []
        for xi, s in enumerate(stages):
            row = next((r for r in div if r["stage"] == s and r["protocol"] == proto), None)
            if row is None:
                continue
            x = xi + i * width
            if isinstance(row.get("rmsd"), (int, float)):
                xs.append(x)
                vals.append(row["rmsd"])
            else:
                missing.append(x)  # unmatched: no bar, annotated instead of drawn as zero
        bars = ax.bar(xs, vals, width, label=proto)
        color = bars[0].get_facecolor() if len(bars) else None
        for x in missing:
            ax.text(x, ax.get_ylim()[1] * 0.02, "unmatched", rotation=90, ha="center",
                    va="bottom", fontsize=7, color=color or "k")
    ax.set_xticks([x + width * (len(protocols) - 1) / 2 for x in range(len(stages))])
    ax.set_xticklabels(stages)
    ax.set_ylabel("RMSD vs reference (Å)")
    ax.set_title("Geometric divergence by stage")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # --- one panel per site stage: energy ranking relative to each chain's OWN minimum --
    # Absolute energies from different heads are not comparable (per-atom reference
    # offset), but each chain's spread over its own sites is — so plot dE from own min.
    # Vacancy and adsorbate get separate axes: their site indices live on different scales.
    groups: dict[tuple, list] = defaultdict(list)
    for c in cands:
        groups[(c["stage"], c["model"], c["protocol"])].append(c)

    for ax, stage in zip(axes[1:], site_stages):
        keys = sorted(k for k in groups if k[0] == stage)
        labels = sorted({r["site_index"] for k in keys for r in groups[k]})
        pos = {s: i for i, s in enumerate(labels)}  # categorical: even spacing
        for key in keys:
            rows = sorted(groups[key], key=lambda r: r["site_index"])
            emin = min(r["energy"] for r in rows)
            ax.plot(
                [pos[r["site_index"]] for r in rows],
                [r["energy"] - emin for r in rows],
                marker="o", ms=5, lw=1.2, alpha=0.8, label=f"{key[1]}/{key[2]}",
            )
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, fontsize=8)
        ax.set_xlabel("originating site index")
        ax.set_ylabel("energy above that chain's own min (eV)")
        ax.set_title(f"{stage} site ranking (0 = the site it picked)")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

    fig.suptitle(f"{rundir.name}   {summary.get('reference', '')} vs "
                 f"{', '.join(summary.get('candidates', []))}", fontsize=10)
    fig.tight_layout()
    outdir = rundir / "analysis"
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / "report.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main(rundir: Path, do_plot: bool) -> None:
    if not rundir.is_dir():
        raise SystemExit(f"not a run directory: {rundir}")
    summary, div, cands = _load(rundir)

    print_header(rundir, summary, div)
    print_divergence(div)
    print_sites(summary, cands)

    if do_plot and (div or cands):
        print(f"plot: {plot(rundir, summary, div, cands)}\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("rundir", type=Path, help="a run directory (contains summary.json)")
    ap.add_argument("--no-plot", action="store_true", help="text report only")
    args = ap.parse_args()
    main(args.rundir, not args.no_plot)
