"""Plot the TiO2(110) oxygen-vacancy formation energy benchmark against DFT.

Reads each model's ``<rundir>/<model>/vacancy/rankings.csv`` (written independently per
model, so unaffected by ``run_stage.py``'s practice of truncating the shared
``candidates.jsonl`` on every invocation), pulls the O2c (bridging) and O3c (in-plane)
rows' ``e_vac_eV``, and plots both beside their literature DFT values. Also prints each
model's MAE/RMSD across the two sites -- a 2-point statistic, small but real, since both
sites are relaxed by the same funnel at no extra cost.

    python scripts/studies/ovfe/ovfe_tio2_benchmark_plot.py runs/ovfe_tio2_benchmark

Literature reference: Kowalski, Meyer & Marx, Phys. Rev. B 79, 115410 (2009), Table I,
triplet state, oxidizing limit (Delta*mu_O = 0, i.e. mu_O = E(O2)/2 from their own PBE
calculation) -- the same O-rich convention as this codebase's OXYGEN_REFERENCE, and the
same 4x2 supercell / dilute (1/8 ML) vacancy concentration as this repo's SlabConfig
defaults, so no unit conversion or concentration correction is needed.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# site_label -> (literature E_vac eV, display name, categorical color)
# Kowalski et al. Table I, triplet, Delta*mu_O = 0. Colors: dataviz skill palette.md,
# light mode, fixed categorical order (slots 1 and 2).
SITES = {
    "O2c": {"lit": 3.02, "name": "bridging O (O2c)", "color": "#2a78d6"},
    "O3c": {"lit": 4.00, "name": "in-plane O (O3c)", "color": "#eb6834"},
}
LIT_CITATION = "Kowalski, Meyer & Marx, PRB 79, 115410 (2009), Table I, triplet state"

MODELS = ["MACE-mh1-omat", "MACE-mh1-oc20", "UMA-omat", "UMA-oc22", "CHGNet-0.3.0", "Orb-v2",
          "SevenNet-omni-omat24"]

LIT_COLOR = "#898781"  # muted ink -- signals "reference", not a model prediction
INK = "#0b0b0b"
SECONDARY_INK = "#52514e"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"


def site_e_vac(rundir: Path, model: str) -> dict[str, float]:
    path = rundir / model / "vacancy" / "rankings.csv"
    if not path.exists():
        return {}
    with path.open() as f:
        return {
            row["site_label"]: float(row["e_vac_eV"])
            for row in csv.DictReader(f)
            if row["site_label"] in SITES and row["e_vac_eV"]
        }


def error_stats(rundir: Path) -> dict[str, dict]:
    """model -> {errors: {site: model - lit}, mae, rmse}, skipping sites not yet relaxed."""
    stats = {}
    for model in MODELS:
        found = site_e_vac(rundir, model)
        errors = {site: found[site] - SITES[site]["lit"] for site in found}
        if errors:
            mae = sum(abs(e) for e in errors.values()) / len(errors)
            rmse = math.sqrt(sum(e * e for e in errors.values()) / len(errors))
        else:
            mae = rmse = None
        stats[model] = {"found": found, "errors": errors, "mae": mae, "rmse": rmse,
                         "n": len(errors)}
    return stats


def plot(rundir: Path, stats: dict[str, dict]) -> Path:
    group_labels = ["DFT\n(literature)"] + MODELS
    site_keys = list(SITES)
    n_groups, n_sites = len(group_labels), len(site_keys)
    width = 0.8 / n_sites

    fig, ax = plt.subplots(figsize=(11, 6))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    for si, site in enumerate(site_keys):
        color = SITES[site]["color"]
        offset = (si - (n_sites - 1) / 2) * width
        xs = [g + offset for g in range(n_groups)]
        vals, present = [], []
        for gi, group in enumerate(group_labels):
            if gi == 0:
                v = SITES[site]["lit"]
            else:
                v = stats[group]["found"].get(site)
            present.append(v is not None)
            vals.append(v if v is not None else 0)
        bars = ax.bar(xs, vals, width=width * 0.92, color=color, edgecolor="none")
        for x, v, ok, gi in zip(xs, vals, present, range(n_groups)):
            if gi == 0:
                bars[gi].set_hatch("///")
                bars[gi].set_edgecolor(SURFACE)
                bars[gi].set_alpha(0.85)
            if ok:
                ax.text(x, v + 0.05, f"{v:.2f}", ha="center", va="bottom",
                         fontsize=8, color=INK, rotation=0)
            else:
                bars[gi].set_alpha(0.12)
                ax.text(x, 0.05, "pending", ha="center", va="bottom",
                         fontsize=7, color=SECONDARY_INK, rotation=90)
        ax.axhline(SITES[site]["lit"], color=color, lw=1.0, ls="--", alpha=0.5, zorder=0)

    ax.set_xticks(range(n_groups))
    ax.set_xticklabels(group_labels, fontsize=9, color=INK, rotation=20, ha="right")
    ax.set_ylabel("E$_{vac}$ (eV)", fontsize=11, color=INK)
    ax.set_title(
        "TiO$_2$(110) oxygen-vacancy formation energy vs. DFT\n"
        "O-rich limit ($\\mu_O$ = E(O$_2$)/2), 4×2 supercell, 1/8 ML",
        fontsize=12, color=INK,
    )
    ax.margins(y=0.15)
    handles = [plt.Rectangle((0, 0), 1, 1, color=SITES[s]["color"]) for s in site_keys]
    ax.legend(handles, [SITES[s]["name"] for s in site_keys], fontsize=9,
              frameon=False, loc="upper left")
    ax.text(0.5, -0.14, LIT_CITATION, transform=ax.transAxes, ha="center",
            fontsize=7.5, color=SECONDARY_INK)

    ax.grid(axis="y", color=GRID, lw=0.8, zorder=-1)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)
    ax.tick_params(colors=SECONDARY_INK)

    fig.tight_layout()
    outdir = rundir / "analysis"
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / "ovfe_benchmark.png"
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    return path


def print_table(stats: dict[str, dict]) -> None:
    print(f"\n{'model':16s}{'O2c eV':>10s}{'O3c eV':>10s}{'MAE eV':>10s}{'RMSE eV':>10s}")
    for model in MODELS:
        s = stats[model]
        o2c = s["found"].get("O2c")
        o3c = s["found"].get("O3c")
        fmt = lambda v: f"{v:.3f}" if v is not None else "-"
        mae = f"{s['mae']:.3f}" if s["mae"] is not None else "-"
        rmse = f"{s['rmse']:.3f}" if s["rmse"] is not None else "-"
        note = "" if s["n"] == 2 else f"  (n={s['n']})"
        print(f"{model:16s}{fmt(o2c):>10s}{fmt(o3c):>10s}{mae:>10s}{rmse:>10s}{note}")
    print(f"\nliterature (Kowalski '09): O2c={SITES['O2c']['lit']:.2f} eV, "
          f"O3c={SITES['O3c']['lit']:.2f} eV")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("rundir", type=Path)
    args = ap.parse_args()
    st = error_stats(args.rundir)
    print_table(st)
    out = plot(args.rundir, st)
    print(f"\nwrote {out}")
