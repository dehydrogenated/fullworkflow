"""Plot the CO2-adsorption chemical-trend benchmark: E_ads and O-C-O angle across oxides.

Reads results.jsonl written by co2_adsorption_benchmark.py and plots two figures against
the literature (110)-plane trend (Chavez-Rocha et al., Molecules 2023, 28, 1776):

  1. E_ads vs. oxide, oxides ordered by literature binding strength (strongest first) so a
     line that preserves the same left-to-right ordering as literature is reproducing the
     RANKING even if the absolute values differ (PBEsol-D3 vs. these MLIPs' training data).
  2. O-C-O angle vs. oxide, same ordering -- strong binders bend CO2 away from linear
     (~129 deg for TiO2), weak binders stay near-linear (~179 deg for IrO2).

    python scripts/co2_adsorption_plot.py runs/co2_ads_benchmark
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from co2_adsorption_benchmark import MODELS, OXIDES

LIT_CITATION = "Chavez-Rocha et al., Molecules 2023, 28, 1776, Tables 1 & 2 ((110) plane, PBEsol-D3/TZP)"
LIT_COLOR = "#898781"
MODEL_COLOR = {"UMA-omat": "#1baf7a", "MACE-mh1-omat": "#2a78d6"}
INK = "#0b0b0b"
SECONDARY_INK = "#52514e"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"

# Oxides ordered by literature binding strength, strongest first -- the ranking axis.
ORDER = sorted(OXIDES, key=lambda o: OXIDES[o]["lit_e_ads_eV"])


def load(rundir: Path) -> dict[tuple[str, str], dict]:
    path = rundir / "results.jsonl"
    out = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            out[(r["model"], r["oxide"])] = r
    return out


def _style_axes(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=-1)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)
    ax.tick_params(colors=SECONDARY_INK)


def plot_energy(rundir: Path, data: dict) -> Path:
    fig, ax = plt.subplots(figsize=(9, 6))
    fig.patch.set_facecolor(SURFACE)
    xs = range(len(ORDER))

    lit_y = [OXIDES[o]["lit_e_ads_eV"] for o in ORDER]
    ax.plot(xs, lit_y, marker="o", ms=8, lw=2, color=LIT_COLOR, label="literature (PBEsol-D3)",
             ls="--")
    for x, y in zip(xs, lit_y):
        ax.text(x, y - 0.06, f"{y:.2f}", ha="center", va="top", fontsize=8, color=LIT_COLOR)

    for model in MODELS:
        ys, xs_present = [], []
        for x, o in zip(xs, ORDER):
            r = data.get((model, o))
            if r is None:
                continue
            xs_present.append(x)
            ys.append(r["e_ads_eV"])
        ax.plot(xs_present, ys, marker="o", ms=8, lw=2, color=MODEL_COLOR[model], label=model)
        for x, y in zip(xs_present, ys):
            ax.text(x, y + 0.06, f"{y:.2f}", ha="center", va="bottom", fontsize=8,
                     color=MODEL_COLOR[model])

    ax.set_xticks(list(xs))
    ax.set_xticklabels(ORDER, fontsize=10, color=INK)
    ax.set_ylabel("E$_{ads}$(CO$_2$) (eV)", fontsize=11, color=INK)
    ax.set_title("CO$_2$ adsorption energy across rutile oxides\n"
                  "oxides ordered by literature binding strength (strongest → weakest)",
                  fontsize=12, color=INK)
    ax.legend(fontsize=9, frameon=False)
    ax.text(0.5, -0.14, LIT_CITATION, transform=ax.transAxes, ha="center", fontsize=7.5,
             color=SECONDARY_INK)
    _style_axes(ax)

    fig.tight_layout()
    outdir = rundir / "analysis"
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / "co2_e_ads_trend.png"
    fig.savefig(path, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_angle(rundir: Path, data: dict) -> Path:
    fig, ax = plt.subplots(figsize=(9, 6))
    fig.patch.set_facecolor(SURFACE)
    xs = range(len(ORDER))

    lit_y = [OXIDES[o]["lit_angle"] for o in ORDER]
    ax.plot(xs, lit_y, marker="o", ms=8, lw=2, color=LIT_COLOR, label="literature (PBEsol-D3)",
             ls="--")
    for x, y in zip(xs, lit_y):
        ax.text(x, y - 3, f"{y:.0f}°", ha="center", va="top", fontsize=8, color=LIT_COLOR)

    for model in MODELS:
        ys, xs_present = [], []
        for x, o in zip(xs, ORDER):
            r = data.get((model, o))
            if r is None:
                continue
            xs_present.append(x)
            ys.append(r["oco_angle_deg"])
        ax.plot(xs_present, ys, marker="o", ms=8, lw=2, color=MODEL_COLOR[model], label=model)
        for x, y in zip(xs_present, ys):
            ax.text(x, y + 3, f"{y:.0f}°", ha="center", va="bottom", fontsize=8,
                     color=MODEL_COLOR[model])

    ax.axhline(180, color=SECONDARY_INK, lw=1.0, ls=":", alpha=0.6)
    ax.text(len(ORDER) - 0.5, 181, "linear (180°)", ha="right", va="bottom", fontsize=7.5,
             color=SECONDARY_INK)
    ax.set_xticks(list(xs))
    ax.set_xticklabels(ORDER, fontsize=10, color=INK)
    ax.set_ylabel("O-C-O angle (deg)", fontsize=11, color=INK)
    ax.set_ylim(115, 190)
    ax.set_title("CO$_2$ bending across rutile oxides\n"
                  "stronger binding → more activation/bending away from linear",
                  fontsize=12, color=INK)
    ax.legend(fontsize=9, frameon=False, loc="lower right")
    ax.text(0.5, -0.14, LIT_CITATION, transform=ax.transAxes, ha="center", fontsize=7.5,
             color=SECONDARY_INK)
    _style_axes(ax)

    fig.tight_layout()
    outdir = rundir / "analysis"
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / "co2_angle_trend.png"
    fig.savefig(path, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return path


def main(rundir: Path) -> None:
    data = load(rundir)
    print(f"{'model':16s}{'oxide':8s}{'E_ads eV':>10s}{'O-C-O deg':>11s}")
    for model in MODELS:
        for o in ORDER:
            r = data.get((model, o))
            if r is None:
                print(f"{model:16s}{o:8s}{'pending':>10s}{'':>11s}")
            else:
                print(f"{model:16s}{o:8s}{r['e_ads_eV']:>10.3f}{r['oco_angle_deg']:>11.1f}")
    print(f"{'literature':16s}" + "")
    for o in ORDER:
        print(f"{'':16s}{o:8s}{OXIDES[o]['lit_e_ads_eV']:>10.3f}{OXIDES[o]['lit_angle']:>11.1f}")

    p1 = plot_energy(rundir, data)
    p2 = plot_angle(rundir, data)
    print(f"\nwrote {p1}\nwrote {p2}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("rundir", type=Path)
    main(ap.parse_args().rundir)
