"""How much of each model's E_vac error comes from its O2 gas reference specifically.

Three mu_O variants per model:

  raw        This model's own relaxed O2 total energy, uncorrected.
  corrected  Thermodynamic-cycle correction through THIS model's own H2 and H2O (see
             ``oxygen_chemical_potential_corrected`` in energetics.py) -- stays entirely on
             this model's own energy scale, only the experimental water formation enthalpy
             (a portable reaction energy) is external.
  lit_style  Kowalski, Meyer & Marx (PRB 79, 115410, 2009) do not publish an absolute E(O2)
             or mu_O anywhere -- every oxygen number in that paper (5.87/6.24/5.26 eV) is
             already a difference/reaction energy entangled with other species on THEIR
             code's own scale, so there is no bare value to transplant into an MLIP's E_vac
             (same reason two heads of one checkpoint can disagree by tens of eV on an
             identical cell -- see energetics.py). What IS extractable is the SIZE of the
             correction they needed: their own raw PBE D_e(O2) overbinds relative to the
             experimental value they anchor to, by LIT_RAW_DE - LIT_EXPT_DE. This applies
             that fixed shift uniformly to every model's raw mu_O -- a naive "assume every
             model has PBE's own O2 error" baseline, deliberately approximate (their actual
             corrected D_e depends on their own H2/H2O accuracy, which is not published),
             for comparison against each model's own proper (per-model) correction.

Since E_defective - E_pristine does not depend on mu_O, every corrected E_vac for a site
already relaxed by run_stage.py is just E_vac_raw + (mu_O_variant - mu_O_raw) -- no
slab/defect re-relaxation needed.

    python scripts/studies/ovfe/ovfe_o2_correction.py runs/ovfe_tio2_benchmark
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from oxide_workflow import pipeline
from oxide_workflow.backends import get_backend
from oxide_workflow.config import RunConfig
from oxide_workflow.energetics import (
    WATER_FORMATION_ENTHALPY_EXP,
    oxygen_chemical_potential,
    oxygen_chemical_potential_corrected,
)

MODELS = ["MACE-mh1-omat", "MACE-mh1-oc20", "UMA-omat", "UMA-oc22", "CHGNet-0.3.0", "Orb-v2",
          "SevenNet-omni-omat24"]
SITES = {
    "O2c": {"lit": 3.02, "name": "bridging O (O2c)", "color": "#2a78d6"},
    "O3c": {"lit": 4.00, "name": "in-plane O (O3c)", "color": "#eb6834"},
}
LIT_CITATION = "Kowalski, Meyer & Marx, PRB 79, 115410 (2009), Table I, triplet state"

# Kowalski et al.'s own raw PBE O2 dissociation energy, and the experimental value their
# correction anchors toward (both Sec. II of the paper). Neither is mu_O itself, but their
# difference is the approximate size of the correction they needed.
LIT_RAW_DE = 5.87
LIT_EXPT_DE = 5.26
LIT_MU_O_SHIFT = (LIT_RAW_DE - LIT_EXPT_DE) / 2  # +0.305 eV, added to raw mu_O

VARIANTS = ["raw", "corrected", "lit_style"]
VARIANT_LABEL = {
    "raw": "raw (model's own O2)",
    "corrected": "corrected (H2/H2O cycle, per model)",
    "lit_style": "lit-style (literature's correction size, applied uniformly)",
}

INK = "#0b0b0b"
SECONDARY_INK = "#52514e"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"
VARIANT_ALPHA = {"raw": 0.35, "corrected": 0.7, "lit_style": 1.0}


def site_e_vac_raw(rundir: Path, model: str) -> dict[str, float]:
    path = rundir / model / "vacancy" / "rankings.csv"
    if not path.exists():
        return {}
    with path.open() as f:
        return {
            row["site_label"]: float(row["e_vac_eV"])
            for row in csv.DictReader(f)
            if row["site_label"] in SITES and row["e_vac_eV"]
        }


def gather(rundir: Path) -> dict[str, dict]:
    cfg = RunConfig()
    out = {}
    for model in MODELS:
        backend = get_backend(model)
        mu_raw = oxygen_chemical_potential(backend, cfg, pipeline.relax)
        mu_corr = oxygen_chemical_potential_corrected(backend, cfg, pipeline.relax)
        mu_lit = mu_raw + LIT_MU_O_SHIFT
        raw = site_e_vac_raw(rundir, model)
        e_vac = {
            "raw": raw,
            "corrected": {s: e + (mu_corr - mu_raw) for s, e in raw.items()},
            "lit_style": {s: e + (mu_lit - mu_raw) for s, e in raw.items()},
        }
        errors = {
            v: {s: e_vac[v][s] - SITES[s]["lit"] for s in e_vac[v]} for v in VARIANTS
        }
        stats = {}
        for v in VARIANTS:
            errs = list(errors[v].values())
            stats[v] = {
                "mae": sum(abs(e) for e in errs) / len(errs) if errs else None,
                "rmse": math.sqrt(sum(e * e for e in errs) / len(errs)) if errs else None,
                "n": len(errs),
            }
        out[model] = {"mu": {"raw": mu_raw, "corrected": mu_corr, "lit_style": mu_lit},
                      "e_vac": e_vac, "stats": stats}
    return out


def print_table(data: dict[str, dict]) -> None:
    print(f"WATER_FORMATION_ENTHALPY_EXP = {WATER_FORMATION_ENTHALPY_EXP} eV (NIST, ZPE-corrected)")
    print(f"LIT_MU_O_SHIFT (estimate) = {LIT_MU_O_SHIFT:+.3f} eV "
          f"(from raw D_e={LIT_RAW_DE} eV vs. experimental-anchored D_e~{LIT_EXPT_DE} eV)\n")
    print(f"{'model':16s}{'mu_O raw':>10s}{'mu_O corr':>11s}{'mu_O lit':>10s}   "
          f"{'E_vac O2c (raw/corr/lit)':>28s}{'E_vac O3c (raw/corr/lit)':>28s}")
    for model, d in data.items():
        mu = d["mu"]
        ev = d["e_vac"]
        def fmt_site(site):
            if site not in ev["raw"]:
                return "pending"
            return f"{ev['raw'][site]:.2f}/{ev['corrected'][site]:.2f}/{ev['lit_style'][site]:.2f}"
        print(f"{model:16s}{mu['raw']:>10.3f}{mu['corrected']:>11.3f}{mu['lit_style']:>10.3f}   "
              f"{fmt_site('O2c'):>28s}{fmt_site('O3c'):>28s}")


def print_error_table(data: dict[str, dict]) -> None:
    print(f"\n{'model':16s}" + "".join(f"{v + ' MAE':>14s}{v + ' RMSE':>15s}" for v in VARIANTS))
    for model, d in data.items():
        row = f"{model:16s}"
        for v in VARIANTS:
            s = d["stats"][v]
            mae = f"{s['mae']:.3f}" if s["mae"] is not None else "-"
            rmse = f"{s['rmse']:.3f}" if s["rmse"] is not None else "-"
            row += f"{mae:>14s}{rmse:>15s}"
        print(row)


def plot_evac(rundir: Path, data: dict[str, dict]) -> Path:
    site_keys = list(SITES)
    fig, axes = plt.subplots(1, len(site_keys), figsize=(14, 6), sharey=True)
    fig.patch.set_facecolor(SURFACE)
    width = 0.24

    for ax, site in zip(axes, site_keys):
        ax.set_facecolor(SURFACE)
        color = SITES[site]["color"]
        labels = ["DFT\n(literature)"] + MODELS

        lit = SITES[site]["lit"]
        ax.bar([0], [lit], width=width * 3, color=color, alpha=0.85, hatch="///", edgecolor=SURFACE)
        ax.text(0, lit + 0.05, f"{lit:.2f}", ha="center", va="bottom", fontsize=8, color=INK)

        for i, model in enumerate(MODELS, start=1):
            if site not in data[model]["e_vac"]["raw"]:
                ax.text(i, 0.05, "pending", ha="center", va="bottom", fontsize=7,
                         color=SECONDARY_INK, rotation=90)
                continue
            for j, v in enumerate(VARIANTS):
                x = i + (j - 1) * width
                val = data[model]["e_vac"][v][site]
                ax.bar(x, val, width * 0.92, color=color, alpha=VARIANT_ALPHA[v], edgecolor="none")
                ax.text(x, val + 0.05, f"{val:.2f}", ha="center", va="bottom", fontsize=6.5, color=INK)

        ax.axhline(lit, color=color, lw=1.0, ls="--", alpha=0.5, zorder=0)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=8, color=INK, rotation=20, ha="right")
        ax.set_title(SITES[site]["name"], fontsize=11, color=INK)
        ax.grid(axis="y", color=GRID, lw=0.8, zorder=-1)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color(GRID)
        ax.tick_params(colors=SECONDARY_INK)

    axes[0].set_ylabel("E$_{vac}$ (eV)", fontsize=11, color=INK)
    handles = [plt.Rectangle((0, 0), 1, 1, color=SECONDARY_INK, alpha=VARIANT_ALPHA[v]) for v in VARIANTS]
    fig.legend(handles, [VARIANT_LABEL[v] for v in VARIANTS], fontsize=8.5, frameon=False,
               loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.06))
    fig.suptitle("O2 gas-reference correction: does it explain the E_vac gap?",
                  fontsize=12, color=INK, y=1.15)
    fig.text(0.5, -0.06, LIT_CITATION, ha="center", fontsize=7.5, color=SECONDARY_INK)

    fig.tight_layout()
    outdir = rundir / "analysis"
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / "ovfe_o2_correction.png"
    fig.savefig(path, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_error_metrics(rundir: Path, data: dict[str, dict]) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.patch.set_facecolor(SURFACE)
    width = 0.24
    variant_color = {"raw": "#898781", "corrected": "#1baf7a", "lit_style": "#4a3aa7"}

    for ax, metric in zip(axes, ("mae", "rmse")):
        ax.set_facecolor(SURFACE)
        for j, v in enumerate(VARIANTS):
            xs, vals = [], []
            for i, model in enumerate(MODELS):
                val = data[model]["stats"][v][metric]
                if val is None:
                    continue
                x = i + (j - 1) * width
                xs.append(x)
                vals.append(val)
                ax.text(x, val + 0.02, f"{val:.2f}", ha="center", va="bottom",
                         fontsize=7, color=INK)
            ax.bar(xs, vals, width * 0.92, color=variant_color[v], label=VARIANT_LABEL[v])
        ax.set_xticks(range(len(MODELS)))
        ax.set_xticklabels(MODELS, fontsize=8, color=INK, rotation=20, ha="right")
        ax.set_ylabel(f"{metric.upper()} vs. literature (eV)", fontsize=10, color=INK)
        ax.set_title(metric.upper(), fontsize=11, color=INK)
        ax.grid(axis="y", color=GRID, lw=0.8, zorder=-1)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color(GRID)
        ax.tick_params(colors=SECONDARY_INK)

    handles = [plt.Rectangle((0, 0), 1, 1, color=variant_color[v]) for v in VARIANTS]
    fig.legend(handles, [VARIANT_LABEL[v] for v in VARIANTS], fontsize=8.5, frameon=False,
               loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.06))
    fig.suptitle("Does correcting the O2 reference actually improve OVFE accuracy?",
                  fontsize=12, color=INK, y=1.14)

    fig.tight_layout()
    outdir = rundir / "analysis"
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / "ovfe_o2_correction_error_metrics.png"
    fig.savefig(path, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return path


def main(rundir: Path) -> None:
    data = gather(rundir)
    print_table(data)
    print_error_table(data)
    p1 = plot_evac(rundir, data)
    p2 = plot_error_metrics(rundir, data)
    print(f"\nwrote {p1}\nwrote {p2}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("rundir", type=Path)
    main(ap.parse_args().rundir)
