"""Aggregate the OC22 model sweep: divergence by region, OVFE vs DFT, ranking fidelity.

    python scripts/oc22_report.py runs/oc22_sweep
    python scripts/oc22_report.py runs/oc22_sweep --plot report.png
    python scripts/oc22_report.py runs/oc22_sweep --seedset chain3d

Reads every ``<seedset>/<model>/divergence.jsonl`` written by ``oc22_diverge.py``. Pure
post-processing — relaxes nothing, so it is cheap to re-run while a sweep is still going.

WHAT IS AGGREGATED, AND WHAT IS NOT. ``oc22_diverge.py`` flags reconstruction and
dissociation as categorical failures rather than divergence values: a slab that found a
different minimum describes a different structure, and averaging its displacement in
reports a model error that is really a structure mismatch. Flagged rows are therefore
excluded from every mean here and counted separately, so a model that "wins" by
reconstructing half its systems cannot hide it.

REGIONS, NOT STAGES. OC22 divergence decomposes one relaxation by the dataset's own
``tags`` (0 subsurface, 1 surface, 2 adsorbate). That is a different axis from the
pipeline's bulk/slab/vacancy/adsorbate stages, and mixing the two vocabularies causes
real confusion — "adsorbate" here means the atoms, not the stage.

OVFE. OC22 carries several clean slabs per facet differing only in oxygen count at
identical metal count, which is an oxygen vacancy with a DFT energy attached:

    E_vac = [E(reduced) - E(stoichiometric) + n_vac * mu_O] / n_vac     (per vacancy)

mu_O = E(O2)/2 from the model's own gas reference, so the term cancels the way
``energetics.vacancy_formation_energy`` intends. n_vac varies from 2 to 16 across the
seeded facets, so it must scale mu_O rather than be assumed to be one — and the result is
reported per vacancy so facets with different n are comparable.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
REGIONS = ("subsurface", "surface", "adsorbate")


def load(root: Path) -> list[dict]:
    """Every divergence row under <root>/<seedset>/<model>/divergence.jsonl."""
    rows = []
    for path in sorted(root.glob("*/*/divergence.jsonl")):
        seedset = path.parent.parent.name
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            r["seedset"] = seedset
            rows.append(r)
    return rows


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def fmt(x, w=8, p=3):
    return f"{x:{w}.{p}f}" if isinstance(x, float) else f"{'-':>{w}}"


def divergence_table(rows: list[dict]) -> None:
    """Mean displacement per region, per model. Flagged systems excluded from means."""
    models = sorted({r["model"] for r in rows})
    print("\n" + "=" * 78)
    print("DIVERGENCE FROM DFT  (A, mean over systems; flagged systems excluded)")
    print("=" * 78)
    print(f"{'model':17s}{'n':>4}{'flag':>5}{'overall':>10}"
          + "".join(f"{reg:>12s}" for reg in REGIONS))
    stats = {}
    for m in models:
        mine = [r for r in rows if r["model"] == m]
        good = [r for r in mine if not r["flags"]]
        row = {reg: mean([r.get(f"mean_{reg}") for r in good]) for reg in REGIONS}
        row["overall"] = mean([r["mean_displacement"] for r in good])
        stats[m] = row
        print(f"{m:17s}{len(good):>4}{len(mine) - len(good):>5}{fmt(row['overall'], 10)}"
              + "".join(fmt(row[reg], 12) for reg in REGIONS))

    print("\nclosest model per region:")
    for reg in ("overall",) + REGIONS:
        ranked = sorted(((v[reg], m) for m, v in stats.items() if v[reg] is not None))
        if ranked:
            best, second = ranked[0], ranked[1] if len(ranked) > 1 else (None, "-")
            margin = f"  (next {second[1]} +{second[0] - best[0]:.3f})" if second[0] else ""
            print(f"  {reg:12s} {best[1]:17s} {best[0]:.3f} A{margin}")
    return stats


def energy_table(rows: list[dict]) -> None:
    """Total-energy error vs DFT. Absolute values carry a per-atom offset, so the useful
    number is the SPREAD: a tight distribution means a constant offset that cancels from
    any energy difference, a wide one means the model is genuinely inconsistent."""
    print("\n" + "=" * 78)
    print("TOTAL-ENERGY ERROR  (e_mlip - e_dft, eV/atom; spread matters, not the mean)")
    print("=" * 78)
    print(f"{'model':17s}{'n':>4}{'mean':>10}{'std':>10}{'min':>10}{'max':>10}")
    for m in sorted({r["model"] for r in rows}):
        good = [r for r in rows if r["model"] == m and not r["flags"]]
        per = [(r["e_mlip"] - r["e_dft"]) / r["natoms"] for r in good
               if r.get("e_mlip") is not None and r.get("e_dft") is not None]
        if not per:
            continue
        mu = sum(per) / len(per)
        sd = math.sqrt(sum((x - mu) ** 2 for x in per) / len(per)) if len(per) > 1 else 0.0
        print(f"{m:17s}{len(per):>4}{mu:>10.4f}{sd:>10.4f}{min(per):>10.4f}{max(per):>10.4f}")


def spearman(a: list[float], b: list[float]) -> float | None:
    """Rank correlation, ties averaged. None when fewer than 3 points."""
    n = len(a)
    if n < 3:
        return None

    def ranks(v):
        order = sorted(range(n), key=lambda i: v[i])
        rk = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                rk[order[k]] = avg
            i = j + 1
        return rk

    ra, rb = ranks(a), ranks(b)
    ma, mb = sum(ra) / n, sum(rb) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    da = math.sqrt(sum((x - ma) ** 2 for x in ra))
    db = math.sqrt(sum((y - mb) ** 2 for y in rb))
    return num / (da * db) if da and db else None


def ranking_table(rows: list[dict], summaries: dict) -> None:
    """Does the model order adsorption the way DFT does?

    dE = E(system) - E(its own clean slab), for a FIXED adsorbate across surfaces. The
    gas-phase term is a per-species constant that shifts every surface equally, so it
    cancels from the ordering — which is the only ranking OC22 supports, since it ships
    no gas references. Never compare across adsorbate species.
    """
    print("\n" + "=" * 78)
    print("RANKING FIDELITY  (one adsorbate across surfaces; flagged systems excluded)")
    print("=" * 78)
    by_model_ads = collections.defaultdict(list)
    for r in rows:
        if r["flags"] or r.get("ads") is None:
            continue
        info = summaries.get(r["seedset"], {}).get(r["system"])
        if not info:
            continue
        base = next((x for x in rows
                     if x["model"] == r["model"] and x["seedset"] == r["seedset"]
                     and x["sid"] == info.get("slab_sid") and not x["flags"]), None)
        if base is None:
            continue
        by_model_ads[(r["model"], r["ads"])].append(
            (r["system"], r["e_mlip"] - base["e_mlip"], r["e_dft"] - base["e_dft"]))

    print(f"{'model':17s}{'ads':5s}{'n':>3}{'exact':>7}{'spearman':>10}"
          f"{'MAE(eV)':>10}{'DFT spread':>12}")
    thin = []
    for (m, ads), group in sorted(by_model_ads.items()):
        if len(group) < 2:
            # Say so rather than printing nothing: an empty table reads like a bug, and
            # the real cause is usually the validity filters leaving one surface standing.
            thin.append(f"{m}/{ads} (n={len(group)})")
            continue
        mlip = [g[1] for g in group]
        dft = [g[2] for g in group]
        exact = ([g[0] for g in sorted(group, key=lambda g: g[1])]
                 == [g[0] for g in sorted(group, key=lambda g: g[2])])
        rho = spearman(mlip, dft)
        mae = mean([abs(x - y) for x, y in zip(mlip, dft)])
        print(f"{m:17s}{ads:5s}{len(group):>3}{str(exact):>7}"
              f"{fmt(rho, 10) if rho is not None else '     n/a':>10}"
              f"{mae:>10.3f}{max(dft) - min(dft):>12.3f}")
    if thin:
        print(f"\n  too few surfaces to rank: {', '.join(thin)}")
        print("  (needs >=2 surfaces sharing one adsorbate, each with an unflagged "
              "clean slab; >=3 for spearman)")


def vacancy_pairs(rows: list[dict], summaries: dict) -> list[dict]:
    """Find (stoichiometric, reduced) clean-slab pairs on the same facet.

    A pair is two clean slabs of the same bulk and Miller index with the SAME metal count
    and DIFFERENT oxygen count — which makes their difference an oxygen vacancy rather
    than a different termination or a thicker slab. Compositions come from the seeded
    POSCARs, since the OC22 metadata does not carry them.

    Returns one dict per (model, facet, pair) with everything the energy needs:
    ``n_vac``, and the MLIP and DFT energies of both slabs.
    """
    from pymatgen.core import Structure

    comps: dict[tuple[str, int], tuple[str, int, int]] = {}
    for seedset, summary in summaries.items():
        for name, info in summary.items():
            if info.get("ads") is not None:
                continue
            p = REPO / "data" / "oc22" / seedset / f"{name}_dft_relaxed.vasp"
            if not p.exists():
                continue
            c = Structure.from_file(p).composition.get_el_amt_dict()
            metal = [e for e in c if e != "O"]
            if len(metal) != 1:
                continue
            comps[(seedset, info["sid"])] = (metal[0], int(c[metal[0]]), int(c["O"]))

    out = []
    by_model = collections.defaultdict(list)
    for r in rows:
        if r["flags"] or r.get("ads") is not None:
            continue
        key = (r["seedset"], r["sid"])
        if key not in comps:
            continue
        info = summaries[r["seedset"]][r["system"]]
        M, nM, nO = comps[key]
        by_model[(r["model"], r["seedset"], info["bulk_id"],
                  tuple(info["miller_index"]), M, nM)].append((nO, r))

    for (model, seedset, bulk, miller, M, nM), items in sorted(by_model.items()):
        if len({nO for nO, _ in items}) < 2:
            continue
        items.sort(key=lambda x: -x[0])
        (nO_hi, rich), (nO_lo, poor) = items[0], items[-1]
        out.append(dict(model=model, seedset=seedset, bulk=bulk, miller=miller,
                        metal=M, n_metal=nM, n_vac=nO_hi - nO_lo,
                        pristine=rich, defective=poor))
    return out


def ovfe_table(rows: list[dict], summaries: dict, mu_o: float | None) -> None:
    """Oxygen vacancy formation energy per vacancy, model vs DFT."""
    pairs = vacancy_pairs(rows, summaries)
    print("\n" + "=" * 78)
    print("OXYGEN VACANCY FORMATION ENERGY  (eV per vacancy, vs DFT)")
    print("=" * 78)
    if not pairs:
        print("  no (stoichiometric, reduced) slab pairs found — is chain3d in the sweep?")
        return
    # The ERROR column needs no oxygen reference at all. Applying one mu_O to both sides
    # makes it cancel exactly, leaving [dE_MLIP - dE_DFT]/n_vac — the surface term alone,
    # which is the quantity that says whether the model gets vacancy formation right.
    # Absolute E_vac does need mu_O, and only then are the columns filled.
    if mu_o is None:
        print("  --mu-o not given: absolute E_vac blank, but dE and error are")
        print("  reference-free and still valid.")
    print(f"{'model':17s}{'facet':20s}{'n_vac':>6}{'dE/vac MLIP':>13}{'dE/vac DFT':>12}"
          f"{'error':>9}{'E_vac MLIP':>12}{'E_vac DFT':>11}")
    for p in pairs:
        facet = f"{p['metal']}/{p['bulk']}{''.join(map(str, p['miller']))}"
        n = p["n_vac"]
        d_mlip = (p["defective"]["e_mlip"] - p["pristine"]["e_mlip"]) / n
        d_dft = (p["defective"]["e_dft"] - p["pristine"]["e_dft"]) / n
        e_mlip = ovfe(p["defective"]["e_mlip"], p["pristine"]["e_mlip"], mu_o, n)
        e_dft = ovfe(p["defective"]["e_dft"], p["pristine"]["e_dft"], mu_o, n)
        print(f"{p['model']:17s}{facet:20s}{n:>6}{d_mlip:>13.3f}{d_dft:>12.3f}"
              f"{d_mlip - d_dft:>9.3f}{fmt(e_mlip, 12)}{fmt(e_dft, 11)}")


def ovfe(e_defective: float, e_pristine: float,
         mu_o: float | None, n_vac: int) -> float | None:
    """Oxygen vacancy formation energy, per vacancy, in eV.

    ``e_defective`` and ``e_pristine`` are total energies of two slabs of the same facet
    that differ by ``n_vac`` oxygen atoms and nothing else. ``mu_o`` is the oxygen
    chemical potential, E(O2)/2 from the SAME calculator that produced the two energies —
    the per-atom offset only cancels when every term comes from one source, which is the
    same argument as ``energetics.vacancy_formation_energy``.

    Return ``None`` when ``mu_o`` is unavailable, so a missing gas reference reports
    nothing rather than something wrong.

    ``n_vac`` scales the chemical-potential term because that term pays for every oxygen
    removed, not one; the whole expression is then divided by ``n_vac`` so facets that
    differ by 16 vacancies and by 2 land on the same axis. Positive by construction, and
    smaller means the vacancy is easier to form.
    """
    if mu_o is None:
        return None
    return (e_defective - e_pristine + n_vac * mu_o) / n_vac


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("root", help="sweep directory holding <seedset>/<model>/divergence.jsonl")
    ap.add_argument("--seedset", help="restrict to one seed set")
    ap.add_argument("--plot", help="write a PNG summary here")
    ap.add_argument("--mu-o", type=float,
                    help="oxygen chemical potential E(O2)/2 in eV, from the model's own "
                         "gas reference in runs/_gas_refs; required for absolute E_vac")
    args = ap.parse_args()

    root = Path(args.root)
    if not root.is_absolute():
        root = Path.cwd() / root
    rows = load(root)
    if args.seedset:
        rows = [r for r in rows if r["seedset"] == args.seedset]
    if not rows:
        raise SystemExit(f"no divergence.jsonl found under {root}")

    seedsets = sorted({r["seedset"] for r in rows})
    summaries = {}
    for s in seedsets:
        p = REPO / "data" / "oc22" / s / "summary.json"
        summaries[s] = json.loads(p.read_text()) if p.exists() else {}

    print(f"{len(rows)} rows | {len({r['model'] for r in rows})} models "
          f"| seed sets: {', '.join(seedsets)}")
    flagged = collections.Counter(f for r in rows for f in r["flags"])
    if flagged:
        print(f"excluded by flag: {dict(flagged)}")

    divergence_table(rows)
    energy_table(rows)
    ranking_table(rows, summaries)
    ovfe_table(rows, summaries, args.mu_o)

    if args.plot:
        make_plot(rows, Path(args.plot))
        print(f"\nwrote {args.plot}")


def make_plot(rows: list[dict], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = sorted({r["model"] for r in rows})
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    width = 0.8 / len(REGIONS)
    for k, reg in enumerate(REGIONS):
        vals = [mean([r.get(f"mean_{reg}") for r in rows
                      if r["model"] == m and not r["flags"]]) or 0 for m in models]
        axes[0].bar([i + k * width for i in range(len(models))], vals, width, label=reg)
    axes[0].set_xticks([i + 0.4 - width / 2 for i in range(len(models))])
    axes[0].set_xticklabels(models, rotation=30, ha="right", fontsize=8)
    axes[0].set_ylabel("mean displacement from DFT (A)")
    axes[0].set_title("Divergence by region")
    axes[0].legend(fontsize=8)

    for m in models:
        good = [r for r in rows if r["model"] == m and not r["flags"]]
        per = [(r["e_mlip"] - r["e_dft"]) / r["natoms"] for r in good]
        if per:
            axes[1].scatter([m] * len(per), per, s=14, alpha=0.6)
    axes[1].set_ylabel("(e_mlip - e_dft) / atom  (eV)")
    axes[1].set_title("Energy error spread (offset cancels; width is the signal)")
    axes[1].tick_params(axis="x", rotation=30, labelsize=8)
    axes[1].axhline(0, color="k", lw=0.5)

    fig.tight_layout()
    fig.savefig(path, dpi=150)


if __name__ == "__main__":
    main()
