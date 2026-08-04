"""Dry-check every material before committing CPU to it — no relaxation, seconds per run.

    python scripts/validate_materials.py                  # everything that resolves offline
    python scripts/validate_materials.py mp-2657 mp-856
    python scripts/validate_materials.py --miller 101

Builds the whole structural chain (bulk -> slab -> vacancy -> adsorbate placement) with the
MLIP never invoked, and reports what each stage produced. Every failure it catches would
otherwise surface *hours* into a real run:

- the configured ``termination_index`` selecting a different physical surface for a
  different material (the silent one — you get plausible numbers for the wrong facet);
- a slab that is off-stoichiometry or polar, which makes its energy meaningless;
- zero vacancy sites (chain dies at stage 3) or zero adsorption sites (dies at stage 4).

VERDICT is OK / WARN / FAIL. FAIL means the chain cannot complete. WARN means it will run
but something needs a human decision before the numbers are trustworthy.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from pymatgen.core import Structure
from pymatgen.core.surface import SlabGenerator
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from oxide_workflow.config import ADSORBATE_FRAGMENTS, RunConfig
from oxide_workflow.stages import (
    adsorbate_candidates,
    apply_bottom_freeze,
    oxygen_vacancy_candidates,
)
from oxide_workflow.structures import available, get_structure, structure_provenance


def _terminations(bulk: Structure, cfg) -> list[dict]:
    """Every termination the configured facet offers, with the properties that matter.

    ``termination_index`` is an index into *this* list — a pymatgen ordering, not a
    literature quantity. Literature names a physical surface ("the stoichiometric
    bridging-oxygen (110)"); which integer that lands on is what this checks.
    """
    gen = SlabGenerator(
        bulk,
        miller_index=cfg.slab.miller_index,
        min_slab_size=cfg.slab.min_slab_size,
        min_vacuum_size=cfg.slab.min_vacuum_size,
        lll_reduce=cfg.slab.lll_reduce,
        center_slab=cfg.slab.center_slab,
    )
    bulk_o = bulk.composition.get_atomic_fraction("O")
    out = []
    for i, slab in enumerate(gen.get_slabs()):
        st = Structure.from_sites(slab.sites)
        o_frac = st.composition.get_atomic_fraction("O")
        out.append({
            "index": i,
            "formula": st.composition.reduced_formula,
            "n_sites": len(st),
            "o_fraction": round(o_frac, 4),
            "stoichiometric": abs(o_frac - bulk_o) < 1e-3,
            "polar": bool(slab.is_polar()),
            "symmetric": bool(slab.is_symmetric()),
        })
    return out


def check(identifier: str, cfg: RunConfig) -> dict:
    """Build the full structural chain for one material. Never raises — records instead."""
    res: dict = {"material": identifier, "verdict": "OK", "problems": [], "warnings": []}
    try:
        bulk = get_structure(identifier)
        res["provenance"] = structure_provenance(identifier)
    except Exception as exc:  # noqa: BLE001
        res["verdict"] = "FAIL"
        res["problems"].append(f"cannot load: {type(exc).__name__}: {exc}")
        return res

    res["formula"] = bulk.composition.reduced_formula
    res["n_bulk_sites"] = len(bulk)
    try:
        sga = SpacegroupAnalyzer(bulk)
        res["spacegroup"] = sga.get_space_group_symbol()
        res["spacegroup_number"] = sga.get_space_group_number()
    except Exception:  # noqa: BLE001
        res["spacegroup"] = "?"
    a, b, c = bulk.lattice.abc
    res["lattice_abc"] = [round(x, 4) for x in (a, b, c)]

    # ---- terminations ------------------------------------------------------------------
    try:
        terms = _terminations(bulk, cfg)
    except Exception as exc:  # noqa: BLE001
        res["verdict"] = "FAIL"
        res["problems"].append(f"slab generation failed: {type(exc).__name__}: {exc}")
        return res

    res["n_terminations"] = len(terms)
    res["terminations"] = terms
    idx = cfg.slab.termination_index
    if idx >= len(terms):
        res["verdict"] = "FAIL"
        res["problems"].append(
            f"termination_index={idx} out of range ({len(terms)} available)"
        )
        return res

    chosen = terms[idx]
    res["chosen_termination"] = chosen
    if not chosen["stoichiometric"]:
        res["verdict"] = "FAIL"
        res["problems"].append(
            f"termination {idx} is off-stoichiometry "
            f"(O fraction {chosen['o_fraction']} vs bulk) — its energy is not comparable"
        )
    if chosen["polar"]:
        res["verdict"] = "FAIL"
        res["problems"].append(f"termination {idx} is polar — divergent surface energy")
    if not chosen["symmetric"]:
        res["warnings"].append(
            f"termination {idx} is asymmetric; TiO2(110) picks the symmetric slab here, so "
            "this material's index may map to a different physical surface"
        )
    # Which index WOULD be the TiO2-like choice, if not this one.
    ideal = [t["index"] for t in terms if t["stoichiometric"] and not t["polar"] and t["symmetric"]]
    res["stoichiometric_symmetric_indices"] = ideal
    if ideal and idx not in ideal:
        res["warnings"].append(
            f"termination_index={idx} is not in {ideal}, which match TiO2(110)'s surface type"
        )

    # ---- slab -> vacancy -> adsorbate --------------------------------------------------
    try:
        slab = Structure.from_sites(
            SlabGenerator(
                bulk, miller_index=cfg.slab.miller_index,
                min_slab_size=cfg.slab.min_slab_size, min_vacuum_size=cfg.slab.min_vacuum_size,
                lll_reduce=cfg.slab.lll_reduce, center_slab=cfg.slab.center_slab,
            ).get_slabs()[idx].sites
        )
        nx, ny = cfg.slab.supercell
        if (nx, ny) != (1, 1):
            slab.make_supercell([nx, ny, 1])
        apply_bottom_freeze(slab, cfg.slab.freeze_bottom_fraction)
        res["n_slab_sites"] = len(slab)
    except Exception as exc:  # noqa: BLE001
        res["verdict"] = "FAIL"
        res["problems"].append(f"slab build failed: {type(exc).__name__}: {exc}")
        return res

    try:
        vacs = oxygen_vacancy_candidates(
            slab, freeze_bottom_fraction=cfg.slab.freeze_bottom_fraction,
            max_sites=cfg.slab.max_vacancy_sites,
        )
        res["n_vacancy_sites"] = len(vacs)
        res["vacancy_classes"] = sorted({v.site_id["symmetry_class"] for v in vacs})
    except Exception as exc:  # noqa: BLE001
        res["verdict"] = "FAIL"
        res["problems"].append(f"vacancy enumeration failed: {type(exc).__name__}: {exc}")
        return res

    try:
        ads = adsorbate_candidates(
            vacs[0].structure, cfg.adsorbate,
            freeze_bottom_fraction=cfg.slab.freeze_bottom_fraction,
        )
        by_type: dict[str, int] = {}
        for c in ads:
            k = c.site_id["symmetry_class"]
            by_type[k] = by_type.get(k, 0) + 1
        res["n_adsorbate_sites"] = len(ads)
        res["adsorbate_sites_by_type"] = by_type
    except Exception as exc:  # noqa: BLE001
        res["verdict"] = "FAIL"
        res["problems"].append(f"adsorbate placement failed: {type(exc).__name__}: {exc}")
        return res

    # A run's cost is dominated by these two counts (one relaxation each, per model).
    res["relaxations_per_model"] = 2 + res["n_vacancy_sites"] + res["n_adsorbate_sites"]

    if res["verdict"] == "OK" and res["warnings"]:
        res["verdict"] = "WARN"
    return res


def main(identifiers: list[str], cfg: RunConfig, out: Path | None) -> None:
    identifiers = identifiers or available()
    adsorbate = "".join(cfg.adsorbate.species)
    print(f"facet {''.join(map(str, cfg.slab.miller_index))}  "
          f"termination_index {cfg.slab.termination_index}  "
          f"supercell {cfg.slab.supercell}  adsorbate {adsorbate}")
    print(f"checking {len(identifiers)} material(s) — no relaxation is run\n")

    results = [check(i, cfg) for i in identifiers]

    hdr = (f"{'material':14s}{'formula':9s}{'spg':10s}{'terms':>6s}{'pick':>5s}"
           f"{'stoich':>7s}{'polar':>6s}{'symm':>6s}{'vac':>5s}{'ads':>5s}{'relax':>7s}  verdict")
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        if "chosen_termination" in r:
            t = r["chosen_termination"]
            row = (f"{r['material']:14s}{r.get('formula','?'):9s}{r.get('spacegroup','?'):10s}"
                   f"{r.get('n_terminations',0):>6d}{t['index']:>5d}"
                   f"{str(t['stoichiometric']):>7s}{str(t['polar']):>6s}{str(t['symmetric']):>6s}"
                   f"{r.get('n_vacancy_sites',0):>5d}{r.get('n_adsorbate_sites',0):>5d}"
                   f"{r.get('relaxations_per_model',0):>7d}  {r['verdict']}")
        else:
            row = (f"{r['material']:14s}{r.get('formula','?'):9s}{r.get('spacegroup','?'):10s}"
                   f"{'-':>6s}{'-':>5s}{'-':>7s}{'-':>6s}{'-':>6s}{'-':>5s}{'-':>5s}{'-':>7s}"
                   f"  {r['verdict']}")
        print(row)

    for r in results:
        if r["problems"] or r["warnings"]:
            print(f"\n[{r['material']}] {r['verdict']}")
            for p in r["problems"]:
                print(f"   FAIL  {p}")
            for w in r["warnings"]:
                print(f"   WARN  {w}")

    ok = [r for r in results if r["verdict"] == "OK"]
    warn = [r for r in results if r["verdict"] == "WARN"]
    fail = [r for r in results if r["verdict"] == "FAIL"]
    print(f"\n{len(ok)} OK, {len(warn)} WARN, {len(fail)} FAIL")

    runnable = ok + warn
    if runnable:
        total = sum(r.get("relaxations_per_model", 0) for r in runnable) * 2  # ref + 1 candidate
        print(f"\n~{total} relaxations for {len(runnable)} material(s) "
              f"(reference + 1 candidate, full_pipeline only)")
        print(f"at ~8 min/relaxation that is roughly {total * 8 / 60:.0f} h of CPU")
        print("\nrunnable: " + " ".join(r["material"] for r in runnable))

    if out:
        out.write_text(json.dumps(results, indent=2))
        print(f"\nfull detail: {out}")


if __name__ == "__main__":
    base = RunConfig()
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("materials", nargs="*", help="identifiers (default: everything available)")
    ap.add_argument("--miller", help='facet as "110" or "1,1,-1"')
    ap.add_argument("--termination-index", type=int, default=base.slab.termination_index)
    ap.add_argument("--adsorbate", choices=sorted(ADSORBATE_FRAGMENTS), default="CO")
    ap.add_argument("--out", type=Path, default=Path("runs/validation.json"))
    a = ap.parse_args()

    from oxide_workflow.pipeline import _cli_miller

    cfg = base
    if a.miller:
        cfg = replace(cfg, slab=replace(cfg.slab, miller_index=_cli_miller(a.miller)))
    cfg = replace(cfg, slab=replace(cfg.slab, termination_index=a.termination_index))
    species, coords = ADSORBATE_FRAGMENTS[a.adsorbate]
    cfg = replace(cfg, adsorbate=replace(cfg.adsorbate, species=species, coords=coords))

    a.out.parent.mkdir(parents=True, exist_ok=True)
    main(a.materials, cfg, a.out)
