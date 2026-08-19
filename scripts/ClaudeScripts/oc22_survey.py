"""Survey the OC22 metadata for rutile systems. Reads only; downloads nothing.

    python scripts/oc22_survey.py                     # the report
    python scripts/oc22_survey.py --miller 1 1 0      # restrict to one facet
    python scripts/oc22_survey.py --element Ti        # restrict to bulks containing Ti
    python scripts/oc22_survey.py --naive-control     # what string matching would do
    python scripts/oc22_survey.py --select --write    # apply the policy, save the subset

Prerequisite (9.3 MB, MD5 13dc06c6510346d8a7f614d5b26c8ffa):

    curl -o data/oc22/oc22_metadata.pkl \\
      https://dl.fbaipublicfiles.com/opencatalystproject/data/oc22/oc22_metadata.pkl

Why this runs before any bulk download: OC22's S2EF-Total split is a single 20 GB
tarball (71 GB unpacked) with no per-split option, so "grab it and look" is not
available. The metadata pickle is the whole 62 331-system index at 9.3 MB, and every
field we filter on lives in it. We decide what to fetch from here.

RUTILE IDENTIFICATION — two disjoint families, found two different ways.

1. Binary MO2 (``kind="binary"``). Composition does not imply structure type: OC22
   carries SnO2 as I4_1/amd, Pnnm and Pbcn, and its only pure TiO2 bulk (mp-554278) is
   C2/m. So these are matched on ``bulk_id`` against mp-ids that were resolved by
   *spacegroup* query (P4_2/mnm, #136) in ``scripts/core/fetch_rutiles.py``. That check
   already ran and is recorded in each file's ``query`` field, so no MP key is needed.
   Verified against MP's OPTIMADE endpoint: every MO2 bulk OC22 uses that is NOT in
   this set is genuinely a different polymorph, so the filter has no false negatives.
   Beware mp-550172 (SnO2, Pnnm) — CaCl2-type, a *slightly distorted* rutile that a
   composition filter takes and a spacegroup filter correctly rejects.

2. Ordered ternary MM'O4 (``kind="ternary"``). ``bulk_id`` is documented as a Materials
   Project ID, but 173 of them are not: they are author-assigned strings ending in
   ``-rutile`` (e.g. ``TiIrO4-rutile``), covering 4 318 systems. Cross-referencing
   against MP cannot see these at all — they have no mp-id to look up. These are 50/50
   cation-ordered rutiles, so they are rutile-structured but NOT the binary chemistry
   the rest of this repo models; ``kind`` keeps them separable.

   CONDITIONAL: the ternary label is OC22's own naming, taken on trust here because
   there is no mp-id to verify against. Confirm it against real geometry once slab
   structures are in hand.

Pure rutile TiO2 (mp-2657) does not appear in OC22 at all. Rutile *Ti* oxides do, as
232 systems across eight ternaries.
"""

from __future__ import annotations

import argparse
import collections
import json
import pickle
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
METADATA = REPO / "data" / "oc22" / "oc22_metadata.pkl"
STRUCTURES = REPO / "data" / "structures"
RUTILE_SPACEGROUP = 136
TERNARY_SUFFIX = "-rutile"


def rutile_mp_ids() -> dict[str, str]:
    """{mp_id: formula} for every local structure resolved by the spacegroup-136 query.

    Reads the provenance recorded at fetch time rather than re-deriving symmetry: the
    ``query`` string is the evidence that this id was picked *because* it is rutile.
    A structure fetched some other way (LaCoO3, added by mp-id) has no such string and
    is correctly skipped.
    """
    found = {}
    for path in sorted(STRUCTURES.glob("*.json")):
        meta = json.loads(path.read_text())
        if f"spacegroup_number={RUTILE_SPACEGROUP}" in meta.get("query", ""):
            found[meta["identifier"]] = meta.get("formula", "?")
    return found


def load_metadata() -> dict[int, dict]:
    if not METADATA.exists():
        raise SystemExit(f"missing {METADATA}\nsee the docstring for the curl command")
    return pickle.loads(METADATA.read_bytes())


def classify(meta: dict[int, dict], mp_rutiles: dict[str, str]) -> dict[int, dict]:
    """Rutile systems, each tagged with ``kind`` and a display ``formula``."""
    out = {}
    for sid, v in meta.items():
        bulk = v["bulk_id"]
        if bulk in mp_rutiles:
            out[sid] = {**v, "kind": "binary", "formula": mp_rutiles[bulk]}
        elif bulk.endswith(TERNARY_SUFFIX):
            out[sid] = {**v, "kind": "ternary", "formula": bulk[: -len(TERNARY_SUFFIX)]}
    return out


def ranking_sets(subset: dict[int, dict]) -> dict[str, list[int]]:
    """{ads_symbols: [sid, ...]} for nads=1 systems, grouped by adsorbate species.

    These are the only ranking tests OC22 can support without gas-phase references.
    Ranking *different* adsorbates on one slab would need a gas reference per species,
    which OC22 does not ship. Ranking *one* adsorbate across different surfaces does
    not: E_ads = E(system) - E(clean slab) - E(gas), and for a fixed species E(gas) is
    a constant that shifts every member of the set equally, so it cancels out of the
    ordering. The question becomes "which surface binds CO most strongly", which is
    answerable with no references at all.

    Every member's clean slab must be selected alongside it or E_ads is not computable.
    """
    groups: dict[str, list[int]] = collections.defaultdict(list)
    for sid, v in subset.items():
        if v["nads"] == 1:
            groups[v["ads_symbols"]].append(sid)
    return dict(groups)


def select_systems(subset: dict[int, dict]) -> list[int]:
    """Choose the 10-20 systems this validation run actually uses.

    ``subset`` is already filtered (by --miller / --element if given). Each value carries
    the OC22 fields plus ``kind`` ("binary" | "ternary") and ``formula``.

    Scale of the unfiltered pool: 4 370 rutile systems over 176 bulks — 52 binary
    (IrO2/RuO2/PbO2 only) and 4 318 ternary. 89% carry a traj_id. Restricting to the
    pipeline's own (1,1,0) facet leaves 83 systems over 26 bulks, of which 36 are clean
    slabs and 34 are nads=1.

    Returns a list of sids. Clean slabs referenced by any selected adsorbate system must
    appear in the returned list too, or that system's E_ads cannot be formed.
    """
    # TODO(human): implement the selection policy.
    raise NotImplementedError("select_systems: see TODO(human)")


def report(subset: dict[int, dict]) -> None:
    kinds = collections.Counter(v["kind"] for v in subset.values())
    bulks = {v["bulk_id"] for v in subset.values()}
    clean = {s for s, v in subset.items() if v["nads"] == 0}
    single = {s: v for s, v in subset.items() if v["nads"] == 1}

    print(f"\n{len(subset)} rutile systems over {len(bulks)} bulks "
          f"(binary {kinds['binary']}, ternary {kinds['ternary']})")
    print(f"  clean slabs {len(clean)} | nads=1 {len(single)} | "
          f"nads>1 {len(subset) - len(clean) - len(single)}")

    # traj_id gates the 34 GB raw-trajectory download: a system without one has no
    # DFT trajectory to seed from or diverge against, whatever else we fetch.
    print(f"  traj_id present {sum(1 for v in subset.values() if v.get('traj_id'))}"
          f"/{len(subset)}")
    print(f"\nmiller indices: "
          f"{dict(collections.Counter(str(v['miller_index']) for v in subset.values()).most_common(8))}")
    print(f"adsorbates (nads=1): "
          f"{dict(collections.Counter(v['ads_symbols'] for v in single.values()).most_common())}")

    print(f"\n{'bulk':18s}{'kind':9s}{'n':>4s}{'clean':>7s}{'nads=1':>8s}{'traj':>6s}")
    rows = sorted(bulks, key=lambda b: -sum(1 for v in subset.values() if v["bulk_id"] == b))
    for bulk in rows[:15]:
        vs = [v for v in subset.values() if v["bulk_id"] == bulk]
        print(f"{bulk:18s}{vs[0]['kind']:9s}{len(vs):>4d}"
              f"{sum(1 for v in vs if v['nads'] == 0):>7d}"
              f"{sum(1 for v in vs if v['nads'] == 1):>8d}"
              f"{sum(1 for v in vs if v.get('traj_id')):>6d}")
    if len(rows) > 15:
        print(f"  ... and {len(rows) - 15} more bulks")

    sets = ranking_sets(subset)
    print(f"\nranking sets (same adsorbate across surfaces, gas ref cancels):")
    for ads, sids in sorted(sets.items(), key=lambda kv: -len(kv[1])):
        usable = sum(1 for s in sids
                     if subset[s]["slab_sid"] in clean and subset[s].get("traj_id"))
        print(f"  {ads:5s} {len(sids):4d} surfaces  ({usable} with clean slab in set AND traj_id)")


def naive_control(meta: dict[int, dict], mp_rutiles: dict[str, str],
                  subset: dict[int, dict]) -> None:
    """What a bulk_symbols string match would have selected, and why that is wrong.

    It fails in both directions at once, which is the point: it sweeps in wrong-polymorph
    bulks *and* misses the ternary rutiles, whose bulk_symbols never look like "MO2".
    """
    formulas = set(mp_rutiles.values())
    hits = collections.defaultdict(set)
    for v in meta.values():
        # bulk_symbols is a supercell formula (Ti8O16), so compare reduced composition
        counts: collections.Counter = collections.Counter()
        sym, num = "", ""
        for ch in v["bulk_symbols"] + "$":
            if ch.isupper() or ch == "$":
                if sym:
                    counts[sym] += int(num or 1)
                sym, num = ch, ""
            elif ch.islower():
                sym += ch
            else:
                num += ch
        if len(counts) == 2 and "O" in counts:
            cation = next(e for e in counts if e != "O")
            if counts["O"] == 2 * counts[cation] and f"{cation}O2" in formulas:
                hits[f"{cation}O2"].add(v["bulk_id"])
    print("\nnaive bulk_symbols match — distinct bulk_ids per formula:")
    for formula in sorted(hits):
        ids = hits[formula]
        wrong = len(ids - set(mp_rutiles))
        element = formula[:-2]
        missed = sum(1 for v in subset.values()
                     if v["kind"] == "ternary" and v["formula"].startswith(element))
        print(f"  {formula:6s} {len(ids):3d} bulk_ids -> {wrong} wrong-polymorph swept in, "
              f"{missed} ternary-rutile systems missed")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--miller", nargs=3, type=int, metavar=("H", "K", "L"),
                    help="restrict to one facet, e.g. --miller 1 1 0")
    ap.add_argument("--element", help="restrict to bulks containing this element, e.g. Ti")
    ap.add_argument("--kind", choices=["binary", "ternary"], help="restrict to one family")
    ap.add_argument("--naive-control", action="store_true",
                    help="show what composition string matching would have selected")
    ap.add_argument("--select", action="store_true",
                    help="apply select_systems() and report the chosen subset")
    ap.add_argument("--write", action="store_true",
                    help="write the subset to data/oc22/rutile_systems.json")
    args = ap.parse_args()

    meta = load_metadata()
    mp_rutiles = rutile_mp_ids()
    if not mp_rutiles:
        raise SystemExit(f"no spacegroup-{RUTILE_SPACEGROUP} structures in {STRUCTURES}; "
                         "run scripts/core/fetch_rutiles.py first")
    print(f"loaded {len(meta)} OC22 systems")
    subset = classify(meta, mp_rutiles)
    full = subset

    if args.kind:
        subset = {s: v for s, v in subset.items() if v["kind"] == args.kind}
    if args.miller:
        want = tuple(args.miller)
        subset = {s: v for s, v in subset.items() if tuple(v["miller_index"]) == want}
    if args.element:
        # match whole element symbols: "Ti" must not match "Tl", and formula is the
        # bulk label (TiIrO4) rather than a supercell count string
        el = args.element
        subset = {s: v for s, v in subset.items()
                  if any(v["formula"][i:i + len(el)] == el
                         and not v["formula"][i + len(el):i + len(el) + 1].islower()
                         for i in range(len(v["formula"])) if v["formula"][i].isupper())}
    if not subset:
        raise SystemExit("no systems match those filters")

    report(subset)
    if args.naive_control:
        naive_control(meta, mp_rutiles, full)

    if args.select:
        chosen = select_systems(subset)
        picked = {s: subset[s] for s in chosen}
        n_clean = sum(1 for v in picked.values() if v["nads"] == 0)
        print(f"\nselected {len(chosen)} systems "
              f"({n_clean} clean slabs, {len(chosen) - n_clean} adsorbate systems)")
        # A selected adsorbate system whose clean slab was left out has no E_ads.
        orphans = [s for s, v in picked.items()
                   if v["nads"] > 0 and v["slab_sid"] not in picked]
        print(f"orphaned (clean slab not selected): {len(orphans)}"
              f"{' -> ' + str(orphans) if orphans else ''}")
        no_traj = [s for s, v in picked.items() if not v.get("traj_id")]
        print(f"selected without traj_id: {len(no_traj)}")
        for ads, sids in sorted(ranking_sets(picked).items(), key=lambda kv: -len(kv[1])):
            print(f"  ranking set {ads:5s} {len(sids)} surfaces "
                  f"{sorted(picked[s]['formula'] for s in sids)}")
        subset = picked

    if args.write:
        out = METADATA.parent / "rutile_systems.json"
        out.write_text(json.dumps(
            {str(s): {**v, "miller_index": list(v["miller_index"])}
             for s, v in subset.items()}, indent=2, sort_keys=True))
        print(f"\nwrote {len(subset)} systems -> {out.relative_to(REPO)}")


if __name__ == "__main__":
    main()
