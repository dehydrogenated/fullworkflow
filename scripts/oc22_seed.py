"""Extract OC22 systems from the IS2RE-Total LMDBs into seed + reference geometries.

    # NOTE: runs in the fairchem env — needs torch to rehydrate the pickled tensors
    /opt/anaconda3/envs/fairchem/bin/python scripts/oc22_seed.py --sids 42982 44149

Prerequisites (~530 MB, both gitignored; MD5s are the published ones):

    curl -o data/oc22/oc22_metadata.pkl \\
      https://dl.fbaipublicfiles.com/opencatalystproject/data/oc22/oc22_metadata.pkl
    curl -o data/oc22/is2res_lmdbs.tar.gz \\
      https://dl.fbaipublicfiles.com/opencatalystproject/data/oc22/is2res_total_train_val_test_lmdbs.tar.gz
    tar xzf data/oc22/is2res_lmdbs.tar.gz -C data/oc22/

WHY IS2RE AND NOT S2EF. Seeded divergence needs exactly two frames — the initial
geometry to seed from, and the DFT-converged geometry to compare against. IS2RE-Total
stores precisely that pair (`pos`, `pos_relaxed`) in 109 MB. S2EF-Total is 20 GB for
every intermediate frame, and the raw trajectories are 34 GB. The ColabFit HuggingFace
mirrors of both are unusable for this: they flatten trajectories into independent
per-frame rows and keep no frame index, so "frame 0" is not recoverable.

The cost is that IS2RE carries NO FORCES. Force parity — evaluating stored DFT forces
against the calculator to isolate plumbing from referencing — cannot be done from this
file and needs S2EF.

`fixed` is all-zero throughout OC22 (verified, not assumed): they relaxed unconstrained.
The POSCARs written here therefore carry no selective_dynamics, which is what ASE reads
as "no constraints" — matching their protocol. Relax with --freeze 0 equivalently.
"""

from __future__ import annotations

import argparse
import io
import json
import pickle
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
LMDB_ROOT = (REPO / "data/oc22/is2res_total_train_val_test_lmdbs/data/oc22/is2re-total")
METADATA = REPO / "data/oc22/oc22_metadata.pkl"
SPLITS = ("train", "val_id", "val_ood", "test_id", "test_ood")


class DataShim:
    """Stand-in for torch_geometric ``Data``.

    The LMDB values were pickled by PyG 1.x; modern PyG refuses to rehydrate them
    ("created by an older version of PyG"), and pinning an ancient PyG next to torch 2.x
    is not worth it. The pickle stream only needs *some* class to attach its state to,
    and the state dict is the data — PyG's accessors were never used. Only torch is
    needed, for the tensor rebuild ops the stream calls directly.
    """

    def __setstate__(self, state):
        if isinstance(state, dict) and set(state) == {"_store"}:
            state = state["_store"]
        self.__dict__.update(state or {})


class _Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        return DataShim if module.startswith("torch_geometric") else super().find_class(module, name)


def find_sids(wanted: set[int], splits=SPLITS) -> dict[int, tuple]:
    """{sid: (split, lmdb_name, DataShim)}; stops as soon as every sid is found."""
    import lmdb

    found: dict[int, tuple] = {}
    for split in splits:
        for path in sorted((LMDB_ROOT / split).glob("*.lmdb")):
            env = lmdb.open(str(path), subdir=False, readonly=True, lock=False,
                            readahead=False, meminit=False)
            try:
                # close only after the transaction has exited, or its __exit__ aborts
                # against a dropped handle
                with env.begin() as txn:
                    for key, _ in txn.cursor():
                        if not key.decode(errors="replace").isdigit():
                            continue
                        obj = _Unpickler(io.BytesIO(txn.get(key))).load()
                        sid = getattr(obj, "sid", None)
                        if sid is not None and int(sid) in wanted and int(sid) not in found:
                            found[int(sid)] = (split, path.name, obj)
            finally:
                env.close()
            if len(found) == len(wanted):
                return found
    return found


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sids", nargs="+", type=int, required=True, help="OC22 system ids")
    ap.add_argument("--out", default="data/oc22/oh_systems", help="output directory")
    args = ap.parse_args()

    from ase import Atoms
    from ase.io import write

    meta = pickle.loads(METADATA.read_bytes())
    out = REPO / args.out
    out.mkdir(parents=True, exist_ok=True)

    wanted = set(args.sids)
    print(f"scanning {LMDB_ROOT} for {len(wanted)} sids ...")
    found = find_sids(wanted)
    missing = wanted - set(found)
    print(f"found {len(found)}/{len(wanted)}")

    arrays, summary = {}, {}
    for sid in sorted(found):
        split, fname, obj = found[sid]
        d, m = obj.__dict__, meta[sid]
        # Test-split entries exist but have their labels withheld — no reference to
        # diverge against, so they are not usable seeds.
        if d.get("pos_relaxed") is None or d.get("y_relaxed") is None:
            print(f"  sid {sid}: in {split}, LABELS WITHHELD -> skipped")
            continue

        Z = d["atomic_numbers"].numpy().astype(int)
        cell = d["cell"].numpy().reshape(3, 3)
        pos0, posR = d["pos"].numpy(), d["pos_relaxed"].numpy()
        tags, fixed = d["tags"].numpy().astype(int), d["fixed"].numpy().astype(int)
        if fixed.any():
            print(f"  sid {sid}: WARNING {fixed.sum()} atoms flagged fixed — "
                  "the unconstrained assumption does not hold here")

        name = (f"{sid}_{m['bulk_id'].replace('mp-', 'mp')}_"
                f"{''.join(map(str, m['miller_index']))}_{m.get('ads_symbols', 'clean')}")
        for tag, P in (("initial", pos0), ("dft_relaxed", posR)):
            write(out / f"{name}_{tag}.vasp",
                  Atoms(numbers=Z, positions=P, cell=cell, pbc=True),
                  format="vasp", sort=False)  # sort=False keeps OC22's atom order
        for key, arr in (("tags", tags), ("cell", cell), ("pos0", pos0), ("posR", posR)):
            arrays[f"{name}_{key}"] = arr
        summary[name] = {
            "sid": sid, "split": split, "lmdb": fname, "natoms": int(len(Z)),
            "nads": m["nads"], "ads": m.get("ads_symbols"), "bulk_id": m["bulk_id"],
            # the clean slab this sits on; the only reliable way to pair an adsorbate
            # system with its reference, since the two carry different sids
            "slab_sid": m.get("slab_sid"),
            "miller_index": list(m["miller_index"]), "y_relaxed": float(d["y_relaxed"]),
            "n_fixed": int(fixed.sum()),
            "tag_counts": {int(k): int(v) for k, v in zip(*np.unique(tags, return_counts=True))},
        }
        print(f"  {name}: {len(Z)} atoms, split={split}, "
              f"tags={summary[name]['tag_counts']}, y_relaxed={d['y_relaxed']:.4f}")

    np.savez(out / "reference.npz", **arrays)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(f"\nwrote {2 * len(summary)} POSCARs + reference.npz + summary.json -> {out}")
    if missing:
        print(f"not found in any split: {sorted(missing)}")


if __name__ == "__main__":
    main()
