"""Gas-phase reference energy comparison across models -- the raw numbers behind every
E_ads in the MO2 benchmark, surfaced directly rather than left buried in runs/_gas_refs/.

Five fragments, each just its own direct relaxed isolated-species energy -- "normal"
prediction, no correction -- since only O2's specific GGA overbinding pathology needs
special handling, not H/CO/H2O/OH: O (bare atom), H (bare atom), H2O, CO, OH.

Oxygen additionally gets its corrected reference (nothing else does): a bare isolated O
atom's own energy is unreliable open-shell DFT/MLIP territory, so instead of using E(O)
directly, O*'s adsorption reference is solved via the same H2/H2O thermodynamic cycle
oxygen_chemical_potential_corrected() uses (Kowalski, Meyer & Marx 2009):

    O_ref_corrected = E(H2O) - E(H2) + WATER_FORMATION_ENTHALPY_EXP

E(H2) is computed to feed that formula but isn't shown as its own column -- it was never
one of the fragments asked for, just an ingredient for oxygen's correction specifically.
O_ref_raw (E(H2O)-E(H2), no +DeltaHf) is kept alongside it since that's the literal
Comer et al. convention mo2_adsorption_benchmark.py's O* leg still uses for the literature
comparison column.

Cheap: every fragment here is a handful of atoms, seconds per relaxation, and
gas_reference_energy() already caches per (model, fragment, fmax) so a repeat run costs
nothing. Runs locally for whatever models have a working env; skips (not fatal) any model
whose backend/env isn't available here -- run again on Sockeye for the rest.

    python scripts/studies/gas_reference_report.py
    python scripts/studies/gas_reference_report.py --models Orb-v2 CHGNet-0.3.0
    python scripts/studies/gas_reference_report.py --out runs/gas_refs_report.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from oxide_workflow import pipeline  # noqa: E402
from oxide_workflow.backends import REGISTRY, get_backend  # noqa: E402
from oxide_workflow.config import ADSORBATE_FRAGMENTS, RunConfig  # noqa: E402
from oxide_workflow.energetics import WATER_FORMATION_ENTHALPY_EXP, gas_reference_energy  # noqa: E402

FRAGMENTS = ["O", "H", "H2O", "CO", "OH"]
DEFAULT_MODELS = [
    "CHGNet-0.3.0", "Orb-v2",
    "MACE-mh1-omat", "MACE-mh1-oc20", "MACE-mh1-matpes",
    "UMA-omat", "UMA-oc20", "UMA-oc22",
    "UMA-M-omat", "UMA-M-oc20",
    "SevenNet-omni-omat24", "SevenNet-omni-oc20", "SevenNet-omni-oc22", "SevenNet-omni-mpa",
    "eSEN-30M-OAM",
]  # matches the 15-model sweep roster (sockeye_mo2_sweep_{3d,4d,5d}.slurm)


def one_model(model: str, fmax: float) -> dict:
    backend = get_backend(model)
    cfg = RunConfig(relax=RunConfig().relax)
    from dataclasses import replace
    cfg = replace(cfg, relax=replace(cfg.relax, fmax=fmax))

    row = {"model": model}
    energies = {}
    for frag in FRAGMENTS:
        species, coords = ADSORBATE_FRAGMENTS[frag]
        e = gas_reference_energy(backend, cfg, pipeline.relax, species=species, coords=coords)
        energies[frag] = e
        row[f"E_{frag}_eV"] = e

    # H2 only as an ingredient for the two literature-matching references below -- not one
    # of the requested fragments, so not its own column.
    h2_species, h2_coords = ADSORBATE_FRAGMENTS["H2"]
    e_h2 = gas_reference_energy(backend, cfg, pipeline.relax, species=h2_species, coords=h2_coords)

    # O*: Comer et al. Table S3 reaction "H2O(g) - H2(g) + * -> O*". Corrected version
    # routes around GGA's O2-specific overbinding via the same H2/H2O thermodynamic cycle
    # oxygen_chemical_potential_corrected() uses (Kowalski, Meyer & Marx 2009).
    row["O_ref_raw_eV"] = energies["H2O"] - e_h2
    row["O_ref_corrected_eV"] = row["O_ref_raw_eV"] + WATER_FORMATION_ENTHALPY_EXP

    # OH*: Comer et al. Table S3 reaction "H2O(g) - 0.5H2(g) + * -> HO*" -- same water-
    # splitting convention as O*, just half the H2. No O2 term anywhere in this formula, so
    # there is nothing for the correction to fix; corrected == raw here, kept as its own
    # column only for schema parity with O_ref_corrected, not because it differs.
    row["OH_ref_raw_eV"] = energies["H2O"] - 0.5 * e_h2
    row["OH_ref_corrected_eV"] = row["OH_ref_raw_eV"]
    return row


def main(models: list[str], fmax: float, out: Path | None) -> None:
    rows = []
    for model in models:
        print(f"[{model}]", flush=True)
        try:
            rows.append(one_model(model, fmax))
        except Exception as e:
            print(f"  SKIPPED: {str(e)[:200]}", flush=True)

    if not rows:
        print("\nno models succeeded (no working local env for any of these?)")
        return

    cols = ["model"] + [f"E_{f}_eV" for f in FRAGMENTS] + [
        "O_ref_raw_eV", "O_ref_corrected_eV", "OH_ref_raw_eV", "OH_ref_corrected_eV",
    ]
    labels = [c.replace("_eV", "").replace("E_", "") for c in cols[1:]]
    w = max(10, max(len(lbl) for lbl in labels) + 2)
    header = f"{'model':22s}" + "".join(f"{lbl:>{w}s}" for lbl in labels)
    print(f"\n{header}")
    print("-" * len(header))
    for row in rows:
        print(f"{row['model']:22s}" + "".join(f"{row[c]:>{w}.3f}" for c in cols[1:]))

    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=cols)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nwrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--models", nargs="+", choices=list(REGISTRY), default=DEFAULT_MODELS)
    ap.add_argument("--fmax", type=float, default=0.02)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    main(a.models, a.fmax, a.out)
