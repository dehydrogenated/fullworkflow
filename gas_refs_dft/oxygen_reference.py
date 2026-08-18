#!/usr/bin/env python3
"""
The Mixed-ML test for oxygen, in one table.

Mixed-ML takes E_sys and E_slab from the MLIP but E_gas from DFT. Whether that
is an improvement or a category error comes down to one number per model:

    delta = mu_O^MLIP - mu_O^DFT

If delta is small, the model's gas-phase scale already agrees with PBE and
Mixed-ML buys you nothing. If it is large, Mixed-ML is either the fix or the
symptom -- large means the model was trained on a different functional (RPBE for
the OC20/OC22 heads), and swapping in a PBE gas term does not repair that, it
just moves the mismatch. Only the PBE-scale heads (MACE-mh1-omat, MACE-mh1-mp,
UMA-omat) are legitimate candidates for this correction.

Two independent routes to mu_O, both computable from the same three molecules:

    direct :  1/2 * E(O2)
    cycle  :  E(H2O) - E(H2) + WATER_FORMATION_ENTHALPY_EXP

Their difference on the DFT side is this settings' PBE O2 overbinding, measured
rather than quoted. Their difference on the ML side is the same thing for the
model. Reporting all four numbers is the point of this script.

Run on the login node (needs the `oxw` env), after check_gas_refs.sh is clean:
    python3 oxygen_reference.py --model MACE-mh1-omat
"""
import argparse
import os
import sys

from ase.io import read

# One source of truth for the cycle constant. energy_schemes.py in this folder
# carries 2.46; oxide_workflow carries 2.51 with a citation, and the DFT and ML
# sides must use the same value or the comparison below is meaningless.
from oxide_workflow.energetics import WATER_FORMATION_ENTHALPY_EXP

GASDIR = "gas_references"
SCHEMES = ("direct", "cycle")


def oxygen_reference(e_o2, e_h2o, e_h2, scheme="cycle"):
    """Energy of one O atom (mu_O), in eV, on whatever scale the inputs came from.

    All three energies must come from the SAME source -- all DFT, or all from one
    MLIP head. Mixing them is exactly the error this script exists to measure.

    scheme="direct" -> 1/2 * E(O2). Simple, and carries PBE's O2 overbinding.
    scheme="cycle"  -> E(H2O) - E(H2) + WATER_FORMATION_ENTHALPY_EXP. Substitutes
                       the badly-described O2 for two molecules PBE handles well,
                       absorbing the difference into an experimental reaction
                       energy (so it stays on the input scale).
    """
    if scheme == "direct":
        return 0.5 * e_o2
    if scheme == "cycle":
        # The constant is defined for H2 + 1/2 O2 -> H2O, so this expression
        # already yields one O atom -- no halving, unlike the direct route.
        return e_h2o - e_h2 + WATER_FORMATION_ENTHALPY_EXP
    # Not a silent default: a typo'd scheme returning the direct value would look
    # like a plausible number while quietly changing what is being measured.
    raise ValueError(f"unknown scheme {scheme!r}; expected one of {SCHEMES}")


def read_dft_energies():
    """Final total energy per species from the completed VASP runs."""
    out = {}
    for name in ("O2", "H2O", "H2"):
        path = os.path.join(GASDIR, name, "OUTCAR")
        if not os.path.exists(path):
            sys.exit(f"missing {path} -- run the SLURM job first.")
        out[name] = read(path, index=-1).get_potential_energy()
    return out


def read_ml_energies(model):
    """Same three molecules, same DFT-relaxed geometries, evaluated by the MLIP.

    Single-point on the DFT geometry, not a fresh MLIP relaxation: relaxing again
    folds a geometry error into what is supposed to be a pure energy-scale
    comparison.
    """
    from pymatgen.io.ase import AseAtomsAdaptor

    from oxide_workflow.backends import REGISTRY, relax

    backend = REGISTRY[model]
    out = {}
    for name in ("O2", "H2O", "H2"):
        atoms = read(os.path.join(GASDIR, name, "OUTCAR"), index=-1)
        structure = AseAtomsAdaptor.get_structure(atoms)
        # max_steps=0 makes this a single-point evaluation; fmax is then unused
        # but still required, since relax() forces every caller to state a target.
        out[name] = relax(structure, backend, fmax=0.02, max_steps=0,
                          optimizer="FIRE").energy
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="MACE-mh1-omat",
                    help="PBE-scale heads only; an OC20/OC22 head measures a "
                         "functional mismatch, not a Mixed-ML correction")
    args = ap.parse_args()

    dft = read_dft_energies()
    ml = read_ml_energies(args.model)

    print(f"\n{'species':8} {'E_DFT (eV)':>14} {'E_ML (eV)':>14} {'diff (eV)':>12}")
    print("-" * 52)
    for name in ("O2", "H2O", "H2"):
        print(f"{name:8} {dft[name]:14.4f} {ml[name]:14.4f} {ml[name] - dft[name]:12.4f}")

    print(f"\nmu_O per scheme   (cycle constant = {WATER_FORMATION_ENTHALPY_EXP} eV)")
    print(f"{'scheme':8} {'DFT (eV)':>14} {'ML (eV)':>14} {'delta (eV)':>12}")
    print("-" * 52)
    mu = {}
    for scheme in SCHEMES:
        mu_dft = oxygen_reference(dft["O2"], dft["H2O"], dft["H2"], scheme)
        mu_ml = oxygen_reference(ml["O2"], ml["H2O"], ml["H2"], scheme)
        mu[scheme] = (mu_dft, mu_ml)
        print(f"{scheme:8} {mu_dft:14.4f} {mu_ml:14.4f} {mu_ml - mu_dft:12.4f}")

    print(f"\ndirect - cycle:  DFT {mu['direct'][0] - mu['cycle'][0]:+.3f} eV"
          f"   ML {mu['direct'][1] - mu['cycle'][1]:+.3f} eV")
    print("  On the DFT side this gap is PBE's O2 overbinding at these settings.")
    print("  If the ML gap is much larger, the model's O2 is the problem, not its scale.")

    d = abs(mu["cycle"][1] - mu["cycle"][0])
    print(f"\nverdict for {args.model}, cycle route: |delta| = {d:.3f} eV")
    if d < 0.10:
        print("  Full-ML is fine. Mixed-ML buys nothing and adds a scale to reconcile.")
    elif d < 0.50:
        print("  Mixed-ML is worth testing, or Full-ML with delta subtracted per species.")
    else:
        print("  Too large for a reference offset -- check the head is PBE-scale.")


if __name__ == "__main__":
    main()
