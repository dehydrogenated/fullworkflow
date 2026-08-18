#!/usr/bin/env python3
"""
Generate VASP inputs for gas-phase reference molecules using Materials Project
(MPtrj / OMat24 / MACE-MP) conventions.

Why these settings: MACE-MP-0/MPA, SevenNet-0, CHGNet, eSEN-MPtrj, Orb and the
OMat24-trained models all predict *raw, uncorrected* VASP PBE(+U) total energies
on the Materials Project scale. For a Mixed-ML adsorption energy to be
meaningful, E_gas^DFT must sit on that same scale.

Run:  python make_gas_refs.py
Then: copy your POTCARs in (see README) and run VASP in each folder.
"""
import os
import sys

import numpy as np
from ase.build import molecule
from ase.io import write

OUTDIR = "gas_references"

# Asymmetric box: breaks accidental degeneracies and avoids image interaction.
# ~12 A of vacuum in every direction is plenty for neutral closed-shell species.
BOX = (15.0, 15.5, 16.0)

# name -> (ASE g2 key, NUPDOWN, comment)
#
# The oxygen set: everything needed to reference an O atom on a DFT scale, and
# nothing else. Both oxygen routes fall out of these three --
#   direct :  1/2 * E(O2)
#   cycle  :  E(H2O) - E(H2) + dG      (Norskov; avoids PBE's O2 overbinding)
# -- so running all three prices the overbinding error instead of quoting it.
#
#   NUPDOWN fixes the total spin. Getting this wrong is THE classic gas-reference
#   bug: without NUPDOWN=2, VASP will happily converge O2 to the singlet and hand
#   you an energy that is ~1 eV too high.
SPECIES = {
    "O2":    ("O2",       2, "TRIPLET ground state - NUPDOWN=2 is mandatory"),
    "H2O":   ("H2O",      0, "water - MvK product, and the numerator of the O reference cycle"),
    "H2":    ("H2",       0, "hydrogen - denominator of the cycle; never appears in the reaction itself"),
}

# Propane-ODH species. Not needed for an oxygen reference, so they are not built
# by default -- their generated folders live in ../gas_references_parked/.
# Rebuild them with:  python make_gas_refs.py --all
PARKED = {
    "C3H8":  ("C3H8",     0, "propane - ODH reactant"),
    "C3H6":  ("C3H6_Cs",  0, "propene (Cs, propylene) - ODH product. NOT C3H6_D3h (cyclopropane)"),
    "CO2":   ("CO2",      0, "total oxidation product"),
    "CO":    ("CO",       0, "optional"),
    "C3H7":  ("C3H7",     1, "propyl RADICAL, doublet - only if you reference the radical directly"),
}

INCAR_TEMPLATE = """SYSTEM = {name} gas-phase reference (MP conventions)
! ---- matched to Materials Project / MPtrj static+relax settings ----
PREC   = Accurate
ENCUT  = 520          ! MP standard. MUST match your slab calculations.
EDIFF  = 1E-6
EDIFFG = -0.02
ALGO   = Fast
NELM   = 200
NELMIN = 6
LASPH  = .TRUE.       ! MP default
LREAL  = .FALSE.      ! exact projection - cheap for a molecule, avoids LREAL=Auto noise
LWAVE  = .FALSE.
LCHARG = .FALSE.

! ---- relaxation: ions only, NEVER the cell ----
IBRION = 2
ISIF   = 2            ! do NOT use ISIF=3 - you would collapse the vacuum box
NSW    = 200

! ---- molecule-in-a-box overrides of the MP solid-state defaults ----
ISMEAR = 0            ! MP uses -5 (tetrahedron) for solids; illegal at a single k-point
SIGMA  = 0.01
ISYM   = 0            ! avoid symmetry-locking the molecule into a saddle point

! ---- spin ----
ISPIN   = 2
NUPDOWN = {nupdown}          ! {spin_note}
MAGMOM  = {magmom}

! ---- NO Hubbard U ----
! MP applies +U only to transition metals in oxides/fluorides. These molecules
! contain none, so LDAU is off. See README for what this means for E_vac(O).
LDAU = .FALSE.
"""

KPOINTS_TEMPLATE = """Gamma-only - isolated molecule in a large box
0
Gamma
1 1 1
0 0 0
"""


def main():
    species = dict(SPECIES)
    if "--all" in sys.argv:
        species.update(PARKED)
    os.makedirs(OUTDIR, exist_ok=True)
    rows = []
    for name, (g2key, nupdown, comment) in species.items():
        atoms = molecule(g2key)
        atoms.set_cell(BOX)
        atoms.set_pbc(True)
        atoms.center()

        d = os.path.join(OUTDIR, name)
        os.makedirs(d, exist_ok=True)

        # Sort atoms by chemical symbol OURSELVES, then write with sort=False.
        # Relying on the writer's internal sort and separately deriving the
        # element order from the unsorted Atoms object is how you end up with a
        # POTCAR whose order silently disagrees with the POSCAR - VASP will not
        # complain, it will just use the wrong potential for every atom.
        order = np.argsort(atoms.get_chemical_symbols(), kind="stable")
        atoms = atoms[order]

        write(os.path.join(d, "POSCAR"), atoms, format="vasp",
              direct=True, sort=False, vasp5=True)

        spin_note = {0: "closed shell / singlet",
                     1: "doublet radical",
                     2: "TRIPLET - do not remove"}[nupdown]
        # small non-zero starting moment helps the spin-polarized species converge
        init = 1.0 if nupdown > 0 else 0.0
        magmom = f"{len(atoms)}*{init}"

        with open(os.path.join(d, "INCAR"), "w") as f:
            f.write(INCAR_TEMPLATE.format(name=name, nupdown=nupdown,
                                          spin_note=spin_note, magmom=magmom))
        with open(os.path.join(d, "KPOINTS"), "w") as f:
            f.write(KPOINTS_TEMPLATE)

        # which POTCARs to cat together, in POSCAR species order
        symbols = []
        for s in atoms.get_chemical_symbols():
            if s not in symbols:
                symbols.append(s)
        with open(os.path.join(d, "POTCAR_ORDER.txt"), "w") as f:
            f.write("PBE_54 POTCARs, concatenated in this order:\n")
            f.write("  " + "  ".join(symbols) + "\n")
            f.write("MP uses the plain H, C, O PAW_PBE.54 potentials for these elements.\n")

        rows.append((name, len(atoms), nupdown, comment))

    print(f"{'folder':8} {'natoms':>6} {'NUPDOWN':>7}  note")
    print("-" * 78)
    for r in rows:
        print(f"{r[0]:8} {r[1]:6d} {r[2]:7d}  {r[3]}")
    print(f"\nWrote {len(rows)} calculations to ./{OUTDIR}/")


if __name__ == "__main__":
    main()
