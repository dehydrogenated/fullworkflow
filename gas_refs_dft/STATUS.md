# READ THIS FIRST — how this folder relates to what the repo already does

Written by a Cowork session that did **not** know this repo existed. It was built
on the assumption that the workflow runs VASP DFT. That assumption is wrong in an
important way, and the repo already contains a *better* answer to part of the
problem. Do not run any of this until you have read the following.

## What this folder is

VASP inputs + scripts to compute gas-phase reference energies (C3H8, C3H6, H2O,
H2, O2, CO2, CO, C3H7) in Materials Project conventions, so an MLIP total-energy
prediction can be turned into an adsorption energy:

    E_ads = [ E_sys^ML - E_slab^ML - n * E_gas^DFT ] / n      ("Mixed-ML")

See NOTES.md for the full reasoning, the MP-convention traps (O2 triplet, POTCAR
ordering, MP2020 corrections, +U cancellation), and evidence that no such
reference table exists online.

## Why it may be the wrong tool here

**1. This repo runs no VASP.** "vasp" appears only as a file format (`input.vasp`,
`relaxed.vasp` = POSCAR). Every backend is an MLIP. It is unconfirmed whether the
group even has a VASP build or license on Sockeye. Check `module spider vasp`
before anything else — if it's empty, this folder is dead weight until that's
sorted.

**2. `oxide_workflow/energetics.py` already solves the O2 reference problem, and
solves it on a sounder principle.** `oxygen_chemical_potential_corrected` does a
thermodynamic-cycle correction through *the model's own* H2 and H2O, staying
entirely on that model's own energy scale, importing only the experimental water
formation enthalpy — a portable *reaction* energy.

That is strictly better than mixing a DFT number into an ML energy scale, and
`scripts/ovfe_o2_correction.py` already says why in its docstring: there is no
bare absolute value that can be transplanted between scales, and two heads of one
checkpoint can disagree by tens of eV on an identical cell.

The same objection applies to Mixed-ML generally. `E_sys^ML` contains the
adsorbate's intramolecular energy on the *model's* scale; subtracting `E_gas^DFT`
subtracts it on the *DFT* scale. That mismatch does not cancel — it is a residual
Mixed-ML cannot remove, and the existing same-scale cycle avoids it entirely.

## So when is this folder actually useful?

Only for things the same-scale trick can't reach:

- **Benchmarking.** Ground-truth DFT `E_ads` on a small subset, to measure how far
  the MLIP-only numbers actually are from DFT. That needs real DFT regardless of
  referencing scheme.
- **Generating DFT training data** for fine-tuning — which is what the Work Learn
  proposal describes. Gas references are needed to define the targets.
- **Species the cycle doesn't cover.** The H2O/H2 cycle handles oxygen. It does
  not give you C3H8 or C3H6 references for propane ODH adsorption energies. If
  those are needed on a DFT scale, they have to be computed.

For oxygen vacancy formation energies specifically: **use the existing
`oxygen_chemical_potential_corrected` path, not `energy_schemes.e_vac_O2` in this
folder.** The existing one is better and is already integrated.

## If you do proceed

1. `module spider vasp` — confirm VASP exists. If not, stop.
2. Confirm the POTCAR library path and that it's the **.54** set, not 5.2.
3. Copy this folder to `/scratch/st-akkiraju-1/$USER/` — `/arc/project` is
   read-only on compute nodes and VASP writes in place.
4. `./make_potcars.sh`, then `sbatch submit_gas_refs.slurm`, then
   `./check_gas_refs.sh`.

Account `st-akkiraju-1` and the scratch guard are already baked into the slurm
script. The `module load vasp` line is still a placeholder.
