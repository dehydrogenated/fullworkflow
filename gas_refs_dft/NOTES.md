# Gas-phase DFT references in MP conventions

Everything you need to plug a `E_gas^DFT` into your adsorption energies and cleanly
separate Mixed-ML from Full-ML.

---

## 0. A naming heads-up before anything else

You're using these definitions:

| Your term | E_sys | E_slab | E_gas |
|---|---|---|---|
| **Mixed-ML** | ML | ML | **DFT** |
| **Full-ML** | ML | ML | **ML** |

OC22 (Chanussot/Tran, the paper in your project) uses them differently — their
*Full-ML* still takes the gas from DFT:

| OC22 term | E_sys | E_slab | E_gas |
|---|---|---|---|
| Mixed-ML | ML | DFT | DFT |
| Full-ML | ML | ML | DFT |

Your scheme is the sensible one for a perovskite screening workflow. Just define
your terms explicitly in any writeup, because a reviewer who knows OC22 will read
"Full-ML" as "DFT gas" and get confused.

---

## 1. What "MP conventions" actually means, and the trap

MACE-MP-0/MPA, CHGNet, SevenNet-0, Orb, eSEN-MPtrj and the OMat24-trained models
all predict **raw, uncorrected VASP PBE(+U) total energies** on the Materials
Project scale. Your DFT gas reference must sit on that same scale or you bake a
constant offset into every adsorption energy.

**The trap:** the formation energies you see on the MP website are *not* raw. MP
applies `MP2020Compatibility` corrections — anion corrections, a fitted O₂
correction, and a GGA/GGA+U mixing correction — before display. The MLIPs were
trained on the **uncorrected** numbers.

> **Do not pull gas energies from the MP API and do not apply
> `MP2020Compatibility`.** Compute them yourself with the inputs here.

Settings that matter, all baked into the generated INCARs:

| Setting | Value | Why |
|---|---|---|
| `ENCUT` | 520 | MP standard. **Must match your slab calculations.** |
| POTCARs | `PBE_54`, plain `H` `C` `O` | MP's choice for these elements |
| `LASPH` | `.TRUE.` | MP default; changes molecular energies non-trivially |
| `ISMEAR` | `0`, `SIGMA=0.01` | MP uses `-5` for solids — illegal at a single k-point |
| KPOINTS | Γ-only | isolated molecule in a 15 Å box |
| `ISIF` | `2` | `3` would collapse your vacuum box |
| `ISYM` | `0` | stops symmetry from locking the molecule at a saddle point |
| `LDAU` | off | no transition metals in these molecules (see §3) |
| `NUPDOWN` | **2 for O₂** | see below |

### The O₂ landmine

O₂ has a **triplet** ground state. Without `NUPDOWN = 2`, VASP will converge to
the singlet and hand you an energy roughly **1 eV too high** — silently, with no
warning. Every oxygen vacancy formation energy downstream is then wrong by 0.5 eV.
The generated `O2/INCAR` sets this. Don't remove it.

---

## 2. Files

```
make_gas_refs.py          -> writes gas_references/{C3H8,C3H6,H2O,H2,O2,CO2,CO,C3H7}/
make_potcars.sh           -> builds POTCARs from your licensed PBE_54 library
submit_gas_refs.slurm     -> Sockeye job, runs all 8 sequentially
check_gas_refs.sh         -> verifies convergence + spin, dumps the energy dict
gas_offset_diagnostic.py  -> after DFT: measures delta = E_gas^MLIP - E_gas^DFT
energy_schemes.py         -> the adsorption + vacancy formulas, and an error budget
gas_references/           -> POSCAR / INCAR / KPOINTS / POTCAR_ORDER.txt
```

**POTCARs are not included** — they're licensed. `make_potcars.sh` builds them
from your own VASP distribution.

Note `C3H6` is built from ASE's `C3H6_Cs` — that's **propene**. `C3H6_D3h` is
cyclopropane. Easy to grab the wrong one.

---

## 3. The one place this bites you: oxygen vacancy formation energy

For an **adsorption** energy, the `+U` on your Cr/Mn/Fe/Co/Ni B-site appears in
both `E_sys` and `E_slab` and cancels cleanly. Fine.

For an **oxygen vacancy** energy it does *not* cancel — you're subtracting a
plain-GGA O₂ molecule from a GGA+U oxide, on top of PBE's well-known O₂
overbinding. This is exactly your MvK step, so it matters.

**Use the H₂O/H₂ reference instead:**

```
½ O₂  ≡  E(H₂O) − E(H₂) + 2.46 eV
```

This is Nørskov's standard substitution (it's in the *Fundamental Concepts in
Heterogeneous Catalysis* chapter you have). It replaces the badly-described O₂
with two molecules PBE handles well and absorbs the difference into an
experimentally fitted constant. It's why `H2` is in the species list even though
it never appears in your reaction. `energy_schemes.e_vac_H2O()` implements it.

---

## 4. The workflow

On Sockeye, start to finish:

```bash
python make_gas_refs.py                              # 1. build inputs

export VASP_PSP_DIR=/arc/project/st-<PI>-1/vasp/potpaw_PBE.54
./make_potcars.sh                                    # 2. build POTCARs

# 3. edit submit_gas_refs.slurm: --account, VASP_PSP_DIR, module lines
sbatch submit_gas_refs.slurm                         # 4. run (~20 min total)

./check_gas_refs.sh                                  # 5. verify + get energies
python gas_offset_diagnostic.py                      # 6. get delta per species
```

**Three things to edit in `submit_gas_refs.slurm` before submitting:**

1. `--account=st-CHANGEME-1` → your allocation code
2. `VASP_PSP_DIR` → your POTCAR path
3. The `module load` lines — **verify these with `module spider vasp` on
   Sockeye.** I wrote plausible defaults (`gcc/9.4.0`, `openmpi/4.1.1`,
   `vasp/6.3.2`) but I have no way to check Sockeye's actual software stack, so
   treat them as placeholders rather than gospel.

Notes baked into the script: it uses `vasp_gam` (correct and ~2× faster for
Γ-only), runs on only 8 cores (these are 2–11 atom systems; more cores makes
VASP *slower* and can crash on band divisibility), pins `OMP_NUM_THREADS=1`, and
drops a `DONE` file per folder so a timed-out job can just be resubmitted.

Then continue with the ML side:

3. Run your MLIP relaxations for `E_sys` and `E_slab`.
   **Relax the clean slab first, place the adsorbate on that relaxed slab, then
   relax the combined system — in series.** Relaxing them in parallel from
   different starting points is what destroys the slab-error cancellation you're
   relying on (OC22 Fig. 10).
4. `python gas_offset_diagnostic.py` → per-species `delta`.
5. Read the answer off the table:
   - `|delta| < 0.1 eV` → Full-ML is fine, use the ML gas.
   - `0.1–0.5 eV` → use Mixed-ML, or Full-ML with `delta` subtracted
     (`energy_schemes.e_ads_full_ml_corrected`).
   - `> 0.5 eV` → Mixed-ML, and double-check your functional matches the model's
     training set. A delta this large usually means a settings mismatch, not
     model error.

---

## 5. Sanity checks before you trust any of it

Run `energy_schemes.error_budget()` on 10–20 DFT-validated systems:

- **Error grows with slab size?** Your slab error isn't cancelling → inconsistent
  slab reference. Go back to step 3.
- **Error is a constant shift per adsorbate species?** That's the gas /
  intramolecular reference offset → correctable with `delta`.
- **Error is scattered with no structure?** That's genuine adsorbate–surface bond
  error. Not correctable. It's the floor of what this MLIP can do, and it's your
  argument for DFT-validating the top candidates.

For context on what "good" looks like: the 2512.16702 benchmark found the best
models (eSEN, Orb, UMA) reach 0.1–0.3 eV RMSE on energy differences — and that
degrades ~1.6× once you relax *in* the MLIP rather than evaluating on DFT-relaxed
geometries. Plan for ~0.2–0.3 eV, which is fine for screening and not fine for
microkinetics.
