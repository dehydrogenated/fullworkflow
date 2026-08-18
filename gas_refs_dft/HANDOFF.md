# Handoff to Claude Code on Sockeye

## Getting the files onto Sockeye

From your Mac:

```bash
scp -r ~/gas_references_MP <cwl>@sockeye.arc.ubc.ca:/scratch/st-<PI>-1/<cwl>/
```

Then on Sockeye:

```bash
cd /scratch/st-<PI>-1/<cwl>/gas_references_MP
claude
```

Run it from **scratch**, not your home directory — VASP writes WAVECAR/CHGCAR and
home quotas on Sockeye are small. (`LWAVE`/`LCHARG` are already off in the INCARs,
so this is mostly about the OUTCARs, but scratch is the right habit.)

Claude Code picks up `CLAUDE.md` automatically from the working directory. You do
not need to paste any of that context in.

---

## First prompt to paste

```
Read CLAUDE.md first — it has the full context for this project.

Before we run anything, resolve the three unverified placeholders flagged in
section 5. You're on the Sockeye login node so you can check these directly
rather than guessing:

  1. The real VASP module names (module spider vasp) — and whether the build
     provides vasp_gam
  2. My allocation code for --account
  3. Where our group's POTCAR library lives, and confirm it's the .54 set

Report what you find before editing submit_gas_refs.slurm. If any of them are
ambiguous, tell me rather than picking one.
```

That's deliberately scoped to verification only. Don't let it submit jobs on the
first turn — a bad POTCAR set produces energies that look completely reasonable and
are silently wrong.

---

## Once verification is done

Reasonable follow-up prompts, roughly in order:

```
Update submit_gas_refs.slurm with what you found, then do a dry run: build the
POTCARs and confirm every folder has POSCAR, INCAR, KPOINTS, POTCAR with matching
species order. Don't submit yet.
```

```
Submit the job. When it finishes, run check_gas_refs.sh and walk me through the
output — especially whether O2 came out as a triplet.
```

```
Install MACE and run gas_offset_diagnostic.py. Interpret the deltas for me: is
this a constant offset or is it species-dependent, and which scheme should I use?
```

---

## What to watch for

**Don't let it raise the core count.** 8 ranks is deliberate for 2–11 atom systems.
More cores makes VASP slower here and can crash on band divisibility.

**If `check_gas_refs.sh` flags O₂ as not-a-triplet, stop.** That's a ~1 eV error
that propagates into every oxygen vacancy energy. It means `NUPDOWN = 2` didn't
take, not that the check is over-sensitive.

**If it wants to pull gas energies from the Materials Project API instead of
running DFT — no.** CLAUDE.md §3 explains why at length. MP's molecule database is
Q-Chem, and MP's O₂ entries are solid crystals.

**Confirm .54 vs 5.2 POTCARs explicitly.** This is the one mistake that produces
no error message, no warning, and completely wrong energies that look fine.

---

## Context that lives outside this repo

If the Claude Code session needs the wider picture, these are in the
"Catalysis + MLIP Research" project (not on Sockeye):

- `claude/gas-phase-reference-conventions.md` — the README, saved for future sessions
- `2206.08917v3.pdf` — OC22 paper, Eqs. 4–5 and Table 8 (the Mixed/Full-ML numbers)
- `2512.16702v1.pdf` — the MLIP benchmark, gas-phase error correction in §2.4
- `Propane_ODH_MainText_v01_KA.docx` — the manuscript this feeds into
- `Fundamental Concepts in Heterogeneous Catalysis` (Nørskov) — the ½O₂ ≡ H₂O − H₂
  substitution

The Claude Code session won't have access to these, so paste in specifics if a
question turns on one of them.
