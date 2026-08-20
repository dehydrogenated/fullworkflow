# Candidate models — Comer et al. 2022 MO2 O*/OH* adsorption sweep

Every entry below is a `REGISTRY` key in `oxide_workflow/backends.py`. Reference is the
literature's own **DFT+U (PBE+U) values**, not any of these models — every model here,
`MACE-mh1-omat` included, is an independent candidate scored against Comer's numbers, not
against each other. (That's a different design from `pipeline.py`'s own reference/candidate
divergence benchmark, which *does* designate `MACE-mh1-omat` as a self-consistency
reference for a different, smaller 3-material comparison — don't conflate the two.)

15 models, driven by `scripts/ClaudeScripts/mo2_adsorption_benchmark.py` via
`scripts/slurm/sockeye_mo2_sweep_{3d,4d,5d}.slurm`.

| Requested                          | `REGISTRY` key(s)                                                                          | env        | checkpoint |
|---|---|---|---|
| CHGNet 0.3.0                       | `CHGNet-0.3.0`                                                                              | chgnet     | ships in pip package |
| Orb-v2                             | `Orb-v2`                                                                                    | orb        | `orb-v2-20241011.ckpt` |
| MACE-mh1 (OMat, OC20, r2SCAN)      | `MACE-mh1-omat`, `MACE-mh1-oc20`, `MACE-mh1-matpes`                                         | mace-clean | `mace-mh-1.model` |
| UMA-S (OMAT, OC20, OC22)           | `UMA-omat`, `UMA-oc20`, `UMA-oc22`                                                          | fairchem   | `uma-s-1p2.pt` |
| UMA-M (OMAT, OC20)                 | `UMA-M-omat`, `UMA-M-oc20`                                                                  | fairchem   | `uma-m-1p1.pt` (gated HF download) |
| SevenNet-omni (OMat24, OC20, OC22, MPtrj+Alex) | `SevenNet-omni-omat24`, `SevenNet-omni-oc20`, `SevenNet-omni-oc22`, `SevenNet-omni-mpa` | sevenn     | `sevennet-omni.pth` |
| eSEN-30M-OAM                       | `eSEN-30M-OAM`                                                                              | esen       | `esen_30m_oam.pt` |

Every checkpoint above is confirmed staged at `$PROJECT/models/` on Sockeye and every env
confirmed created, as of 2026-08-19 (see CLAUDE.md's Sockeye section for the live status —
re-verify there before trusting this, per that section's own caveat about drift).

## The two deliberate outliers

The sweep is mostly organized around a clean omat/oc20/oc22 axis (generalist-materials vs.
catalysis-specialist training data, same functional where possible). Two models break that
symmetry on purpose:

- **`MACE-mh1-matpes`** — trained on r2SCAN, not PBE, unlike everything else in the sweep
  and unlike Comer's own PBE+U reference. Included specifically to test whether a different
  (in principle more accurate) DFT functional systematically outperforms the PBE-trained
  models despite the reference mismatch — a real question, not an oversight. Read its
  numbers as "does r2SCAN training help," not as apples-to-apples with the rest.
- **`SevenNet-omni-mpa`** — MPtrj+Alexandria, PBE (so no functional mismatch), included as a
  novel-dataset generalist point. Has no MACE/UMA equivalent registered on this exact axis,
  so it's a standalone comparison rather than part of the omat/oc20/oc22 triangle.

## Materials

All 33 rutile-structure MO2 oxides from Comer et al. 2022 Table S3, split three ways by
row (`sockeye_mo2_sweep_3d/4d/5d.slurm`, 11 oxides each):

- **3d**: TiO2, VO2, CrO2, MnO2, FeO2, CoO2, NiO2, CuO2, ZnO2, GaO2, GeO2
- **4d**: ZrO2, NbO2, MoO2, TcO2, RuO2, RhO2, PdO2, AgO2, CdO2, InO2, SnO2
- **5d**: HfO2, TaO2, WO2, ReO2, OsO2, IrO2, PtO2, AuO2, HgO2, TlO2, PbO2

32 of 33 come from Comer's own committed relaxed structures (`data/structures/O_OH_AdsStructures/`,
fetched via `scripts/core/fetch_comer2022_structures.py`); ZrO2 has no
`final_with_calculator.json` in their repo and falls back to Materials Project.

## `ALL_CANDIDATES` doesn't cover this list

`ALL_CANDIDATES` (backends.py) is built from `MACE_HEADS + UMA_TASKS + CHGNET_MODELS +
ORB_MODELS` only — it omits SevenNet, eSEN, and UMA-M entirely, and includes MACE/UMA heads
(matpes aside) that aren't part of this sweep either (spice, omol, odac, omc — wrong domain,
molecular/MOF chemistry not periodic oxide surfaces). The sweep scripts pass their `MODELS`
array explicitly rather than relying on it.

## Known caveats

- **Same nominal dataset, different functional underneath.** `MACE-mh1-oc20`'s head is
  `oc20_usemppbe` (PBE-relabeled OC20 structures), whereas `UMA-oc20`/`SevenNet-omni-oc20`
  use the original OC20 task, computed at RPBE. Worth calling out explicitly in any report
  that groups these by dataset name rather than functional.
- **`UMA-M-omat`/`UMA-M-oc20`** depend on the gated `uma-m-1p1.pt` checkpoint and have a
  known predictor-construction hang on some setups (fairchem#2095) that isn't a config
  mistake if hit.
- **Slab geometry fix (2026-08-19)**: every model's `make_slab()` output previously carried
  a spurious lattice shear from `SlabGenerator`'s LLL reduction (see the commit "Fix
  spurious lattice shear from SlabGenerator's LLL reduction") — magnitude was
  chemistry-dependent, from negligible (Orb-v2's TiO2) to ~1.2 A (eSEN's TiO2). Any run from
  before that fix should be treated as built on slightly-to-severely warped starting
  geometry and not compared directly against post-fix results.
