#!/bin/bash
#SBATCH --job-name=co2-retest
#SBATCH --account=<YOUR_ALLOCATION>        # e.g. st-someuser-1 -- replace before submitting
#SBATCH --partition=<YOUR_GPU_PARTITION>   # e.g. gpu_v100 -- check `sinfo` for what your allocation has access to
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00                    # 10 (model,oxide) pairs x 18 relaxations each; GPU should clear this comfortably
#SBATCH --output=co2_retest_%j.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=<YOUR_EMAIL>

# CO2 adsorption retest (5 azimuthal orientations + larger standoff + early-stop desorb
# check) across the rutile family. See scripts/co2_adsorption_retest.py for the full
# rationale. Run from the repo root: sbatch scripts/sockeye_co2_retest.slurm

set -euo pipefail

# --- Sockeye-specific paths -- adjust these to match your allocation's layout -----------
# backends.py reads these three at import time (defaults are for the local Mac dev setup):
export OXW_CONDA_BASE=/arc/project/<YOUR_ALLOCATION>/conda   # wherever your envs live here
export OXW_MODEL_DIR=/arc/project/<YOUR_ALLOCATION>/models   # mace-mh-1.model + uma-s-1p2.pt
export OXW_DEVICE=cuda

REPO=/arc/project/<YOUR_ALLOCATION>/fullworkflow             # path to this repo on Sockeye
cd "$REPO"

# --- one-time setup reminders (skip if already done on this allocation) -----------------
# 1. Orchestrator env (pymatgen, this package): same as CLAUDE.md's `pip install -e ".[dev]"`
# 2. Model envs this job needs: mace-clean, fairchem (UMA-omat + MACE-mh1-omat only --
#    the chgnet/orb envs are NOT required for this particular retest)
# 3. Model checkpoints present under $OXW_MODEL_DIR: mace-mh-1.model, uma-s-1p2.pt
# 4. If reusing the Mac run's already-relaxed bulk/slab structures (saves real time --
#    only the adsorbate stage differs in this retest), rsync the existing run directory
#    here first, e.g.:
#      rsync -av <mac>:fullworkflow/runs/co2_ads_benchmark/ "$REPO/runs/co2_ads_benchmark/"
#    then point the command below at that same --outdir so pipeline._relax_record's resume
#    logic picks up the existing bulk/slab leaves instead of re-relaxing them.

module load gcc cuda   # adjust to whatever Sockeye's module names actually are

python scripts/co2_adsorption_retest.py runs/co2_ads_benchmark --fmax 0.05

echo "done -> runs/co2_ads_benchmark/results_retest.jsonl"
