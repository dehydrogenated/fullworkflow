#!/bin/bash
#SBATCH --account=st-akkiraju-1-gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=0:15:00
#SBATCH --job-name=orb-chgnet-gpu-smoke
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#
# One-shot smoke test: does the cu126 torch reinstall in the orb/chgnet envs actually fix
# the "no kernel image is available for execution on the device" crash on Sockeye's V100s
# (compute capability 7.0)? Bulk stage only, 1x1 supercell, fmax=0.1 -- this project's own
# documented smoke-test/quick-check settings (see CLAUDE.md), chosen purely for speed: this
# only needs to trigger one real GPU kernel launch per model to prove the fix works, not
# produce usable science. Throwaway output (runs/gpu_smoke), not meant to be kept or synced.
#
# Both models tested in one job so a single sbatch answers the question for both envs;
# `&&`/`||` per model (not a bare loop body) so model 2 still runs even if model 1 crashes --
# set -e doesn't trigger inside a && / || chain, only if the whole compound command fails,
# and echo never fails, so this job always reaches both models and reports pass/fail for each.
#
# PREREQUISITES:
#   1. Repo up to date: git pull origin main
#   2. torch reinstalled from the cu126 wheel index in BOTH "orb" and "chgnet" envs first --
#      this smoke test is what tells you whether that worked, it doesn't do the reinstall
#
#   cd /scratch/st-akkiraju-1/$USER
#   sbatch /arc/project/st-akkiraju-1/ssong18/fullworkflow/scripts/slurm/sockeye_orb_chgnet_gpu_smoke.slurm

set -euo pipefail

source /arc/project/st-akkiraju-1/ssong18/fullworkflow/scripts/slurm/_env.sh

for model in Orb-v2 CHGNet-0.3.0; do
    echo "=== $model ==="
    python "$REPO/scripts/core/run_stage.py" bulk \
        --model "$model" --supercell 1,1 --fmax 0.1 \
        --outdir "$SLURM_SUBMIT_DIR/runs/gpu_smoke/$model" \
        && echo "$model: OK -- GPU path works" \
        || echo "$model: FAILED -- see error above"
done
