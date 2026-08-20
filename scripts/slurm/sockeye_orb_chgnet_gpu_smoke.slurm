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

PROJECT=/arc/project/st-akkiraju-1/ssong18
REPO="$PROJECT/fullworkflow"

export OXW_CONDA_BASE="$PROJECT/miniforge3"
export OXW_MODEL_DIR="$PROJECT/models"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"

# nvidia-smi, not CUDA_VISIBLE_DEVICES -- see sockeye_job.sh for the full incident this
# guards against (a stale inherited env var resolving OXW_DEVICE to cuda on a job that
# never requested a GPU). This job DOES request one (--gres=gpu:1 above), so this should
# resolve to cuda; if it prints "device cpu" instead, that's a scheduling problem to chase
# down before it's even a torch/CUDA-kernel question.
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L 2>/dev/null | grep -q "GPU"; then
    export OXW_DEVICE=cuda
else
    export OXW_DEVICE=cpu
fi

export XDG_CACHE_HOME="${OXW_CACHE:-$SLURM_SUBMIT_DIR/.cache}"
export TRITON_CACHE_DIR="$XDG_CACHE_HOME/triton"
export TORCHINDUCTOR_CACHE_DIR="$XDG_CACHE_HOME/torchinductor"
export HF_HOME="$XDG_CACHE_HOME/huggingface"
export MPLCONFIGDIR="$XDG_CACHE_HOME/matplotlib"
export FAIRCHEM_CACHE_DIR="$XDG_CACHE_HOME/fairchem"
export CACHED_PATH_CACHE_ROOT="$XDG_CACHE_HOME/cached_path"  # orb-models' own cache lib -- ignores XDG_CACHE_HOME, has its own var
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$HF_HOME" "$MPLCONFIGDIR" \
         "$FAIRCHEM_CACHE_DIR" "$CACHED_PATH_CACHE_ROOT"

source "$OXW_CONDA_BASE/etc/profile.d/conda.sh"
conda activate oxw

cd "$SLURM_SUBMIT_DIR"

echo "job $SLURM_JOB_ID on $(hostname), device $OXW_DEVICE"

for model in Orb-v2 CHGNet-0.3.0; do
    echo "=== $model ==="
    python "$REPO/scripts/core/run_stage.py" bulk \
        --model "$model" --supercell 1,1 --fmax 0.1 \
        --outdir "$SLURM_SUBMIT_DIR/runs/gpu_smoke/$model" \
        && echo "$model: OK -- GPU path works" \
        || echo "$model: FAILED -- see error above"
done
