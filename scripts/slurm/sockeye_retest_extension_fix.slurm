#!/bin/bash
#SBATCH --account=st-akkiraju-1
#SBATCH --partition=cascade
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --job-name=retest-ext
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#
# Re-runs the 7 legs that didn't converge in the medium test, after the worker_relax.py
# extension-logic fix (compare against the anchor's start distance, not a rolling
# per-window baseline). scripts/ClaudeScripts/retest_extension_fix.py has the exact list
# and old numbers for comparison. Points at the medium test's OWN output directory so
# bulk/slab resume from the already-converged cached leaves (pipeline._resume) instead of
# recomputing -- verified locally: first relax_bulk_slab call ~67s, second (resumed) call
# ~0.1s, identical energy. Only the adsorbate leg (which always relaxes fresh, no resume
# check on that path) actually spends new compute.
#
# GPU recommended -- 7 legs, should be quick either way, but no reason not to:
#   sbatch --account=st-akkiraju-1-gpu --partition=gpu --gres=gpu:1 --time=1:00:00 \
#          /arc/project/st-akkiraju-1/ssong18/fullworkflow/scripts/slurm/sockeye_retest_extension_fix.slurm
#
# PREREQUISITES:
#   1. Repo up to date: git pull origin main (needs the extension-logic fix commit)
#   2. runs/mo2_medium_test/ must already exist from the medium test job -- this reuses it,
#      does not recreate it from scratch
#
#   cd /scratch/st-akkiraju-1/$USER
#   sbatch /arc/project/st-akkiraju-1/ssong18/fullworkflow/scripts/slurm/sockeye_retest_extension_fix.slurm

set -euo pipefail

PROJECT=/arc/project/st-akkiraju-1/ssong18
REPO="$PROJECT/fullworkflow"

export OXW_CONDA_BASE="$PROJECT/miniforge3"
export OXW_MODEL_DIR="$PROJECT/models"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"

# nvidia-smi, not CUDA_VISIBLE_DEVICES -- that env var is inherited from the submitting
# shell by sbatch's default --export=ALL, so a stray value from an earlier interactive GPU
# session can make a --partition=cascade job (no --gres=gpu) wrongly resolve to cuda. See
# sockeye_job.sh for the full incident. nvidia-smi respects SLURM's cgroup isolation instead.
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

echo "job $SLURM_JOB_ID on $(hostname), ${SLURM_CPUS_PER_TASK:-?} cpus, device $OXW_DEVICE"

OUT="$SLURM_SUBMIT_DIR/runs/mo2_medium_test"
test -d "$OUT" || {
    echo "missing $OUT -- this reuses the medium test's own output dir, run that first" >&2
    exit 1
}

python "$REPO/scripts/ClaudeScripts/retest_extension_fix.py" "$OUT"

echo
echo "appended to: $OUT/results.jsonl (dedupe by keeping the LAST row per"
echo "  (model, oxide, facet, adsorbate) if analyzing alongside the original medium-test rows)"
