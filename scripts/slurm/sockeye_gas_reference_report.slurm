#!/bin/bash
#SBATCH --account=st-akkiraju-1
#SBATCH --partition=cascade
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --job-name=gas-refs
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#
# Gas-phase reference energy comparison across all 15 models -- NOT a dependency of the
# main sweep (mo2_adsorption_benchmark.py already computes each model's own H2/H2O
# references internally, self-consistently, same-calculator-cancellation style; see
# energetics.py's own docstring for why borrowing a foreign model's gas energies would be
# wrong). This is a separate diagnostic: how much do models' own gas-phase predictions
# actually disagree with each other, surfaced directly rather than left buried inside each
# sweep run's per-model cache.
#
# Cheap regardless of device -- every fragment (O, H, H2O, CO, OH) is 1-3 atoms, seconds
# per relaxation, 15 models x 5 fragments = 75 relaxations total. CPU default is plenty;
# GPU wouldn't meaningfully speed this up given how small each system is.
#
# PREREQUISITES:
#   1. Repo up to date: git pull origin main
#   2. All 15 models' envs/checkpoints already required by the sweep jobs -- same
#      requirements, nothing extra needed here
#
#   cd /scratch/st-akkiraju-1/$USER
#   sbatch /arc/project/st-akkiraju-1/ssong18/fullworkflow/scripts/slurm/sockeye_gas_reference_report.slurm

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

OUT="$SLURM_SUBMIT_DIR/runs/gas_reference_report.csv"
python "$REPO/scripts/ClaudeScripts/gas_reference_report.py" --out "$OUT" "$@"

echo
echo "results in scratch: $OUT"
echo "to keep it, run this on the LOGIN node (scratch is purged on a timer):"
D="$PROJECT/runs/gas_reference_report"
echo "    mkdir -p $D && cp $OUT $D/ && cp $SLURM_SUBMIT_DIR/slurm-${SLURM_JOB_ID}.{out,err} $D/"
echo "then pull it to your laptop with:"
echo "    scripts/slurm/sync_sockeye_runs.sh $(basename "$D")"
