#!/bin/bash
#SBATCH --account=st-akkiraju-1
#SBATCH --partition=cascade
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --job-name=chgnet-test
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#
# Smoke test for CHGNet-0.3.0 on a freshly-created "chgnet" env (pip install chgnet==0.4.2).
# No separate checkpoint file to stage here, unlike Orb/UMA/eSEN: the 0.3.0 pretrained
# weights ship bundled inside the pip package itself (site-packages/chgnet/pretrained/
# 0.3.0/), so a successful `pip install` on the login node already fully stages it -- the
# only thing worth checking before running is that the env exists at all.
#
# Deliberately tiny: TiO2 only, both facets, both adsorbates -- 4 legs, not the full
# 33-oxide sweep.
#
# PREREQUISITES:
#   1. Repo up to date: git pull origin main
#   2. "chgnet" env exists with chgnet==0.4.2 + ase installed
#
#   cd /scratch/st-akkiraju-1/$USER
#   sbatch /arc/project/st-akkiraju-1/ssong18/fullworkflow/scripts/slurm/sockeye_chgnet_test.slurm

set -euo pipefail

PROJECT=/arc/project/st-akkiraju-1/ssong18
REPO="$PROJECT/fullworkflow"

export OXW_CONDA_BASE="$PROJECT/miniforge3"
export OXW_MODEL_DIR="$PROJECT/models"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"

if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
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

test -d "$OXW_CONDA_BASE/envs/chgnet" || {
    echo "missing conda env: $OXW_CONDA_BASE/envs/chgnet -- create it first (chgnet==0.4.2, ase)" >&2
    exit 1
}

OUT="$SLURM_SUBMIT_DIR/runs/chgnet_test"
python "$REPO/scripts/ClaudeScripts/mo2_adsorption_benchmark.py" "$OUT" \
    --oxides TiO2 --facets 110 100 --adsorbates O OH --models CHGNet-0.3.0 --fmax 0.02 "$@"

echo
echo "results in scratch: $OUT/results.jsonl"
echo "to keep them, run this on the LOGIN node (scratch is purged on a timer):"
D="$PROJECT/runs/$(date +%Y%m%d-%H%M%S)_chgnet_test"
echo "    mkdir -p $D && rsync -a $OUT/ $D/ && cp $SLURM_SUBMIT_DIR/slurm-${SLURM_JOB_ID}.{out,err} $D/"
echo "then pull it to your laptop with:"
echo "    scripts/slurm/sync_sockeye_runs.sh $(basename "$D")"
