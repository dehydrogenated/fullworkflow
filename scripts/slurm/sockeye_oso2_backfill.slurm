#!/bin/bash
#SBATCH --account=st-akkiraju-1
#SBATCH --partition=cascade
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --job-name=oso2-backfill
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#
# Backfills OsO2 across all 15 models, both facets, both adsorbates -- the
# undercoordinated_metal_site() fix (commit "Fix undercoordinated_metal_site() dropping Os
# sites as if they were oxygen") landed after the main 5d sweep and the CHGNet/Orb-v2 CPU
# backfill were already running with old code, so every model that reaches OsO2 in either
# of those jobs will still hit "no metal ontop site found" until this reruns it fresh. Not
# model-specific -- undercoordinated_metal_site() is pure geometry, called identically
# regardless of backend, so this affects every model that gets to OsO2's site-finding step,
# not just MACE-mh1-omat (which just happened to be first in line).
#
# Points at runs/mo2_sweep_5d -- the SAME directory the main 5d sweep already writes to --
# so bulk/slab resumes automatically wherever a model already got that far (pipeline._resume,
# same mechanism as the extension-fix retest and the CHGNet/Orb-v2 backfill). Only the
# adsorbate leg, which always relaxes fresh, spends new compute. Tiny scope: 1 oxide x 2
# facets x 15 models x 4 relaxations = 120 relaxations max, most far fewer once resume kicks
# in -- CPU is plenty, no need to fight the GPU queue for this.
#
# PREREQUISITES:
#   1. Repo up to date: git pull origin main (needs the Os fix commit)
#
#   cd /scratch/st-akkiraju-1/$USER
#   sbatch /arc/project/st-akkiraju-1/ssong18/fullworkflow/scripts/slurm/sockeye_oso2_backfill.slurm

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

MODELS=(
    CHGNet-0.3.0 Orb-v2
    MACE-mh1-omat MACE-mh1-oc20 MACE-mh1-matpes
    UMA-omat UMA-oc20 UMA-oc22
    UMA-M-omat UMA-M-oc20
    SevenNet-omni-omat24 SevenNet-omni-oc20 SevenNet-omni-oc22 SevenNet-omni-mpa
    eSEN-30M-OAM
)

OUT="$SLURM_SUBMIT_DIR/runs/mo2_sweep_5d"
test -d "$OUT" || {
    echo "missing $OUT -- this resumes bulk/slab from the main 5d sweep's own output, run that first" >&2
    exit 1
}

python "$REPO/scripts/ClaudeScripts/mo2_adsorption_benchmark.py" "$OUT" \
    --oxides OsO2 --facets 110 100 --adsorbates O OH \
    --models "${MODELS[@]}" --fmax 0.02 "$@"

echo
echo "appended to: $OUT/results.jsonl (dedupe by keeping the LAST row per"
echo "  (model, oxide, facet, adsorbate) alongside the main 5d sweep's own rows)"
