#!/bin/bash
#SBATCH --account=st-akkiraju-1-gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
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
# sites as if they were oxygen") landed after the main 5d sweep started, so any model that
# reached OsO2 with old code still hit "no metal ontop site found". Not model-specific --
# undercoordinated_metal_site() is pure geometry, called identically regardless of backend.
#
# GPU now, for every model -- the orb/chgnet torch cu126 reinstall (see
# sockeye_orb_chgnet_gpu_smoke.slurm) closed the last gap; every model in MODELS below has a
# confirmed-working CUDA path as of 2026-08-20.
#
# Deliberately its OWN output folder (runs/oso2_gpu_backfill), NOT runs/mo2_sweep_5d --
# the main 5d sweep job is still actively running (on SevenNet as of this writing) and
# actively appending to that directory's results.jsonl. Concurrent appends to the same file
# are safe (confirmed earlier this session, POSIX atomic small-write semantics), but a
# separate directory means this job doesn't touch bulk/slab CONTCARs the main job might still
# be reading/writing for OTHER oxides in the same row, and keeps this run's own output
# trivially inspectable in isolation while both are in flight. Costs a bit of extra compute
# (bulk+slab redone from scratch per model rather than resumed via pipeline._resume, since
# there's no prior OsO2 output in this fresh directory) -- fine, OsO2 is a single oxide and
# GPU makes the whole thing cheap regardless.
#
# Merge afterward: concatenate this directory's results.jsonl onto mo2_sweep_5d's own, same
# last-row-wins dedup rule used everywhere else this session --
#   cat runs/oso2_gpu_backfill/results.jsonl >> runs/mo2_sweep_5d/results.jsonl
# (or dedupe at read time instead of physically merging, same effect either way).
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

# nvidia-smi, not CUDA_VISIBLE_DEVICES -- see sockeye_job.sh for the full incident. This job
# requests a GPU (--gres=gpu:1 above), so this should resolve to cuda; if it prints
# "device cpu" instead, that's a scheduling problem to chase down first.
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

OUT="$SLURM_SUBMIT_DIR/runs/oso2_gpu_backfill"
mkdir -p "$OUT"

python "$REPO/scripts/ClaudeScripts/mo2_adsorption_benchmark.py" "$OUT" \
    --oxides OsO2 --facets 110 100 --adsorbates O OH \
    --models "${MODELS[@]}" --fmax 0.02 "$@"

echo
echo "results in: $OUT/results.jsonl"
echo "merge into the main 5d sweep on the LOGIN node once both are done:"
echo "    cat $OUT/results.jsonl >> \$PROJECT/runs/mo2_sweep_5d/results.jsonl"
echo "(or just pass both files to any script that already dedupes by last-row-wins, e.g."
echo " periodic_trend_report.py -- no physical merge required)"
