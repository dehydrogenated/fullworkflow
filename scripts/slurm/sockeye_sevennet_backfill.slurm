#!/bin/bash
#SBATCH --account=st-akkiraju-1-gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --job-name=sevennet-backfill
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#
# The main mo2-sweep-3d/4d/5d jobs (12656383/84/85) died at their 20h walltime before
# reaching the back of the model roster for rows 3d and 4d -- confirmed against the pulled
# results (data/ClaudeArtifacts/O_OH_Trendline/{3d,4d}_results.jsonl on the laptop,
# 2026-08-20): row 3d has ZERO rows for SevenNet-omni-oc22 and SevenNet-omni-mpa; row 4d
# has SevenNet-omni-oc22 at 22/44 (exactly half) and SevenNet-omni-mpa at ZERO. Row 5d
# reached both already (confirmed complete) so isn't in scope here.
#
# Only these 2 models, only these 2 rows. Not eSEN (already fully backfilled separately,
# 44/44 in every row) and not the earlier SevenNet variants (omni-omat24/oc20, already
# converged in the rows that needed them).
#
# Own output folder (runs/sevennet_gpu_backfill_<row>), not runs/mo2_sweep_<row> -- the
# main jobs are dead now so there's no live-write collision to avoid, but scratch is purged
# on a timer and there's no guarantee runs/mo2_sweep_<row> still exists there with usable
# bulk/slab CONTCARs to resume from (unlike when the eSEN backfill ran, while the main jobs
# were still alive and scratch was fresh). Safer to assume nothing carries over and just
# redo bulk/slab for these 2 models fresh -- cheap on GPU regardless. Merge into the
# existing *_results.jsonl afterward, same last-row-wins rule as every other backfill this
# session.
#
# PREREQUISITES:
#   1. Repo up to date: git pull origin main
#   2. "sevenn" env exists, sevennet-omni.pth staged at $PROJECT/models/ (checked below)
#
#   cd /scratch/st-akkiraju-1/$USER
#   sbatch /arc/project/st-akkiraju-1/ssong18/fullworkflow/scripts/slurm/sockeye_sevennet_backfill.slurm 3d
#   sbatch /arc/project/st-akkiraju-1/ssong18/fullworkflow/scripts/slurm/sockeye_sevennet_backfill.slurm 4d

set -euo pipefail

ROW="${1:?usage: sbatch sockeye_sevennet_backfill.slurm <3d|4d>}"
shift  # remaining "$@" (if any) still forwards to mo2_adsorption_benchmark.py below

case "$ROW" in
    3d) OXIDES=(TiO2 VO2 CrO2 MnO2 FeO2 CoO2 NiO2 CuO2 ZnO2 GaO2 GeO2) ;;
    4d) OXIDES=(ZrO2 NbO2 MoO2 TcO2 RuO2 RhO2 PdO2 AgO2 CdO2 InO2 SnO2) ;;
    *) echo "unknown row: $ROW -- expected 3d or 4d (5d already has both models complete)" >&2; exit 1 ;;
esac

PROJECT=/arc/project/st-akkiraju-1/ssong18
REPO="$PROJECT/fullworkflow"

export OXW_CONDA_BASE="$PROJECT/miniforge3"
export OXW_MODEL_DIR="$PROJECT/models"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"

# nvidia-smi, not CUDA_VISIBLE_DEVICES -- see sockeye_job.sh for the full incident. This
# job requests a GPU (--gres=gpu:1 above), so this should resolve to cuda.
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

echo "job $SLURM_JOB_ID on $(hostname), ${SLURM_CPUS_PER_TASK:-?} cpus, device $OXW_DEVICE, row $ROW"

test -d "$OXW_CONDA_BASE/envs/sevenn" || {
    echo "missing conda env: $OXW_CONDA_BASE/envs/sevenn" >&2
    exit 1
}
test -f "$OXW_MODEL_DIR/sevennet-omni.pth" || {
    echo "missing checkpoint: $OXW_MODEL_DIR/sevennet-omni.pth" >&2
    exit 1
}

OUT="$SLURM_SUBMIT_DIR/runs/sevennet_gpu_backfill_$ROW"
mkdir -p "$OUT"

python "$REPO/scripts/ClaudeScripts/mo2_adsorption_benchmark.py" "$OUT" \
    --oxides "${OXIDES[@]}" --facets 110 100 --adsorbates O OH \
    --models SevenNet-omni-oc22 SevenNet-omni-mpa --fmax 0.02 "$@"

echo
echo "results in: $OUT/results.jsonl"
echo "to keep them, run this on the LOGIN node (scratch is purged on a timer):"
D="$PROJECT/runs/sevennet_gpu_backfill_$ROW"
echo "    mkdir -p $D && rsync -a $OUT/ $D/ && cp $SLURM_SUBMIT_DIR/slurm-${SLURM_JOB_ID}.{out,err} $D/"
echo "then pull it to your laptop and merge into data/ClaudeArtifacts/O_OH_Trendline/${ROW}_results.jsonl"
echo "(cat onto the end, or hand both files to anything that dedupes by last-row-wins already)"
