#!/bin/bash
#SBATCH --account=st-akkiraju-1
#SBATCH --partition=cascade
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --job-name=esen-skew
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#
# Breadth check for the lattice-shear fix (stages.py's get_orthogonal_c_slab call),
# with eSEN specifically -- it produced the worst observed pre-fix case (TiO2(110):
# b_z=-1.19 A vs Orb-v2's ~-0.003 A on the same facet), so it's the model most likely to
# expose a chemistry this fix doesn't actually cover. Bulk + slab only, no vacancy, no
# adsorbate -- scripts/studies/mo2/esen_slab_skew_check.py reads back each relaxed slab's
# lattice matrix and reports a_z/b_z directly, no adsorption energetics involved. Cheap on
# purpose: the shear pymatgen introduces pre-fix is chemistry-dependent (driven by each
# oxide's own converged bulk lattice constants), so this is about running eSEN's own bulk
# relaxation across all 33 oxides and confirming every one comes out clean, not about
# proving the fix once more on TiO2.
#
# For GPU (recommended -- eSEN is the slowest model in this project's own timing logs, and
# this job still runs 33 oxides x 2 facets x 2 relaxations = 132 relaxations):
#   sbatch --account=st-akkiraju-1-gpu --partition=gpu --gres=gpu:1 --time=2:00:00 \
#          /arc/project/st-akkiraju-1/ssong18/fullworkflow/scripts/studies/mo2/sockeye_esen_skew_check.slurm
#
# PREREQUISITES:
#   1. Repo up to date: git pull origin main
#   2. "esen" env exists, esen_30m_oam.pt staged at $PROJECT/models/ (checked below)
#
#   cd /scratch/st-akkiraju-1/$USER
#   sbatch /arc/project/st-akkiraju-1/ssong18/fullworkflow/scripts/studies/mo2/sockeye_esen_skew_check.slurm

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

test -d "$OXW_CONDA_BASE/envs/esen" || {
    echo "missing conda env: $OXW_CONDA_BASE/envs/esen" >&2
    exit 1
}
test -f "$OXW_MODEL_DIR/esen_30m_oam.pt" || {
    echo "missing checkpoint: $OXW_MODEL_DIR/esen_30m_oam.pt" >&2
    exit 1
}

OUT="$SLURM_SUBMIT_DIR/runs/esen_skew_check"
python "$REPO/scripts/studies/mo2/esen_slab_skew_check.py" "$OUT" --facets 110 100 "$@"

echo
echo "report in scratch: $OUT/skew_report.jsonl"
echo "to keep it, run this on the LOGIN node (scratch is purged on a timer):"
D="$PROJECT/runs/$(date +%Y%m%d-%H%M%S)_esen_skew_check"
echo "    mkdir -p $D && rsync -a $OUT/ $D/ && cp $SLURM_SUBMIT_DIR/slurm-${SLURM_JOB_ID}.{out,err} $D/"
echo "then pull it to your laptop with:"
echo "    scripts/slurm/sync_sockeye_runs.sh $(basename "$D")"
