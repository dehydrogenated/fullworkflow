#!/bin/bash
#SBATCH --account=st-akkiraju-1
#SBATCH --partition=cascade
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --job-name=run-stage
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#
# scripts/run_stage.py on Sockeye -- one stage (bulk/slab/vacancy/adsorbate), one model,
# skipping every stage after the one requested. For OVFE-only comparisons this avoids the
# adsorbate sweep that dominates a full oxide_workflow.pipeline run. Submit from /scratch,
# with an explicit --outdir per model so parallel submissions don't collide:
#
#   cd /scratch/st-akkiraju-1/$USER
#   sbatch /arc/project/st-akkiraju-1/ssong18/fullworkflow/scripts/sockeye_run_stage.slurm \
#          vacancy --model UMA-M-omat --outdir runs/ovfe_uma_m_omat
#
#   sbatch --account=st-akkiraju-1-gpu --partition=gpu --gres=gpu:1 --time=0:30:00 \
#          /arc/project/st-akkiraju-1/ssong18/fullworkflow/scripts/sockeye_run_stage.slurm \
#          vacancy --model UMA-M-oc20 --outdir runs/ovfe_uma_m_oc20

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
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$HF_HOME" "$MPLCONFIGDIR" \
         "$FAIRCHEM_CACHE_DIR"

source "$OXW_CONDA_BASE/etc/profile.d/conda.sh"
conda activate oxw

cd "$SLURM_SUBMIT_DIR"

echo "job $SLURM_JOB_ID on $(hostname), ${SLURM_CPUS_PER_TASK:-?} cpus, device $OXW_DEVICE"

test -f "$OXW_MODEL_DIR/mace-mh-1.model" || {
    echo "missing checkpoint: $OXW_MODEL_DIR/mace-mh-1.model" >&2
    exit 1
}
test -f "$OXW_MODEL_DIR/uma-m-1p1.pt" || echo "note: uma-m-1p1.pt not staged (needed for UMA-M-* models)"

# Everything after the script path forwards straight to run_stage.py -- stage, --model,
# --outdir, and any geometry/relax overrides, exactly like running it locally.
python "$REPO/scripts/run_stage.py" "$@"

echo
echo "results in scratch under the --outdir you passed above"
echo "to keep them, run this on the LOGIN node (scratch is purged on a timer):"
echo "    mkdir -p $PROJECT/runs && rsync -a \$SLURM_SUBMIT_DIR/<outdir>/ $PROJECT/runs/\$(date +%Y%m%d-%H%M%S)_<outdir>/"
