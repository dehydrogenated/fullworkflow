#!/bin/bash
#SBATCH --account=st-akkiraju-1-gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --job-name=ovfe-vac
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#
# OVFE convergence vs. vacuum thickness (min_vacuum_size = 5,8,13,16,20,25,30,35,40,50 A),
# on the 3 top-performing OVFE-benchmark models (UMA-omat, eSEN-30M-OAM, Orb-v2). Bulk
# relaxed once per model, reused across every vacuum value; supercell/height/freeze held
# at the current SlabConfig default (4x2, 4 trilayers -- see
# ovfe_convergence_common.py's BASE_SLAB comment). See
# scripts/ClaudeScripts/ovfe_vacuum_sweep.py for the full rationale. A model with a
# missing checkpoint/env is skipped, not fatal -- the other two still complete.
#
# GPU by default (per Sean, 2026-08-20) -- UMA-omat dominates wall-clock and the -gpu
# account/partition should help it the most; --time=4:00:00 is carried over from the CPU
# estimate unverified against an actual GPU run, so treat it as a conservative ceiling,
# not a measured budget.
#
#   cd /scratch/st-akkiraju-1/$USER
#   sbatch /arc/project/st-akkiraju-1/ssong18/fullworkflow/scripts/slurm/sockeye_ovfe_vacuum_sweep.slurm

set -euo pipefail

PROJECT=/arc/project/st-akkiraju-1/ssong18
REPO="$PROJECT/fullworkflow"

export OXW_CONDA_BASE="$PROJECT/miniforge3"
export OXW_MODEL_DIR="$PROJECT/models"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

# eSEN OOMed mid-sweep on the height/freeze sweeps at ~384+ atoms even with a
# fresh subprocess per relaxation -- the driver-reported gap between allocated and
# reserved-but-unallocated memory in that crash suggests fragmentation, not just
# raw need. This is PyTorch's own suggested mitigation from that error message.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MKL_NUM_THREADS="$OMP_NUM_THREADS"

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
export CACHED_PATH_CACHE_ROOT="$XDG_CACHE_HOME/cached_path"
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$HF_HOME" "$MPLCONFIGDIR" \
         "$FAIRCHEM_CACHE_DIR" "$CACHED_PATH_CACHE_ROOT"

source "$OXW_CONDA_BASE/etc/profile.d/conda.sh"
conda activate oxw

cd "$SLURM_SUBMIT_DIR"

echo "job $SLURM_JOB_ID on $(hostname), ${SLURM_CPUS_PER_TASK:-?} cpus, device $OXW_DEVICE"

# All three MODELS in ovfe_convergence_common.py are needed for this sweep to be complete
# -- note (not fatal) rather than exit, since run_sweep() itself now skips a model whose
# checkpoint/env is missing and continues with the rest.
for ckpt in uma-s-1p2.pt esen_30m_oam.pt orb-v2-20241011.ckpt; do
    test -f "$OXW_MODEL_DIR/$ckpt" || echo "note: $OXW_MODEL_DIR/$ckpt not staged -- that model's rows will be marked failed"
done

OUT="$SLURM_SUBMIT_DIR/runs/ovfe_vacuum_sweep"
python "$REPO/scripts/ClaudeScripts/ovfe_vacuum_sweep.py" "$OUT"

echo
echo "results in scratch: $OUT/results.jsonl"
echo "to keep them, run this on the LOGIN node (scratch is purged on a timer):"
D="$PROJECT/runs/$(date +%Y%m%d-%H%M%S)_ovfe_vacuum_sweep"
echo "    mkdir -p $D && rsync -a $OUT/ $D/ && cp $SLURM_SUBMIT_DIR/slurm-${SLURM_JOB_ID}.{out,err} $D/"
