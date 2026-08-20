#!/bin/bash
#SBATCH --account=st-akkiraju-1-gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=3:00:00
#SBATCH --job-name=co-h-ads
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#
# CO + H adsorption smoke test, TiO2(110) vs. Andriuc, Siron & Persson 2025 and IrO2(110)
# CO vs. Pope, Yun et al. 2025 -- see scripts/ClaudeScripts/co_h_adsorption_benchmark.py
# for the rationale and the settings differences between the two papers. One relaxation
# per (oxide, adsorbate, model): CO ontop the undercoordinated metal cation (Ti5c/Ircus),
# H ontop the bridging O2c site (its real preferred site per Andriuc et al.'s own
# finding, not the metal). seed_standoff=0.5 deliberately not tighter -- a prior local
# probe showed CO pushed in tighter relaxes straight back out to ~4 A instead of binding.
# Default: 4 models x (TiO2 CO + TiO2 H + IrO2 CO) = 12 pairs, still small and fast.
# Needs the -gpu account (a different allocation from the CPU jobs elsewhere in this repo
# -- Sockeye rejects --gres=gpu otherwise). Submit from /scratch:
#
#   cd /scratch/st-akkiraju-1/$USER
#   sbatch /arc/project/st-akkiraju-1/ssong18/fullworkflow/scripts/slurm/sockeye_co_h_adsorption_benchmark.slurm
#
# Extra sbatch args after the script path forward straight through:
#   sbatch sockeye_co_h_adsorption_benchmark.slurm --oxides IrO2 --adsorbates CO
#   sbatch sockeye_co_h_adsorption_benchmark.slurm --models UMA-omat UMA-oc22

set -euo pipefail

PROJECT=/arc/project/st-akkiraju-1/ssong18
REPO="$PROJECT/fullworkflow"

export OXW_CONDA_BASE="$PROJECT/miniforge3"
export OXW_MODEL_DIR="$PROJECT/models"

# Compute nodes have no outbound network -- everything here uses locally-staged
# checkpoints, none of which needs a runtime download, but these stop anything that would
# try from hanging until the walltime runs out instead of failing fast.
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

# $HOME and /arc/project are read-only on compute nodes; only /scratch is writable, so
# every library that caches under $HOME by default has to be redirected or it dies
# mid-run (triton, torch inductor, huggingface, matplotlib via pymatgen, fairchem
# separately since it ignores XDG_CACHE_HOME).
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
python -c "import oxide_workflow, sys; print('python', sys.version.split()[0])"

for ckpt in mace-mh-1.model uma-s-1p2.pt sevennet-omni.pth; do
    test -f "$OXW_MODEL_DIR/$ckpt" || {
        echo "missing checkpoint: $OXW_MODEL_DIR/$ckpt" >&2
        exit 1
    }
done

OUT="$SLURM_SUBMIT_DIR/runs/co_h_ads_benchmark"

STATUS=0
python "$REPO/scripts/ClaudeScripts/co_h_adsorption_benchmark.py" "$OUT" "$@" \
    || STATUS=$?

echo
echo "results in scratch: $OUT/results.jsonl"
echo "to keep them, run this on the LOGIN node (scratch is purged on a timer):"
D="$PROJECT/runs/$(date +%Y%m%d-%H%M%S)_co_h_ads_benchmark"
echo "    mkdir -p $D && rsync -a $OUT/ $D/ && cp $SLURM_SUBMIT_DIR/slurm-${SLURM_JOB_ID}.{out,err} $D/"

exit $STATUS
