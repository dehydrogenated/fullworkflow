#!/bin/bash
#SBATCH --account=st-akkiraju-1
#SBATCH --partition=cascade
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=3:00:00
#SBATCH --job-name=h2o-ads-cpu
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#
# CHGNet-0.3.0 + Orb-v2 companion to sockeye_h2o_adsorption_benchmark.slurm -- same
# script, same settings (nudge=1.0, fmax=0.01, TiO2/RuO2/IrO2), CPU forced directly rather
# than detected. Both models' "orb"/"chgnet" envs pull an unpinned torch+cu130 with no
# kernels for Sockeye's V100s (CC 7.0) -- confirmed real on the mo2_adsorption_benchmark
# sweep (see sockeye_chgnet_orb_cpu_backfill.slurm's own comment; same root cause, same
# fix). Both models measured running fine on CPU there (Orb-v2 at 0.45 s/step), so
# OXW_DEVICE is hardcoded here rather than trusting nvidia-smi detection -- nothing to
# misdetect since we're not asking for a GPU at all (no -gpu account, no --gres=gpu).
#
# Points at the SAME runs/h2o_adsorption_benchmark directory the GPU job writes to, so
# results.jsonl accumulates every model's rows together and bulk/slab resumes for free if
# this happens to run after the GPU job already relaxed a shared (oxide, model) pair
# (it won't, since CHGNet/Orb-v2 are model-namespaced, but the mechanism is free either
# way). Submit from /scratch, no GPU account needed:
#
#   cd /scratch/st-akkiraju-1/$USER
#   sbatch /arc/project/st-akkiraju-1/ssong18/fullworkflow/scripts/slurm/sockeye_h2o_adsorption_benchmark_cpu.slurm
#
# Extra sbatch args after the script path forward straight through:
#   sbatch sockeye_h2o_adsorption_benchmark_cpu.slurm --oxides RuO2 IrO2

set -euo pipefail

PROJECT=/arc/project/st-akkiraju-1/ssong18
REPO="$PROJECT/fullworkflow"

export OXW_CONDA_BASE="$PROJECT/miniforge3"
export OXW_MODEL_DIR="$PROJECT/models"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"

export OXW_DEVICE=cpu  # hardcoded on purpose -- see header, no detection needed here

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

for env in chgnet orb; do
    test -d "$OXW_CONDA_BASE/envs/$env" || {
        echo "missing conda env: $OXW_CONDA_BASE/envs/$env" >&2
        exit 1
    }
done
test -f "$OXW_MODEL_DIR/orb-v2-20241011.ckpt" || {
    echo "missing checkpoint: $OXW_MODEL_DIR/orb-v2-20241011.ckpt" >&2
    exit 1
}

OUT="$SLURM_SUBMIT_DIR/runs/h2o_adsorption_benchmark"

STATUS=0
python "$REPO/scripts/ClaudeScripts/h2o_adsorption_benchmark.py" "$OUT" \
    --models CHGNet-0.3.0 Orb-v2 --fmax 0.01 "$@" \
    || STATUS=$?

echo
echo "results in scratch: $OUT/results.jsonl (same file the GPU job writes to)"
echo "to keep them, run this on the LOGIN node (scratch is purged on a timer):"
D="$PROJECT/runs/$(date +%Y%m%d-%H%M%S)_h2o_adsorption_benchmark_cpu"
echo "    mkdir -p $D && rsync -a $OUT/ $D/ && cp $SLURM_SUBMIT_DIR/slurm-${SLURM_JOB_ID}.{out,err} $D/"

exit $STATUS
