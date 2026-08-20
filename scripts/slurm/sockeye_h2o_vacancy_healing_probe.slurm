#!/bin/bash
#SBATCH --account=st-akkiraju-1
#SBATCH --partition=cascade
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --job-name=h2o-vac-heal
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#
# H2O vacancy-healing probe on TiO2(110): does water's own O fill a bridging-O2c
# vacancy while a proton transfers to a *different*, still-intact neighboring O2c?
# UMA-M-omat only, x 2 placement guesses (direct_fill, bridge_site), each relaxed and
# then nudged apart and re-relaxed -- see scripts/h2o_vacancy_healing_probe.py for the
# rationale. 6 relaxations total (bulk + slab + vacancy + 2 placements x 2 relaxations
# each). The vacancy-build and min-image-direction pieces were smoke-tested locally
# (MACE-mh1-omat, loose fmax, 15 steps) before this was written -- see conversation
# history for the periodic-boundary bug that test caught and fixed. Submit from /scratch:
#
#   cd /scratch/st-akkiraju-1/$USER
#   sbatch /arc/project/st-akkiraju-1/ssong18/fullworkflow/scripts/slurm/sockeye_h2o_vacancy_healing_probe.slurm
#
#   sbatch --account=st-akkiraju-1-gpu --partition=gpu --gres=gpu:1 --time=1:30:00 \
#          /arc/project/st-akkiraju-1/ssong18/fullworkflow/scripts/slurm/sockeye_h2o_vacancy_healing_probe.slurm
#
# UMA-M-omat needs uma-m-1p1.pt staged in $OXW_MODEL_DIR by hand (gated on HuggingFace,
# not auto-fetchable -- see backends.py's UMA_M_CKPT comment) -- a DIFFERENT checkpoint
# from UMA-oc22/UMA-omat's uma-s-1p2.pt. Also per backends.py: facebookresearch/
# fairchem#2095 reports uma-m-1p1 predictor construction stalling indefinitely on some
# setups (uma-s loads fine) -- if the job looks stuck at model load with no progress,
# that known issue, not necessarily a config mistake here, is the first thing to check.
#
# Extra sbatch args after the script path forward straight through, e.g.:
#   sbatch sockeye_h2o_vacancy_healing_probe.slurm --models UMA-M-omat UMA-oc22

set -euo pipefail

PROJECT=/arc/project/st-akkiraju-1/ssong18
REPO="$PROJECT/fullworkflow"

export OXW_CONDA_BASE="$PROJECT/miniforge3"
export OXW_MODEL_DIR="$PROJECT/models"

# Compute nodes have no outbound network -- this only ever uses locally-staged
# checkpoints (fairchem env), which needs no runtime download, but these stop anything
# that would try from hanging until the walltime runs out instead of failing fast.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"

# Ask SLURM what it actually granted rather than assuming -- see sockeye_job.sh for why.
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
echo "conda base $OXW_CONDA_BASE"
echo "models     $OXW_MODEL_DIR"
python -c "import oxide_workflow, sys; print('python', sys.version.split()[0])"

# Fail before burning walltime if the checkpoint never made it over.
test -f "$OXW_MODEL_DIR/uma-m-1p1.pt" || {
    echo "missing checkpoint: $OXW_MODEL_DIR/uma-m-1p1.pt" >&2
    echo "(gated on HuggingFace -- download by hand, see backends.py's UMA_M_CKPT comment)" >&2
    exit 1
}

# Absolute, and it matters: /arc/project is read-only from compute nodes, so a relative
# outdir landing inside the repo would fail to write. This keeps it in scratch instead.
OUT="$SLURM_SUBMIT_DIR/runs/h2o_vacancy_healing_probe"

# If the Mac run's bulk/slab leaves were rsynced into $OUT already, _relax_record's
# resume logic picks them up for free and only the vacancy/adsorbate stages actually
# rerun. Nothing breaks if they aren't there -- it just relaxes bulk/slab fresh.
# --models hardcoded to just UMA-M-omat; "$@" still lets it be overridden.
STATUS=0
python "$REPO/scripts/ClaudeScripts/h2o_vacancy_healing_probe.py" "$OUT" --models UMA-M-omat "$@" \
    || STATUS=$?

echo
echo "results in scratch: $OUT/results.jsonl"
echo "to keep them, run this on the LOGIN node (scratch is purged on a timer):"
D="$PROJECT/runs/$(date +%Y%m%d-%H%M%S)_h2o_vacancy_healing_probe"
echo "    mkdir -p $D && rsync -a $OUT/ $D/ && cp $SLURM_SUBMIT_DIR/slurm-${SLURM_JOB_ID}.{out,err} $D/"

exit $STATUS
