#!/bin/bash
#SBATCH --account=st-akkiraju-1
#SBATCH --partition=cascade
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --job-name=co2-retest
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#
# CO2 adsorption retest (5 azimuthal orientations + larger standoff + early-stop desorb
# check) across the rutile family — see scripts/co2_adsorption_retest.py for the rationale.
#
# Same environment contract as sockeye_job.sh (read that one first if anything here is
# surprising) — reused directly rather than re-derived, since it's already the
# battle-tested version:
#   - submit from /scratch, NOT from the repo in /arc/project (rejected outright)
#   - GPU needs a *different* account (st-akkiraju-1-gpu) and --gres=gpu, granted at
#     submission time — command-line flags beat the #SBATCH defaults above:
#
#     cd /scratch/st-akkiraju-1/$USER
#     sbatch /arc/project/st-akkiraju-1/ssong18/fullworkflow/scripts/sockeye_co2_retest.slurm
#
#     sbatch --account=st-akkiraju-1-gpu --partition=gpu --gres=gpu:1 --time=6:00:00 \
#            /arc/project/st-akkiraju-1/ssong18/fullworkflow/scripts/sockeye_co2_retest.slurm
#
# 180 relaxations (3 sites x 6 orientations x 5 oxides x 2 models), early-stop pruning the
# doomed ones at step 100 instead of running to 300 — GPU strongly recommended over cascade.

set -euo pipefail

PROJECT=/arc/project/st-akkiraju-1/ssong18
REPO="$PROJECT/fullworkflow"

export OXW_CONDA_BASE="$PROJECT/miniforge3"
export OXW_MODEL_DIR="$PROJECT/models"

# Compute nodes have no outbound network — this only ever uses UMA-omat/MACE-mh1-omat
# (mace-clean/fairchem envs), neither of which needs a runtime download, but these stop
# anything that would try from hanging until the walltime runs out.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"

# Ask SLURM what it actually granted rather than assuming — see sockeye_job.sh for why.
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    export OXW_DEVICE=cuda
else
    export OXW_DEVICE=cpu
fi

# $HOME and /arc/project are read-only on compute nodes; every library that caches under
# $HOME has to be redirected or it dies mid-run (triton, torch inductor, huggingface,
# matplotlib via pymatgen, fairchem separately since it ignores XDG_CACHE_HOME).
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
echo "conda base $OXW_CONDA_BASE"
echo "models     $OXW_MODEL_DIR"
python -c "import oxide_workflow, sys; print('python', sys.version.split()[0])"

test -f "$OXW_MODEL_DIR/mace-mh-1.model" || {
    echo "missing checkpoint: $OXW_MODEL_DIR/mace-mh-1.model" >&2
    exit 1
}
test -f "$OXW_MODEL_DIR/uma-s-1p2.pt" || {
    echo "missing checkpoint: $OXW_MODEL_DIR/uma-s-1p2.pt" >&2
    exit 1
}

# Absolute, and it matters: /arc/project is read-only from compute nodes, so a relative
# outdir landing inside the repo would fail to write. This keeps it in scratch instead —
# same reason as oc22_diverge.py's --out in sockeye_oc22.sh.
OUT="$SLURM_SUBMIT_DIR/runs/co2_ads_benchmark"

# If the Mac run's bulk/slab leaves were rsynced into $OUT already, _relax_record's resume
# logic picks them up for free and only the (standoff/orientation-affected) adsorbate stage
# actually reruns. Nothing breaks if they aren't there — it just relaxes bulk/slab fresh.
STATUS=0
python "$REPO/scripts/co2_adsorption_retest.py" "$OUT" --fmax 0.02 || STATUS=$?

echo
echo "results in scratch: $OUT/results_retest.jsonl"
echo "to keep them, run this on the LOGIN node (scratch is purged on a timer):"
D="$PROJECT/runs/$(date +%Y%m%d-%H%M%S)_co2_retest"
echo "    mkdir -p $D && rsync -a $OUT/ $D/ && cp $SLURM_SUBMIT_DIR/slurm-${SLURM_JOB_ID}.{out,err} $D/"

exit $STATUS
