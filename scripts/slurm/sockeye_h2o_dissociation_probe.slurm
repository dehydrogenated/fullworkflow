#!/bin/bash
#SBATCH --account=st-akkiraju-1-gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --job-name=h2o-dissoc
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#
# H2O dissociation probe on TiO2(110): does a well-oriented starting guess find the
# dissociated state (Ti5c-OH + O2c-H) on its own? SevenNet-omni-omat24 only, x 4
# orientation guesses (vertical, vertical_flipped, bisector, single_h_ti_facing), each
# relaxed and then nudged apart and re-relaxed -- see scripts/h2o_dissociation_probe.py
# for the rationale, and its docstrings for nudge_apart()/is_dissociated(). The
# nudge-apart + second-relaxation path was smoke-tested locally (loose fmax, 15 steps,
# MACE-mh1-omat) before this was written: no crashes, atom count preserved, geometry/
# energy moved sensibly. 10 relaxations total (bulk + slab + 4 orientations x 2
# relaxations each). Needs the -gpu account (a different allocation from the CPU jobs
# elsewhere in this repo -- Sockeye rejects --gres=gpu otherwise). Submit from /scratch:
#
#   cd /scratch/st-akkiraju-1/$USER
#   sbatch /arc/project/st-akkiraju-1/ssong18/fullworkflow/scripts/slurm/sockeye_h2o_dissociation_probe.slurm
#
# Extra sbatch args after the script path forward straight through, e.g. to add models
# back in (both would then need a CUDA-built torch in their own conda env too):
#   sbatch sockeye_h2o_dissociation_probe.slurm --models SevenNet-omni-omat24 UMA-oc22

set -euo pipefail

PROJECT=/arc/project/st-akkiraju-1/ssong18
REPO="$PROJECT/fullworkflow"

export OXW_CONDA_BASE="$PROJECT/miniforge3"
export OXW_MODEL_DIR="$PROJECT/models"

# Compute nodes have no outbound network -- this only ever uses locally-staged
# checkpoints (mace-clean/fairchem/sevenn envs), none of which needs a runtime download,
# but these stop anything that would try from hanging until the walltime runs out.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"

# Ask SLURM what it actually granted rather than assuming -- see sockeye_job.sh for why.
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
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
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$HF_HOME" "$MPLCONFIGDIR" \
         "$FAIRCHEM_CACHE_DIR"

source "$OXW_CONDA_BASE/etc/profile.d/conda.sh"
conda activate oxw

cd "$SLURM_SUBMIT_DIR"

echo "job $SLURM_JOB_ID on $(hostname), ${SLURM_CPUS_PER_TASK:-?} cpus, device $OXW_DEVICE"
echo "conda base $OXW_CONDA_BASE"
echo "models     $OXW_MODEL_DIR"
python -c "import oxide_workflow, sys; print('python', sys.version.split()[0])"

# Fail before burning walltime if the checkpoint never made it over.
test -f "$OXW_MODEL_DIR/sevennet-omni.pth" || {
    echo "missing checkpoint: $OXW_MODEL_DIR/sevennet-omni.pth" >&2
    exit 1
}

# Absolute, and it matters: /arc/project is read-only from compute nodes, so a relative
# outdir landing inside the repo would fail to write. This keeps it in scratch instead.
OUT="$SLURM_SUBMIT_DIR/runs/h2o_dissociation_probe"

# If the Mac run's bulk/slab leaves were rsynced into $OUT already, _relax_record's
# resume logic picks them up for free and only the adsorbate stage actually reruns.
# Nothing breaks if they aren't there -- it just relaxes bulk/slab fresh. --models
# hardcoded to just SevenNet; "$@" still lets it be overridden at submission time.
STATUS=0
python "$REPO/scripts/ClaudeScripts/h2o_dissociation_probe.py" "$OUT" --models SevenNet-omni-omat24 "$@" \
    || STATUS=$?

echo
echo "results in scratch: $OUT/results.jsonl"
echo "to keep them, run this on the LOGIN node (scratch is purged on a timer):"
D="$PROJECT/runs/$(date +%Y%m%d-%H%M%S)_h2o_dissociation_probe"
echo "    mkdir -p $D && rsync -a $OUT/ $D/ && cp $SLURM_SUBMIT_DIR/slurm-${SLURM_JOB_ID}.{out,err} $D/"

exit $STATUS
