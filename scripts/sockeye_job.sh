#!/bin/bash
#SBATCH --account=st-akkiraju-1
#SBATCH --partition=cascade
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=3:00:00
#SBATCH --job-name=oxw
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#
# One SLURM job = one full pipeline run on one node.
#
# relax() is a blocking, serial subprocess.run, so this asks for a single task and lets
# torch use the cores via threads. Nothing here parallelises across nodes; that is a
# later change to relax(), not to this script.
#
# Submit from a directory in /scratch, NOT from the repo in /arc/project — Sockeye
# rejects the job outright ("Submitting jobs from directories residing in /arc/project is
# not allowed"). Project space is backed up and snapshotted for durability; scratch is the
# parallel filesystem meant to absorb job I/O. Only the *working directory* is restricted,
# so the script itself stays in the repo and is named by absolute path:
#
#     cd /scratch/st-akkiraju-1/$USER
#     sbatch /arc/project/st-akkiraju-1/$USER/fullworkflow/scripts/sockeye_job.sh
#     sbatch /arc/.../sockeye_job.sh --material mp-825 --protocol seeded
#
# Anything after the script name is forwarded to the pipeline verbatim. The package is
# installed editable, so it imports from /arc/project no matter where the job runs.
#
# Consequence: --outdir is relative to $SLURM_SUBMIT_DIR, so results land in scratch,
# which is purged on a timer. See the end of this script.

set -euo pipefail

PROJECT=/arc/project/st-akkiraju-1/ssong18

# The two machine-specific paths backends.py reads (defaults are the local Mac layout).
export OXW_CONDA_BASE="$PROJECT/miniforge3"
export OXW_MODEL_DIR="$PROJECT/models"

# Compute nodes have no outbound network. Anything that would reach out at runtime has to
# be pre-staged on the login node — the checkpoints above, and every pip install. These
# stop MACE/torch/HF from trying anyway, so a mistake fails loudly instead of hanging
# until the walltime runs out.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Match torch's thread count to what the scheduler actually gave us; left unset, torch
# grabs every core on the node and thrashes against the other jobs sharing it.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"

# Ask SLURM what it gave us rather than taking a flag. CUDA_VISIBLE_DEVICES is set by the
# scheduler only when --gres=gpu was granted, so this is the authoritative answer and it
# cannot drift from reality:
#
#     sbatch --partition=gpu --gres=gpu:1 ...   -> cuda
#     sbatch --partition=cascade ...            -> cpu
#
# A forgotten flag would otherwise mean paying for a GPU node and running on its CPUs, and
# nothing in the output would say so.
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    export OXW_DEVICE=cuda
else
    export OXW_DEVICE=cpu
fi

source "$OXW_CONDA_BASE/etc/profile.d/conda.sh"
conda activate oxw

cd "$SLURM_SUBMIT_DIR"

echo "job $SLURM_JOB_ID on $(hostname), ${SLURM_CPUS_PER_TASK:-?} cpus"
echo "conda base $OXW_CONDA_BASE"
echo "models     $OXW_MODEL_DIR"
echo "device     $OXW_DEVICE"
python -c "import oxide_workflow, sys; print('python', sys.version.split()[0])"

# Fail before burning walltime if the checkpoint never made it over.
test -f "$OXW_MODEL_DIR/mace-mh-1.model" || {
    echo "missing checkpoint: $OXW_MODEL_DIR/mace-mh-1.model" >&2
    exit 1
}

# `|| STATUS=$?` rather than a bare call: `set -e` would abort the script the moment the
# pipeline exited non-zero, skipping the archive below — losing exactly the crashed run
# that is worth keeping. The real exit code is re-raised at the very end.
STATUS=0
srun python -m oxide_workflow.pipeline "$@" || STATUS=$?

# Archive to durable space. Results land in scratch (see the header), which is purged by
# last-access time, so a run left only there disappears silently. Unconditional on
# purpose: a failed run is usually the one you most want to inspect.
PROJECT_RUNS=/arc/project/st-akkiraju-1/ssong18/runs

# Defaults mirror the pipeline's own, so the label is right even when nothing was passed.
MATERIAL=rutile-tio2   # RunConfig.polymorph
ADSORBATE=H            # AdsorbateConfig.species
OUTDIR=runs/latest     # pipeline default
prev=""
for arg in "$@"; do
    case "$arg" in                      # --flag=value form
        --material=*)  MATERIAL="${arg#*=}" ;;
        --adsorbate=*) ADSORBATE="${arg#*=}" ;;
        --outdir=*)    OUTDIR="${arg#*=}" ;;
    esac
    case "$prev" in                     # --flag value form
        --material)  MATERIAL="$arg" ;;
        --adsorbate) ADSORBATE="$arg" ;;
        --outdir)    OUTDIR="$arg" ;;
    esac
    prev="$arg"
done

# --material may be a path to a CIF, so keep the leaf and strip anything that would break
# a directory name. Sorts chronologically because the timestamp leads.
TAG=$(basename "$MATERIAL")
TAG=${TAG//[^A-Za-z0-9._-]/-}
DEST="$PROJECT_RUNS/$(date +%Y%m%d-%H%M%S)_${TAG}_${ADSORBATE}"

# Guarded by `if` so a failed copy cannot trip `set -e` and swallow $STATUS.
if mkdir -p "$DEST" && rsync -a "$OUTDIR"/ "$DEST"/; then
    cp -f "slurm-${SLURM_JOB_ID}.out" "slurm-${SLURM_JOB_ID}.err" "$DEST"/ 2>/dev/null || true
    echo "archived to $DEST"
else
    echo "WARNING: archive to $DEST failed — results remain only in scratch" >&2
fi

exit $STATUS
