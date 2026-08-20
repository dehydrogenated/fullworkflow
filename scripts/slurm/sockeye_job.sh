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
#     sbatch /arc/project/st-akkiraju-1/$USER/fullworkflow/scripts/slurm/sockeye_job.sh
#     sbatch /arc/.../sockeye_job.sh --material mp-825 --protocol seeded
#
# Anything after the script name is forwarded to the pipeline verbatim. The package is
# installed editable, so it imports from /arc/project no matter where the job runs.
#
# GPU runs need a *different allocation* — Sockeye rejects --gres=gpu unless the account
# ends in -gpu — and command-line flags beat the #SBATCH lines below, so no edit is needed:
#
#     sbatch --account=st-akkiraju-1-gpu --partition=gpu --gres=gpu:1 --time=0:30:00 \
#            /arc/.../sockeye_job.sh --candidates UMA-oc22 ...
#
# BOTH worker envs need a CUDA torch for any GPU run — not just the one holding the
# candidate. RunConfig.reference is MACE-mh1-omat, so every run loads MACE first to build
# the ground truth, whatever --candidates says. A CPU-only mace-clean therefore fails even
# on a pure-UMA run, with "Attempting to deserialize object on a CUDA device but
# torch.cuda.is_available() is False".
#
# Install torch from cu126 in both: the GPUs are V100 (sm_70) and newer CUDA indexes have
# dropped Volta, while older ones (cu124) top out below the torch~=2.8.0 that
# fairchem-core 2.21.0 requires. cu126 is the overlap.
#
# /arc/project is mounted READ-ONLY on compute nodes, so the archive step below cannot
# succeed from inside a job — it warns and continues by design. Copy from the login node:
#     rsync -a /scratch/st-akkiraju-1/$USER/runs/ /arc/project/st-akkiraju-1/$USER/runs/
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

# Query the hardware directly rather than trusting CUDA_VISIBLE_DEVICES. This used to check
# that variable instead, on the theory that SLURM sets it only when --gres=gpu was granted --
# true of the scheduler's OWN behavior, but wrong in practice: sbatch inherits the entire
# submitting shell's environment by default (no --export=NONE here), so a CUDA_VISIBLE_DEVICES
# left over from an earlier interactive GPU session on the login node rides straight into a
# job that never requested one. Confirmed: a --partition=cascade job (no --gres=gpu at all)
# hit the identical "CUDA error: no kernel image is available for execution on the device"
# crash as the real GPU jobs, because OXW_DEVICE still resolved to cuda from the stale
# inherited value. nvidia-smi respects SLURM's own cgroup isolation, so -L correctly reports
# nothing for a job that wasn't granted a GPU regardless of what the shell exported:
#
#     sbatch --partition=gpu --gres=gpu:1 ...   -> cuda
#     sbatch --partition=cascade ...            -> cpu, even with a stale env var inherited
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L 2>/dev/null | grep -q "GPU"; then
    export OXW_DEVICE=cuda
else
    export OXW_DEVICE=cpu
fi

# Compute nodes mount BOTH $HOME and /arc/project read-only — scratch is the only writable
# filesystem. Every library that caches under $HOME therefore has to be redirected or it
# dies mid-run: triton (JIT-compiles fused GPU kernels, killed the first GPU run with
# "Read-only file system: /home/…/.triton"), torch inductor, huggingface, and matplotlib,
# which pymatgen pulls in. Kept outside $SLURM_SUBMIT_DIR/runs so compiled kernels survive
# between jobs — otherwise every run recompiles from scratch.
export XDG_CACHE_HOME="${OXW_CACHE:-$SLURM_SUBMIT_DIR/.cache}"
export TRITON_CACHE_DIR="$XDG_CACHE_HOME/triton"
export TORCHINDUCTOR_CACHE_DIR="$XDG_CACHE_HOME/torchinductor"
export HF_HOME="$XDG_CACHE_HOME/huggingface"
export MPLCONFIGDIR="$XDG_CACHE_HOME/matplotlib"
# fairchem ignores XDG_CACHE_HOME — fairchem/core/_config.py reads FAIRCHEM_CACHE_DIR and
# otherwise hardcodes ~/.cache/fairchem, then calls os.makedirs at *import* time. So it
# fails the moment the worker imports it, before any model is touched.
export FAIRCHEM_CACHE_DIR="$XDG_CACHE_HOME/fairchem"
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$HF_HOME" "$MPLCONFIGDIR" \
         "$FAIRCHEM_CACHE_DIR"

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

# The copy cannot happen here: /arc/project is read-only from compute nodes, so this used
# to warn on every single run for something that can never succeed. Print the command
# instead — it runs from the login node, where project IS writable.
echo
echo "results in scratch: $SLURM_SUBMIT_DIR/$OUTDIR"
echo "to keep them, run this on the LOGIN node:"
echo "    mkdir -p $DEST && rsync -a $SLURM_SUBMIT_DIR/$OUTDIR/ $DEST/ && cp $SLURM_SUBMIT_DIR/slurm-${SLURM_JOB_ID}.{out,err} $DEST/"

exit $STATUS
