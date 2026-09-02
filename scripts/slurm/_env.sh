# Shared Sockeye environment contract -- sourced by every job script in this repo.
#
#   set -euo pipefail
#   source /arc/project/st-akkiraju-1/ssong18/fullworkflow/scripts/slurm/_env.sh
#
# Sourced with an absolute path on purpose: SLURM copies the batch script into a spool
# directory before running it, so ${BASH_SOURCE[0]} points at /var/spool/... and cannot be
# used to find the repo. $PROJECT is fixed on Sockeye, so hardcoding it is the honest option.
#
# Source this AFTER `set -euo pipefail` so a missing/broken env file aborts the job instead
# of running it half-configured.
#
# Compute nodes are far more restricted than the login node, and every export below exists
# because one of those restrictions bit an actual run:
#   - no outbound network      -> anything that reaches out must fail fast, not hang to walltime
#   - $HOME and /arc/project are READ-ONLY -> every library cache must be redirected to scratch
#   - GPUs are granted by the scheduler, not by a flag in a file
#
# Overrides a job may set BEFORE sourcing:
#   OXW_DEVICE=cpu    pin the device (see the detection block below)
#   OXW_ENV=<name>    activate a conda env other than `oxw`
#   OXW_CACHE=<dir>   cache root, default $SLURM_SUBMIT_DIR/.cache

: "${SLURM_SUBMIT_DIR:?_env.sh must be sourced from inside a SLURM job (SLURM_SUBMIT_DIR unset)}"

PROJECT=/arc/project/st-akkiraju-1/ssong18
REPO="$PROJECT/fullworkflow"

export OXW_CONDA_BASE="$PROJECT/miniforge3"
export OXW_MODEL_DIR="$PROJECT/models"

# Compute nodes have no outbound network. Without these, a library that tries to reach out
# hangs until the walltime expires instead of failing immediately.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"

# eSEN OOMed mid-sweep at ~384+ atoms even with a fresh subprocess per relaxation; the
# driver-reported gap between allocated and reserved-but-unallocated memory pointed at
# fragmentation rather than raw need. This is PyTorch's own suggested mitigation from that
# error message. Inert on CPU, so it is set unconditionally.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Device comes from what the scheduler actually granted, never from a default -- so a
# forgotten --gres=gpu shows up as "device cpu" in the banner below instead of silently
# paying for a GPU node and running on its CPUs.
#
# nvidia-smi, NOT CUDA_VISIBLE_DEVICES: that env var is inherited from the submitting shell
# by sbatch's default --export=ALL, so a stray value left over from an earlier interactive
# GPU session can make a --partition=cascade job (no --gres=gpu) wrongly resolve to cuda.
# See sockeye_job.sh for the full incident. nvidia-smi respects SLURM's cgroup isolation.
#
# A job may pin OXW_DEVICE before sourcing -- the case that needs it is having a GPU but
# being unable to use it (Orb-v2, CHGNet-0.3.0: no kernels for this V100's torch build).
if [ -z "${OXW_DEVICE:-}" ]; then
    if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L 2>/dev/null | grep -q "GPU"; then
        export OXW_DEVICE=cuda
    else
        export OXW_DEVICE=cpu
    fi
fi

# $HOME and /arc/project are read-only from a compute node, so every library that caches
# under $HOME by default has to be pointed at scratch or it dies mid-run. fairchem and
# cached_path (pulled in by orb-models) each ignore XDG_CACHE_HOME and need their own var --
# confirmed by an actual "OSError: Read-only file system", not assumed.
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
conda activate "${OXW_ENV:-oxw}"

cd "$SLURM_SUBMIT_DIR"

echo "job ${SLURM_JOB_ID:-?} on $(hostname), ${SLURM_CPUS_PER_TASK:-?} cpus, device $OXW_DEVICE, env ${OXW_ENV:-oxw}"

# Scratch is purged on a timer, so every job ends by printing the copy that makes its
# results durable. It can only be run from the LOGIN node -- /arc/project is read-only
# from inside a job -- which is why this prints a command instead of running it.
#   keep_results "$OUT" <run-name>
keep_results() {
    local out="$1" name="$2"
    local dest="$PROJECT/runs/$(date +%Y%m%d-%H%M%S)_$name"
    echo
    echo "results in scratch: $out"
    echo "to keep them, run this on the LOGIN node (scratch is purged on a timer):"
    echo "    mkdir -p $dest && rsync -a $out/ $dest/ && cp $SLURM_SUBMIT_DIR/slurm-${SLURM_JOB_ID:-JOBID}.{out,err} $dest/"
}
