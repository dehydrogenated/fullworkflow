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
# Submit from the repo root:
#     sbatch scripts/sockeye_job.sh
#     sbatch scripts/sockeye_job.sh --material mp-825 --protocol seeded
# Anything after the script name is forwarded to the pipeline verbatim.

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

source "$OXW_CONDA_BASE/etc/profile.d/conda.sh"
conda activate oxw

cd "$SLURM_SUBMIT_DIR"

echo "job $SLURM_JOB_ID on $(hostname), ${SLURM_CPUS_PER_TASK:-?} cpus"
echo "conda base $OXW_CONDA_BASE"
echo "models     $OXW_MODEL_DIR"
python -c "import oxide_workflow, sys; print('python', sys.version.split()[0])"

# Fail before burning walltime if the checkpoint never made it over.
test -f "$OXW_MODEL_DIR/mace-mh-1.model" || {
    echo "missing checkpoint: $OXW_MODEL_DIR/mace-mh-1.model" >&2
    exit 1
}

srun python -m oxide_workflow.pipeline "$@"
