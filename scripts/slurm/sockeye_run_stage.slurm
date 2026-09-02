#!/bin/bash
#SBATCH --account=st-akkiraju-1
#SBATCH --partition=cascade
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --job-name=run-stage
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#
# One stage only (bulk/slab/vacancy/adsorbate), skipping everything after it -- for
# OVFE-only comparisons this avoids the adsorbate sweep that dominates a full pipeline run.
# See scripts/core/run_stage.py for what the stages do and which flags it takes.
#
# CPU by default. Pass the GPU account/partition/gres at submission time to switch --
# command-line flags beat the #SBATCH defaults above, so no editing is needed.
#
#   cd /scratch/st-akkiraju-1/$USER
#   sbatch /arc/project/st-akkiraju-1/ssong18/fullworkflow/scripts/slurm/sockeye_run_stage.slurm \
#          vacancy --model UMA-M-omat --outdir runs/ovfe_uma_m_omat
#
#   sbatch --account=st-akkiraju-1-gpu --partition=gpu --gres=gpu:1 --time=0:30:00 \
#          /arc/project/st-akkiraju-1/ssong18/fullworkflow/scripts/slurm/sockeye_run_stage.slurm \
#          vacancy --model UMA-M-oc20 --outdir runs/ovfe_uma_m_oc20

set -euo pipefail
source /arc/project/st-akkiraju-1/ssong18/fullworkflow/scripts/slurm/_env.sh

test -f "$OXW_MODEL_DIR/mace-mh-1.model" || {
    echo "missing checkpoint: $OXW_MODEL_DIR/mace-mh-1.model" >&2
    exit 1
}
test -f "$OXW_MODEL_DIR/uma-m-1p1.pt" || echo "note: uma-m-1p1.pt not staged (needed for UMA-M-* models)"

# Everything after the script path forwards straight to run_stage.py -- stage, --model,
# --outdir, and any geometry/relax overrides, exactly like running it locally.
python "$REPO/scripts/core/run_stage.py" "$@"

# --outdir is caller-supplied, so this job can't name the destination for keep_results.
echo
echo "results in scratch under the --outdir you passed above"
echo "to keep them, run this on the LOGIN node (scratch is purged on a timer):"
echo "    mkdir -p $PROJECT/runs && rsync -a \$SLURM_SUBMIT_DIR/<outdir>/ $PROJECT/runs/\$(date +%Y%m%d-%H%M%S)_<outdir>/"
