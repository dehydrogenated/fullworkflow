#!/bin/bash
#SBATCH --account=st-akkiraju-1-gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=4:00:00  # was 2h for TiO2 alone; doubled for the default 3-oxide sweep
#SBATCH --job-name=h2o-dissoc
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#
# H2O dissociation probe: does a well-oriented starting guess find the dissociated state
# (M5c-OH + O2c-H) on its own? Covers TiO2/RuO2/IrO2 (--oxides, default all 3) --
# Gonzalez et al. 2019's three materials -- SevenNet-omni-omat24 only by default. Default
# orientation is "bisector" + a DIAGONAL_NUDGE rigid translate toward O2c (see the Python
# script's own docstring/comments): single_h_ti_facing was tried and repulsed instead of
# adsorbing, rotation-only bisector adsorbed but left the approaching H ~1.95 A out after
# a full relaxation, so the translate closes part of that gap before relaxation starts.
# Each (oxide, model) pair is independent and failure-isolated -- one pair erroring
# doesn't stop the rest or lose their rows, unlike before this was added. Needs the -gpu
# account (a different allocation from the CPU jobs elsewhere in this repo -- Sockeye
# rejects --gres=gpu otherwise). Submit from /scratch:
#
#   cd /scratch/st-akkiraju-1/$USER
#   sbatch /arc/project/st-akkiraju-1/ssong18/fullworkflow/scripts/studies/molecular/sockeye_h2o_dissociation_probe.slurm
#
# Extra sbatch args after the script path forward straight through, e.g. to narrow to one
# material or add models back in (each extra model needs a CUDA-built torch in its own
# conda env too):
#   sbatch sockeye_h2o_dissociation_probe.slurm --oxides RuO2 IrO2
#   sbatch sockeye_h2o_dissociation_probe.slurm --models SevenNet-omni-omat24 UMA-oc22

set -euo pipefail

source /arc/project/st-akkiraju-1/ssong18/fullworkflow/scripts/slurm/_env.sh
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
python "$REPO/scripts/studies/molecular/h2o_dissociation_probe.py" "$OUT" --models SevenNet-omni-omat24 "$@" \
    || STATUS=$?

echo
echo "results in scratch: $OUT/results.jsonl"
echo "to keep them, run this on the LOGIN node (scratch is purged on a timer):"
D="$PROJECT/runs/$(date +%Y%m%d-%H%M%S)_h2o_dissociation_probe"
echo "    mkdir -p $D && rsync -a $OUT/ $D/ && cp $SLURM_SUBMIT_DIR/slurm-${SLURM_JOB_ID}.{out,err} $D/"

exit $STATUS
