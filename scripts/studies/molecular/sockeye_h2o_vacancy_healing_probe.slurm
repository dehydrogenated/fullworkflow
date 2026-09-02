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
# then nudged apart and re-relaxed -- see scripts/studies/molecular/h2o_vacancy_healing_probe.py for the
# rationale. 6 relaxations total (bulk + slab + vacancy + 2 placements x 2 relaxations
# each). The vacancy-build and min-image-direction pieces were smoke-tested locally
# (MACE-mh1-omat, loose fmax, 15 steps) before this was written -- see conversation
# history for the periodic-boundary bug that test caught and fixed. Submit from /scratch:
#
#   cd /scratch/st-akkiraju-1/$USER
#   sbatch /arc/project/st-akkiraju-1/ssong18/fullworkflow/scripts/studies/molecular/sockeye_h2o_vacancy_healing_probe.slurm
#
#   sbatch --account=st-akkiraju-1-gpu --partition=gpu --gres=gpu:1 --time=1:30:00 \
#          /arc/project/st-akkiraju-1/ssong18/fullworkflow/scripts/studies/molecular/sockeye_h2o_vacancy_healing_probe.slurm
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

source /arc/project/st-akkiraju-1/ssong18/fullworkflow/scripts/slurm/_env.sh
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
python "$REPO/scripts/studies/molecular/h2o_vacancy_healing_probe.py" "$OUT" --models UMA-M-omat "$@" \
    || STATUS=$?

echo
echo "results in scratch: $OUT/results.jsonl"
echo "to keep them, run this on the LOGIN node (scratch is purged on a timer):"
D="$PROJECT/runs/$(date +%Y%m%d-%H%M%S)_h2o_vacancy_healing_probe"
echo "    mkdir -p $D && rsync -a $OUT/ $D/ && cp $SLURM_SUBMIT_DIR/slurm-${SLURM_JOB_ID}.{out,err} $D/"

exit $STATUS
