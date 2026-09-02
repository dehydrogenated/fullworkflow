#!/bin/bash
#SBATCH --account=st-akkiraju-1-gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=10:00:00
#SBATCH --job-name=propane-propene-ads-v2
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#
# Propane and propene (propylene) adsorption on rutile TiO2/RuO2/IrO2(110) vs. van Hout et
# al. 2026 (GAME-Net-Ox, Digital Discovery 5, 407-414) -- see
# scripts/studies/molecular/propane_propene_adsorption_benchmark.py for the full rationale, the
# literature numbers, and how they were identified from the paper's own relaxed CONTCARs
# (downloaded from ioChem-BD, not tabulated in the paper's text). NOT a multi-site screen:
# one fixed orientation each (vertical terminal-CH3-down for propane, side-on through the
# C=C for propene -- see config.py's ADSORBATE_FRAGMENTS comments), placed once on the
# single undercoordinated metal-cation ontop site (top_M5c equivalent), same convention as
# co_h_adsorption_benchmark.py. 3 models x 3 oxides x 2 adsorbates = 18 (oxide, adsorbate,
# model) combos, bulk/slab shared between Propane and Propene per (oxide, model) pair, on
# our 4x2 pristine slab (bigger than the paper's p(2x2), deliberately, to avoid the
# self-image clash a 1x1-cell smoke test hit during development -- see conversation).
#
# v2 of this job: the first run (runs/propane_propene_ads_benchmark) used
# seed_standoff=1.5, which -- because this codebase adds seed_standoff ON TOP OF a
# covalent-radius-sum estimate, not as a total distance the way the paper's own 1.5-3 A
# figure is -- seeded every anchor ~3.2-3.4 A out, past the edge of a physisorption well
# that only extends to ~3.0-3.5 A, so most pairs just sat there and called it "converged"
# without ever sliding in. Fixed to seed_standoff=0.5 (matching
# co_h_adsorption_benchmark.py's own validated value) and fmax tightened to 0.01 (from this
# repo's usual 0.02) to be sure FIRE actually finds these shallow wells rather than stopping
# early in a near-flat region. Writes to a NEW output folder
# (runs/propane_propene_ads_v2) rather than overwriting the first (invalid) run's results.
# 10h budget, up from the first run's 6h, since fmax=0.01 needs more steps per pair to
# converge than 0.02 did -- if it's not enough, rerun the leftover (oxide, adsorbate,
# model) combos directly with the --oxides/--adsorbates/--models flags below rather than
# widening this file's time limit blindly. Needs the -gpu account (different allocation
# from the CPU jobs elsewhere in this repo -- Sockeye rejects --gres=gpu otherwise). Submit
# from /scratch:
#
#   cd /scratch/st-akkiraju-1/$USER
#   sbatch /arc/project/st-akkiraju-1/ssong18/fullworkflow/scripts/studies/molecular/sockeye_propane_propene_adsorption_benchmark.slurm
#
# Extra sbatch args after the script path forward straight through -- e.g. to calibrate
# timing on a small slice first:
#   sbatch sockeye_propane_propene_adsorption_benchmark.slurm --oxides TiO2 --models Orb-v2
#   sbatch sockeye_propane_propene_adsorption_benchmark.slurm --adsorbates Propane

set -euo pipefail

source /arc/project/st-akkiraju-1/ssong18/fullworkflow/scripts/slurm/_env.sh
python -c "import oxide_workflow, sys; print('python', sys.version.split()[0])"

# Only the checkpoints this specific benchmark's three models need -- Orb-v2, eSEN-30M-OAM,
# UMA-M-oc20 -- not the full roster every model-agnostic job here checks.
for ckpt in orb-v2-20241011.ckpt esen_30m_oam.pt uma-m-1p1.pt; do
    test -f "$OXW_MODEL_DIR/$ckpt" || {
        echo "missing checkpoint: $OXW_MODEL_DIR/$ckpt" >&2
        exit 1
    }
done

OUT="$SLURM_SUBMIT_DIR/runs/propane_propene_ads_v2"

STATUS=0
python "$REPO/scripts/studies/molecular/propane_propene_adsorption_benchmark.py" "$OUT" "$@" \
    || STATUS=$?

echo
echo "results in scratch: $OUT/results.jsonl"
echo "to keep them, run this on the LOGIN node (scratch is purged on a timer):"
D="$PROJECT/runs/$(date +%Y%m%d-%H%M%S)_propane_propene_ads_v2"
echo "    mkdir -p $D && rsync -a $OUT/ $D/ && cp $SLURM_SUBMIT_DIR/slurm-${SLURM_JOB_ID}.{out,err} $D/"

exit $STATUS
