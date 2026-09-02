#!/bin/bash
#SBATCH --account=st-akkiraju-1
#SBATCH --partition=cascade
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --job-name=retest-ext
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#
# Re-runs the 7 legs that didn't converge in the medium test, after the worker_relax.py
# extension-logic fix (compare against the anchor's start distance, not a rolling
# per-window baseline). scripts/studies/mo2/retest_extension_fix.py has the exact list
# and old numbers for comparison. Points at the medium test's OWN output directory so
# bulk/slab resume from the already-converged cached leaves (pipeline._resume) instead of
# recomputing -- verified locally: first relax_bulk_slab call ~67s, second (resumed) call
# ~0.1s, identical energy. Only the adsorbate leg (which always relaxes fresh, no resume
# check on that path) actually spends new compute.
#
# GPU recommended -- 7 legs, should be quick either way, but no reason not to:
#   sbatch --account=st-akkiraju-1-gpu --partition=gpu --gres=gpu:1 --time=1:00:00 \
#          /arc/project/st-akkiraju-1/ssong18/fullworkflow/scripts/studies/mo2/sockeye_retest_extension_fix.slurm
#
# PREREQUISITES:
#   1. Repo up to date: git pull origin main (needs the extension-logic fix commit)
#   2. runs/mo2_medium_test/ must already exist from the medium test job -- this reuses it,
#      does not recreate it from scratch
#
#   cd /scratch/st-akkiraju-1/$USER
#   sbatch /arc/project/st-akkiraju-1/ssong18/fullworkflow/scripts/studies/mo2/sockeye_retest_extension_fix.slurm

set -euo pipefail

source /arc/project/st-akkiraju-1/ssong18/fullworkflow/scripts/slurm/_env.sh

OUT="$SLURM_SUBMIT_DIR/runs/mo2_medium_test"
test -d "$OUT" || {
    echo "missing $OUT -- this reuses the medium test's own output dir, run that first" >&2
    exit 1
}

python "$REPO/scripts/studies/mo2/retest_extension_fix.py" "$OUT"

echo
echo "appended to: $OUT/results.jsonl (dedupe by keeping the LAST row per"
echo "  (model, oxide, facet, adsorbate) if analyzing alongside the original medium-test rows)"
