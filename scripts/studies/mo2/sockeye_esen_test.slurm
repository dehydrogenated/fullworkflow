#!/bin/bash
#SBATCH --account=st-akkiraju-1
#SBATCH --partition=cascade
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --job-name=esen-test
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#
# Smoke test for the new eSEN-30M-OAM loader (backends.py's "esen" loader, OCPCalculator
# via fairchem-core==1.10.0 in its own "esen" env) -- built against the documented 1.10.0
# source but never actually run, since that means installing fairchem-core==1.10.0
# somewhere, and the only place that's supposed to happen is this env, not the dev laptop.
# This is the first real test of it.
#
# Deliberately tiny: TiO2 only (the most-exercised, best-understood oxide all session),
# both facets, both adsorbates -- 4 legs, not the full 33-oxide sweep. If this doesn't
# converge cleanly, the sweep jobs won't either, so this is worth running and checking
# before touching sockeye_mo2_sweep_*.slurm at all.
#
# PREREQUISITES:
#   1. Repo up to date: git pull origin main
#   2. "esen" env exists with fairchem-core==1.10.0 + ase installed (see backends.py's
#      ESEN_CKPT/_esen() comment, or the setup commands from earlier in this project's
#      history, for why it's a separate env from "fairchem")
#   3. esen_30m_oam.pt staged at $PROJECT/models/ (checked below, fatal if missing)
#
#   cd /scratch/st-akkiraju-1/$USER
#   sbatch /arc/project/st-akkiraju-1/ssong18/fullworkflow/scripts/studies/mo2/sockeye_esen_test.slurm

set -euo pipefail

source /arc/project/st-akkiraju-1/ssong18/fullworkflow/scripts/slurm/_env.sh

test -d "$OXW_CONDA_BASE/envs/esen" || {
    echo "missing conda env: $OXW_CONDA_BASE/envs/esen -- create it first (fairchem-core==1.10.0, ase)" >&2
    exit 1
}
test -f "$OXW_MODEL_DIR/esen_30m_oam.pt" || {
    echo "missing checkpoint: $OXW_MODEL_DIR/esen_30m_oam.pt" >&2
    exit 1
}

OUT="$SLURM_SUBMIT_DIR/runs/esen_test"
python "$REPO/scripts/studies/mo2/mo2_adsorption_benchmark.py" "$OUT" \
    --oxides TiO2 --facets 110 100 --adsorbates O OH --models eSEN-30M-OAM --fmax 0.02 "$@"

echo
echo "results in scratch: $OUT/results.jsonl"
echo "to keep them, run this on the LOGIN node (scratch is purged on a timer):"
D="$PROJECT/runs/$(date +%Y%m%d-%H%M%S)_esen_test"
echo "    mkdir -p $D && rsync -a $OUT/ $D/ && cp $SLURM_SUBMIT_DIR/slurm-${SLURM_JOB_ID}.{out,err} $D/"
echo "then pull it to your laptop with:"
echo "    scripts/slurm/sync_sockeye_runs.sh $(basename "$D")"
