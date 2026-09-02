#!/bin/bash
#SBATCH --account=st-akkiraju-1
#SBATCH --partition=cascade
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --job-name=gas-refs
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#
# Gas-phase reference energy comparison across all 15 models -- NOT a dependency of the
# main sweep (mo2_adsorption_benchmark.py already computes each model's own H2/H2O
# references internally, self-consistently, same-calculator-cancellation style; see
# energetics.py's own docstring for why borrowing a foreign model's gas energies would be
# wrong). This is a separate diagnostic: how much do models' own gas-phase predictions
# actually disagree with each other, surfaced directly rather than left buried inside each
# sweep run's per-model cache.
#
# Cheap regardless of device -- every fragment (O, H, H2O, CO, OH) is 1-3 atoms, seconds
# per relaxation, 15 models x 5 fragments = 75 relaxations total. CPU default is plenty;
# GPU wouldn't meaningfully speed this up given how small each system is.
#
# PREREQUISITES:
#   1. Repo up to date: git pull origin main
#   2. All 15 models' envs/checkpoints already required by the sweep jobs -- same
#      requirements, nothing extra needed here
#
#   cd /scratch/st-akkiraju-1/$USER
#   sbatch /arc/project/st-akkiraju-1/ssong18/fullworkflow/scripts/studies/sockeye_gas_reference_report.slurm

set -euo pipefail

source /arc/project/st-akkiraju-1/ssong18/fullworkflow/scripts/slurm/_env.sh

OUT="$SLURM_SUBMIT_DIR/runs/gas_reference_report.csv"
python "$REPO/scripts/studies/gas_reference_report.py" --out "$OUT" "$@"

echo
echo "results in scratch: $OUT"
echo "to keep it, run this on the LOGIN node (scratch is purged on a timer):"
D="$PROJECT/runs/gas_reference_report"
echo "    mkdir -p $D && cp $OUT $D/ && cp $SLURM_SUBMIT_DIR/slurm-${SLURM_JOB_ID}.{out,err} $D/"
echo "then pull it to your laptop with:"
echo "    scripts/slurm/sync_sockeye_runs.sh $(basename "$D")"
