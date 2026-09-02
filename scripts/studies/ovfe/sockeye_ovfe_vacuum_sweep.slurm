#!/bin/bash
#SBATCH --account=st-akkiraju-1-gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --job-name=ovfe-vac
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#
# OVFE convergence vs. vacuum thickness (min_vacuum_size = 5,8,13,16,20,25,30,35,40,50 A),
# on the 3 top-performing OVFE-benchmark models (UMA-omat, eSEN-30M-OAM, Orb-v2). Bulk
# relaxed once per model, reused across every vacuum value; supercell/height/freeze held
# at the current SlabConfig default (4x2, 4 trilayers -- see
# ovfe_convergence_common.py's BASE_SLAB comment). See
# scripts/studies/ovfe/ovfe_vacuum_sweep.py for the full rationale. A model with a
# missing checkpoint/env is skipped, not fatal -- the other two still complete.
#
# GPU by default (per Sean, 2026-08-20) -- UMA-omat dominates wall-clock and the -gpu
# account/partition should help it the most; --time=4:00:00 is carried over from the CPU
# estimate unverified against an actual GPU run, so treat it as a conservative ceiling,
# not a measured budget.
#
#   cd /scratch/st-akkiraju-1/$USER
#   sbatch /arc/project/st-akkiraju-1/ssong18/fullworkflow/scripts/studies/ovfe/sockeye_ovfe_vacuum_sweep.slurm

set -euo pipefail

source /arc/project/st-akkiraju-1/ssong18/fullworkflow/scripts/slurm/_env.sh

# All three MODELS in ovfe_convergence_common.py are needed for this sweep to be complete
# -- note (not fatal) rather than exit, since run_sweep() itself now skips a model whose
# checkpoint/env is missing and continues with the rest.
for ckpt in uma-s-1p2.pt esen_30m_oam.pt orb-v2-20241011.ckpt; do
    test -f "$OXW_MODEL_DIR/$ckpt" || echo "note: $OXW_MODEL_DIR/$ckpt not staged -- that model's rows will be marked failed"
done

OUT="$SLURM_SUBMIT_DIR/runs/ovfe_vacuum_sweep"
python "$REPO/scripts/studies/ovfe/ovfe_vacuum_sweep.py" "$OUT"

echo
echo "results in scratch: $OUT/results.jsonl"
echo "to keep them, run this on the LOGIN node (scratch is purged on a timer):"
D="$PROJECT/runs/$(date +%Y%m%d-%H%M%S)_ovfe_vacuum_sweep"
echo "    mkdir -p $D && rsync -a $OUT/ $D/ && cp $SLURM_SUBMIT_DIR/slurm-${SLURM_JOB_ID}.{out,err} $D/"
