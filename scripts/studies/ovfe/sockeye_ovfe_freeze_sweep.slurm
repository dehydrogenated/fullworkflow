#!/bin/bash
#SBATCH --account=st-akkiraju-1-gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --job-name=ovfe-freeze
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#
# OVFE convergence vs. how much of the slab is frozen at bulk positions (0 through 7 of 8
# trilayers), on the 3 top-performing OVFE-benchmark models (UMA-omat, eSEN-30M-OAM,
# Orb-v2). Slab fixed at 8 trilayers for this sweep (separate from ovfe_height_sweep.py);
# supercell/vacuum held at the current SlabConfig default (4x2, min_vacuum_size=20.0).
# Stops at 7/8, not 8/8 -- freezing every atom would also freeze the vacancy site's own
# neighbors, so the defective structure could never relax.
#
# GPU by default (per Sean, 2026-08-20) -- --time=6:00:00 is carried over from a CPU
# estimate, not a measured GPU budget; treat it as a ceiling.
#
#   cd /scratch/st-akkiraju-1/$USER
#   sbatch /arc/project/st-akkiraju-1/ssong18/fullworkflow/scripts/studies/ovfe/sockeye_ovfe_freeze_sweep.slurm

set -euo pipefail

source /arc/project/st-akkiraju-1/ssong18/fullworkflow/scripts/slurm/_env.sh

for ckpt in uma-s-1p2.pt esen_30m_oam.pt orb-v2-20241011.ckpt; do
    test -f "$OXW_MODEL_DIR/$ckpt" || echo "note: $OXW_MODEL_DIR/$ckpt not staged -- that model's rows will be marked failed"
done

OUT="$SLURM_SUBMIT_DIR/runs/ovfe_freeze_sweep"
python "$REPO/scripts/studies/ovfe/ovfe_freeze_sweep.py" "$OUT"

echo
echo "results in scratch: $OUT/results.jsonl"
echo "to keep them, run this on the LOGIN node (scratch is purged on a timer):"
D="$PROJECT/runs/$(date +%Y%m%d-%H%M%S)_ovfe_freeze_sweep"
echo "    mkdir -p $D && rsync -a $OUT/ $D/ && cp $SLURM_SUBMIT_DIR/slurm-${SLURM_JOB_ID}.{out,err} $D/"
