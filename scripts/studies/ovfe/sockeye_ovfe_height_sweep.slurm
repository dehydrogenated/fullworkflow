#!/bin/bash
#SBATCH --account=st-akkiraju-1-gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --job-name=ovfe-height
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#
# OVFE convergence vs. slab thickness (2,4,6,8,10 trilayers), on the 3 top-performing
# OVFE-benchmark models (UMA-omat, eSEN-30M-OAM, Orb-v2). Supercell/vacuum held at the
# current SlabConfig default (4x2, min_vacuum_size=20.0). freeze_bottom_fraction stays
# fixed at 0.5 throughout -- trilayer counts are restricted to even numbers specifically so
# that fraction always lands on a real trilayer boundary. See
# scripts/studies/ovfe/ovfe_height_sweep.py for how the min_slab_size values were derived
# (measured achievable-thickness boundaries, not a naive trilayer_thickness*k multiply --
# that overshoots SlabGenerator's own rounding and silently produces the wrong trilayer
# count).
#
# GPU by default (per Sean, 2026-08-20) -- --time=6:00:00 is carried over from a CPU
# estimate, not a measured GPU budget; treat it as a ceiling.
#
#   cd /scratch/st-akkiraju-1/$USER
#   sbatch /arc/project/st-akkiraju-1/ssong18/fullworkflow/scripts/studies/ovfe/sockeye_ovfe_height_sweep.slurm

set -euo pipefail

source /arc/project/st-akkiraju-1/ssong18/fullworkflow/scripts/slurm/_env.sh

for ckpt in uma-s-1p2.pt esen_30m_oam.pt orb-v2-20241011.ckpt; do
    test -f "$OXW_MODEL_DIR/$ckpt" || echo "note: $OXW_MODEL_DIR/$ckpt not staged -- that model's rows will be marked failed"
done

OUT="$SLURM_SUBMIT_DIR/runs/ovfe_height_sweep"
python "$REPO/scripts/studies/ovfe/ovfe_height_sweep.py" "$OUT"

echo
echo "results in scratch: $OUT/results.jsonl"
echo "to keep them, run this on the LOGIN node (scratch is purged on a timer):"
D="$PROJECT/runs/$(date +%Y%m%d-%H%M%S)_ovfe_height_sweep"
echo "    mkdir -p $D && rsync -a $OUT/ $D/ && cp $SLURM_SUBMIT_DIR/slurm-${SLURM_JOB_ID}.{out,err} $D/"
