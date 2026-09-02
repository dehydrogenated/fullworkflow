#!/bin/bash
#SBATCH --account=st-akkiraju-1-gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --job-name=ovfe-scell
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#
# OVFE convergence vs. lateral supercell size (1x1 up to 8x8, 13 points), on the 3
# top-performing OVFE-benchmark models (UMA-omat, eSEN-30M-OAM, Orb-v2). Height/vacuum
# held at the current SlabConfig default (4 trilayers, min_vacuum_size=20.0 -- see
# ovfe_convergence_common.py's BASE_SLAB comment). This is the expensive sweep: atom count
# scales with nx*ny, from 192 at the 4x2 baseline to ~1500 at 8x8. GPU by default (per
# Sean, 2026-08-20) -- 8h/64G is a generous guess, not a measured budget; check the first
# job's actual pace before assuming it holds for the largest supercells.
#
#   cd /scratch/st-akkiraju-1/$USER
#   sbatch /arc/project/st-akkiraju-1/ssong18/fullworkflow/scripts/studies/ovfe/sockeye_ovfe_supercell_sweep.slurm

set -euo pipefail

source /arc/project/st-akkiraju-1/ssong18/fullworkflow/scripts/slurm/_env.sh

for ckpt in uma-s-1p2.pt esen_30m_oam.pt orb-v2-20241011.ckpt; do
    test -f "$OXW_MODEL_DIR/$ckpt" || echo "note: $OXW_MODEL_DIR/$ckpt not staged -- that model's rows will be marked failed"
done

OUT="$SLURM_SUBMIT_DIR/runs/ovfe_supercell_sweep"
python "$REPO/scripts/studies/ovfe/ovfe_supercell_sweep.py" "$OUT"

echo
echo "results in scratch: $OUT/results.jsonl"
echo "to keep them, run this on the LOGIN node (scratch is purged on a timer):"
D="$PROJECT/runs/$(date +%Y%m%d-%H%M%S)_ovfe_supercell_sweep"
echo "    mkdir -p $D && rsync -a $OUT/ $D/ && cp $SLURM_SUBMIT_DIR/slurm-${SLURM_JOB_ID}.{out,err} $D/"
