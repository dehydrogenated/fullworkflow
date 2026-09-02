#!/bin/bash
#SBATCH --account=st-akkiraju-1-gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --job-name=ovfe-tio2-backfill
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#
# Fills in the 5 models missing from the local TiO2(110) oxygen-vacancy formation-energy
# benchmark (runs/ovfe_tio2_benchmark on the laptop -- 10 of 15 models already have local
# output there; see scripts/studies/ovfe/ovfe_tio2_benchmark_plot.py for the literature
# reference it's plotted against, Kowalski/Meyer/Marx PRB 79, 115410 (2009)). Started this
# locally first but two of the five aren't runnable on the laptop right now -- UMA-oc20's
# checkpoint isn't currently staged there, and there's no local "esen" conda env at all --
# so all five just run here instead, where everything's already staged and GPU now works
# for every model (orb/chgnet's cu126 fix was the last gap, and neither of those two is
# even needed for this particular backfill).
#
# Single-stage tool (scripts/core/run_stage.py vacancy), not the sweep script -- this
# benchmark is one material (mp-2657 rutile TiO2), one facet (110, the default), one
# supercell (4x2, the default). Only --vacuum is non-default: 12 A, not the usual 20 A,
# matching every other model already in runs/ovfe_tio2_benchmark so all 15 points end up
# comparable. run_stage.py vacancy with no --from runs the full bulk->slab->vacancy chain
# in one call and writes its own <outdir>/<model>/vacancy/rankings.csv per model --
# ovfe_tio2_benchmark_plot.py reads exactly that path, so no renaming needed afterward,
# just drop these 5 model folders in alongside the 10 already there.
#
# Own output folder (runs/ovfe_tio2_gpu_backfill), same reasoning as the OsO2 backfill --
# pull it down and copy the 5 model subfolders into the laptop's runs/ovfe_tio2_benchmark/
# rather than writing into a directory you're also touching locally.
#
# PREREQUISITES:
#   1. Repo up to date: git pull origin main
#   2. "mace-clean", "fairchem", "sevenn", "esen" envs all exist (checked below)
#   3. mace-mh-1.model, uma-s-1p2.pt, sevennet-omni.pth, esen_30m_oam.pt staged at
#      $PROJECT/models/ (checked below)
#
#   cd /scratch/st-akkiraju-1/$USER
#   sbatch /arc/project/st-akkiraju-1/ssong18/fullworkflow/scripts/studies/ovfe/sockeye_ovfe_tio2_backfill.slurm

set -euo pipefail

source /arc/project/st-akkiraju-1/ssong18/fullworkflow/scripts/slurm/_env.sh

for env in mace-clean fairchem sevenn esen; do
    test -d "$OXW_CONDA_BASE/envs/$env" || {
        echo "missing conda env: $OXW_CONDA_BASE/envs/$env" >&2
        exit 1
    }
done
for ckpt in mace-mh-1.model uma-s-1p2.pt sevennet-omni.pth esen_30m_oam.pt; do
    test -f "$OXW_MODEL_DIR/$ckpt" || {
        echo "missing checkpoint: $OXW_MODEL_DIR/$ckpt" >&2
        exit 1
    }
done

OUT="$SLURM_SUBMIT_DIR/runs/ovfe_tio2_gpu_backfill"
mkdir -p "$OUT"

MODELS=(MACE-mh1-matpes UMA-oc20 SevenNet-omni-oc22 SevenNet-omni-mpa eSEN-30M-OAM)

for model in "${MODELS[@]}"; do
    echo "=== $model ==="
    python "$REPO/scripts/core/run_stage.py" vacancy \
        --model "$model" --vacuum 12 --outdir "$OUT"
    echo
done

echo
echo "results in: $OUT/<model>/vacancy/rankings.csv, one per model"
echo "to keep them, run this on the LOGIN node (scratch is purged on a timer):"
D="$PROJECT/runs/ovfe_tio2_gpu_backfill"
echo "    mkdir -p $D && rsync -a $OUT/ $D/ && cp $SLURM_SUBMIT_DIR/slurm-${SLURM_JOB_ID}.{out,err} $D/"
echo "then pull it to your laptop and copy each model's folder into runs/ovfe_tio2_benchmark/:"
echo "    scripts/slurm/sync_sockeye_runs.sh ovfe_tio2_gpu_backfill"
