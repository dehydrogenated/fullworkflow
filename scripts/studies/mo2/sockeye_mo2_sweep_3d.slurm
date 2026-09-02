#!/bin/bash
#SBATCH --account=st-akkiraju-1
#SBATCH --partition=cascade
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --job-name=mo2-sweep-3d
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#
# Full O*/OH* adsorption sweep vs. Comer et al. 2022 -- 3d-row oxides (Ti, V, Cr, Mn, Fe,
# Co, Ni, Cu, Zn, Ga, Ge), both facets, all 15 currently-accessible models spanning the
# omat/oc20/oc22 training-data axis (generalist vs. catalysis-specialist comparison), plus
# two deliberate outliers: MACE-mh1-matpes (r2SCAN, not PBE -- included to see whether a
# different DFT functional systematically beats PBE-trained models against Comer's PBE+U
# reference, not because it's apples-to-apples) and SevenNet-omni-mpa (MPtrj+Alexandria,
# PBE -- a novel-dataset generalist head with no MACE/UMA equivalent on this axis).
# Sibling jobs: sockeye_mo2_sweep_4d.slurm, sockeye_mo2_sweep_5d.slurm (paper's own row
# split -- 11 oxides each, equally sized). eSEN-30M-OAM (30M, OMat24+sAlex+MPtrj -- current Matbench Discovery leaderboard
# topper) runs in its own "esen" env, fairchem-core==1.10.0 -- its public checkpoint is
# in fairchem's legacy format, which the "fairchem" env's current 2.x dropped support
# for. See backends.py's ESEN_CKPT/_esen() comment for the full story.
#
# ~1320 relaxations this job alone (11 oxides x 2 facets x 15 models x (2 bulk/slab +
# 2 adsorbate)) -- rough estimate 23-35h serial wall-clock, scaled from this session's
# own measured per-relaxation times. results.jsonl is written one row at a time, so a walltime kill
# loses nothing already completed; just resubmit (a fresh outdir, or accept the append and
# only rerun what's missing) to pick up the rest.
#
# PREREQUISITES -- every one of these is a hard requirement, not optional, because this
# job's model list uses all of them:
#   1. Repo up to date: git pull origin main
#   2. All 5 checkpoints staged at $PROJECT/models/ (checked below, job exits if any
#      missing): mace-mh-1.model, uma-s-1p2.pt, uma-m-1p1.pt (separately HF-gated from
#      uma-s), sevennet-omni.pth, orb-v2-20241011.ckpt (plain public download, stage it
#      the same way you did for the pilot job)
#
#   cd /scratch/st-akkiraju-1/$USER
#   sbatch /arc/project/st-akkiraju-1/ssong18/fullworkflow/scripts/studies/mo2/sockeye_mo2_sweep_3d.slurm

set -euo pipefail

source /arc/project/st-akkiraju-1/ssong18/fullworkflow/scripts/slurm/_env.sh

for ckpt in mace-mh-1.model uma-s-1p2.pt uma-m-1p1.pt sevennet-omni.pth orb-v2-20241011.ckpt esen_30m_oam.pt; do
    test -f "$OXW_MODEL_DIR/$ckpt" || {
        echo "missing checkpoint: $OXW_MODEL_DIR/$ckpt -- this job's model list needs it, stage it before resubmitting" >&2
        exit 1
    }
done

MODELS=(
    CHGNet-0.3.0 Orb-v2
    MACE-mh1-omat MACE-mh1-oc20 MACE-mh1-matpes
    UMA-omat UMA-oc20 UMA-oc22
    UMA-M-omat UMA-M-oc20
    SevenNet-omni-omat24 SevenNet-omni-oc20 SevenNet-omni-oc22 SevenNet-omni-mpa
    eSEN-30M-OAM
)
OXIDES=(TiO2 VO2 CrO2 MnO2 FeO2 CoO2 NiO2 CuO2 ZnO2 GaO2 GeO2)

OUT="$SLURM_SUBMIT_DIR/runs/mo2_sweep_3d"
python "$REPO/scripts/studies/mo2/mo2_adsorption_benchmark.py" "$OUT" \
    --oxides "${OXIDES[@]}" --facets 110 100 --adsorbates O OH \
    --models "${MODELS[@]}" --fmax 0.02 "$@"

echo
echo "results in scratch: $OUT/results.jsonl"
echo "to keep them, run this on the LOGIN node (scratch is purged on a timer):"
D="$PROJECT/runs/mo2_sweep_3d"  # fixed name (no timestamp) -- predictable target for sync_sockeye_runs.sh,
# and safe to resubmit into: rsync -a just updates it, same append-safety as results.jsonl itself
echo "    mkdir -p $D && rsync -a $OUT/ $D/ && cp $SLURM_SUBMIT_DIR/slurm-${SLURM_JOB_ID}.{out,err} $D/"
echo "then pull it to your laptop with:"
echo "    scripts/slurm/sync_sockeye_runs.sh $(basename "$D")"
