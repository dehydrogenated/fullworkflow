#!/bin/bash
#SBATCH --account=st-akkiraju-1
#SBATCH --partition=cascade
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --job-name=mo2-sweep-4d
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#
# Full O*/OH* adsorption sweep vs. Comer et al. 2022 -- 4d-row oxides (Zr, Nb, Mo, Tc, Ru, Rh, Pd, Ag, Cd, In, Sn), both facets, all 15 currently-accessible models spanning the
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
#   sbatch /arc/project/st-akkiraju-1/ssong18/fullworkflow/scripts/slurm/sockeye_mo2_sweep_4d.slurm

set -euo pipefail

PROJECT=/arc/project/st-akkiraju-1/ssong18
REPO="$PROJECT/fullworkflow"

export OXW_CONDA_BASE="$PROJECT/miniforge3"
export OXW_MODEL_DIR="$PROJECT/models"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"

# nvidia-smi, not CUDA_VISIBLE_DEVICES -- that env var is inherited from the submitting
# shell by sbatch's default --export=ALL, so a stray value from an earlier interactive GPU
# session can make a --partition=cascade job (no --gres=gpu) wrongly resolve to cuda. See
# sockeye_job.sh for the full incident. nvidia-smi respects SLURM's cgroup isolation instead.
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L 2>/dev/null | grep -q "GPU"; then
    export OXW_DEVICE=cuda
else
    export OXW_DEVICE=cpu
fi

export XDG_CACHE_HOME="${OXW_CACHE:-$SLURM_SUBMIT_DIR/.cache}"
export TRITON_CACHE_DIR="$XDG_CACHE_HOME/triton"
export TORCHINDUCTOR_CACHE_DIR="$XDG_CACHE_HOME/torchinductor"
export HF_HOME="$XDG_CACHE_HOME/huggingface"
export MPLCONFIGDIR="$XDG_CACHE_HOME/matplotlib"
export FAIRCHEM_CACHE_DIR="$XDG_CACHE_HOME/fairchem"
export CACHED_PATH_CACHE_ROOT="$XDG_CACHE_HOME/cached_path"  # orb-models' own cache lib -- ignores XDG_CACHE_HOME, has its own var
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$HF_HOME" "$MPLCONFIGDIR" \
         "$FAIRCHEM_CACHE_DIR" "$CACHED_PATH_CACHE_ROOT"

source "$OXW_CONDA_BASE/etc/profile.d/conda.sh"
conda activate oxw

cd "$SLURM_SUBMIT_DIR"

echo "job $SLURM_JOB_ID on $(hostname), ${SLURM_CPUS_PER_TASK:-?} cpus, device $OXW_DEVICE"

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
OXIDES=(ZrO2 NbO2 MoO2 TcO2 RuO2 RhO2 PdO2 AgO2 CdO2 InO2 SnO2)

OUT="$SLURM_SUBMIT_DIR/runs/mo2_sweep_4d"
python "$REPO/scripts/ClaudeScripts/mo2_adsorption_benchmark.py" "$OUT" \
    --oxides "${OXIDES[@]}" --facets 110 100 --adsorbates O OH \
    --models "${MODELS[@]}" --fmax 0.02 "$@"

echo
echo "results in scratch: $OUT/results.jsonl"
echo "to keep them, run this on the LOGIN node (scratch is purged on a timer):"
D="$PROJECT/runs/mo2_sweep_4d"  # fixed name (no timestamp) -- predictable target for sync_sockeye_runs.sh,
# and safe to resubmit into: rsync -a just updates it, same append-safety as results.jsonl itself
echo "    mkdir -p $D && rsync -a $OUT/ $D/ && cp $SLURM_SUBMIT_DIR/slurm-${SLURM_JOB_ID}.{out,err} $D/"
echo "then pull it to your laptop with:"
echo "    scripts/slurm/sync_sockeye_runs.sh $(basename "$D")"
