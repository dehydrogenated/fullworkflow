#!/bin/bash
#SBATCH --account=st-akkiraju-1
#SBATCH --partition=cascade
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --job-name=mo2-medium
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#
# Medium test, one step up from the individual per-model smoke tests already run this
# session (sockeye_chgnet_test, sockeye_esen_test, the Orb-v2 pilot). Those covered
# CHGNet/Orb-v2/eSEN on TiO2/FeO2/CrO2; this deliberately covers the opposite corner --
# every remaining model FAMILY that has never run through this benchmark script at all
# (MACE, UMA-S, UMA-M, SevenNet), including both brand-new registry entries added tonight
# (MACE-mh1-matpes, SevenNet-omni-mpa), on oxides this script has never touched either.
#
# UMA-M-omat is the highest-risk untested item here: its checkpoint (uma-m-1p1.pt) is a
# gated HF download with presence on Sockeye unconfirmed as of data/models.md, and it has a
# known predictor-construction hang on some setups (fairchem#2095) that isn't a config
# mistake if hit -- exactly the kind of thing worth finding out now, not 6h into a sweep job.
#
# ZrO2 is the only one of the 33 oxides with no Comer-sourced structure (their repo has no
# final_with_calculator.json for it) -- it's the sole oxide that exercises the Materials
# Project structure-fetch fallback in mp_id_for_formula() instead of the CIF-read path every
# other oxide uses, so it's worth confirming that fallback still produces a sane geometry
# after tonight's lattice-shear fix. RuO2 (4d, well-characterized rutile) rounds out facet
# coverage on a chemistry none of this session's local runs have touched.
#
# 6 models x 2 oxides x 2 facets x 2 adsorbates = 48 relaxation legs, x (2 bulk/slab pairs
# per oxide/facet) = 24 more -- comparable order of magnitude to the original design.
#
# PREREQUISITES:
#   1. Repo up to date: git pull origin main (needs the lattice-shear fix + 15-model commits)
#   2. "mace-clean", "fairchem", "sevenn" envs all exist (checked below) -- these predate
#      this session and were never in question, unlike orb/chgnet/esen
#   3. mace-mh-1.model, uma-s-1p2.pt, sevennet-omni.pth staged at $PROJECT/models/ (checked
#      below, fatal if missing). uma-m-1p1.pt is ALSO checked but its presence was flagged
#      unconfirmed in data/models.md -- if this job dies on that check, that's the answer,
#      go stage it and resubmit rather than treating it as a surprise.
#
#   cd /scratch/st-akkiraju-1/$USER
#   sbatch /arc/project/st-akkiraju-1/ssong18/fullworkflow/scripts/studies/mo2/sockeye_mo2_medium_test.slurm

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

for env in mace-clean fairchem sevenn; do
    test -d "$OXW_CONDA_BASE/envs/$env" || {
        echo "missing conda env: $OXW_CONDA_BASE/envs/$env" >&2
        exit 1
    }
done
for ckpt in mace-mh-1.model uma-s-1p2.pt uma-m-1p1.pt sevennet-omni.pth; do
    test -f "$OXW_MODEL_DIR/$ckpt" || {
        echo "missing checkpoint: $OXW_MODEL_DIR/$ckpt -- if this is uma-m-1p1.pt, its presence was unconfirmed going in; stage it (gated HF download) and resubmit" >&2
        exit 1
    }
done

OUT="$SLURM_SUBMIT_DIR/runs/mo2_medium_test"
python "$REPO/scripts/studies/mo2/mo2_adsorption_benchmark.py" "$OUT" \
    --oxides ZrO2 RuO2 --facets 110 100 --adsorbates O OH \
    --models MACE-mh1-omat MACE-mh1-matpes UMA-oc22 UMA-M-omat SevenNet-omni-omat24 SevenNet-omni-mpa \
    --fmax 0.02 "$@"

echo
echo "results in scratch: $OUT/results.jsonl"
echo "to keep them, run this on the LOGIN node (scratch is purged on a timer):"
D="$PROJECT/runs/$(date +%Y%m%d-%H%M%S)_mo2_medium_test"
echo "    mkdir -p $D && rsync -a $OUT/ $D/ && cp $SLURM_SUBMIT_DIR/slurm-${SLURM_JOB_ID}.{out,err} $D/"
echo "then pull it to your laptop with:"
echo "    scripts/slurm/sync_sockeye_runs.sh $(basename "$D")"
