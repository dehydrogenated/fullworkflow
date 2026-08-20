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
# Medium test, one step up from the individual per-model smoke tests (sockeye_chgnet_test,
# sockeye_esen_test, the Orb-v2 pilot) -- those each proved one model runs at all; this
# proves the pipeline is CORRECT after tonight's lattice-shear fix (stages.py's
# _orthogonalize_inplane_lattice), at full fidelity (fmax=0.02, no shortcuts), across
# multiple oxides. The shear SlabGenerator introduces is chemistry-dependent (tiny for
# some bulk lattice constants, ~1.2 A for others -- see the commit this job is meant to
# validate), so oxide diversity matters here more than model count: the fix lives in
# make_slab(), shared by every model, so one model proving the geometry is clean already
# validates it structurally -- CHGNet/Orb-v2/eSEN are included because those three had
# separate deployment blockers fixed this session and haven't been run end-to-end together
# since.
#
# TiO2 (known-good control) + CrO2 + FeO2, both facets, both adsorbates, 3 models ->
# 3 oxides x 2 facets x 3 models = 18 bulk/slab pairs, x 2 adsorbates = 36 relaxation legs.
#
# PREREQUISITES:
#   1. Repo up to date: git pull origin main (needs the lattice-shear fix commit)
#   2. "chgnet", "orb", "esen" envs all exist (checked below)
#   3. orb-v2-20241011.ckpt and esen_30m_oam.pt staged at $PROJECT/models/ (checked below;
#      CHGNet's weights ship inside the package, no checkpoint file needed)
#
#   cd /scratch/st-akkiraju-1/$USER
#   sbatch /arc/project/st-akkiraju-1/ssong18/fullworkflow/scripts/slurm/sockeye_mo2_medium_test.slurm

set -euo pipefail

PROJECT=/arc/project/st-akkiraju-1/ssong18
REPO="$PROJECT/fullworkflow"

export OXW_CONDA_BASE="$PROJECT/miniforge3"
export OXW_MODEL_DIR="$PROJECT/models"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"

if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
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

for env in chgnet orb esen; do
    test -d "$OXW_CONDA_BASE/envs/$env" || {
        echo "missing conda env: $OXW_CONDA_BASE/envs/$env" >&2
        exit 1
    }
done
for ckpt in orb-v2-20241011.ckpt esen_30m_oam.pt; do
    test -f "$OXW_MODEL_DIR/$ckpt" || {
        echo "missing checkpoint: $OXW_MODEL_DIR/$ckpt" >&2
        exit 1
    }
done

OUT="$SLURM_SUBMIT_DIR/runs/mo2_medium_test"
python "$REPO/scripts/ClaudeScripts/mo2_adsorption_benchmark.py" "$OUT" \
    --oxides TiO2 CrO2 FeO2 --facets 110 100 --adsorbates O OH \
    --models CHGNet-0.3.0 Orb-v2 eSEN-30M-OAM --fmax 0.02 "$@"

echo
echo "results in scratch: $OUT/results.jsonl"
echo "to keep them, run this on the LOGIN node (scratch is purged on a timer):"
D="$PROJECT/runs/$(date +%Y%m%d-%H%M%S)_mo2_medium_test"
echo "    mkdir -p $D && rsync -a $OUT/ $D/ && cp $SLURM_SUBMIT_DIR/slurm-${SLURM_JOB_ID}.{out,err} $D/"
echo "then pull it to your laptop with:"
echo "    scripts/slurm/sync_sockeye_runs.sh $(basename "$D")"
