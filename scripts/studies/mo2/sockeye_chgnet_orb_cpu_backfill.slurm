#!/bin/bash
#SBATCH --account=st-akkiraju-1
#SBATCH --partition=cascade
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --job-name=chgnet-orb-cpu
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#
# CHGNet-0.3.0 + Orb-v2 backfill, CPU forced directly rather than detected. Both models'
# "orb"/"chgnet" envs pulled in an unpinned torch==2.13.0+cu130 that has no kernels for
# Sockeye's V100s (CC 7.0) -- confirmed real, same crash on both models. The nvidia-smi
# device-detection fix (see sockeye_job.sh) should handle this correctly now, but this job
# doesn't need to trust detection at all: we already have real Sockeye timing data proving
# both models run fine on CPU (Orb-v2 measured at 0.45 s/step, comparable to several GPU-tier
# models), so OXW_DEVICE is just hardcoded here -- nothing to misdetect.
#
# Points at the SAME runs/mo2_sweep_<row> directory the main 15-model sweep already writes
# to, so bulk/slab resumes automatically wherever another model already got that far
# (pipeline._resume) and this only spends new compute on CHGNet/Orb-v2 specifically.
#
# PREREQUISITES:
#   1. Repo up to date: git pull origin main
#   2. "chgnet" and "orb" envs exist, orb-v2-20241011.ckpt staged at $PROJECT/models/
#      (checked below; CHGNet's weights ship inside the package)
#
#   cd /scratch/st-akkiraju-1/$USER
#   sbatch /arc/project/st-akkiraju-1/ssong18/fullworkflow/scripts/studies/mo2/sockeye_chgnet_orb_cpu_backfill.slurm 3d
#   sbatch /arc/project/st-akkiraju-1/ssong18/fullworkflow/scripts/studies/mo2/sockeye_chgnet_orb_cpu_backfill.slurm 4d
#   sbatch /arc/project/st-akkiraju-1/ssong18/fullworkflow/scripts/studies/mo2/sockeye_chgnet_orb_cpu_backfill.slurm 5d

set -euo pipefail

ROW="${1:?usage: sbatch sockeye_chgnet_orb_cpu_backfill.slurm <3d|4d|5d>}"
shift  # remaining "$@" (if any) still forwards to mo2_adsorption_benchmark.py below

case "$ROW" in
    3d) OXIDES=(TiO2 VO2 CrO2 MnO2 FeO2 CoO2 NiO2 CuO2 ZnO2 GaO2 GeO2) ;;
    4d) OXIDES=(ZrO2 NbO2 MoO2 TcO2 RuO2 RhO2 PdO2 AgO2 CdO2 InO2 SnO2) ;;
    5d) OXIDES=(HfO2 TaO2 WO2 ReO2 OsO2 IrO2 PtO2 AuO2 HgO2 TlO2 PbO2) ;;
    *) echo "unknown row: $ROW -- expected 3d, 4d, or 5d" >&2; exit 1 ;;
esac

PROJECT=/arc/project/st-akkiraju-1/ssong18
REPO="$PROJECT/fullworkflow"

export OXW_CONDA_BASE="$PROJECT/miniforge3"
export OXW_MODEL_DIR="$PROJECT/models"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"

export OXW_DEVICE=cpu  # hardcoded on purpose -- see header, no detection needed here

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

echo "job $SLURM_JOB_ID on $(hostname), ${SLURM_CPUS_PER_TASK:-?} cpus, device $OXW_DEVICE, row $ROW"

for env in chgnet orb; do
    test -d "$OXW_CONDA_BASE/envs/$env" || {
        echo "missing conda env: $OXW_CONDA_BASE/envs/$env" >&2
        exit 1
    }
done
test -f "$OXW_MODEL_DIR/orb-v2-20241011.ckpt" || {
    echo "missing checkpoint: $OXW_MODEL_DIR/orb-v2-20241011.ckpt" >&2
    exit 1
}

OUT="$SLURM_SUBMIT_DIR/runs/mo2_sweep_$ROW"
test -d "$OUT" || {
    echo "missing $OUT -- this resumes bulk/slab from the main sweep's own output, run that first" >&2
    exit 1
}

python "$REPO/scripts/studies/mo2/mo2_adsorption_benchmark.py" "$OUT" \
    --oxides "${OXIDES[@]}" --facets 110 100 --adsorbates O OH \
    --models CHGNet-0.3.0 Orb-v2 --fmax 0.02 "$@"

echo
echo "appended to: $OUT/results.jsonl (dedupe by keeping the LAST row per"
echo "  (model, oxide, facet, adsorbate) alongside the main sweep's own rows)"
