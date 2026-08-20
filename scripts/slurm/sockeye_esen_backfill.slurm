#!/bin/bash
#SBATCH --account=st-akkiraju-1-gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --job-name=esen-backfill
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#
# Hedge against the main 15-model mo2-sweep jobs running out of their 20h walltime before
# reaching eSEN -- it's last in the per-row model order and the slowest model in this
# project's own timing logs (see sockeye_esen_skew_check.slurm), so a naive linear
# extrapolation from the models that finished first already predicted it wouldn't make it
# (13h elapsed / 316 legs done / 6.8h left for 256 legs still queued, at row 3d's own
# observed pace -- checked 2026-08-20 via squeue, not assumed). Submitted proactively,
# running in parallel with the main sweep rather than waiting for it to die at the time
# limit, so this doesn't cost extra wall-clock versus a reactive backfill.
#
# Points at the SAME runs/mo2_sweep_<row> directory the main sweep already writes to, so
# bulk/slab resumes automatically if eSEN happened to get partway through before this job
# starts (pipeline._resume, same mechanism as the CHGNet/Orb-v2 CPU backfill). Only the
# adsorbate legs, which always relax fresh, spend new compute.
#
# GPU, not CPU-forced -- unlike CHGNet/Orb-v2 (which crashed on the V100s' torch build, see
# sockeye_chgnet_orb_cpu_backfill.slurm), eSEN has no known GPU incompatibility and is
# explicitly recommended to run on GPU in sockeye_esen_skew_check.slurm given how slow it
# already is; forcing CPU here would make the exact problem this job exists to hedge against
# worse, not better. Uses the normal nvidia-smi detection since this job legitimately
# requests --gres=gpu:1 at submission (baked into #SBATCH above, no flag needed at submit
# time, unlike the CPU-forced backfill's opposite failure mode).
#
# PREREQUISITES:
#   1. Repo up to date: git pull origin main
#   2. "esen" env exists, esen_30m_oam.pt staged at $PROJECT/models/ (checked below)
#
#   cd /scratch/st-akkiraju-1/$USER
#   sbatch /arc/project/st-akkiraju-1/ssong18/fullworkflow/scripts/slurm/sockeye_esen_backfill.slurm 3d
#   sbatch /arc/project/st-akkiraju-1/ssong18/fullworkflow/scripts/slurm/sockeye_esen_backfill.slurm 4d
#   sbatch /arc/project/st-akkiraju-1/ssong18/fullworkflow/scripts/slurm/sockeye_esen_backfill.slurm 5d

set -euo pipefail

ROW="${1:?usage: sbatch sockeye_esen_backfill.slurm <3d|4d|5d>}"
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

echo "job $SLURM_JOB_ID on $(hostname), ${SLURM_CPUS_PER_TASK:-?} cpus, device $OXW_DEVICE, row $ROW"

test -d "$OXW_CONDA_BASE/envs/esen" || {
    echo "missing conda env: $OXW_CONDA_BASE/envs/esen" >&2
    exit 1
}
test -f "$OXW_MODEL_DIR/esen_30m_oam.pt" || {
    echo "missing checkpoint: $OXW_MODEL_DIR/esen_30m_oam.pt" >&2
    exit 1
}

OUT="$SLURM_SUBMIT_DIR/runs/mo2_sweep_$ROW"
test -d "$OUT" || {
    echo "missing $OUT -- this resumes bulk/slab from the main sweep's own output, run that first" >&2
    exit 1
}

python "$REPO/scripts/ClaudeScripts/mo2_adsorption_benchmark.py" "$OUT" \
    --oxides "${OXIDES[@]}" --facets 110 100 --adsorbates O OH \
    --models eSEN-30M-OAM --fmax 0.02 "$@"

echo
echo "appended to: $OUT/results.jsonl (dedupe by keeping the LAST row per"
echo "  (model, oxide, facet, adsorbate) alongside the main sweep's own rows)"
