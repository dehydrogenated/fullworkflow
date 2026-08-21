#!/bin/bash
#SBATCH --account=st-akkiraju-1-gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --job-name=propane-propene-ads
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#
# Propane and propene (propylene) adsorption on rutile TiO2/RuO2/IrO2(110) vs. van Hout et
# al. 2026 (GAME-Net-Ox, Digital Discovery 5, 407-414) -- see
# scripts/ClaudeScripts/propane_propene_adsorption_benchmark.py for the full rationale, the
# literature numbers, and how they were identified from the paper's own relaxed CONTCARs
# (downloaded from ioChem-BD, not tabulated in the paper's text). NOT a multi-site screen:
# one fixed orientation each (vertical terminal-CH3-down for propane, side-on through the
# C=C for propene -- see config.py's ADSORBATE_FRAGMENTS comments), placed once on the
# single undercoordinated metal-cation ontop site (top_M5c equivalent), same convention as
# co_h_adsorption_benchmark.py. 3 models x 3 oxides x 2 adsorbates = 18 (oxide, adsorbate,
# model) combos, bulk/slab shared between Propane and Propene per (oxide, model) pair, on
# our 4x2 pristine slab (bigger than the paper's p(2x2), deliberately, to avoid the
# self-image clash a 1x1-cell smoke test hit during development -- see conversation). 6h
# budget calibrated off sockeye_h2o_adsorption_benchmark.slurm's 7h for 42 single-site
# pairs (this job has fewer combos but bigger molecules); if it's not enough, rerun the
# leftover (oxide, adsorbate, model) combos directly with the --oxides/--adsorbates/
# --models flags below rather than widening this file's time limit blindly. Needs the -gpu
# account (different allocation from the CPU jobs elsewhere in this repo -- Sockeye rejects
# --gres=gpu otherwise). Submit from /scratch:
#
#   cd /scratch/st-akkiraju-1/$USER
#   sbatch /arc/project/st-akkiraju-1/ssong18/fullworkflow/scripts/slurm/sockeye_propane_propene_adsorption_benchmark.slurm
#
# Extra sbatch args after the script path forward straight through -- e.g. to calibrate
# timing on a small slice first:
#   sbatch sockeye_propane_propene_adsorption_benchmark.slurm --oxides TiO2 --models Orb-v2
#   sbatch sockeye_propane_propene_adsorption_benchmark.slurm --adsorbates Propane

set -euo pipefail

PROJECT=/arc/project/st-akkiraju-1/ssong18
REPO="$PROJECT/fullworkflow"

export OXW_CONDA_BASE="$PROJECT/miniforge3"
export OXW_MODEL_DIR="$PROJECT/models"

# Compute nodes have no outbound network -- everything here uses locally-staged
# checkpoints, none of which needs a runtime download, but these stop anything that would
# try from hanging until the walltime runs out instead of failing fast.
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

# $HOME and /arc/project are read-only on compute nodes; only /scratch is writable, so
# every library that caches under $HOME by default has to be redirected or it dies
# mid-run (triton, torch inductor, huggingface, matplotlib via pymatgen, fairchem
# separately since it ignores XDG_CACHE_HOME, cached_path -- pulled in by orb-models,
# ignores XDG_CACHE_HOME too and needs its own CACHED_PATH_CACHE_ROOT).
export XDG_CACHE_HOME="${OXW_CACHE:-$SLURM_SUBMIT_DIR/.cache}"
export TRITON_CACHE_DIR="$XDG_CACHE_HOME/triton"
export TORCHINDUCTOR_CACHE_DIR="$XDG_CACHE_HOME/torchinductor"
export HF_HOME="$XDG_CACHE_HOME/huggingface"
export MPLCONFIGDIR="$XDG_CACHE_HOME/matplotlib"
export FAIRCHEM_CACHE_DIR="$XDG_CACHE_HOME/fairchem"
export CACHED_PATH_CACHE_ROOT="$XDG_CACHE_HOME/cached_path"
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$HF_HOME" "$MPLCONFIGDIR" \
         "$FAIRCHEM_CACHE_DIR" "$CACHED_PATH_CACHE_ROOT"

source "$OXW_CONDA_BASE/etc/profile.d/conda.sh"
conda activate oxw

cd "$SLURM_SUBMIT_DIR"

echo "job $SLURM_JOB_ID on $(hostname), ${SLURM_CPUS_PER_TASK:-?} cpus, device $OXW_DEVICE"
python -c "import oxide_workflow, sys; print('python', sys.version.split()[0])"

# Only the checkpoints this specific benchmark's three models need -- Orb-v2, eSEN-30M-OAM,
# UMA-M-oc20 -- not the full roster every model-agnostic job here checks.
for ckpt in orb-v2-20241011.ckpt esen_30m_oam.pt uma-m-1p1.pt; do
    test -f "$OXW_MODEL_DIR/$ckpt" || {
        echo "missing checkpoint: $OXW_MODEL_DIR/$ckpt" >&2
        exit 1
    }
done

OUT="$SLURM_SUBMIT_DIR/runs/propane_propene_ads_benchmark"

STATUS=0
python "$REPO/scripts/ClaudeScripts/propane_propene_adsorption_benchmark.py" "$OUT" "$@" \
    || STATUS=$?

echo
echo "results in scratch: $OUT/results.jsonl"
echo "to keep them, run this on the LOGIN node (scratch is purged on a timer):"
D="$PROJECT/runs/$(date +%Y%m%d-%H%M%S)_propane_propene_ads_benchmark"
echo "    mkdir -p $D && rsync -a $OUT/ $D/ && cp $SLURM_SUBMIT_DIR/slurm-${SLURM_JOB_ID}.{out,err} $D/"

exit $STATUS
