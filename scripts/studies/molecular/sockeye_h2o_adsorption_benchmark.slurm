#!/bin/bash
#SBATCH --account=st-akkiraju-1-gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=7:00:00  # 14 models x 3 oxides x 1 nudge; margin for UMA-M-* running
# noticeably slower than the rest (confirmed on Sockeye, not a hang -- it's last in
# MODELS regardless, so its rows land last without delaying anything else's)
#SBATCH --job-name=h2o-ads
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#
# H2O molecular adsorption on TiO2/RuO2/IrO2(110) vs. Gonzalez et al. 2019 (ACS Omega 4,
# 2989-2999) Table 3 -- see scripts/studies/molecular/h2o_adsorption_benchmark.py for the
# rationale. One relaxation per (oxide, model): no dissociation probing, no nudge-apart
# re-relaxation (that's h2o_dissociation_probe.py's job, a separate track). Default models
# are the full roster (14: MACE-mh1 x3, UMA-S x2, UMA-M x2, SevenNet-omni x4, eSEN,
# CHGNet-0.3.0, Orb-v2) spanning 6 conda envs -- fine in one job, since relax() always
# launches each model in its own env via subprocess; only the submitting shell needs oxw.
# Orb-v2/CHGNet-0.3.0 run on GPU directly now (their envs' torch build was fixed to be
# CUDA-compatible) -- sockeye_h2o_adsorption_benchmark_cpu.slurm is no longer needed for
# new runs, kept only as a fallback. Each (oxide, model) pair is independent and
# failure-isolated. Needs the -gpu account (a different allocation from the CPU jobs
# elsewhere in this repo -- Sockeye rejects --gres=gpu otherwise). Submit from /scratch:
#
#   cd /scratch/st-akkiraju-1/$USER
#   sbatch /arc/project/st-akkiraju-1/ssong18/fullworkflow/scripts/studies/molecular/sockeye_h2o_adsorption_benchmark.slurm
#
# Extra sbatch args after the script path forward straight through:
#   sbatch sockeye_h2o_adsorption_benchmark.slurm --oxides RuO2 IrO2
#   sbatch sockeye_h2o_adsorption_benchmark.slurm --models UMA-omat UMA-oc22

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

# Ask SLURM what it actually granted rather than assuming -- see sockeye_job.sh for why.
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
# separately since it ignores XDG_CACHE_HOME).
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
echo "conda base $OXW_CONDA_BASE"
echo "models     $OXW_MODEL_DIR"
python -c "import oxide_workflow, sys; print('python', sys.version.split()[0])"

# Fail before burning walltime if a checkpoint the default --models needs never made it
# over. Only checks the default list's checkpoints; a custom --models overriding this
# doesn't get the same guard. uma-m-1p1.pt is gated (manual HF download, see backends.py)
# -- if this fails on that one specifically, that's the checkpoint to go stage by hand.
for ckpt in mace-mh-1.model uma-s-1p2.pt uma-m-1p1.pt sevennet-omni.pth esen_30m_oam.pt orb-v2-20241011.ckpt; do
    test -f "$OXW_MODEL_DIR/$ckpt" || {
        echo "missing checkpoint: $OXW_MODEL_DIR/$ckpt" >&2
        exit 1
    }
done

# Absolute, and it matters: /arc/project is read-only from compute nodes, so a relative
# outdir landing inside the repo would fail to write. This keeps it in scratch instead.
OUT="$SLURM_SUBMIT_DIR/runs/h2o_adsorption_benchmark"

STATUS=0
python "$REPO/scripts/studies/molecular/h2o_adsorption_benchmark.py" "$OUT" "$@" \
    || STATUS=$?

echo
echo "results in scratch: $OUT/results.jsonl"
echo "to keep them, run this on the LOGIN node (scratch is purged on a timer):"
D="$PROJECT/runs/$(date +%Y%m%d-%H%M%S)_h2o_adsorption_benchmark"
echo "    mkdir -p $D && rsync -a $OUT/ $D/ && cp $SLURM_SUBMIT_DIR/slurm-${SLURM_JOB_ID}.{out,err} $D/"

exit $STATUS
