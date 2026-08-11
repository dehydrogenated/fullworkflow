#!/bin/bash
#SBATCH --account=st-akkiraju-1
#SBATCH --partition=cascade
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --job-name=oc22
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#
# Sweep several models through the OC22 seeded-divergence comparison in one job.
#
# Same environment contract as sockeye_job.sh — submit from /scratch, GPU needs the -gpu
# account, both worker envs need CUDA torch from cu126. Deliberately a separate script
# rather than a refactor of that one: this was written the evening of an overnight run and
# breaking the working pipeline job to share twenty lines was not worth it. Unify later.
#
#     cd /scratch/st-akkiraju-1/$USER
#     sbatch --account=st-akkiraju-1-gpu --partition=gpu --gres=gpu:1 \
#            /arc/.../scripts/sockeye_oc22.sh
#     sbatch ... /arc/.../scripts/sockeye_oc22.sh MACE-mh1-omat UMA-oc22   # pick models
#
# Model names are read from "$@"; with none given it runs the six below. Each model writes
# its own runs/oc22_sweep/<model>/divergence.jsonl, so a model that dies does not cost the
# others — the loop records the failure, carries on, and re-raises at the end.

set -uo pipefail   # NOT -e: one model failing must not end the sweep

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

# $HOME and /arc/project are read-only on compute nodes; every library that caches under
# $HOME has to be redirected or it dies mid-run. See sockeye_job.sh for the full story.
export XDG_CACHE_HOME="${OXW_CACHE:-$SLURM_SUBMIT_DIR/.cache}"
export TRITON_CACHE_DIR="$XDG_CACHE_HOME/triton"
export TORCHINDUCTOR_CACHE_DIR="$XDG_CACHE_HOME/torchinductor"
export HF_HOME="$XDG_CACHE_HOME/huggingface"
export MPLCONFIGDIR="$XDG_CACHE_HOME/matplotlib"
export FAIRCHEM_CACHE_DIR="$XDG_CACHE_HOME/fairchem"
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$HF_HOME" "$MPLCONFIGDIR" \
         "$FAIRCHEM_CACHE_DIR"

source "$OXW_CONDA_BASE/etc/profile.d/conda.sh"
conda activate oxw
cd "$SLURM_SUBMIT_DIR"

MODELS=("$@")
if [ ${#MODELS[@]} -eq 0 ]; then
    # Three MACE heads and three UMA tasks, paired so the comparison is interpretable:
    # omat vs omat isolates architecture on identical training data; oc20 vs oc20 does the
    # same for surface-trained heads; mp and oc22 are the odd ones out on each side.
    # UMA-oc22 has seen these systems in training — read it as an upper bound.
    MODELS=(MACE-mh1-omat MACE-mh1-oc20 MACE-mh1-mp UMA-oc22 UMA-oc20 UMA-omat)
fi

# Three seed sets, run per model.
#   chain3d        the stage chain, and the reason this exists. OC22 carries several
#                  clean slabs per facet that differ ONLY in oxygen count — same metal
#                  count, fewer O — which is an oxygen vacancy with a DFT energy. Pairing
#                  them gives a DFT oxygen-vacancy formation energy to check a model
#                  against, and pairing either with an O adsorbate on the same facet
#                  continues the chain. 46 systems, 8 facets, 5 metals; several are
#                  series rather than pairs, so multiple vacancy concentrations per
#                  surface. OVFE itself is post-processing on e_mlip/e_dft — no extra
#                  relaxation needed here.
#   row3d_O        the ranking experiment: one adsorbate (O) across nine 3d metals.
#   rutile_systems H/OH/CO on IrO2 and PbO2 so the adsorbate axis is not one species.
SEEDSETS=(chain3d row3d_O rutile_systems)
OUT=runs/oc22_sweep

echo "job $SLURM_JOB_ID on $(hostname), ${SLURM_CPUS_PER_TASK:-?} cpus, device $OXW_DEVICE"
echo "models ${MODELS[*]}"
echo "seed sets ${SEEDSETS[*]}"
for S in "${SEEDSETS[@]}"; do
    test -d "$REPO/data/oc22/$S" || { echo "no seeds at $REPO/data/oc22/$S — git pull?" >&2; exit 1; }
done

FAILED=()
for S in "${SEEDSETS[@]}"; do
    for M in "${MODELS[@]}"; do
        echo
        echo "================ $S / $M ================"
        if python "$REPO/scripts/oc22_diverge.py" --model "$M" \
                  --seeds "$REPO/data/oc22/$S" --out "$OUT/$S/$M"; then
            echo "---- $S/$M ok"
        else
            echo "---- $S/$M FAILED (rc=$?) — continuing" >&2
            FAILED+=("$S/$M")
        fi
    done
done

echo
echo "results in scratch: $SLURM_SUBMIT_DIR/$OUT"
echo "to keep them, run this on the LOGIN node:"
D="$PROJECT/runs/$(date +%Y%m%d-%H%M%S)_oc22_sweep"
echo "    mkdir -p $D && rsync -a $SLURM_SUBMIT_DIR/$OUT/ $D/ && cp $SLURM_SUBMIT_DIR/slurm-${SLURM_JOB_ID}.{out,err} $D/"

if [ ${#FAILED[@]} -gt 0 ]; then
    echo "FAILED models: ${FAILED[*]}" >&2
    exit 1
fi
echo "all ${#MODELS[@]} models completed"
