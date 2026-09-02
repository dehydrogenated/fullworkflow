#!/bin/bash
#SBATCH --account=st-akkiraju-1-gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --job-name=co-h-ads
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#
# CO + H adsorption, TiO2(110) vs. Andriuc, Siron & Persson 2025 and IrO2(110) CO vs.
# Pope, Yun et al. 2025 -- see scripts/studies/molecular/co_h_adsorption_benchmark.py for the
# rationale and the settings differences between the two papers. CO ontop the
# undercoordinated metal cation (Ti5c/Ircus), H ontop the bridging O2c site (its real
# preferred site per Andriuc et al.'s own finding, not the metal). seed_standoff=0.5
# deliberately not tighter -- a prior local probe showed CO pushed in tighter relaxes
# straight back out to ~4 A instead of binding. Default: full 14-model roster x (TiO2 CO
# + TiO2 H + IrO2 CO) = 42 pairs; bulk/slab is shared between TiO2's CO and H legs per
# model, not re-relaxed per adsorbate. Orb-v2/CHGNet-0.3.0 run on GPU directly now (their
# envs' torch build was fixed to be CUDA-compatible). UMA-M-* last in MODELS since it's
# the slowest -- everything else finishes and writes first regardless of how long it
# takes. Needs the -gpu account (a different allocation from the CPU jobs elsewhere in
# this repo -- Sockeye rejects --gres=gpu otherwise). Submit from /scratch:
#
#   cd /scratch/st-akkiraju-1/$USER
#   sbatch /arc/project/st-akkiraju-1/ssong18/fullworkflow/scripts/studies/molecular/sockeye_co_h_adsorption_benchmark.slurm
#
# Extra sbatch args after the script path forward straight through:
#   sbatch sockeye_co_h_adsorption_benchmark.slurm --oxides IrO2 --adsorbates CO
#   sbatch sockeye_co_h_adsorption_benchmark.slurm --models UMA-omat UMA-oc22

set -euo pipefail

source /arc/project/st-akkiraju-1/ssong18/fullworkflow/scripts/slurm/_env.sh
python -c "import oxide_workflow, sys; print('python', sys.version.split()[0])"

for ckpt in mace-mh-1.model uma-s-1p2.pt uma-m-1p1.pt sevennet-omni.pth esen_30m_oam.pt orb-v2-20241011.ckpt; do
    test -f "$OXW_MODEL_DIR/$ckpt" || {
        echo "missing checkpoint: $OXW_MODEL_DIR/$ckpt" >&2
        exit 1
    }
done

OUT="$SLURM_SUBMIT_DIR/runs/co_h_ads_benchmark"

STATUS=0
python "$REPO/scripts/studies/molecular/co_h_adsorption_benchmark.py" "$OUT" "$@" \
    || STATUS=$?

echo
echo "results in scratch: $OUT/results.jsonl"
echo "to keep them, run this on the LOGIN node (scratch is purged on a timer):"
D="$PROJECT/runs/$(date +%Y%m%d-%H%M%S)_co_h_ads_benchmark"
echo "    mkdir -p $D && rsync -a $OUT/ $D/ && cp $SLURM_SUBMIT_DIR/slurm-${SLURM_JOB_ID}.{out,err} $D/"

exit $STATUS
