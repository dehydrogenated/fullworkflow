#!/bin/bash
#SBATCH --account=st-akkiraju-1
#SBATCH --partition=cascade
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --job-name=o-ads-bench
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#
# O-atom adsorption vs. Zhao & Kulik (JPCL 2019, 10, 5090) -- MACE-mh1-omat, UMA-omat,
# UMA-oc22 x 7 rutile oxides (Ti, V, Mo, Ru, Rh, Ir, Pt), single ontop-metal site each.
# See scripts/studies/mo2/o_adsorption_benchmark.py for the rationale. Submit from /scratch:
#
#   cd /scratch/st-akkiraju-1/$USER
#   sbatch /arc/project/st-akkiraju-1/ssong18/fullworkflow/scripts/studies/mo2/sockeye_o_ads_benchmark.slurm
#
#   sbatch --account=st-akkiraju-1-gpu --partition=gpu --gres=gpu:1 --time=3:00:00 \
#          /arc/project/st-akkiraju-1/ssong18/fullworkflow/scripts/studies/mo2/sockeye_o_ads_benchmark.slurm

set -euo pipefail

source /arc/project/st-akkiraju-1/ssong18/fullworkflow/scripts/slurm/_env.sh

test -f "$OXW_MODEL_DIR/mace-mh-1.model" || {
    echo "missing checkpoint: $OXW_MODEL_DIR/mace-mh-1.model" >&2
    exit 1
}
test -f "$OXW_MODEL_DIR/uma-s-1p2.pt" || {
    echo "missing checkpoint: $OXW_MODEL_DIR/uma-s-1p2.pt" >&2
    exit 1
}
# Optional extras -- only fatal if you actually pass a --models flag that needs them.
test -f "$OXW_MODEL_DIR/sevennet-omni.pth" || echo "note: sevennet-omni.pth not staged (needed for SevenNet-omni-* models)"
test -f "$OXW_MODEL_DIR/uma-m-1p1.pt" || echo "note: uma-m-1p1.pt not staged (needed for UMA-M-* models)"

OUT="$SLURM_SUBMIT_DIR/runs/o_ads_benchmark"
# Extra sbatch args after the script path forward straight through, e.g.:
#   sbatch sockeye_o_ads_benchmark.slurm --models UMA-oc20 MACE-mh1-oc20
python "$REPO/scripts/studies/mo2/o_adsorption_benchmark.py" "$OUT" --fmax 0.02 "$@"

echo
python "$REPO/scripts/studies/mo2/o_ads_report.py" "$OUT"

echo
echo "results in scratch: $OUT/results.jsonl"
echo "to keep them, run this on the LOGIN node (scratch is purged on a timer):"
D="$PROJECT/runs/$(date +%Y%m%d-%H%M%S)_o_ads_benchmark"
echo "    mkdir -p $D && rsync -a $OUT/ $D/ && cp $SLURM_SUBMIT_DIR/slurm-${SLURM_JOB_ID}.{out,err} $D/"
