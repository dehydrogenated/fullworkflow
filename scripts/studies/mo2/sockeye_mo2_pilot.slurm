#!/bin/bash
#SBATCH --account=st-akkiraju-1
#SBATCH --partition=cascade
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --job-name=mo2-pilot
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#
# O*/OH* adsorption vs. Comer et al. 2022 (JPCC 126, 7903) -- generalization test on a
# brand-new oxide (FeO2, never touched locally) plus TiO2 as a known-good control, both
# facets, Orb-v2 only (fastest model, per prior timing measurements on this repo's own
# runs/ovfe_tio2_benchmark logs). Verifies the facet-dependent OH seed orientation fix
# (scripts/studies/mo2/mo2_adsorption_benchmark.py's OH_COORDS_BY_FACET) generalizes
# beyond the two oxides it was derived from. Submit from /scratch:
#
#   cd /scratch/st-akkiraju-1/$USER
#   sbatch /arc/project/st-akkiraju-1/ssong18/fullworkflow/scripts/studies/mo2/sockeye_mo2_pilot.slurm
#
# IMPORTANT -- run this ONCE on the LOGIN node first, before ever submitting this job.
# Orb-v2's weights are fetched+cached on first use (unlike MACE/UMA, which load from a
# pre-staged checkpoint file) -- compute nodes have no outbound network, so an uncached
# Orb-v2 job will hang until the walltime runs out instead of failing fast:
#
#   conda activate oxw   # or: source $PROJECT/miniforge3/etc/profile.d/conda.sh && conda activate orb
#   python -c "from orb_models.forcefield.pretrained import orb_v2; orb_v2()"

set -euo pipefail

source /arc/project/st-akkiraju-1/ssong18/fullworkflow/scripts/slurm/_env.sh

test -d "$OXW_CONDA_BASE/envs/orb" || {
    echo "missing conda env: $OXW_CONDA_BASE/envs/orb -- create it first (orb-models, ase)" >&2
    exit 1
}
test -f "$OXW_MODEL_DIR/orb-v2-20241011.ckpt" || {
    echo "missing checkpoint: $OXW_MODEL_DIR/orb-v2-20241011.ckpt" >&2
    exit 1
}

OUT="$SLURM_SUBMIT_DIR/runs/mo2_pilot_sockeye"
# Extra sbatch args after the script path forward straight through, e.g.:
#   sbatch sockeye_mo2_pilot.slurm --oxides FeO2
python "$REPO/scripts/studies/mo2/mo2_adsorption_benchmark.py" "$OUT" \
    --oxides TiO2 FeO2 --facets 110 100 --adsorbates O OH --models Orb-v2 --fmax 0.02 "$@"

echo
echo "results in scratch: $OUT/results.jsonl"
echo "to keep them, run this on the LOGIN node (scratch is purged on a timer):"
D="$PROJECT/runs/$(date +%Y%m%d-%H%M%S)_mo2_pilot_sockeye"
echo "    mkdir -p $D && rsync -a $OUT/ $D/ && cp $SLURM_SUBMIT_DIR/slurm-${SLURM_JOB_ID}.{out,err} $D/"
echo "then pull it to your laptop with:"
echo "    scripts/slurm/sync_sockeye_runs.sh $(basename "$D")"
