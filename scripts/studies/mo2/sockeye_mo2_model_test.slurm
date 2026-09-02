#!/bin/bash
#SBATCH --account=st-akkiraju-1
#SBATCH --partition=cascade
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --job-name=mo2-model-test
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#
# Cheap smoke test for ONE model before committing it to the full sweep. TiO2 only (the
# most-exercised, best-understood oxide here), both facets, both adsorbates -- 4 legs
# rather than 33 oxides. If a model doesn't converge cleanly here, the sweep jobs won't
# either, so this is worth running before touching sockeye_mo2_sweep_*.slurm at all.
#
# Generalized from the per-model smoke tests that onboarded CHGNet and eSEN. No env or
# checkpoint preflight: backends.relax() already fails with the backend name, the conda
# env name and the worker's stderr, which is more informative than a check here could be
# and doesn't need this file to track what lives in backends.py's REGISTRY.
#
# PREREQUISITES: repo up to date (git pull origin main), and the model's conda env +
# checkpoint staged from the LOGIN node -- compute nodes have no outbound network.
#
#   cd /scratch/st-akkiraju-1/$USER
#   sbatch .../scripts/studies/mo2/sockeye_mo2_model_test.slurm eSEN-30M-OAM
#   sbatch .../scripts/studies/mo2/sockeye_mo2_model_test.slurm CHGNet-0.3.0
#
# Anything after the model name forwards to mo2_adsorption_benchmark.py.

set -euo pipefail

MODEL="${1:?usage: sbatch sockeye_mo2_model_test.slurm <MODEL-NAME> [extra args]}"
shift

source /arc/project/st-akkiraju-1/ssong18/fullworkflow/scripts/slurm/_env.sh

OUT="$SLURM_SUBMIT_DIR/runs/model_test_$MODEL"
python "$REPO/scripts/studies/mo2/mo2_adsorption_benchmark.py" "$OUT" \
    --oxides TiO2 --facets 110 100 --adsorbates O OH --models "$MODEL" --fmax 0.02 "$@"

keep_results "$OUT" "model_test_$MODEL"
echo "then pull it to your laptop with:"
echo "    scripts/core/sync_sockeye_runs.sh <the directory name above>"
