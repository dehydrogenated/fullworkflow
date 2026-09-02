#!/bin/bash
#SBATCH --account=st-akkiraju-1-gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --job-name=oso2-backfill
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#
# Backfills OsO2 across all 15 models, both facets, both adsorbates -- the
# undercoordinated_metal_site() fix (commit "Fix undercoordinated_metal_site() dropping Os
# sites as if they were oxygen") landed after the main 5d sweep started, so any model that
# reached OsO2 with old code still hit "no metal ontop site found". Not model-specific --
# undercoordinated_metal_site() is pure geometry, called identically regardless of backend.
#
# GPU now, for every model -- the orb/chgnet torch cu126 reinstall (see
# sockeye_orb_chgnet_gpu_smoke.slurm) closed the last gap; every model in MODELS below has a
# confirmed-working CUDA path as of 2026-08-20.
#
# Deliberately its OWN output folder (runs/oso2_gpu_backfill), NOT runs/mo2_sweep_5d --
# the main 5d sweep job is still actively running (on SevenNet as of this writing) and
# actively appending to that directory's results.jsonl. Concurrent appends to the same file
# are safe (confirmed earlier this session, POSIX atomic small-write semantics), but a
# separate directory means this job doesn't touch bulk/slab CONTCARs the main job might still
# be reading/writing for OTHER oxides in the same row, and keeps this run's own output
# trivially inspectable in isolation while both are in flight. Costs a bit of extra compute
# (bulk+slab redone from scratch per model rather than resumed via pipeline._resume, since
# there's no prior OsO2 output in this fresh directory) -- fine, OsO2 is a single oxide and
# GPU makes the whole thing cheap regardless.
#
# Merge afterward: concatenate this directory's results.jsonl onto mo2_sweep_5d's own, same
# last-row-wins dedup rule used everywhere else this session --
#   cat runs/oso2_gpu_backfill/results.jsonl >> runs/mo2_sweep_5d/results.jsonl
# (or dedupe at read time instead of physically merging, same effect either way).
#
# PREREQUISITES:
#   1. Repo up to date: git pull origin main (needs the Os fix commit)
#
#   cd /scratch/st-akkiraju-1/$USER
#   sbatch /arc/project/st-akkiraju-1/ssong18/fullworkflow/scripts/studies/mo2/sockeye_oso2_backfill.slurm

set -euo pipefail

source /arc/project/st-akkiraju-1/ssong18/fullworkflow/scripts/slurm/_env.sh

MODELS=(
    CHGNet-0.3.0 Orb-v2
    MACE-mh1-omat MACE-mh1-oc20 MACE-mh1-matpes
    UMA-omat UMA-oc20 UMA-oc22
    UMA-M-omat UMA-M-oc20
    SevenNet-omni-omat24 SevenNet-omni-oc20 SevenNet-omni-oc22 SevenNet-omni-mpa
    eSEN-30M-OAM
)

OUT="$SLURM_SUBMIT_DIR/runs/oso2_gpu_backfill"
mkdir -p "$OUT"

python "$REPO/scripts/studies/mo2/mo2_adsorption_benchmark.py" "$OUT" \
    --oxides OsO2 --facets 110 100 --adsorbates O OH \
    --models "${MODELS[@]}" --fmax 0.02 "$@"

echo
echo "results in: $OUT/results.jsonl"
echo "merge into the main 5d sweep on the LOGIN node once both are done:"
echo "    cat $OUT/results.jsonl >> \$PROJECT/runs/mo2_sweep_5d/results.jsonl"
echo "(or just pass both files to any script that already dedupes by last-row-wins, e.g."
echo " periodic_trend_report.py -- no physical merge required)"
