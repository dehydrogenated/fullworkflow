#!/bin/bash
#SBATCH --account=st-akkiraju-1
#SBATCH --job-name=gas_refs_O
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#
# Gas-phase DFT reference energies (MP conventions) for the oxygen reference:
# O2, H2O, H2. All three run sequentially in one job -- each takes minutes, so
# one queue wait beats three. CPU only: at 2-3 atoms a GPU node is pure waste.
#
# SUBMIT FROM SCRATCH. $HOME and /arc/project are read-only on compute nodes,
# and VASP writes OUTCAR/CONTCAR in-place next to the inputs.
#
#   cp -r /arc/project/st-akkiraju-1/ssong18/fullworkflow/gas_refs_dft \
#         /scratch/st-akkiraju-1/$USER/
#   cd /scratch/st-akkiraju-1/$USER/gas_refs_dft
#   export OXW_VASP_MODULE=<what preflight_sockeye.sh reported>
#   export VASP_PSP_DIR=<what preflight_sockeye.sh reported>
#   ./make_potcars.sh && sbatch --export=ALL submit_gas_refs.slurm
#
# Run ./preflight_sockeye.sh on the LOGIN node first. It resolves the VASP
# module name, the allocation, and the POTCAR path -- all three were guesses.

set -euo pipefail

# --- guard: refuse to run anywhere the filesystem is read-only ---------------
case "$SLURM_SUBMIT_DIR" in
    /scratch/*) ;;
    *) echo "ERROR: submit from /scratch. $SLURM_SUBMIT_DIR is read-only on compute nodes." >&2
       exit 1 ;;
esac

cd "$SLURM_SUBMIT_DIR"
GASDIR=gas_references

# --- environment -------------------------------------------------------------
module purge

# No default here on purpose. A wrong-but-plausible module name is worse than a
# missing one: VASP would run under whatever the site's `vasp` alias points at,
# possibly a different version or POTCAR convention, and the energies would look
# entirely reasonable while sitting on the wrong scale.
if [[ -z "${OXW_VASP_MODULE:-}" ]]; then
    echo "ERROR: OXW_VASP_MODULE is not set." >&2
    echo "  Run ./preflight_sockeye.sh on the login node, then:" >&2
    echo "    export OXW_VASP_MODULE=vasp/<version-it-reported>" >&2
    echo "    sbatch --export=ALL submit_gas_refs.slurm" >&2
    exit 1
fi
module load "$OXW_VASP_MODULE" || {
    echo "ERROR: 'module load $OXW_VASP_MODULE' failed on the compute node." >&2
    echo "  Login and compute nodes do not always expose the same module tree." >&2
    exit 1; }

export OMP_NUM_THREADS=1        # VASP 6 is OpenMP-aware; unset looks like a hang

# Gamma-only -> vasp_gam is correct and ~2x faster than vasp_std.
if command -v vasp_gam &> /dev/null; then VASP_BIN=$(command -v vasp_gam)
elif command -v vasp_std &> /dev/null; then VASP_BIN=$(command -v vasp_std)
                                           echo "NOTE: vasp_gam not found, using vasp_std"
else
    echo "ERROR: module '$OXW_VASP_MODULE' loaded but provides no vasp_gam/vasp_std." >&2
    exit 1
fi
echo "module: $OXW_VASP_MODULE"
echo "binary: $VASP_BIN   ranks: $SLURM_NTASKS"

# --- input completeness ------------------------------------------------------
# Check everything before running anything: a missing POTCAR on the third
# molecule should not surface after the first two have already burned an hour.
missing=0
for dir in "$GASDIR"/*/; do
    for f in POSCAR INCAR KPOINTS POTCAR; do
        [ -f "$dir/$f" ] || { echo "ERROR: $dir is missing $f"; missing=1; }
    done
done
if [ "$missing" -ne 0 ]; then
    echo "Run ./make_potcars.sh (with VASP_PSP_DIR set) and re-submit." >&2
    exit 1
fi

# --- run ---------------------------------------------------------------------
for dir in "$GASDIR"/*/; do
    name=$(basename "$dir")
    [ -f "$dir/DONE" ] && { echo "[$(date +%T)] $name done, skipping"; continue; }
    echo "===== [$(date +%T)] $name ====="
    pushd "$dir" > /dev/null
    mpirun -np "$SLURM_NTASKS" "$VASP_BIN" > vasp.log 2>&1 || {
        echo "  FAILED - see $dir/vasp.log"; popd > /dev/null; continue; }
    if grep -q "reached required accuracy" OUTCAR; then
        echo "  converged. E = $(grep 'free  energy   TOTEN' OUTCAR | tail -1 | awk '{print $5}') eV"
        touch DONE
    else
        echo "  WARNING: ionic relaxation did NOT converge"
    fi
    popd > /dev/null
done

echo
echo "done. now run: ./check_gas_refs.sh"
echo
echo "scratch is purged on a timer. From the LOGIN node (/arc/project is"
echo "read-only from inside a job), copy the results somewhere durable:"
echo "  rsync -a --include='*/' --include='OUTCAR' --include='CONTCAR' \\"
echo "        --include='vasp.log' --exclude='*' \\"
echo "        $SLURM_SUBMIT_DIR/$GASDIR/ \\"
echo "        /arc/project/st-akkiraju-1/ssong18/runs/gas_refs_dft/"
