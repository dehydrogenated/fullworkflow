#!/bin/bash
# Verify the gas reference calculations and print the energy table you will
# paste into your workflow. Run this AFTER the SLURM job finishes.
#
# Checks, per molecule:
#   - did the ionic relaxation converge
#   - did the electronic loop converge (no NELM hit on the last step)
#   - is the spin state what you asked for  <-- this is the one that catches O2
#   - final total energy

set -uo pipefail
GASDIR="${GASDIR:-gas_references}"

printf "%-8s %14s %10s %8s  %s\n" "species" "E_DFT (eV)" "mag" "ionic" "status"
printf -- "----------------------------------------------------------------\n"

allgood=1
for dir in "$GASDIR"/*/; do
    name=$(basename "$dir")
    out="$dir/OUTCAR"

    if [[ ! -f "$out" ]]; then
        printf "%-8s %14s %10s %8s  %s\n" "$name" "-" "-" "-" "NOT RUN"
        allgood=0
        continue
    fi

    energy=$(grep "free  energy   TOTEN" "$out" | tail -1 | awk '{print $5}')
    [[ -z "$energy" ]] && energy="-"

    # total magnetization of the final step
    mag=$(grep "number of electron" "$out" | tail -1 | awk '{print $6}')
    [[ -z "$mag" ]] && mag="0.000"

    status="ok"
    bond_note=""

    if ! grep -q "reached required accuracy" "$out"; then
        status="IONIC NOT CONVERGED"
        allgood=0
    fi

    # electronic non-convergence on the final ionic step
    nelm=$(grep -m1 "NELM   =" "$out" | sed 's/.*NELM   = *\([0-9]*\).*/\1/')
    last_iter=$(grep -E "^(DAV|RMM|EDDAV):" "$out" | tail -1 | awk '{print $2}' | tr -d ':')
    if [[ -n "${nelm:-}" && -n "${last_iter:-}" ]] && (( last_iter >= nelm )); then
        status="ELECTRONIC NOT CONVERGED"
        allgood=0
    fi

    # --- the O2 spin check ---
    # O2 must come out as a triplet, total moment ~2.0 mu_B. If it is ~0 you
    # converged the singlet and the energy is ~1 eV too high, which will poison
    # every oxygen vacancy formation energy downstream.
    #
    # Caveat: NUPDOWN=2 *constrains* the moment, so this confirms the tag was
    # parsed, not that the physics is right. The independent check is the bond
    # length below -- geometry is not constrained by NUPDOWN, so a triplet that
    # relaxed to a singlet-like 1.20 A is a real signal.
    if [[ "$name" == "O2" ]]; then
        m=$(printf "%.1f" "$mag" 2>/dev/null || echo 0)
        if awk "BEGIN{exit !($m < 1.5)}"; then
            status="*** O2 IS NOT A TRIPLET - CHECK NUPDOWN=2 ***"
            allgood=0
        fi
    fi

    # --- geometry sanity: bond length of the diatomics ---
    # PBE reference values: O2 1.23 A (triplet), H2 0.75 A. A relaxation that
    # converged cleanly onto the wrong structure passes every check above.
    case "$name" in
        O2) expect=1.23 ;;
        H2) expect=0.75 ;;
        *)  expect="" ;;
    esac
    if [[ -n "$expect" && -f "$dir/CONTCAR" ]]; then
        d=$(python3 - "$dir/CONTCAR" <<'PYEOF'
import sys
from ase.io import read
a = read(sys.argv[1], format="vasp")
print(f"{a.get_distance(0, 1, mic=True):.3f}")
PYEOF
) || d=""
        if [[ -n "$d" ]] && awk "BEGIN{exit !(($d - $expect) < -0.05 || ($d - $expect) > 0.05)}"; then
            status="BOND LENGTH $d A, expected ~$expect A"
            allgood=0
        elif [[ -n "$d" ]]; then
            bond_note="  (r = $d A)"
        fi
    fi

    printf "%-8s %14s %10s %8s  %s%s\n" "$name" "$energy" "$mag" \
        "$(grep -c 'reached required accuracy' "$out")" "$status" "$bond_note"
done

echo
if [[ $allgood -eq 1 ]]; then
    echo "All references clean. Next: python3 oxygen_reference.py"
else
    echo "Fix the flagged runs before using these numbers. A bad gas reference is"
    echo "a CONSTANT offset on every adsorption energy - it will not look like an"
    echo "error, it will look like a trend."
fi

# Dump a python dict you can paste straight into your workflow
echo
echo "# --- paste into your script ---"
echo "E_GAS_DFT = {"
for dir in "$GASDIR"/*/; do
    name=$(basename "$dir")
    [[ -f "$dir/OUTCAR" ]] || continue
    energy=$(grep "free  energy   TOTEN" "$dir/OUTCAR" | tail -1 | awk '{print $5}')
    [[ -n "$energy" ]] && echo "    \"$name\": $energy,"
done
echo "}"
