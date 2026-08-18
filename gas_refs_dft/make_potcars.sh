#!/bin/bash
# Build POTCARs for every gas reference folder.
#
# Reads the element order out of each folder's POTCAR_ORDER.txt and concatenates
# the matching PBE_54 POTCARs in that exact order. Order matters: it must match
# the species order in the POSCAR or VASP will silently use the wrong potentials.
#
# USAGE:
#   export VASP_PSP_DIR=/path/to/potpaw_PBE.54
#   ./make_potcars.sh
#
# VASP_PSP_DIR must point at the directory that contains one subdirectory per
# element, each holding a POTCAR (or POTCAR.Z / POTCAR.gz). Typical layouts:
#   .../potpaw_PBE.54/H/POTCAR
#   .../potpaw_PBE.54/C/POTCAR
#   .../potpaw_PBE.54/O/POTCAR
#
# On Sockeye the VASP pseudopotentials usually live under your group's project
# space, e.g. /arc/project/st-<PI>-1/vasp/potpaw_PBE.54
# They are licensed - they will NOT be in a public module path.

set -euo pipefail

GASDIR="${GASDIR:-gas_references}"

if [[ -z "${VASP_PSP_DIR:-}" ]]; then
    echo "ERROR: VASP_PSP_DIR is not set."
    echo "  export VASP_PSP_DIR=/arc/project/st-<PI>-1/vasp/potpaw_PBE.54"
    exit 1
fi

if [[ ! -d "$VASP_PSP_DIR" ]]; then
    echo "ERROR: VASP_PSP_DIR does not exist: $VASP_PSP_DIR"
    exit 1
fi

echo "Pseudopotential library: $VASP_PSP_DIR"
echo

# Return the path to a readable POTCAR for one element, decompressing if needed.
fetch_potcar() {
    local sym="$1" dest="$2"
    local base="$VASP_PSP_DIR/$sym"

    if   [[ -f "$base/POTCAR"    ]]; then cat        "$base/POTCAR"    >> "$dest"
    elif [[ -f "$base/POTCAR.gz" ]]; then zcat       "$base/POTCAR.gz" >> "$dest"
    elif [[ -f "$base/POTCAR.Z"  ]]; then zcat       "$base/POTCAR.Z"  >> "$dest"
    else
        echo "  ERROR: no POTCAR found for element '$sym' at $base"
        return 1
    fi
}

nbuilt=0
for dir in "$GASDIR"/*/; do
    [[ -f "$dir/POTCAR_ORDER.txt" ]] || continue
    name=$(basename "$dir")

    # second line of POTCAR_ORDER.txt holds the whitespace-separated symbols
    symbols=$(sed -n '2p' "$dir/POTCAR_ORDER.txt")

    rm -f "$dir/POTCAR"
    printf "%-8s <- " "$name"
    for sym in $symbols; do
        printf "%s " "$sym"
        fetch_potcar "$sym" "$dir/POTCAR"
    done

    # Cross-check against the POSCAR species line (line 6 in VASP 5 format).
    poscar_species=$(sed -n '6p' "$dir/POSCAR" | xargs)
    order_species=$(echo "$symbols" | xargs)
    if [[ "$poscar_species" != "$order_species" ]]; then
        echo ""
        echo "  WARNING: POSCAR species line '$poscar_species' != POTCAR order '$order_species'"
        echo "  Fix this before running - VASP will not catch it for you."
    else
        ntitel=$(grep -c "TITEL" "$dir/POTCAR")
        echo " ok ($ntitel potentials)"
    fi
    nbuilt=$((nbuilt + 1))
done

echo
echo "Built $nbuilt POTCARs."
echo "Sanity check one of them:  grep TITEL $GASDIR/C3H8/POTCAR"
echo "You should see PAW_PBE C and PAW_PBE H, both dated for the .54 set."
