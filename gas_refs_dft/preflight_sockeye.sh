#!/bin/bash
# Answer the three unknowns that block submit_gas_refs.slurm. LOGIN NODE ONLY --
# compute nodes have no outbound network and a stripped module tree, so a check
# that passes there tells you nothing about what the job will actually see.
#
#   ./preflight_sockeye.sh
#
# Read-only: probes and prints, changes nothing. Nothing here submits a job.

echo "host: $(hostname)   user: $USER"
case "$(hostname)" in
    *login*|*sockeye*) ;;
    *) echo "WARNING: this does not look like a Sockeye login node." ;;
esac

# ---------------------------------------------------------------------------
# 1. Does VASP exist here at all, and does it provide vasp_gam?
# ---------------------------------------------------------------------------
echo
echo "=== 1. VASP module ==="
spider_out=$(module spider vasp 2>&1)
if echo "$spider_out" | grep -qi "unable to find\|error"; then
    echo "NOT FOUND. Sockeye exposes no VASP module to you."
    echo "  VASP is licensed per-group and is usually gated behind a module"
    echo "  restricted to licence holders -- 'not found' can mean 'not licensed"
    echo "  to st-akkiraju-1' rather than 'not installed'."
    echo "  STOP HERE and ask Kiran whether the group holds a VASP licence."
    VASP_OK=0
else
    echo "$spider_out" | grep -iE "^\s*vasp" | head -20
    VASP_OK=1
fi

# vasp_gam is what the job wants: Gamma-only, ~2x faster than vasp_std here.
echo
echo "--- binaries on PATH after loading (try each candidate module) ---"
for mod in $(echo "$spider_out" | grep -oE "vasp/[A-Za-z0-9._-]+" | sort -u); do
    ( module purge >/dev/null 2>&1
      if module load "$mod" >/dev/null 2>&1; then
          gam=$(command -v vasp_gam || echo "-")
          std=$(command -v vasp_std || echo "-")
          printf "  %-24s vasp_gam=%s  vasp_std=%s\n" "$mod" "$gam" "$std"
      else
          printf "  %-24s LOAD FAILED (likely a licence-restricted module)\n" "$mod"
      fi )
done

# ---------------------------------------------------------------------------
# 2. Which allocation codes can actually submit?
# ---------------------------------------------------------------------------
echo
echo "=== 2. Allocations ==="
# The CPU account is what this job wants -- these are 2-11 atom molecules and
# VASP on 8 CPU ranks finishes in minutes. Do not send them to a GPU account.
sacctmgr -nP show assoc user="$USER" format=Account,Partition 2>/dev/null | sort -u \
    || echo "  sacctmgr unavailable; fall back to: sshare -U -u $USER"

# ---------------------------------------------------------------------------
# 3. Where is the POTCAR library, and is it the .54 set?
# ---------------------------------------------------------------------------
# This is the one mistake that produces no error, no warning, and completely
# wrong energies. PBE_54 vs PBE_52 differ in the O potential in particular.
echo
echo "=== 3. POTCAR library ==="
found=0
for root in /arc/project/st-akkiraju-1 /arc/software /arc/project/st-akkiraju-1/ssong18; do
    [ -d "$root" ] || continue
    while IFS= read -r hit; do
        found=1
        echo "  candidate: $hit"
        for el in O H; do
            p="$hit/$el/POTCAR"
            [ -f "$p.gz" ] && p="$p.gz"
            if [ -f "$p" ]; then
                # TITEL carries the set's date -- that is what distinguishes
                # .54 from 5.2, since the directory name can lie.
                titel=$(zgrep -m1 TITEL "$p" 2>/dev/null || grep -m1 TITEL "$p" 2>/dev/null)
                printf "    %-2s %s\n" "$el" "${titel:-<unreadable>}"
            else
                printf "    %-2s MISSING\n" "$el"
            fi
        done
    done < <(find "$root" -maxdepth 4 -type d \( -name "potpaw_PBE*" -o -name "*PBE.54*" -o -name "*PBE_54*" \) 2>/dev/null)
done
[ "$found" -eq 0 ] && echo "  none found under the searched roots -- ask Kiran where the group keeps them."

echo
echo "=== verdict ==="
if [ "${VASP_OK:-0}" -eq 0 ]; then
    echo "BLOCKED: no VASP. Nothing else in this folder can run."
else
    echo "Report the three sections above before editing submit_gas_refs.slurm."
    echo "For O, expect a TITEL dated 2015 or later for the .54 set."
fi
