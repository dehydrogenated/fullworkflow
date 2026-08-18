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
# Three distinct outcomes that must not be collapsed into one "not found":
#   (a) no VASP anywhere
#   (b) VASP present but its module is hidden -- licensed software on Sockeye is
#       gated behind a Unix group, and is invisible to `spider` until you are in it
#   (c) VASP present as a group-compiled binary with no module entry at all
# So this asks the module system AND the filesystem, and prints raw output rather
# than a verdict, because only (a) is actually blocking.
echo
echo "=== 1a. is the module system even available in this shell? ==="
if ! command -v module >/dev/null 2>&1 && ! declare -F module >/dev/null 2>&1; then
    echo "  'module' is not defined here. Re-run as:  bash -l ./preflight_sockeye.sh"
    echo "  (everything below in section 1 is meaningless until this is fixed)"
else
    module --version 2>&1 | head -3
fi

echo
echo "=== 1b. raw 'module spider vasp' output ==="
module spider vasp 2>&1 | head -40

echo
echo "=== 1c. other module-system views (a hidden module can show in one, not another) ==="
echo "--- module avail | grep -i vasp ---"
module avail 2>&1 | grep -i vasp || echo "  (nothing)"
echo "--- module --ignore_cache avail | grep -i vasp   (stale spider cache is a real failure mode) ---"
module --ignore_cache avail 2>&1 | grep -i vasp || echo "  (nothing)"
echo "--- module keyword vasp (RAW: a keyword hit is often a module whose NAME lacks the word) ---"
module keyword vasp 2>&1 | head -30
echo "--- regex spider, case-insensitive ---"
module -r spider '.*[Vv][Aa][Ss][Pp].*' 2>&1 | head -30
echo "--- regex avail, case-insensitive ---"
module -r avail '.*[Vv][Aa][Ss][Pp].*' 2>&1 | head -30

echo
echo "=== 1d. VASP binaries on the filesystem (catches a group build with no module) ==="
# /cvmfs is deliberately absent: it is a lazily-mounted network filesystem, so a
# find(1) walk over it stats the entire remote software stack and hangs. Query it
# through the module system (1b/1c) instead. -mount keeps the walk on one
# filesystem; timeout is a backstop for any other slow mount.
for root in /arc/project/st-akkiraju-1 /arc/software /opt/software; do
    [ -d "$root" ] || continue
    echo "--- under $root (60s cap) ---"
    timeout 60 find "$root" -mount -maxdepth 6 \
        \( -name 'vasp_gam' -o -name 'vasp_std' -o -name 'vasp_ncl' -o -name 'vasp' \) \
        -type f 2>/dev/null | head -10
    [ "${PIPESTATUS[0]}" = "124" ] && echo "  (timed out -- not a result either way)"
done

echo
echo "=== 1e. your Unix groups (licensed software is gated by group membership) ==="
# A group named for vasp means the licence exists and you are already in it. No
# such group, but a colleague who can run VASP, means you need to be added --
# that is a support ticket, not a dead end.
id -Gn 2>/dev/null | tr ' ' '\n' | sort

# Only a genuinely empty result across 1b-1d is blocking.
if module spider vasp 2>&1 | grep -qiE "unable to find|error" \
   && ! module avail 2>&1 | grep -qi vasp; then
    VASP_OK=0
else
    VASP_OK=1
fi

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
for root in /arc/project/st-akkiraju-1 /arc/software /opt/software; do
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
    done < <(timeout 60 find "$root" -mount -maxdepth 6 -type d \( -name "potpaw_PBE*" -o -name "*PBE.54*" -o -name "*PBE_54*" \) 2>/dev/null)
done
if [ "$found" -eq 0 ]; then
    echo "  no potpaw_PBE* directory found. Widening to any file literally named POTCAR:"
    for root in /arc/project/st-akkiraju-1 /arc/software; do
        [ -d "$root" ] || continue
        timeout 60 find "$root" -mount -maxdepth 7 -name POTCAR -type f 2>/dev/null | head -10
    done
    echo "  (still nothing -> ask Kiran where the group keeps the pseudopotentials)"
fi

echo
echo "=== verdict ==="
if [ "${VASP_OK:-0}" -eq 0 ]; then
    echo "No VASP found by module OR on the filesystem."
    echo "  This is NOT proof it is unavailable: licensed modules stay hidden until"
    echo "  your account is added to the licence group. Send Kiran / ARC support"
    echo "  sections 1b-1e above and ask which group grants VASP access."
else
    echo "Report the three sections above before editing submit_gas_refs.slurm."
    echo "For O, expect a TITEL dated 2015 or later for the .54 set."
fi
