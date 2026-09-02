#!/usr/bin/env bash
# Pulls run output from Sockeye's durable $PROJECT/runs down to this laptop's runs/,
# so a completed job becomes locally readable/analyzable without a manual rsync
# invocation each time. This is the second hop -- every .slurm job script already
# prints the first hop (scratch -> $PROJECT/runs) to run on the Sockeye login node;
# this script runs from the local machine and does scratch's *destination* -> here.
#
# Usage:
#   scripts/core/sync_sockeye_runs.sh                    # pull the whole remote runs/ tree
#   scripts/core/sync_sockeye_runs.sh mo2_ads_benchmark   # pull just one run subdirectory
set -euo pipefail

REMOTE_HOST="ssong18@sockeye.arc.ubc.ca"  # NOT arc.sockeye.ubc.ca -- that hostname doesn't
# resolve (confirmed via nslookup); the subdomain order in ~/.ssh/config's alias is swapped.
REMOTE_RUNS="/arc/project/st-akkiraju-1/ssong18/fullworkflow/runs"
LOCAL_RUNS="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/runs"

SUBPATH="${1:-}"
rsync -avz --progress \
  "${REMOTE_HOST}:${REMOTE_RUNS}/${SUBPATH:+$SUBPATH/}" \
  "${LOCAL_RUNS}/${SUBPATH:+$SUBPATH/}"
