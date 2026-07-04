#!/usr/bin/env bash
# CAPIM — MEDUSA sanity collection (5 built-in prompts, no dataset download).
# Thin wrapper over ./collect.sh. Run this FIRST on a fresh GPU box to confirm
# the MEDUSA stack loads and the heads emit real scores before the full runs.
#   arg 1 = L (DTP keep count, default 4).
#     bash capim/scripts/gpu/medusa/sanity.sh          # L=4
#     bash capim/scripts/gpu/medusa/sanity.sh 8        # L=8
# Checks after it runs (traces/medusa_sanity_L<L>.json):
#   * steps > 0 and mean_acceptance_rate non-trivial -> heads loaded + working
#   * tree_size ~ L each step (the DTP kept L nodes)
set -euo pipefail
L="${1:-${L:-4}}" DATASETS=sanity \
    exec bash "$(dirname "${BASH_SOURCE[0]:-$0}")/collect.sh"
