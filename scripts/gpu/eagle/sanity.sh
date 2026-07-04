#!/usr/bin/env bash
# CAPIM — EAGLE sanity collection (5 built-in prompts, no dataset download).
# Thin wrapper over ./collect.sh. Run this FIRST on a fresh GPU box to confirm
# the EAGLE stack loads and drafts before committing to the full runs.
#   arg 1 = sigma_th gate (default -1.5); pass -inf to inspect the full tree.
#     bash capim/scripts/gpu/eagle/sanity.sh          # sigma_th=-1.5
#     bash capim/scripts/gpu/eagle/sanity.sh -inf     # ungated full tree
# Checks after it runs (traces/eagle_sanity_s<sigma>.json):
#   * steps > 0 and log_probs VARY (not all -ln|V|) -> draft head loaded (FP16)
#   * tree_size varies step-to-step (dynamic tree, unlike MEDUSA's constant 63)
set -euo pipefail
SIGMA_TH="${1:-${SIGMA_TH:--1.5}}" DATASETS=sanity \
    exec bash "$(dirname "${BASH_SOURCE[0]:-$0}")/collect.sh"
