#!/usr/bin/env bash
# CAPIM — EAGLE trace collection on Alpaca only. Thin wrapper over ./collect.sh.
#   arg 1 = sigma_th gate (default -1.5); quote a list to sweep.
#     bash capim/scripts/gpu/eagle/alpaca.sh              # sigma_th=-1.5
#     bash capim/scripts/gpu/eagle/alpaca.sh -2.0         # sigma_th=-2.0
#     bash capim/scripts/gpu/eagle/alpaca.sh "-1.0 -1.5"  # sweep
# N_PROMPTS / PRECISION env overrides still apply.
set -euo pipefail
SIGMA_TH="${1:-${SIGMA_TH:--1.5}}" DATASETS=alpaca \
    exec bash "$(dirname "${BASH_SOURCE[0]:-$0}")/collect.sh"
