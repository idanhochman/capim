#!/usr/bin/env bash
# CAPIM — MEDUSA trace collection on Alpaca only. Thin wrapper over ./collect.sh.
#   arg 1 = L (DTP keep count, default 4); quote a list to sweep.
#     bash capim/scripts/gpu/medusa/alpaca.sh            # L=4
#     bash capim/scripts/gpu/medusa/alpaca.sh 8          # L=8
#     bash capim/scripts/gpu/medusa/alpaca.sh "4 8 16"   # sweep
# N_PROMPTS / PRECISION env overrides still apply.
set -euo pipefail
L="${1:-${L:-4}}" DATASETS=alpaca \
    exec bash "$(dirname "${BASH_SOURCE[0]:-$0}")/collect.sh"
