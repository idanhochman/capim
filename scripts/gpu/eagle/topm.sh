#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# CAPIM — EAGLE-2 FIXED-BUDGET (top-m) CONTROL traces. Thin wrapper over ./collect.sh.
#
# Why this run exists: our headline "the live gate beats the retrospective ranker at
# equal budget" compares EAGLE+sigma against MEDUSA+DTP, which differ in TWO things --
# the pruning policy AND the drafter (EAGLE is simply the better drafter: 4.98 vs 2.52
# acceptance length before anyone prunes). This run holds the drafter fixed and varies
# ONLY the budget rule, so it isolates the actual contribution.
#
# The control is exact, not approximate. The sigma gate keeps {nodes : cum >= sigma};
# EAGLE-2's own rerank keeps the top-m by that SAME cumulative log-prob. So both select
# from one ordering and differ only in how the cardinality is set:
#     sigma gate -> k chosen by the CONTEXT (adaptive)
#     top-m      -> m fixed by a hyperparameter
# That single variable IS RQ1.
#
# Usage (Kaggle GPU cell, >= sm_75 e.g. T4):
#     !bash capim/scripts/gpu/eagle/topm.sh              # m = 2 4 8 12, alpaca + gsm8k
#     !bash capim/scripts/gpu/eagle/topm.sh "2 4"        # subset
#     !DATASETS=alpaca bash capim/scripts/gpu/eagle/topm.sh
#
#   arg 1 = node budgets m (default "2 4 8 12"). m == verified nodes per step, EXACTLY
#           (mu = m, no variance -- that is what makes it a fixed budget).
#   DATASETS / N_PROMPTS / PRECISION env overrides still apply; keep them at the σ-run
#   defaults (alpaca gsm8k / 100 / int8) or the comparison is not iso.
#
# m = 59 is NOT in the default list because we already have it: EAGLE-2's shipped budget
# is total_token=60 -> 59 nodes, so the existing traces/eagle_<ds>_s-inf.json ARE the
# m=59 point of this curve (their measured mu is 59.00 exactly).
#
# Output: traces/eagle_<dataset>_m<m>.json
# ---------------------------------------------------------------------------
set -euo pipefail

M_LIST="${1:-${M_LIST:-2 4 8 12}}"

# EAGLE caps the tree at total_token-1 nodes (cnets.py: self.total_tokens = total_tokens-1),
# so ask for m+1 to verify m.
TOTAL_TOKEN=""
for m in $M_LIST; do TOTAL_TOKEN="${TOTAL_TOKEN} $((m + 1))"; done

# sigma=-inf: no gate. The tree is then purely EAGLE-2's fixed-budget rerank -- which is
# the whole point of the control.
SIGMA_TH="-inf" TOTAL_TOKEN="${TOTAL_TOKEN# }" \
    exec bash "$(dirname "${BASH_SOURCE[0]:-$0}")/collect.sh"
