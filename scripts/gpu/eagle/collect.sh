#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# CAPIM — EAGLE (CAPIM draft stack) trace collection over Alpaca + GSM8K.
#
# Usage (Kaggle GPU cell, >= sm_75 e.g. T4), cloning WITH submodules:
#     !git clone --recurse-submodules https://github.com/idanhochman/capim.git
#     !bash capim/scripts/gpu/eagle/collect.sh
#
# Prefer the sibling wrappers for the common cases:
#     eagle/alpaca.sh [sigma]   eagle/gsm8k.sh [sigma]   eagle/sanity.sh [sigma]
#
# Parameterized via env vars (all optional — bare invocation = the deliverable run):
#     SIGMA_TH     cumulative-log-prob gate(s); space-separated to SWEEP. default "-1.5"
#                  "-inf" = no gate: the tree is then whatever EAGLE-2's own rerank keeps.
#     TOTAL_TOKEN  EAGLE rerank budget(s); space-separated to SWEEP.      default "60"
#                  Caps the tree at TOTAL_TOKEN-1 nodes, so mu == TOTAL_TOKEN-1 EXACTLY.
#                  60 is EAGLE-2's shipped budget (mu=59). With SIGMA_TH=-inf this is the
#                  fixed-top-m CONTROL -- prefer the ./topm.sh wrapper, which takes m.
#     DATASETS     which sets; space-separated. "sanity" -> the 5 built-in prompts.
#                                                            default "alpaca gsm8k"
#     N_PROMPTS    prompts per dataset.                      default 100
#     PRECISION    int8 | int4 | fp16.                       default int8
#   examples:
#     SIGMA_TH="-1.0 -1.5 -2.0" bash .../eagle/collect.sh   # 3-point sigma sweep
#     DATASETS=alpaca SIGMA_TH=-inf bash .../eagle/collect.sh  # ungated (== top-m, m=59)
#     bash .../eagle/topm.sh "2 4 8 12"                     # the fixed-budget control
#
# Output: traces/eagle_<dataset>_s<sigma>.json for a gate sweep, or _m<m>.json when a
# non-default budget is set (the swept knob is in the name, so a sweep never overwrites).
# 8-bit (LLM.int8): base int8, EAGLE draft head stays FP16 (EaModel loads it separately,
# so it is NOT quantized) -> scores track FP16.
# ---------------------------------------------------------------------------
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]:-$0}")/../env.sh"

# Pinned EAGLE (cb7e084) needs transformers==4.53.1: its Qwen3 import wants symbols
# only in >=4.52/4.53 (use_kernel_forward_from_hub) yet also LossKwargs, dropped in
# later releases, so >= drifts too new and re-breaks. Pin exact. Fairness unaffected.
echo "==> installing EAGLE python deps (transformers 4.53.1) ..."
pip install "transformers==4.53.1" "accelerate>=0.26" bitsandbytes datasets sentencepiece protobuf

SIGMA_TH="${SIGMA_TH:--1.5}"
TOTAL_TOKEN="${TOTAL_TOKEN:-60}"
DATASETS="${DATASETS:-alpaca gsm8k}"
N_PROMPTS="${N_PROMPTS:-100}"
PRECISION="${PRECISION:-int8}"

for ds in $DATASETS; do
  for s in $SIGMA_TH; do
    for tt in $TOTAL_TOKEN; do
      # Name by the knob actually being swept: sigma at the default budget, else the node
      # budget m (both, if a run varies both). Keeps every pre-existing invocation's
      # output name byte-identical.
      if [ "$tt" -eq 60 ]; then tag="s${s}"
      elif [ "$s" = "-inf" ]; then tag="m$((tt - 1))"
      else tag="m$((tt - 1))_s${s}"; fi

      if [ "$ds" = "sanity" ]; then
        sel=(--sanity);                                  out="traces/eagle_sanity_${tag}.json"
      else
        sel=(--dataset "$ds" --n-prompts "$N_PROMPTS");  out="traces/eagle_${ds}_${tag}.json"
      fi
      echo "==> EAGLE on ${ds}  (sigma_th=${s}, budget m=$((tt - 1)), ${PRECISION}) -> ${out}"
      # --sigma-th=$s (equals form): argparse's negative-number matcher only accepts
      # digits, so a bare `--sigma-th -inf` is misread as an option and rejected.
      python -u main.py collect --method eagle "${sel[@]}" \
          --sigma-th="$s" --total-token "$tt" --precision "$PRECISION" --out "$out"
    done
  done
done

echo ""
echo "==> DONE. Traces under traces/ . Sanity check each: log_probs VARY (not all"
echo "    -ln|V|), tree_size varies step-to-step (dynamic), accept > the Medusa run."
