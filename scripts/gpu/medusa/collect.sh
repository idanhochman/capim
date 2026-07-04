#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# CAPIM — MEDUSA (LP-Spec baseline stack) trace collection over Alpaca + GSM8K.
#
# Usage (Kaggle GPU cell, >= sm_75 e.g. T4), cloning WITH submodules:
#     !git clone --recurse-submodules https://github.com/idanhochman/capim.git
#     !bash capim/scripts/gpu/medusa/collect.sh
#
# Prefer the sibling wrappers for the common cases:
#     medusa/alpaca.sh [L]   medusa/gsm8k.sh [L]   medusa/sanity.sh [L]
#
# Parameterized via env vars (all optional — bare invocation = the deliverable run):
#     L          DTP keep count(s); space-separated to SWEEP.  default "4"
#                (this is the COLLECTION-side gate `--L`; the driver's replay knob
#                 is a distinct `L_spec` on `main.py drive`.)
#     DATASETS   which sets; space-separated. "sanity" -> the 5 built-in prompts.
#                                                          default "alpaca gsm8k"
#     N_PROMPTS  prompts per dataset.                       default 100
#     PRECISION  int8 | int4 | fp16.                        default int8
#   examples:
#     L="4 8 16"      bash .../medusa/collect.sh            # 3-point L sweep
#
# Output: traces/medusa_<dataset>_L<L>.json  (L in the name so a sweep never
# overwrites). 8-bit (LLM.int8): base int8, MEDUSA's vocab heads stay FP16 so
# they load and emit real confidence scores. Same precision/prompts as EAGLE.
#
# NOTE: the CLI hardcodes the DTP policy to greedy_headk (MEDUSA's real rule);
# there is currently no flag to record an UNGATED/full-tree MEDUSA trace
# (needed later for the L-characterization sweep, not for this deliverable).
# ---------------------------------------------------------------------------
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]:-$0}")/../env.sh"

# MEDUSA's vendored modeling is known-good on the 4.36.2-era stack and breaks on
# the >=4.50 loader rewrite ("weight is not an nn.Module" under quant). accelerate
# is pinned to the matching release so the old transformers imports cleanly.
echo "==> installing MEDUSA python deps (transformers 4.36.2) ..."
pip install "transformers==4.36.2" "accelerate==0.25.0" bitsandbytes datasets sentencepiece protobuf

L="${L:-4}"
DATASETS="${DATASETS:-alpaca gsm8k}"
N_PROMPTS="${N_PROMPTS:-100}"
PRECISION="${PRECISION:-int8}"

for ds in $DATASETS; do
  for l in $L; do
    if [ "$ds" = "sanity" ]; then
      sel=(--sanity);            out="traces/medusa_sanity_L${l}.json"
    else
      sel=(--dataset "$ds" --n-prompts "$N_PROMPTS"); out="traces/medusa_${ds}_L${l}.json"
    fi
    echo "==> MEDUSA on ${ds}  (L=${l}, ${PRECISION}) -> ${out}"
    python -u main.py collect --method medusa "${sel[@]}" \
        --L "$l" --precision "$PRECISION" --out "$out"
  done
done

echo ""
echo "==> DONE. Traces under traces/ . Sanity check each: mean_acceptance_rate"
echo "    non-trivial (GSM8K typically > Alpaca); tree_size ~ L each step."
