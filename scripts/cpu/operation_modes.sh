#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# CAPIM — CPU-only re-cost of collected traces into the operation-modes table.
#
# Reproduces the headline operation-modes comparison: the 4 CAPIM modes
# + the LP-Spec baseline + autoregressive, per dataset, then prints one merged
# table. No GPU, no deps (the drive path is pure stdlib).
#
#   arg 1 = sigma  (default -1.5)  — MUST match the EAGLE collection gate, since
#                                     it selects traces/eagle_<ds>_s<sigma>.json
#   arg 2 = L      (default 4)     — MUST match the MEDUSA collection keep count,
#                                     selects traces/medusa_<ds>_L<L>.json
#   env DATASETS     which sets, space-separated.  default "alpaca gsm8k"
#   env RESULTS_DIR  where drive JSONs land.       default "results"
#
#   examples:
#     bash capim/scripts/cpu/operation_modes.sh              # sigma=-1.5, L=4
#     bash capim/scripts/cpu/operation_modes.sh -2.0 8       # traces gated at those
#     DATASETS=alpaca bash capim/scripts/cpu/operation_modes.sh
#
# The 4 CAPIM modes: High-perf (NPU draft, mu_th=1), Standard (PIM, 1),
# Low-power (PIM, 4), Super-low-power (PIM, mu_th=64 ~ infinity, >= max tree 59).
# ---------------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

SIGMA="${1:--1.5}"
L="${2:-4}"
DATASETS="${DATASETS:-alpaca gsm8k}"
RESULTS_DIR="${RESULTS_DIR:-results}"

# prefer python3 (python may be absent on a bare CPU box), fall back to python.
PY="$(command -v python3 || command -v python)"
[ -n "$PY" ] || { echo "error: no python3/python on PATH" >&2; exit 1; }

mkdir -p "$RESULTS_DIR"

# --- pre-flight: every trace the drivers will read must already exist ---------
missing=0
for ds in $DATASETS; do
  for f in "traces/eagle_${ds}_s${SIGMA}.json" "traces/medusa_${ds}_L${L}.json"; do
    if [ ! -f "$f" ]; then echo "missing trace: $f" >&2; missing=1; fi
  done
done
if [ "$missing" -ne 0 ]; then
  echo "error: collect the traces first (scripts/gpu/{eagle,medusa}/...) " >&2
  echo "       with the SAME sigma=${SIGMA} / L=${L} this script expects." >&2
  exit 1
fi

# --- drive each dataset: 4 CAPIM modes + LP-Spec + AR -------------------------
outs=()
for ds in $DATASETS; do
  eagle="traces/eagle_${ds}_s${SIGMA}.json"
  medusa="traces/medusa_${ds}_L${L}.json"

  # High-perf (NPU draft, mu_th=1) + AR baseline (AR only needs a trace for its
  # token trajectory; it has no method of its own).
  o1="$RESULTS_DIR/drive_${ds}_capim_npu.json"
  echo "==> [$ds] CAPIM High-perf (NPU, mu=1) + AR"
  "$PY" -u main.py drive --eagle-trace "$eagle" \
      --driver ar capim --draft-device npu --mu-th 1 --out "$o1"

  # Standard / Low-power / Super-low-power (PIM draft, mu_th in {1, 4, 64~inf}).
  o2="$RESULTS_DIR/drive_${ds}_capim_pim.json"
  echo "==> [$ds] CAPIM Standard/Low/Super-low (PIM, mu=1 4 64)"
  "$PY" -u main.py drive --eagle-trace "$eagle" \
      --driver capim --draft-device pim --mu-th 1 4 64 --out "$o2"

  # LP-Spec baseline (MEDUSA + DTP), driver replay knob L_spec = collection L.
  o3="$RESULTS_DIR/drive_${ds}_lpspec.json"
  echo "==> [$ds] LP-Spec (L_spec=${L})"
  "$PY" -u main.py drive --medusa-trace "$medusa" \
      --driver lp_spec --L-spec "$L" --out "$o3"

  outs+=("$o1" "$o2" "$o3")
done

# --- one merged comparison table over everything -----------------------------
echo ""
echo "==> operation-modes comparison (sigma=${SIGMA}, L=${L}):"
echo ""
"$PY" scripts/cpu/print_table.py "${outs[@]}"
