#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# CAPIM — CPU-only CANONICAL drive: re-cost every collected trace into ONE
# self-describing results artifact, results/drive_all.json.
#
# This is the single source of truth the figures + comparison table read.  It
# sweeps the full design grid so the plots never re-drive and never parse trace
# filenames (each record carries dataset, driver, config{mu_th,draft_device} /
# {L_spec}, and collection_gate = the sigma/L the trace was CAUSALLY gated at).
#
# CAUSAL-ONLY: each EAGLE trace is replayed as-gated (--sigma-th=-inf), so the
# swept sigma axis IS the collection gate (which trace file we load), not a
# re-gate at cost time.  Likewise each MEDUSA trace is replayed at L_spec = its
# collection L (>= its tree_size -> pass-through).  No trace is re-thresholded.
#
#   env DATASETS     space-separated.  default "alpaca gsm8k"
#   env SIGMAS       EAGLE gates -> traces/eagle_<ds>_s<sigma>.json
#                                     default "-0.5 -1 -1.5 -2 -2.5 -inf"
#   env MU_THS       CAPIM routing-threshold grid (per (ds,sigma) invocation)
#                                     default "1 2 4 8 12 16 64"
#   env LS           MEDUSA keep counts -> traces/medusa_<ds>_L<L>.json
#                                     default "2 4 8 12 16 64"
#   env RESULTS_DIR  where parts + drive_all.json land.  default "results"
#   env OUT          merged artifact path.  default "$RESULTS_DIR/drive_all.json"
#
#   examples:
#     bash capim/scripts/cpu/drive_all.sh                 # the full canonical run
#     DATASETS=alpaca bash capim/scripts/cpu/drive_all.sh # one dataset
#     SIGMAS="-1.5 -inf" MU_THS="1 4 64" bash .../drive_all.sh  # a quick subset
#
# Per dataset it emits: 1 AR + (|SIGMAS| x |MU_THS| x 2 draft) CAPIM + |LS|
# LP-Spec records.  With the defaults: (1 + 6*7*2 + 6) * 2 = 182 records.
# Each `main.py drive` invocation writes a part under $RESULTS_DIR/parts/;
# they are concatenated into drive_all.json (parts kept as debug artifacts).
# The drive path is pure stdlib -- no GPU, no torch, no deps.
# ---------------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

DATASETS="${DATASETS:-alpaca gsm8k}"
SIGMAS="${SIGMAS:--0.5 -1 -1.5 -2 -2.5 -inf}"
MU_THS="${MU_THS:-1 2 4 8 12 16 64}"
LS="${LS:-2 4 8 12 16 64}"
RESULTS_DIR="${RESULTS_DIR:-results}"
PARTS_DIR="$RESULTS_DIR/parts"
OUT="${OUT:-$RESULTS_DIR/drive_all.json}"

# prefer python3 (python may be absent on a bare CPU box), fall back to python.
PY="$(command -v python3 || command -v python)"
[ -n "$PY" ] || { echo "error: no python3/python on PATH" >&2; exit 1; }

# --- pre-flight: every trace the drivers will read must already exist ---------
missing=0
for ds in $DATASETS; do
  for s in $SIGMAS;  do [ -f "traces/eagle_${ds}_s${s}.json" ]  || { echo "missing: traces/eagle_${ds}_s${s}.json"  >&2; missing=1; }; done
  for l in $LS;      do [ -f "traces/medusa_${ds}_L${l}.json" ] || { echo "missing: traces/medusa_${ds}_L${l}.json" >&2; missing=1; }; done
  [ -f "traces/eagle_${ds}_s-inf.json" ] || { echo "missing: traces/eagle_${ds}_s-inf.json (AR baseline)" >&2; missing=1; }
done
if [ "$missing" -ne 0 ]; then
  echo "error: collect the missing traces first (scripts/gpu/{eagle,medusa}/...)." >&2
  exit 1
fi

# --- fresh parts dir so a stale record can never leak into the merge ----------
rm -rf "$PARTS_DIR"
mkdir -p "$PARTS_DIR"

outs=()
for ds in $DATASETS; do
  # AR baseline: one record per dataset.  SD is lossless -> the decoded text (and
  # thus AR's per-token forward-pass cost) is trace-invariant; we read it off the
  # ungated full-tree EAGLE trace as the canonical trajectory.
  ar_out="$PARTS_DIR/ar_${ds}.json"
  echo "==> [$ds] AR baseline"
  "$PY" -u main.py drive --eagle-trace "traces/eagle_${ds}_s-inf.json" \
      --driver ar --out "$ar_out"
  outs+=("$ar_out")

  # CAPIM: full (sigma x mu_th x draft_device) sweep, causal replay-as-gated.
  # One invocation per (ds,sigma) sweeps mu_th x {npu,pim} = |MU_THS|*2 records.
  for s in $SIGMAS; do
    o="$PARTS_DIR/capim_${ds}_s${s}.json"
    echo "==> [$ds] CAPIM  sigma=${s}  (mu_th={${MU_THS// /,}} x draft={npu,pim})"
    # --sigma-th=-inf (equals form): argparse misreads a bare `-inf` as a flag.
    "$PY" -u main.py drive --eagle-trace "traces/eagle_${ds}_s${s}.json" \
        --driver capim --draft-device npu pim --mu-th $MU_THS \
        --sigma-th=-inf --out "$o"
    outs+=("$o")
  done

  # LP-Spec baseline: full L sweep, causal replay (L_spec = collection L).
  for l in $LS; do
    o="$PARTS_DIR/lpspec_${ds}_L${l}.json"
    echo "==> [$ds] LP-Spec  L=${l}"
    "$PY" -u main.py drive --medusa-trace "traces/medusa_${ds}_L${l}.json" \
        --driver lp_spec --L-spec "$l" --out "$o"
    outs+=("$o")
  done
done

# --- merge every part into the single canonical artifact ----------------------
echo ""
"$PY" - "$OUT" "${outs[@]}" <<'PYEOF'
import json, sys
from collections import Counter
out, parts = sys.argv[1], sys.argv[2:]
recs = []
for p in parts:
    recs.extend(json.load(open(p)))
with open(out, "w") as f:
    json.dump(recs, f, indent=2)
by = Counter((r["driver"], r["dataset"]) for r in recs)
print(f"merged {len(parts)} parts -> {out}  ({len(recs)} records)")
for (drv, ds), n in sorted(by.items()):
    print(f"    {drv:<8} {ds:<7} {n:>4}")
PYEOF