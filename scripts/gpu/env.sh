#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# CAPIM — shared GPU-session prelude for trace collection.
# SOURCE this from eagle/collect.sh / medusa/collect.sh; it is not run standalone.
#
# It:
#   1. locates the repo root (this file lives in <repo>/scripts/gpu/) and cd's there
#      so `python main.py ...` resolves the top-level packages (common/, prompts/, …)
#   2. initialises the sd_repos/ git submodules the collectors import from
#      (EAGLE @ cb7e084 -> `import eagle`, Medusa @ e2a5d20 -> `import medusa`);
#      a fresh clone leaves them EMPTY until inited, and the collector import
#      guard fails otherwise
#   3. prints GPU + RAM so you can confirm the runtime (needs >= sm_75, e.g. T4)
#
# It deliberately does NOT install python deps: EAGLE and MEDUSA pin mutually
# incompatible transformers, so each <method>/collect.sh installs its own beforehand.
# ---------------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"
echo "==> repo root: $REPO_ROOT"

echo "==> initialising sd_repos/ submodules (EAGLE, Medusa) ..."
git submodule update --init --recursive

mkdir -p traces

echo "==> GPU:"; nvidia-smi --query-gpu=name,memory.total --format=csv || echo "  (no nvidia-smi / no GPU)"
echo "==> RAM:"; free -h
