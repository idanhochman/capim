#!/usr/bin/env bash
#
# Build the CAPIM final report (capim/doc/main.tex).
#
# The report uses fontspec (\setmainfont{Arial} / Times New Roman), so it MUST
# be compiled with XeLaTeX, not pdfLaTeX. The full sequence is:
#   xelatex -> bibtex -> makeindex (nomencl List of Symbols) -> xelatex -> xelatex
#
# Skipping the makeindex step compiles the List of Symbols EMPTY.
#
# Usage:
#   ./build.sh          # build main.pdf
#   ./build.sh clean    # remove build artifacts

set -euo pipefail

# Resolve the directory this script lives in, so it works from anywhere.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LATEX_DIR="$SCRIPT_DIR"
MAIN="main"

# TinyTeX (installed without sudo) lives in ~/.TinyTeX; put it on PATH first.
export PATH="$HOME/.TinyTeX/bin/x86_64-linux:$PATH"

if [[ "${1:-}" == "clean" ]]; then
    cd "$LATEX_DIR"
    rm -f "$MAIN".{aux,bbl,blg,log,lof,lot,toc,out,nlo,nls,ilg}
    echo "Cleaned build artifacts in $LATEX_DIR"
    exit 0
fi

if ! command -v xelatex >/dev/null 2>&1; then
    echo "ERROR: xelatex not found. Install TinyTeX (see CLAUDE.md sec 6) and re-run." >&2
    exit 1
fi

cd "$LATEX_DIR"

echo "==> [1/5] xelatex (first pass)"
xelatex -interaction=nonstopmode "$MAIN.tex"

echo "==> [2/5] bibtex"
bibtex "$MAIN"

echo "==> [3/5] makeindex (nomencl List of Symbols)"
makeindex "$MAIN.nlo" -s nomencl.ist -o "$MAIN.nls"

echo "==> [4/5] xelatex (resolve refs/citations)"
xelatex -interaction=nonstopmode "$MAIN.tex"

echo "==> [5/5] xelatex (final pass)"
xelatex -interaction=nonstopmode "$MAIN.tex"

echo
echo "Done: $LATEX_DIR/$MAIN.pdf"
grep -o "Output written on $MAIN.pdf ([0-9]* pages" "$MAIN.log" | sed 's/Output written on/  ->/' || true
