"""
Drift guard for the vendored EAGLE ``topK_genrate`` (capim_ctrl/eagle_topk.py).

The collector installs a VERBATIM copy of upstream ``topK_genrate`` plus one marked
``_capim_scores`` stash (see that module's docstring).  "Verbatim + one line" is only
safe if it STAYS verbatim as the pinned submodule evolves -- so this test re-derives the
guarantee mechanically instead of trusting a comment:

  1. take our vendored function source, delete the marked ``CAPIM STASH BEGIN..END``
     block (the sole sanctioned deviation),
  2. take upstream ``Model.topK_genrate`` from the pinned submodule
     (``sd_repos/EAGLE/eagle/model/cnets1.py``),
  3. parse both to ASTs and assert ``ast.dump`` equality.

Comparing ASTs (not raw text) makes the check robust to comments / blank lines /
indentation while still catching any real code change.  It is GPU-free and imports no
ML deps -- it reads source text only.  If the submodule is absent (no
``--recurse-submodules``), the test SKIPS with an actionable message rather than failing.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_VENDORED = _REPO_ROOT / "capim_ctrl" / "eagle_topk.py"
_UPSTREAM = _REPO_ROOT / "sd_repos" / "EAGLE" / "eagle" / "model" / "cnets1.py"

_BEGIN = "CAPIM STASH BEGIN"
_END = "CAPIM STASH END"


def _find_func(source: str, name: str) -> ast.AST:
    """Return the (first) FunctionDef named ``name`` anywhere in ``source``."""
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"function {name!r} not found")


def _strip_capim_block(source: str) -> str:
    """Remove the marked CAPIM stash block (BEGIN..END inclusive) from our copy."""
    lines = source.splitlines(keepends=True)
    begins = [i for i, ln in enumerate(lines) if _BEGIN in ln]
    ends = [i for i, ln in enumerate(lines) if _END in ln]
    assert len(begins) == 1 and len(ends) == 1, (
        "expected exactly one CAPIM STASH BEGIN/END pair in the vendored copy"
    )
    assert begins[0] < ends[0], "CAPIM STASH markers out of order"
    del lines[begins[0]:ends[0] + 1]
    return "".join(lines)


def test_vendored_topk_matches_upstream():
    if not _UPSTREAM.is_file():
        pytest.skip(
            f"EAGLE submodule missing at {_UPSTREAM} -- run "
            "`git submodule update --init sd_repos/EAGLE`"
        )

    vendored_src = _strip_capim_block(_VENDORED.read_text())
    ours = _find_func(vendored_src, "topK_genrate")
    upstream = _find_func(_UPSTREAM.read_text(), "topK_genrate")

    assert ast.dump(ours) == ast.dump(upstream), (
        "vendored capim_ctrl/eagle_topk.py has drifted from upstream "
        "sd_repos/EAGLE/eagle/model/cnets1.py::topK_genrate -- re-vendor it "
        "(only the marked _capim_scores stash may differ)."
    )
