"""
Path invalidation — the shared trajectory-side gating MECHANISM.

Both collectors turn a per-step *keep decision* into a genuinely-shorter accepted
prefix the SAME way: they overwrite pruned nodes in the model's `retrieve_indices`
with the `-1` padding the verifier already understands.  This module is that one
operation, isolated from the policy that produces the keep set:

  - CAPIM  (`capim_ctrl/collector.py`): keep = {nodes with cumulative_log_prob >= sigma}.
  - LP-Spec(`baselines/lp_spec/collector.py`): keep = DTP greedy top-L from the
    retrospective histogram.

Only the keep set differs; the invalidation is identical, so it lives here once.
`invalidate_paths` takes the keep set as DATA (a set of node indices), never a
threshold -- σ / L stay in the collectors.

Why this makes the trajectory causal (greedy / temperature = 0)
--------------------------------------------------------------
EAGLE/MEDUSA both verify with, in effect,
    posterior_mask   = (candidates[:, 1:] == argmax(logits[:, :-1]))
    accept_length    = cumprod(posterior_mask, dim=1).sum(dim=1).max()
`retrieve_indices` selects the candidate tokens per path, and setting a position to
`-1` is EXACTLY how the two stacks already pad paths shorter than the max tree depth
-- so our injected `-1` is indistinguishable from native padding and rides the same
verified behaviour.  Both build the candidate row by *gathering with `retrieve_indices`*
from a token vector that has a dedicated pad slot appended, so a `-1` (negative index)
selects that pad slot rather than a real drafted token:
  - EAGLE `eagenerate` appends ``padding = -1`` to `draft_tokens` (ea_model.py) and
    then gathers `draft_tokens[0, retrieve_indices]`, so `-1` yields candidate token
    id **-1**.  Since a target argmax is always a valid vocab id >= 0, `-1` can NEVER
    match -> the truncation is EXACT for greedy, no caveat.
  - MEDUSA `generate_candidates` appends a **0** pad slot to `tree_candidates_ext`
    and gathers with `retrieve_indices`, so `-1` yields token id 0 (`<unk>`).  The
    target effectively never argmaxes `<unk>`, so truncation is exact in practice
    (a negligible, output-preserving coincidence at worst -- if it ever matched, the
    accepted token would be exactly what greedy emits, so the sequence is unchanged).
Either way `cumprod` zeroes from that position on, capping `accept_length` at exactly
the first pruned node, and `update_inference_inputs` advances the real KV-cache by
that shorter prefix.  The next step is drafted from the genuinely-shortened context:
the future is the *fixed* future, not a counterfactual stitched on after the fact.

(The `-1` sentinel in EAGLE's legacy `generate_candidates` -- `tree_candidates_ext`
appends -1 -- is a *different* code path from the `eagenerate` loop above; the loop is
what our collector drives, and it reaches the same exact result via the appended
`padding` token.)

Correctness precondition: the keep set must be ANCESTOR-CLOSED (if a node survives,
so do all its ancestors).  Then truncating each path independently at its first
pruned node reconstructs exactly the kept sub-tree, with no cross-path bookkeeping.
Both policies guarantee this -- CAPIM by the monotonicity of cumulative log-prob
with depth, LP-Spec by the connected greedy top-L construction.  `invalidate_paths`
verifies it and raises if a caller passes a malformed (non-closed) keep set.

Shape / index contract (verified against both repos)
----------------------------------------------------
`retrieve_indices` is a 2D int structure: one row per candidate root->leaf path,
one column per depth position, `-1`-padded on the right for short paths.  A value is
a node index into the step's `draft_tokens` (EAGLE) / candidate tensor (MEDUSA):

  - value  0  == the root (the already-accepted "sample" token). ALWAYS survives;
                it is column 0 of every path and is never a gateable draft node.
  - value -1  == existing padding. Passed through (idempotent).
  - value >0  == a drafted node; survives iff it is in `kept_nodes`.

This module is pure python (no torch/numpy) by design: it is unit-testable without a
GPU or the model, and the collectors do the tiny `.tolist()` <-> `torch.tensor()`
bridging on their side (retrieve_indices is at most ~leaves x depth, so the round
trip is negligible and runs once per step).
"""

from __future__ import annotations

from typing import Iterable, List, Sequence


def invalidate_paths(
    retrieve_indices: Sequence[Sequence[int]],
    kept_nodes: Iterable[int],
) -> List[List[int]]:
    """Truncate every candidate path at its first non-kept node.

    Args:
        retrieve_indices: 2D int structure (list of rows, or anything supporting
            two levels of iteration), rows = candidate paths, values = node indices
            into `draft_tokens` with `-1` right-padding.  See the module docstring
            for the full index contract.  NOT mutated -- a new list-of-lists is
            returned (important: MEDUSA's `retrieve_indices` is a persistent buffer).
        kept_nodes: the drafted node indices that survive the gate (the policy's
            output).  Node `0` (root) is always kept implicitly; callers need only
            list surviving draft nodes (>=1).

    Returns:
        A new list-of-lists, same shape, where in each row the first position whose
        node is not kept -- and every position after it -- is set to `-1`.

    Raises:
        ValueError: if the keep set is not ancestor-closed, i.e. some path has a
            kept node appearing *after* a pruned node (a surviving node whose
            ancestor was dropped).  This is a policy bug -- per-path truncation
            cannot represent such a set -- and is surfaced loudly rather than
            silently emitting a malformed sub-tree.
    """
    keep = set(kept_nodes)
    out: List[List[int]] = []
    for row in retrieve_indices:
        new_row: List[int] = []
        alive = True
        for v in row:
            v = int(v)
            if v == -1:
                # existing padding: nothing meaningful follows on this path
                new_row.append(-1)
                alive = False
            elif v == 0 or v in keep:
                if not alive:
                    # a surviving node after a pruned/padded one => not ancestor-closed
                    raise ValueError(
                        f"non-ancestor-closed keep set: node {v} survives after an "
                        f"earlier pruned node in path {list(row)}"
                    )
                new_row.append(v)          # root, or a surviving draft node
            else:
                new_row.append(-1)          # first pruned node -> cut here and after
                alive = False
        out.append(new_row)
    return out
