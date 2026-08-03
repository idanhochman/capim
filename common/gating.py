"""
Path invalidation — the shared trajectory-side gating mechanism.

Both collectors turn a per-step keep decision into a genuinely shorter accepted prefix
the same way: they overwrite pruned nodes in the model's `retrieve_indices` with the
`-1` padding the verifier already understands.  This module is that one operation,
isolated from the policy that produces the keep set:

  - CAPIM  (`capim_ctrl/collector.py`): keep = {nodes with cumulative_log_prob >= sigma}.
  - LP-Spec (`baselines/lp_spec/collector.py`): keep = DTP greedy top-L from the
    retrospective histogram.

Only the keep set differs, so `invalidate_paths` takes it as data (a set of node
indices) and never sees a threshold -- σ and L stay in the collectors.

Why this makes the trajectory causal (greedy / temperature = 0)
--------------------------------------------------------------
EAGLE and MEDUSA both verify with, in effect,
    posterior_mask   = (candidates[:, 1:] == argmax(logits[:, :-1]))
    accept_length    = cumprod(posterior_mask, dim=1).sum(dim=1).max()
`retrieve_indices` selects the candidate tokens per path, and setting a position to `-1`
is how both stacks already pad paths shorter than the max tree depth, so an injected
`-1` is indistinguishable from native padding.  Both gather the candidate row from a
token vector with a pad slot appended, so a negative index selects that pad slot rather
than a real drafted token:
  - EAGLE's `eagenerate` appends `padding = -1` to `draft_tokens`, so `-1` yields
    candidate token id -1.  A target argmax is always a valid vocab id >= 0, so it can
    never match and the truncation is exact for greedy.
  - MEDUSA's `generate_candidates` appends a 0 pad slot, so `-1` yields token id 0
    (`<unk>`).  The target effectively never argmaxes `<unk>`, so truncation is exact in
    practice; if it ever matched, the accepted token would be what greedy emits anyway.
Either way `cumprod` zeroes from that position on, capping `accept_length` at the first
pruned node, and `update_inference_inputs` advances the real KV-cache by that shorter
prefix.  The next step is drafted from the genuinely shortened context, so the recorded
future is the real one rather than a counterfactual stitched on afterwards.

Correctness precondition: the keep set must be ancestor-closed (if a node survives, so
do all its ancestors).  Truncating each path independently at its first pruned node then
reconstructs exactly the kept sub-tree, with no cross-path bookkeeping.  Both policies
guarantee this -- CAPIM by the monotonicity of cumulative log-prob with depth, LP-Spec
by the connected greedy top-L construction -- and `invalidate_paths` raises if a caller
passes a malformed keep set.

Index contract: `retrieve_indices` is a 2D int structure, one row per candidate
root->leaf path, one column per depth position, `-1`-padded on the right.  A value is a
node index into the step's `draft_tokens` (EAGLE) / candidate tensor (MEDUSA):

  - value  0  == the root (the already-accepted sample token).  Always survives; it is
                column 0 of every path and is never a gateable draft node.
  - value -1  == existing padding.  Passed through (idempotent).
  - value >0  == a drafted node; survives iff it is in `kept_nodes`.

Pure python (no torch/numpy) by design, so it is unit-testable without a GPU or the
model; the collectors do the small `.tolist()` <-> `torch.tensor()` bridging on their
side, once per step.
"""

from __future__ import annotations

from typing import Iterable, List, Sequence


def invalidate_paths(
    retrieve_indices: Sequence[Sequence[int]],
    kept_nodes: Iterable[int],
) -> List[List[int]]:
    """Truncate every candidate path at its first non-kept node.

    Args:
        retrieve_indices: 2D int structure (list of rows, or anything supporting two
            levels of iteration), rows = candidate paths, values = node indices into
            `draft_tokens` with `-1` right-padding.  Not mutated -- a new
            list-of-lists is returned, since MEDUSA's `retrieve_indices` is a
            persistent buffer.
        kept_nodes: the drafted node indices that survive the gate.  Node `0` (root)
            is always kept implicitly; callers need only list surviving draft nodes.

    Returns:
        A new list-of-lists, same shape, where in each row the first position whose
        node is not kept -- and every position after it -- is set to `-1`.

    Raises:
        ValueError: if the keep set is not ancestor-closed, i.e. some path has a kept
            node appearing after a pruned one.  Per-path truncation cannot represent
            such a set, so it is surfaced rather than silently emitting a malformed
            sub-tree.
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
