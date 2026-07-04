"""
GPU-free unit tests for the path-invalidation mechanism (common/gating.py).

Hand-built `retrieve_indices` trees, no torch and no model -- this is the load-bearing
correctness check that a pruned node caps a candidate path at exactly that node (the
property the greedy verifier's cumprod relies on).

Run from the repo root:
  python3 -m pytest tests/test_gating.py
  python3 tests/test_gating.py
"""

from __future__ import annotations

from common.gating import invalidate_paths


# A small tree used across several tests.  Node index 0 = root (sample token).
# Two candidate paths sharing node 1:
#   path A: 0 -> 1 -> 2 -> 4
#   path B: 0 -> 1 -> 3            (shorter, right-padded with -1)
TREE = [
    [0, 1, 2, 4],
    [0, 1, 3, -1],
]


def test_keep_all_is_identity():
    kept = {1, 2, 3, 4}
    assert invalidate_paths(TREE, kept) == [
        [0, 1, 2, 4],
        [0, 1, 3, -1],
    ]


def test_prune_leaf_only_truncates_that_position():
    # Drop node 4 (a leaf on path A). Path B untouched.
    kept = {1, 2, 3}
    assert invalidate_paths(TREE, kept) == [
        [0, 1, 2, -1],
        [0, 1, 3, -1],
    ]


def test_prune_interior_truncates_node_and_all_descendants():
    # Drop node 2 (interior on path A): position AND everything after it -> -1.
    kept = {1, 3, 4}   # 4 is nominally kept but unreachable once 2 is gone...
    # ...which makes this keep set non-ancestor-closed -> must raise (see dedicated test).
    # Here we use a closed set instead: drop 2 and its descendant 4 together.
    kept = {1, 3}
    assert invalidate_paths(TREE, kept) == [
        [0, 1, -1, -1],
        [0, 1, 3, -1],
    ]


def test_prune_depth0_node_collapses_path_to_root():
    # Drop node 1 (the shared depth-0 node): both paths collapse to just the root.
    kept = {2, 3, 4}   # nominally kept but all orphaned by dropping 1
    # closed version: keep nothing below the root
    kept = set()
    assert invalidate_paths(TREE, kept) == [
        [0, -1, -1, -1],
        [0, -1, -1, -1],
    ]


def test_root_always_survives_even_if_not_in_kept_nodes():
    # kept_nodes lists only draft nodes (>=1); node 0 must survive implicitly.
    kept = {1, 2, 3, 4}
    assert 0 not in kept
    out = invalidate_paths(TREE, kept)
    assert all(row[0] == 0 for row in out)


def test_existing_padding_is_preserved():
    kept = {1, 3}
    out = invalidate_paths(TREE, kept)
    assert out[1] == [0, 1, 3, -1]   # the trailing -1 stays -1


def test_input_is_not_mutated():
    original = [list(r) for r in TREE]
    _ = invalidate_paths(TREE, {1, 2})
    assert TREE == original          # helper must return a fresh structure


def test_idempotent():
    kept = {1, 2}
    once = invalidate_paths(TREE, kept)
    twice = invalidate_paths(once, kept)
    assert once == twice


def test_non_ancestor_closed_keep_set_raises():
    # Keep node 4 (deep) but drop its ancestor 2 -> per-path truncation cannot
    # represent this; must be surfaced loudly, not silently mangled.
    kept = {1, 3, 4}   # 2 dropped, 4 kept -> orphan
    try:
        invalidate_paths(TREE, kept)
    except ValueError:
        return
    raise AssertionError("expected ValueError for non-ancestor-closed keep set")


def test_realistic_sigma_gate_prefix():
    # A wider EAGLE-like tree; a sigma gate keeps an ancestor-closed frontier.
    #   0 -> 1 -> 2 -> 5
    #   0 -> 1 -> 3
    #   0 -> 4            (a second depth-0 branch)
    tree = [
        [0, 1, 2, 5],
        [0, 1, 3, -1],
        [0, 4, -1, -1],
    ]
    # sigma keeps {1,2,4} (ancestor-closed), drops 3 (leaf) and 5 (leaf).
    kept = {1, 2, 4}
    assert invalidate_paths(tree, kept) == [
        [0, 1, 2, -1],
        [0, 1, -1, -1],
        [0, 4, -1, -1],
    ]


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\n{len(fns)} gating tests passed.")


if __name__ == "__main__":
    _run_all()
