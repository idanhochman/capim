"""
GPU-free dry-run of the EAGLE collector's pure recording core (record_gated_step /
mark_accepted).  No torch, model or GPU -- feeds hand-built list trees, exactly the
``.tolist()``-ed shapes the wrappers pass, so the recording logic is validated before
any GPU time.

Fixture tree (draft-index space, 0 = root):

        root(0)
        /     \\
    n1(1)     n2(2)          depth 0
      |         |
    n3(3)     n4(4)          depth 1

cumulative log-probs:  n1=-0.5  n2=-3.0  n3=-1.0  n4=-5.0
"""

from __future__ import annotations

from capim_ctrl.collector import mark_accepted, record_gated_step

NEG_INF = float("-inf")

# draft_tokens[0]: [root, n1, n2, n3, n4]  (token ids arbitrary but distinct)
DRAFT = [100, 11, 12, 13, 14]
# tree_mask[i][j] = j is ancestor of i (incl. self and root col 0)
TREE_MASK = [
    [True,  False, False, False, False],   # root
    [True,  True,  False, False, False],   # n1: {0,1}
    [True,  False, True,  False, False],   # n2: {0,2}
    [True,  True,  False, True,  False],   # n3: {0,1,3}
    [True,  False, True,  False, True],    # n4: {0,2,4}
]
TREE_POS = [0, 1, 1, 2, 2]                  # root=0; n1,n2 depth0->1; n3,n4 depth1->2
CUM = [-0.5, -3.0, -1.0, -5.0]             # n1, n2, n3, n4
# retrieve_indices: one row per root->leaf path, -1 padded
RETRIEVE = [[0, 1, 3], [0, 2, 4]]


def test_ungated_records_full_tree():
    nodes, edited, remap = record_gated_step(
        DRAFT, RETRIEVE, TREE_MASK, TREE_POS, CUM, NEG_INF,
    )
    assert edited == RETRIEVE                      # no invalidation when sigma=-inf
    assert len(nodes) == 4
    assert remap == {1: 0, 2: 1, 3: 2, 4: 3}
    # depths, tokens, parents (global positions; -1 for root's children)
    assert [n.depth for n in nodes] == [0, 0, 1, 1]
    assert [n.token_id for n in nodes] == [11, 12, 13, 14]
    assert [n.parent_idx for n in nodes] == [-1, -1, 0, 1]
    # log_prob = cum - parent_cum ; n3 = -1.0 - (-0.5) = -0.5 ; n4 = -5.0 - (-3.0) = -2.0
    assert nodes[2].log_prob == -0.5
    assert nodes[3].log_prob == -2.0
    assert all(not n.accepted for n in nodes)


def test_gated_prunes_and_reindexes():
    # sigma = -2.0 : keep n1(-0.5), n3(-1.0) ; prune n2(-3.0) -> n4(-5.0) unreachable
    nodes, edited, remap = record_gated_step(
        DRAFT, RETRIEVE, TREE_MASK, TREE_POS, CUM, -2.0,
    )
    assert remap == {1: 0, 3: 1}
    assert [n.token_id for n in nodes] == [11, 13]
    assert [n.depth for n in nodes] == [0, 1]
    # n3's parent n1 is kept -> global position 0
    assert [n.parent_idx for n in nodes] == [-1, 0]
    # path B truncated at the first pruned node (n2), path A intact
    assert edited == [[0, 1, 3], [0, -1, -1]]


def test_gated_accept_marking():
    nodes, edited, remap = record_gated_step(
        DRAFT, RETRIEVE, TREE_MASK, TREE_POS, CUM, -2.0,
    )
    # winning path A = edited[0] = [0,1,3]; both draft nodes accepted (accept_length=2)
    accepted = mark_accepted(nodes, edited, remap, best_candidate=0, accept_length=2)
    assert accepted == 2
    assert [n.accepted for n in nodes] == [True, True]


def test_accept_stops_at_pruned_node():
    # If the target's argmax lands on the pruned path B, the gated path is [0,-1,-1]:
    # only the root column is real, so no draft node is (wrongly) marked accepted.
    nodes, edited, remap = record_gated_step(
        DRAFT, RETRIEVE, TREE_MASK, TREE_POS, CUM, -2.0,
    )
    accepted = mark_accepted(nodes, edited, remap, best_candidate=1, accept_length=0)
    assert accepted == 0
    assert [n.accepted for n in nodes] == [False, False]


def test_gated_keep_all_when_sigma_below_min():
    # sigma below every score -> keeps everything but STILL goes through invalidate_paths
    nodes, edited, remap = record_gated_step(
        DRAFT, RETRIEVE, TREE_MASK, TREE_POS, CUM, -100.0,
    )
    assert len(nodes) == 4
    assert edited == RETRIEVE                       # nothing truncated
    assert [n.parent_idx for n in nodes] == [-1, -1, 0, 1]
