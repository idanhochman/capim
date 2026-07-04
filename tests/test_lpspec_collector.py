"""
GPU-free dry-run of the LP-Spec / MEDUSA collector's pure core: static-structure
extraction, the DTP keep-set -> node-index mapping, kept-sub-tree recording (with
re-indexed parents), and accept marking.  No torch / model / GPU.

Same fixture tree as the EAGLE collector test (draft-index space, 0 = root):

        root(0)
        /     \\
    n1(1)     n2(2)          depth 0  (layer_idx 0, 1)
      |         |
    n3(3)     n4(4)          depth 1  (layer_idx 0, 1)

Derived from the MEDUSA-style buffers (position_ids + attn_mask), exactly the
``.tolist()``-ed shapes the wrapper reads from ``model.medusa_buffers``.
"""

from __future__ import annotations

from baselines.lp_spec import dtp
from baselines.lp_spec.collector import (
    build_static_structure,
    mark_accepted,
    pos_to_index,
    record_kept,
    structural_step,
)

# medusa_position_ids: root=0; n1,n2 depth0->1; n3,n4 depth1->2
POS_IDS = [0, 1, 1, 2, 2]
# medusa_attn_mask[0,0]: row i True at its ancestors (incl. self and root col 0)
MASK = [
    [True,  False, False, False, False],   # root
    [True,  True,  False, False, False],   # n1: {0,1}
    [True,  False, True,  False, False],   # n2: {0,2}
    [True,  True,  False, True,  False],   # n3: {0,1,3}
    [True,  False, True,  False, True],    # n4: {0,2,4}
]
PRISTINE = [[0, 1, 3], [0, 2, 4]]           # retrieve_indices paths (root->leaf), -1 padded


def test_static_structure():
    meta = build_static_structure(POS_IDS, MASK)
    # v -> (depth, layer_idx, parent_node_index)
    assert meta == {
        1: (0, 0, 0),   # n1: depth0, layer0, parent root
        2: (0, 1, 0),   # n2: depth0, layer1, parent root
        3: (1, 0, 1),   # n3: depth1, layer0, parent n1
        4: (1, 1, 2),   # n4: depth1, layer1, parent n2
    }
    assert pos_to_index(meta) == {(0, 0): 1, (0, 1): 2, (1, 0): 3, (1, 1): 4}


def test_structural_step_for_dtp():
    step = build_static_structure(POS_IDS, MASK)
    ss = structural_step(step)
    assert [n.depth for n in ss.nodes] == [0, 0, 1, 1]
    assert [n.layer_idx for n in ss.nodes] == [0, 1, 0, 1]
    assert [n.parent_idx for n in ss.nodes] == [-1, -1, 0, 1]   # list positions (root -> -1)
    # DTP derivations work on it (k_pred = sibling rank; parents resolve)
    kp = dtp.k_pred_map(ss)
    assert kp[(0, 0)] == 0 and kp[(0, 1)] == 1                  # depth-0 siblings ranked 0,1
    pp = dtp.parent_pos_map(ss)
    assert pp[(1, 0)] == (0, 0) and pp[(1, 1)] == (0, 1)        # n3->n1, n4->n2


def test_record_kept_reindexes_parents():
    meta = build_static_structure(POS_IDS, MASK)
    # keep n1, n3 (a connected branch); n2, n4 pruned
    nodes, remap = record_kept(meta, {1, 3})
    assert remap == {1: 0, 3: 1}
    assert [n.depth for n in nodes] == [0, 1]
    assert [n.layer_idx for n in nodes] == [0, 0]              # STATIC layer_idx preserved
    assert [n.parent_idx for n in nodes] == [-1, 0]           # n3's parent n1 -> new pos 0


def test_dtp_select_maps_to_node_indices():
    meta = build_static_structure(POS_IDS, MASK)
    ss = structural_step(meta)
    p2i = pos_to_index(meta)
    hist = dtp.DTPHist()
    # t == 0 -> cold start returns the FULL static tree
    kept_pos = dtp.select_kept(ss, 0, 4, "greedy_headk", hist)
    kept_idx = {p2i[p] for p in kept_pos}
    assert kept_idx == {1, 2, 3, 4}


def test_gated_accept_marking():
    from common.gating import invalidate_paths
    meta = build_static_structure(POS_IDS, MASK)
    kept_idx = {1, 3}
    edited = invalidate_paths(PRISTINE, kept_idx)
    assert edited == [[0, 1, 3], [0, -1, -1]]                 # path B truncated at pruned n2
    nodes, remap = record_kept(meta, kept_idx)
    # winning path A accepts both kept draft nodes
    assert mark_accepted(nodes, edited, remap, best_candidate=0, accept_length=2) == 2
    assert [n.accepted for n in nodes] == [True, True]
    # winning path B: gated to [0,-1,-1] -> no draft node wrongly accepted
    nodes2, remap2 = record_kept(meta, kept_idx)
    assert mark_accepted(nodes2, edited, remap2, best_candidate=1, accept_length=0) == 0
    assert [n.accepted for n in nodes2] == [False, False]


def test_causal_hist_update_uses_static_kp():
    # After a full-tree step where the whole depth-0 layer accepted, the (head=0,k)
    # stats populate; a later select at L=1 must keep the highest-scoring depth-0 node.
    meta = build_static_structure(POS_IDS, MASK)
    ss = structural_step(meta)
    kp, pp = dtp.k_pred_map(ss), dtp.parent_pos_map(ss)
    hist = dtp.DTPHist()
    # observe: n1 accepted, n2 not (both reachable at depth 0)
    obs_nodes, _ = record_kept(meta, {1, 2, 3, 4})
    obs_nodes[0].accepted = True                              # n1
    obs = structural_step(meta)
    obs.nodes[0].accepted = True
    hist.update(obs, kp, pp)
    # now p(head0,k0) = 1.0 (n1) > p(head0,k1) = 0.0 (n2); top-1 keeps n1
    kept_pos = dtp.select_kept(ss, 1, 1, "greedy_headk", hist, kp, pp)
    assert kept_pos == {(0, 0)}
