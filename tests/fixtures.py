"""
GPU-free synthetic Trace fixtures for unit-testing the cost plane and drivers.

These are NOT part of the schema contract (common/schema.py) — they are test
data generators only.  `make_synthetic_trace` mimics an EAGLE-style dynamic tree;
`make_synthetic_medusa_trace` mimics a MEDUSA static tree with a single accepted
path and stationary per-(head, k) acceptance, so the LP-Spec DTP histogram can
converge.
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional

from common.schema import DecodeStep, TokenNode, Trace


def make_synthetic_trace(
    n_steps: int = 100,
    tree_size: int = 20,
    acceptance_rate: float = 0.4,
    max_depth: int = 4,
    dataset: str = "synthetic",
    seed: int = 42,
) -> Trace:
    """Synthetic EAGLE-style trace (dynamic tree, confidence-bearing nodes)."""
    rng = random.Random(seed)
    steps: List[DecodeStep] = []
    nodes_per_depth = max(1, tree_size // max_depth)

    for i in range(n_steps):
        nodes: List[TokenNode] = []
        global_idx = 0
        prev_layer_global: List[int] = []
        for d in range(max_depth):
            n_nodes = nodes_per_depth if d < max_depth - 1 else tree_size - global_idx
            n_nodes = max(1, n_nodes)
            cur_layer_global: List[int] = []
            for j in range(n_nodes):
                lp = -abs(rng.gauss(2.0, 1.5))
                cum_lp = lp * (d + 1) + rng.gauss(0, 0.2)
                parent = rng.choice(prev_layer_global) if d > 0 else -1
                nodes.append(
                    TokenNode(
                        depth=d,
                        token_id=rng.randint(0, 31999),
                        log_prob=lp,
                        cumulative_log_prob=cum_lp,
                        parent_idx=parent,
                        accepted=rng.random() < acceptance_rate,
                        layer_idx=j,
                    )
                )
                cur_layer_global.append(global_idx)
                global_idx += 1
            prev_layer_global = cur_layer_global

        steps.append(
            DecodeStep(
                step_id=i,
                context_length=128 + i,
                nodes=nodes,
                accepted_length=max(1, int(rng.gauss(2.5, 1.0))),
                dataset=dataset,
                prompt_id=0,
            )
        )

    td = Trace(
        steps=steps,
        model="Vicuna-7B-v1.3",
        sd_method="eagle2",
        metadata={"synthetic": True, "seed": seed, "draft_head": "EAGLE-Vicuna-7B-v1.3"},
    )
    td.compute_summary()
    return td


def balanced_medusa_choices(branching: int = 2, max_depth: int = 3) -> List[List[int]]:
    """MEDUSA-format paths for a balanced tree (every node has `branching` children
    with k = 0..branching-1, down to `max_depth` levels)."""
    choices: List[List[int]] = []

    def grow(prefix: List[int], depth: int) -> None:
        if depth >= max_depth:
            return
        for k in range(branching):
            path = prefix + [k]
            choices.append(path)
            grow(path, depth + 1)

    grow([], 0)
    return choices


def make_synthetic_medusa_trace(
    n_steps: int = 100,
    tree_choices: Optional[List[List[int]]] = None,
    branching: int = 2,
    max_depth: int = 3,
    base_accept_prob: float = 0.7,
    depth_decay: float = 0.85,
    rank_decay: float = 0.55,
    dataset: str = "synthetic",
    seed: int = 42,
) -> Trace:
    """Synthetic MEDUSA trace (static tree, single accepted path) for the LP-Spec
    DTP driver.  Same-parent siblings are stored contiguously in ascending-k order
    so the DTP can derive k; acceptance is one connected chain from the root."""
    rng = random.Random(seed)
    if tree_choices is None:
        tree_choices = balanced_medusa_choices(branching, max_depth)

    paths = sorted([list(p) for p in tree_choices], key=lambda p: (len(p), p))
    max_d = max(len(p) for p in paths) - 1

    layer_paths: List[List[List[int]]] = [[] for _ in range(max_d + 1)]
    for p in paths:
        layer_paths[len(p) - 1].append(p)

    path_to_layeridx: Dict[tuple, int] = {}
    pos_to_global: Dict[tuple, int] = {}
    g = 0
    for d in range(max_d + 1):
        for li, p in enumerate(layer_paths[d]):
            path_to_layeridx[tuple(p)] = li
            pos_to_global[(d, li)] = g
            g += 1
    template = []
    for d in range(max_d + 1):
        for li, p in enumerate(layer_paths[d]):
            if d == 0:
                parent_global = -1
            else:
                parent_global = pos_to_global[(d - 1, path_to_layeridx[tuple(p[:-1])])]
            template.append((d, li, parent_global, p[-1]))

    children: Dict[tuple, List[tuple]] = {}
    for gidx, (d, li, parent_global, k) in enumerate(template):
        children.setdefault((d, parent_global), []).append((gidx, k, li))

    steps: List[DecodeStep] = []
    for i in range(n_steps):
        accepted_ids = set()
        parent_key = -1
        for d in range(max_d + 1):
            sibs = children.get((d, parent_key))
            if not sibs:
                break
            if rng.random() > base_accept_prob * (depth_decay ** d):
                break
            weights = [rank_decay ** k for (_, k, _) in sibs]
            r = rng.random() * sum(weights)
            chosen, acc = sibs[-1], 0.0
            for s, w in zip(sibs, weights):
                acc += w
                if r <= acc:
                    chosen = s
                    break
            gidx, _, _ = chosen
            accepted_ids.add(gidx)
            parent_key = gidx

        nodes: List[TokenNode] = []
        cum_by_gidx: Dict[int, float] = {}
        for gidx, (d, li, parent_global, k) in enumerate(template):
            lp = min(-0.3 - 0.7 * k + rng.gauss(0, 0.1), -1e-4)
            cum = lp if d == 0 else cum_by_gidx[parent_global] + lp
            cum_by_gidx[gidx] = cum
            nodes.append(
                TokenNode(
                    depth=d,
                    token_id=rng.randint(0, 31999),
                    log_prob=lp,
                    cumulative_log_prob=cum,
                    parent_idx=parent_global,
                    accepted=(gidx in accepted_ids),
                    layer_idx=li,
                )
            )

        steps.append(
            DecodeStep(
                step_id=i,
                context_length=128 + i,
                nodes=nodes,
                accepted_length=len(accepted_ids),
                dataset=dataset,
                prompt_id=0,
            )
        )

    td = Trace(
        steps=steps,
        model="Vicuna-7B-v1.3",
        sd_method="medusa",
        metadata={
            "synthetic": True, "static_tree": True, "tree_size": len(template),
            "seed": seed, "draft_head": "medusa-vicuna-7b-v1.3",
        },
    )
    td.compute_summary()
    return td
