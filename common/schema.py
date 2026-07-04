"""
Trace schema — the shared contract between the GPU collectors and the CPU drivers.

A `Trace` is a list of `DecodeStep`s recorded by running an instrumented SD method
(EAGLE-2 or MEDUSA) on the shared Vicuna-7B backbone over Alpaca / GSM8K.  The
σ-gate (CAPIM) / DTP (LP-Spec) fire INSIDE the drafting loop, so a recorded tree
is the *gated, causal* trajectory — not a full tree pruned later.  The drivers
only read `Trace` objects; they never touch the model.

Two identity fields:
  - model     : the backbone, shared across every trace (fairness invariant),
                e.g. "Vicuna-7B-v1.3".
  - sd_method : the speculative-decoding method being compared — the real axis,
                a canonical lowercase id the drivers/collectors dispatch on
                ("eagle2" | "medusa").  The exact draft-head checkpoint is
                provenance and lives in metadata["draft_head"].

Per-node fields:
  - log_prob: per-token log-softmax probability at the node's depth.
  - cumulative_log_prob: sum of log_probs root → node (matches EAGLE-2's cu_scores);
    the σ-characterization figures group acceptance by this.
  - accepted: whether the target accepted this exact token at this step.

This module is the contract ONLY — no model code, no synthetic generators
(those are GPU-free test fixtures and live under tests/).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class TokenNode:
    """One draft token in a single decode step's tree."""

    depth: int                   # 0 = direct child of the root / last accepted token
    token_id: int                # vocabulary index
    log_prob: float              # per-token log-softmax probability at this depth
    cumulative_log_prob: float   # sum of log_probs along the path root → this node
    parent_idx: int              # GLOBAL index of the parent within the step's `nodes`
                                 # list (nodes[parent_idx]); −1 for depth-0 nodes whose
                                 # parent is the root / true accepted token.
    accepted: bool               # True if the target accepted this exact token

    layer_idx: int = 0           # index of this node within its depth layer
    token_str: str = ""          # human-readable decoded token (e.g. " Paris")


@dataclass
class DecodeStep:
    """One speculative-decoding step: a draft tree + how much of it was accepted.

    A step = one draft-tree generation (topK_genrate / MEDUSA heads) followed by
    one verification (evaluate_posterior).
    """

    step_id: int                    # 0-based decode step index
    context_length: int             # tokens in the KV-cache at step start
    nodes: List[TokenNode]          # the draft tree (gated/causal)
    accepted_length: int            # tokens the target accepted (≥1: the bonus token)
    dataset: str                    # "alpaca" | "gsm8k"
    prompt_id: int                  # index of the source prompt within the dataset
    sample_token_id: int = 0        # vocabulary index of the root token (last accepted)
    sample_token_str: str = ""      # human-readable root token

    @property
    def tree_size(self) -> int:
        return len(self.nodes)

    def nodes_at_depth(self, depth: int) -> List[TokenNode]:
        return [n for n in self.nodes if n.depth == depth]

    @property
    def max_depth(self) -> int:
        return max((n.depth for n in self.nodes), default=-1)

    def depth_widths(self) -> List[int]:
        """Number of nodes at each depth 0..max_depth (the tree's per-level widths)."""
        if not self.nodes:
            return []
        widths = [0] * (self.max_depth + 1)
        for n in self.nodes:
            widths[n.depth] += 1
        return widths


@dataclass
class Trace:
    """A full decoded run: the list of steps plus collection metadata.

    Saved as JSON so traces can be re-costed across simulation runs without
    re-running the GPU draft model.
    """

    steps: List[DecodeStep]
    model: str                  # backbone, e.g. "Vicuna-7B-v1.3" (shared across traces)
    sd_method: str              # "eagle2" | "medusa" (the compared axis; drivers dispatch on it)
    metadata: Dict              # free-form: draft_head, sigma_th, temperature, dataset split, ...

    # Summary stats (populated after collection).  mean_tree_size = mean gated μ:
    # varies with sigma_th (EAGLE); fixed by the static L_spec (MEDUSA).
    mean_tree_size: float = 0.0
    mean_accepted_length: float = 0.0
    mean_acceptance_rate: float = 0.0   # accepted / tree_size per step, averaged

    # Genuine per-prompt decoded output (causal gating -> real continuation, not a
    # counterfactual).  One dict per prompt: {"prompt_id", "prompt", "output"}.
    prompt_outputs: List[Dict] = field(default_factory=list)

    def compute_summary(self) -> None:
        if not self.steps:
            return
        n = len(self.steps)
        self.mean_tree_size = sum(s.tree_size for s in self.steps) / n
        self.mean_accepted_length = sum(s.accepted_length for s in self.steps) / n
        rates = [
            sum(1 for nd in s.nodes if nd.accepted) / s.tree_size
            for s in self.steps if s.tree_size > 0
        ]
        self.mean_acceptance_rate = sum(rates) / len(rates) if rates else 0.0

    # -- (de)serialization -------------------------------------------------
    def save(self, path: str) -> None:
        data = {
            "model": self.model,
            "sd_method": self.sd_method,
            "metadata": self.metadata,
            "mean_tree_size": self.mean_tree_size,
            "mean_accepted_length": self.mean_accepted_length,
            "mean_acceptance_rate": self.mean_acceptance_rate,
            "prompt_outputs": self.prompt_outputs,
            "steps": [
                {
                    "step_id": s.step_id,
                    "context_length": s.context_length,
                    "accepted_length": s.accepted_length,
                    "dataset": s.dataset,
                    "prompt_id": s.prompt_id,
                    "sample_token_id": s.sample_token_id,
                    "sample_token_str": s.sample_token_str,
                    "nodes": [
                        {
                            "depth": n.depth,
                            "token_id": n.token_id,
                            "token_str": n.token_str,
                            "log_prob": n.log_prob,
                            "cumulative_log_prob": n.cumulative_log_prob,
                            "parent_idx": n.parent_idx,
                            "accepted": n.accepted,
                            "layer_idx": n.layer_idx,
                        }
                        for n in s.nodes
                    ],
                }
                for s in self.steps
            ],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @staticmethod
    def load(path: str) -> "Trace":
        with open(path, "r") as f:
            data = json.load(f)
        steps = []
        for sd in data["steps"]:
            nodes = [
                TokenNode(
                    depth=nd["depth"],
                    token_id=nd["token_id"],
                    token_str=nd.get("token_str", ""),
                    log_prob=nd["log_prob"],
                    cumulative_log_prob=nd["cumulative_log_prob"],
                    parent_idx=nd["parent_idx"],
                    accepted=nd["accepted"],
                    layer_idx=nd.get("layer_idx", 0),
                )
                for nd in sd["nodes"]
            ]
            steps.append(
                DecodeStep(
                    step_id=sd["step_id"],
                    context_length=sd["context_length"],
                    nodes=nodes,
                    accepted_length=sd["accepted_length"],
                    dataset=sd["dataset"],
                    prompt_id=sd["prompt_id"],
                    sample_token_id=sd.get("sample_token_id", 0),
                    sample_token_str=sd.get("sample_token_str", ""),
                )
            )
        return Trace(
            steps=steps,
            model=data["model"],
            sd_method=data["sd_method"],
            metadata=data["metadata"],
            mean_tree_size=data.get("mean_tree_size", 0.0),
            mean_accepted_length=data.get("mean_accepted_length", 0.0),
            mean_acceptance_rate=data.get("mean_acceptance_rate", 0.0),
            prompt_outputs=data.get("prompt_outputs", []),
        )
