"""
CAPIM controller policy — pure algorithm, no hardware/cost dependencies.

Two responsibilities, both data-dependent control the host runs at kernel
boundaries (the confidence-driven scheduling that is CAPIM's contribution):

  1. prune_tree(step, sigma_th) — the live σ_th confidence gate.  A draft node is
     kept iff its cumulative_log_prob (the log joint probability of its root->node
     path) is >= sigma_th.  cumulative_log_prob is strictly monotone-decreasing
     with depth, so a failing node guarantees all its descendants fail too — one
     independent pass suffices, no parent propagation.  sigma_th = -inf disables the
     gate: a trace already gated inside the GPU drafting loop is replayed as-is, while
     a finite sigma_th applies the gate at re-cost time (e.g. sweeping sigma_th over a
     full-tree trace).

  2. route(mu, mu_th) — the verify execution plan, binary on the gated tree size mu
     (PAPI-style at RLP=1):
       mu <  mu_th -> all-PIM, SEQUENTIAL/additive  (small tree, energy-cheap)
       mu >= mu_th -> CONCURRENT NPU||PIM column-split makespan, attention
                      PIM-pinned (split_attention=False) — larger tree worth the
                      parallel verify, KV stays in-bank.
     mu_th is a speed<->energy mode dial; the driver passes it in.

This module returns a Route plan; the driver turns it into tagged layers + a
cost_forward_pass call.  It never touches Device/cost — that keeps policy and
mechanism cleanly separated.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from common.schema import DecodeStep, TokenNode
from common.type import Device, ExecModel


# --- 1. confidence gate --------------------------------------------------------

def prune_tree(step: DecodeStep, sigma_th: float) -> List[TokenNode]:
    """Surviving nodes after the σ_th gate (cumulative_log_prob >= sigma_th).

    sigma_th = -inf disables the gate (returns the full tree) — the path taken when
    replaying an already-gated trace.
    """
    if not step.nodes:
        return []
    if sigma_th == float("-inf"):
        return list(step.nodes)
    return [n for n in step.nodes if n.cumulative_log_prob >= sigma_th]


# --- 2. verify route -----------------------------------------------------------

@dataclass(frozen=True)
class Route:
    """The verify execution plan the sequencer hands the driver."""
    exec_model: ExecModel       # SEQUENTIAL (all-PIM) or CONCURRENT (makespan)
    fc_device: Device           # device tag for FC layers (used only when SEQUENTIAL)
    split_attention: bool       # CONCURRENT only: column-split attention too?


def route(mu: int, mu_th: int, concurrent_verify: bool = True) -> Route:
    """Map (mu, mu_th) -> the verify execution plan.

    mu <  mu_th : all-PIM, sequential/additive (FC + attention on PIM, NL on NPU).
    mu >= mu_th : concurrent NPU||PIM FC column-split makespan, attention PIM-pinned
                  (split_attention=False).  `concurrent_verify=False` falls back to
                  the legacy all-NPU-FC sequential big-tree route (kept for ablation).
    """
    if mu < mu_th:
        return Route(ExecModel.SEQUENTIAL, Device.PIM, False)
    if concurrent_verify:
        return Route(ExecModel.CONCURRENT, Device.NPU, False)
    return Route(ExecModel.SEQUENTIAL, Device.NPU, False)
