"""
CAPIM driver — re-costs an EAGLE-2 trace into per-step latency/energy.

Per decode step:
  1. DRAFT   - EAGLE head grown D depths; the σ_th gate terminates a branch before
               drafting its descendants, so the drafted width at each depth is the
               number of nodes whose parent survived the gate.  Draft FC/attn on the
               draft_device (PIM default), nonlinear on NPU -> a PIM<->NPU ping-pong
               when draft_device=PIM.  Always sequential (autoregressive chain).
  2. GATE    - mu = surviving tree size (sequencer.prune_tree on cumulative_log_prob).
  3. ROUTE   - sequencer.route(mu, mu_th) -> the verify execution plan.
  4. VERIFY  - one target forward over mu tokens, composed per the plan.
  5. ACCEPT  - committed accepted prefix that survives the gate, + 1 bonus token.

sigma_th supports two modes.  On a trace already gated by the GPU collector, run
with sigma_th = -inf so the gate is a no-op (the deliverable path).  On a full-tree
trace, pass a finite sigma_th to apply the gate at re-cost time — e.g. to sweep
sigma_th over existing traces, or for the full-tree kernel cross-check.

The driver owns all routing policy (its routers + the per-step exec choice from the
sequencer); common/ only costs the tagged layers.
"""

from __future__ import annotations

from dataclasses import dataclass

from common.config import ModelConfig
from common.devices.npu import MobileNPU
from common.devices.pim import LPDDR5PIM
from common.model import build_decoder_layer, build_eagle_draft_step, build_lm_head
from common.schema import DecodeStep, Trace
from common.system import (
    Composed,
    DriverResult,
    StepRecord,
    compose_sequential,
    cost_forward_pass,
    prefill_means,
    tag,
)
from common.type import Device as Dev
from common.type import ExecModel, LayerType
from capim_ctrl.sequencer import prune_tree, route


# --- routers (CAPIM's device-placement policy) ---------------------------------

def router_capim_verify(fc_device: Dev):
    """Verify: FC -> routed device; attention MATMUL -> PIM; nonlinear -> NPU."""
    def _r(layer):
        if layer.type == LayerType.FC:
            return fc_device
        if layer.type == LayerType.MATMUL:
            return Dev.PIM
        return Dev.NPU
    return _r


def router_eagle_draft(layer):
    """Draft on PIM: weights/attention pinned to PIM, nonlinear -> NPU."""
    if layer.type in (LayerType.FC, LayerType.MATMUL):
        return Dev.PIM
    return Dev.NPU


def router_all_npu(layer):
    return Dev.NPU


@dataclass
class CapimConfig:
    sigma_th: float = float("-inf")    # cumulative-log-prob gate (-inf on a gated trace)
    mu_th: int = 4                     # binary route threshold / speed<->energy mode dial
    all_npu: bool = False              # True -> EAGLE-2/NPU ablation
    concurrent_verify: bool = True     # mu>=mu_th -> FC column-split NPU||PIM (attn PIM-pinned)
    draft_device: Dev = Dev.PIM        # where draft FC/attn run (NL always NPU)
    name: str = "CAPIM"


def _generated_by_depth(step: DecodeStep, sigma_th: float):
    """depth -> count of nodes CAPIM actually *generates* under the gate.

    A node is drafted (and so costs) iff its parent was expanded == its parent
    survived the gate (depth-0 nodes hang off the always-expanded root).  By
    monotonicity of cumulative_log_prob this single check is ancestor-closed.  The
    set includes Boundary nodes that fail the gate themselves but were drafted once
    to be scored before their branch is killed — the cost the proactive gate cannot
    avoid.
    """
    if sigma_th == float("-inf"):
        out = {}
        for n in step.nodes:
            out[n.depth] = out.get(n.depth, 0) + 1
        return out
    survived = [n.cumulative_log_prob >= sigma_th for n in step.nodes]
    out = {}
    for n in step.nodes:
        if n.depth == 0:
            generated = True
        else:
            p = n.parent_idx
            generated = 0 <= p < len(step.nodes) and survived[p]
        if generated:
            out[n.depth] = out.get(n.depth, 0) + 1
    return out


def _draft_cost(model, step, sigma_th, draft_on_npu, npu, pim) -> Composed:
    gen = _generated_by_depth(step, sigma_th)
    total = Composed()
    ctx0 = step.context_length
    for depth in sorted(gen):
        width = gen[depth]
        if width <= 0:
            continue
        layers = build_eagle_draft_step(model, width=width, ctx=ctx0 + depth)
        tag(layers, router_all_npu if draft_on_npu else router_eagle_draft)
        total.merge(compose_sequential(layers, npu, pim))
    return total


def _effective_accept(step: DecodeStep, sigma_th: float) -> int:
    """Accepted tokens that also survive the gate (truncated to accepted_length)."""
    surviving_accepted = sum(
        1 for n in step.nodes
        if n.accepted and (sigma_th == float("-inf") or n.cumulative_log_prob >= sigma_th)
    )
    return min(step.accepted_length, surviving_accepted)


def drive(model: ModelConfig, trace: Trace, config: CapimConfig = None,
             npu: MobileNPU = None, pim: LPDDR5PIM = None,
             draft_cache: dict = None) -> DriverResult:
    """draft_cache: optional {step_index -> Composed} reused across calls.  Draft
    cost depends only on (sigma_th, draft_device), NOT mu_th, so a sweep that fixes
    (sigma, draft_device) and varies mu_th can pass ONE cache to skip recomputing
    draft.  The caller must Use a fresh cache per distinct (sigma_th, draft_device).
    """
    config = config or CapimConfig()
    npu = npu or MobileNPU()
    pim = pim or LPDDR5PIM()
    result = DriverResult(driver=config.name, model=model.name)
    result.prefill_time_s, result.prefill_energy_j = prefill_means(model, trace, npu, pim)

    draft_on_npu = config.all_npu or config.draft_device == Dev.NPU

    for i, step in enumerate(trace.steps):
        # 1-2. draft (gated) + gate -> mu
        if draft_cache is not None and i in draft_cache:
            draft = draft_cache[i]
        else:
            draft = _draft_cost(model, step, config.sigma_th, draft_on_npu, npu, pim)
            if draft_cache is not None:
                draft_cache[i] = draft
        mu = max(1, len(prune_tree(step, config.sigma_th)))

        # 3. route -> exec plan
        if config.all_npu:
            router = router_all_npu
            fc_dev = Dev.NPU
            exec_model = ExecModel.SEQUENTIAL
            split_attention = False
        else:
            plan = route(mu, config.mu_th, config.concurrent_verify)
            fc_dev = plan.fc_device
            exec_model = plan.exec_model
            split_attention = plan.split_attention
            router = router_capim_verify(fc_dev)

        # 4. verify: build + tag (driver policy), then cost via common (mechanism)
        block = tag(build_decoder_layer(model, m=mu, ctx=step.context_length), router)
        head = build_lm_head(model, m=mu)
        head.device = fc_dev
        verify = cost_forward_pass(block, head, model.n_layers, npu, pim,
                                   exec_model, split_attention=split_attention)

        # 5. accept
        tokens = _effective_accept(step, config.sigma_th) + 1

        result.steps.append(_combine(step, draft, verify, tokens))
    return result


def _combine(step: DecodeStep, draft: Composed, verify: Composed, tokens: int) -> StepRecord:
    energy = [draft.energy_j[i] + verify.energy_j[i] for i in range(4)]
    tdev = {k: draft.time_by_device.get(k, 0.0) + verify.time_by_device.get(k, 0.0)
            for k in ("NPU", "PIM")}
    ttype = {}
    for d in (draft.time_by_type, verify.time_by_type):
        for k, v in d.items():
            ttype[k] = ttype.get(k, 0.0) + v
    return StepRecord(
        prompt_id=step.prompt_id,
        dataset=step.dataset,
        step_id=step.step_id,
        tokens_emitted=tokens,
        time_s=draft.time_s + verify.time_s,
        energy_j=sum(energy),
        time_by_device=tdev,
        energy_by_component={"off_mem": energy[0], "on_chip": energy[1],
                             "alu": energy[2], "comm": energy[3]},
        time_by_type=ttype,
        crossings=draft.crossings + verify.crossings,
        split_t_npu_bound=draft.split_t_npu_bound + verify.split_t_npu_bound,
        split_t_pim_bound=draft.split_t_pim_bound + verify.split_t_pim_bound,
    )
