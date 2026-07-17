"""
LP-Spec baseline driver — MEDUSA + retrospective DTP (trace-replay) + concurrent
NPU||PIM verification.

Per decode step, replayed from a MEDUSA trace:
  1. DRAFT  : K=5 MEDUSA heads, one parallel shot off a single hidden state (no
              attention, no KV), DAU column-split NPU||PIM like every other GEMM.
  2. SELECT : the DTP picks which nodes to verify from a retrospective per-(head, k)
              acceptance histogram (`dtp`).  Content-blind counterpart to CAPIM's
              live σ_th gate — same greedy ∏ p construction, but the accuracies come
              from PAST verification history, not this step.
  3. VERIFY : ONE target forward over the kept tree (m = |kept|), composed
              CONCURRENTLY — every GEMM column-split NPU||PIM at the DAU ratio, the
              (1-r) output slice gathered over the bus, nonlinear additive on the NPU.
              Attention is column-split too (split_attention=True) — the one place
              LP-Spec differs from CAPIM's large-tree verify.
  4. ACCEPT : the measured accepted path truncated to the kept tree, + 1 bonus.

`L_spec` (LP-Spec's verified tree size) is the swept knob `config.L_spec`; report
LP-Spec as a band over L_spec.

Histogram causality: at step t the selection uses history from steps < t only;
step t's observations are folded in AFTER costing it.  Step 0 is a cold start that
verifies the full static tree.
"""

from __future__ import annotations

from dataclasses import dataclass

from common.config import ModelConfig
from common.devices.npu import MobileNPU
from common.devices.pim import LPDDR5PIM
from common.model import build_decoder_layer, build_lm_head, build_medusa_draft
from common.schema import DecodeStep, Trace
from common.system import (
    compose_concurrent,
    DriverResult,
    StepRecord,
    compose_sequential,
    cost_forward_pass,
    prefill_means,
    tag,
)
from common.type import Device as Dev
from common.type import ExecModel
from baselines.lp_spec import dtp


def router_all_npu(layer):
    return Dev.NPU


@dataclass
class LPSpecConfig:
    L_spec: int = 16                           # verified tree size (swept)
    selection: str = "greedy_headk"            # see dtp.select_kept
    medusa_num_heads: int = 5
    name: str = "LP-Spec"


def _granularity(selection: str) -> str:
    return "node" if selection == "greedy_node" else "headk"


def drive(model: ModelConfig, trace: Trace, config: LPSpecConfig = None,
             npu: MobileNPU = None, pim: LPDDR5PIM = None) -> DriverResult:
    config = config or LPSpecConfig()
    npu = npu or MobileNPU()
    pim = pim or LPDDR5PIM()
    result = DriverResult(driver=config.name, model=model.name)
    result.prefill_time_s, result.prefill_energy_j = prefill_means(model, trace, npu, pim)

    hist = dtp.DTPHist(granularity=_granularity(config.selection))

    for t, step in enumerate(trace.steps):
        kp = dtp.k_pred_map(step)
        pp = dtp.parent_pos_map(step)

        # 1. MEDUSA draft: K heads fired off one hidden state (no inter-head dependency,
        #    no attention, no KV), composed CONCURRENTLY -- the DAU column-splits their
        #    FC weights across NPU||PIM exactly as it does the backbone's.
        #
        #    CORRECTED 2026-07-12 (was pinned all-NPU + composed sequentially).  The heads
        #    are ordinary FC layers over model parameters -- neither prefill nor nonlinear
        #    -- and LP-Spec puts only those two on the NPU:
        #      §VI-A "Prefill stage of LLM inference and nonlinear functions are executed
        #             on the NPU."
        #      §V-C  "During NPU computation, model parameters are fetched from DRAM ranks,
        #             while PIM computation utilizes parameters stored in PIM ranks."
        #    The old all-NPU pin streamed all 740 MB of head weight over the external bus
        #    every iteration -- 57% of iteration latency and 32% of its energy -- which
        #    handicapped the baseline, and did so asymmetrically, since CAPIM's EAGLE draft
        #    is PIM-resident.  With this fix the driver reproduces LP-Spec's published
        #    throughput to 1.02x (74.8 vs 73.4 token/s at L=8, Alpaca).
        #    See scripts/cpu/validate_cost_model.py.
        #
        #    No split_attention argument: the draft emits no MATMUL, so the flag is a
        #    no-op.  No tag() either -- compose_concurrent reads no layer.device.
        heads = build_medusa_draft(model, medusa_num_heads=config.medusa_num_heads)
        draft = compose_concurrent(heads, npu, pim)

        # 2. DTP select (causal: history < t only; step 0 = full-tree cold start)
        kept = dtp.select_kept(step, t, config.L_spec, config.selection, hist, kp, pp)
        m = max(1, len(kept))

        # 3. concurrent verify over the kept tree.  Tags are unused by the makespan
        #    composer, but cost_forward_pass takes tagged layers -> tag harmlessly.
        block = tag(build_decoder_layer(model, m=m, ctx=step.context_length), router_all_npu)
        head = build_lm_head(model, m=m)
        head.device = Dev.NPU
        verify = cost_forward_pass(block, head, model.n_layers, npu, pim,
                                   ExecModel.CONCURRENT, split_attention=True)

        # 4. accept: measured accepted path truncated to the kept tree, + bonus
        tokens = dtp.effective_accept(step, kept) + 1

        result.steps.append(_combine(step, draft, verify, tokens))

        # 5. fold step t into the histogram (AFTER costing -> strict causality)
        hist.update(step, kp, pp)

    return result


def _combine(step: DecodeStep, draft, verify, tokens: int) -> StepRecord:
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
