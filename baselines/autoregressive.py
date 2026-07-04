"""
Autoregressive baseline (lower bound / normalization anchor).

No speculation: one full target forward per emitted token, on the NPU, at the
token's REAL context length.  Replayed from a trace so it generates the same number
of tokens at the same contexts as the SD runs (token-weighted per-token contexts).
"""

from __future__ import annotations

from common.config import ModelConfig
from common.devices.npu import MobileNPU
from common.devices.pim import LPDDR5PIM
from common.model import build_decoder_layer, build_lm_head
from common.schema import Trace
from common.system import (
    DriverResult,
    StepRecord,
    cost_forward_pass,
    prefill_means,
    tag,
)
from common.type import Device as Dev
from common.type import ExecModel


def router_all_npu(layer):
    return Dev.NPU


def drive(model: ModelConfig, trace: Trace,
             npu: MobileNPU = None, pim: LPDDR5PIM = None) -> DriverResult:
    npu = npu or MobileNPU()
    pim = pim or LPDDR5PIM()
    result = DriverResult(driver="Autoregressive", model=model.name)
    result.prefill_time_s, result.prefill_energy_j = prefill_means(model, trace, npu, pim)

    for step in trace.steps:
        # AR emits the same #tokens this step as SD committed (accepted + bonus),
        # one forward each, at growing context.
        n_tokens = step.accepted_length + 1
        time_s = 0.0
        energy = [0.0, 0.0, 0.0, 0.0]
        tdev = {"NPU": 0.0, "PIM": 0.0}
        ttype = {}
        crossings = 0
        for j in range(n_tokens):
            block = tag(build_decoder_layer(model, m=1, ctx=step.context_length + j),
                        router_all_npu)
            head = build_lm_head(model, m=1)
            head.device = Dev.NPU
            c = cost_forward_pass(block, head, model.n_layers, npu, pim,
                                  ExecModel.SEQUENTIAL)
            time_s += c.time_s
            for i in range(4):
                energy[i] += c.energy_j[i]
            for k, v in c.time_by_device.items():
                tdev[k] = tdev.get(k, 0.0) + v
            for k, v in c.time_by_type.items():
                ttype[k] = ttype.get(k, 0.0) + v
            crossings += c.crossings

        result.steps.append(StepRecord(
            prompt_id=step.prompt_id,
            dataset=step.dataset,
            step_id=step.step_id,
            tokens_emitted=n_tokens,
            time_s=time_s,
            energy_j=sum(energy),
            time_by_device=tdev,
            energy_by_component={"off_mem": energy[0], "on_chip": energy[1],
                                 "alu": energy[2], "comm": energy[3]},
            time_by_type=ttype,
            crossings=crossings,
        ))
    return result
