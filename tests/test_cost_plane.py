"""
GPU-free smoke tests for the shared cost plane (common/).

Run from the repo root:
  python3 -m pytest tests/test_cost_plane.py        # if pytest is available
  python3 tests/test_cost_plane.py                  # plain-python fallback
"""

from __future__ import annotations

from common.config import VICUNA_7B
from common.devices.npu import MobileNPU
from common.devices.pim import LPDDR5PIM
from common.model import (
    build_eagle_draft_step,
    build_medusa_draft,
    build_prefill,
    build_verify_pass,
)
from common.schema import Trace
from common.system import (
    compose_concurrent,
    compose_sequential,
    cost_forward_pass,
    tag,
)
from common.type import Device as Dev
from common.type import ExecModel, LayerType, NONLINEAR
from tests.fixtures import make_synthetic_medusa_trace, make_synthetic_trace


def _route_capim(layer):
    """Toy router: FC/MATMUL -> PIM, nonlinear glue -> NPU."""
    return Dev.NPU if layer.type in NONLINEAR else Dev.PIM


def test_layer_cost_positive():
    npu, pim = MobileNPU(), LPDDR5PIM()
    layers = build_verify_pass(VICUNA_7B, m=8, ctx=256)
    for layer in layers:
        if layer.type in NONLINEAR:
            r = npu.cost(layer)
        else:
            r = pim.cost(layer)
        assert r.time_s > 0, f"{layer.name} time not positive"
        assert r.total_energy_j > 0, f"{layer.name} energy not positive"
        assert r.bound in ("compute", "memory", "comm")


def test_compose_sequential_vs_concurrent():
    npu, pim = MobileNPU(), LPDDR5PIM()
    layers = build_verify_pass(VICUNA_7B, m=16, ctx=512)
    tag(layers, _route_capim)
    seq = compose_sequential(layers, npu, pim)
    con = compose_concurrent(layers, npu, pim, split_attention=False)
    assert seq.time_s > 0 and con.time_s > 0
    # concurrent makespan should never exceed the all-additive sequential time
    assert con.time_s <= seq.time_s + 1e-12
    # sequential must have inserted PIM<->NPU crossings (FC<->NL device switches)
    assert seq.crossings > 0


def test_pim_nalu_token_batching():
    """m=1..4 cost one ALU pass; the FC compute time is flat across that band."""
    pim = LPDDR5PIM()
    t1 = pim.cost(build_verify_pass(VICUNA_7B, m=1, ctx=128)[0]).time_s
    t4 = pim.cost(build_verify_pass(VICUNA_7B, m=4, ctx=128)[0]).time_s
    t5 = pim.cost(build_verify_pass(VICUNA_7B, m=5, ctx=128)[0]).time_s
    assert abs(t1 - t4) < 1e-12, "m=1..4 should share one ALU pass"
    assert t5 > t4, "m=5 should need a second ALU pass"


def test_prefill_first_class():
    npu = MobileNPU()
    # prefill is a full forward pass: (block, head) scaled by n_layers, all-NPU.
    block, head = build_prefill(VICUNA_7B, prompt_len=128)
    tag(block, lambda l: Dev.NPU)        # prefill runs on the NPU
    head.device = Dev.NPU
    out = cost_forward_pass(block, head, VICUNA_7B.n_layers, npu, LPDDR5PIM(),
                            ExecModel.SEQUENTIAL)
    assert out.time_s > 0
    assert out.time_by_device["NPU"] > 0
    assert out.time_by_device.get("PIM", 0.0) == 0.0


def test_builders_emit_expected_types():
    eagle = build_eagle_draft_step(VICUNA_7B, width=4, ctx=200)
    assert any(l.name == "fusion_fc" and l.type == LayerType.FC for l in eagle)
    assert any(l.type == LayerType.SOFTMAX for l in eagle)
    medusa = build_medusa_draft(VICUNA_7B, medusa_num_heads=5)
    assert sum(1 for l in medusa if l.name.endswith("_lmhead")) == 5


def test_synthetic_traces_roundtrip(tmp_path=None):
    import os
    import tempfile

    for trace in (make_synthetic_trace(n_steps=10),
                  make_synthetic_medusa_trace(n_steps=10)):
        assert isinstance(trace, Trace)
        assert trace.model == "Vicuna-7B-v1.3"
        assert trace.sd_method in ("eagle2", "medusa")
        assert len(trace.steps) == 10
        d = tmp_path or tempfile.mkdtemp()
        path = os.path.join(str(d), f"{trace.sd_method}.json")
        trace.save(path)
        reloaded = Trace.load(path)
        assert reloaded.sd_method == trace.sd_method
        assert len(reloaded.steps) == len(trace.steps)


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\n{len(fns)} cost-plane tests passed.")


if __name__ == "__main__":
    _run_all()
