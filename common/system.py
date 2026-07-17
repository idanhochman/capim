"""
Composition: turn a device-tagged layer list into a step latency + energy.

A driver emits typed layers (tagged with a device by its router) and composes
their per-op costs with one of two execution models.  Both composers share the
identical Device.cost(), so the comparison stays fair.

  compose_sequential — additive, no PIM/NPU concurrency, with a PIM<->NPU COMM
    crossing at each device switch.  Used for:
      * AR (all-NPU);
      * every CAPIM/LP-Spec DRAFT step (autoregressive dependency chain → no
        FC||attn overlap, so always sequential);
      * CAPIM's verify when μ < μ_th (the all-PIM low-power route).

  compose_concurrent — makespan: every GEMM column-split NPU||PIM at the DAU ratio
    r, PIM's (1-r) output slice gathered over the bus, nonlinear glue additive on
    the NPU.  `split_attention` distinguishes the two users:
      * True  → LP-Spec: attention MATMULs are column-split too (PIM's KV slice is
        gathered each kernel).
      * False → the Attn-PIM ablation: FC is split NPU||PIM but attention stays
        PIM-pinned (its KV slice never crosses the bus), overlapping the FC makespan
        on the PIM side.  NOT CAPIM's default — attention follows μ_th like every
        other GEMM (CapimConfig.split_attention=None); this flag is an override.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Tuple

from common.config import ModelConfig
from common.devices.base import Device, zero_energy
from common.devices.npu import MobileNPU
from common.devices.pim import LPDDR5PIM
from common.model import Layer, build_prefill
from common.schema import Trace
from common.type import Device as Dev
from common.type import ExecModel, LayerType


def _add(dst: List[float], src: List[float]) -> None:
    for i in range(len(dst)):
        dst[i] += src[i]


@dataclass
class Composed:
    """Result of costing a layer list."""
    time_s: float = 0.0
    energy_j: List[float] = field(default_factory=zero_energy)
    time_by_device: Dict[str, float] = field(default_factory=lambda: {"NPU": 0.0, "PIM": 0.0})
    time_by_type: Dict[str, float] = field(default_factory=dict)
    crossings: int = 0
    # Concurrent-only diagnostics: the all-NPU and all-PIM time of the column-split
    # FC/MATMUL kernels (the makespan they actually cost lands in time_by_type).
    # Zero for sequential composition.  Lets the breakdown show where a concurrent
    # makespan sits in the [all-PIM, all-NPU] band (the PIM cost-model check).
    split_t_npu_bound: float = 0.0
    split_t_pim_bound: float = 0.0

    def scale(self, factor: float) -> "Composed":
        return Composed(
            time_s=self.time_s * factor,
            energy_j=[e * factor for e in self.energy_j],
            time_by_device={k: v * factor for k, v in self.time_by_device.items()},
            time_by_type={k: v * factor for k, v in self.time_by_type.items()},
            crossings=int(self.crossings * factor),
            split_t_npu_bound=self.split_t_npu_bound * factor,
            split_t_pim_bound=self.split_t_pim_bound * factor,
        )

    def merge(self, other: "Composed") -> None:
        self.time_s += other.time_s
        _add(self.energy_j, other.energy_j)
        for k, v in other.time_by_device.items():
            self.time_by_device[k] = self.time_by_device.get(k, 0.0) + v
        for k, v in other.time_by_type.items():
            self.time_by_type[k] = self.time_by_type.get(k, 0.0) + v
        self.crossings += other.crossings
        self.split_t_npu_bound += other.split_t_npu_bound
        self.split_t_pim_bound += other.split_t_pim_bound


def _dev_obj(dev: Dev, npu: MobileNPU, pim: LPDDR5PIM) -> Device:
    return pim if dev == Dev.PIM else npu


def compose_sequential(layers: List[Layer], npu: MobileNPU, pim: LPDDR5PIM,
                       count_crossings: bool = True) -> Composed:
    """Additive composition with a COMM crossing at each device switch.

    No PIM/NPU concurrency: latency is the sum of per-layer costs plus a PIM<->NPU
    crossing wherever consecutive layers land on different devices.  Used for AR,
    all CAPIM/LP-Spec draft steps, and CAPIM's small-tree (μ<μ_th) all-PIM verify.
    """
    out = Composed()
    prev: Dev = None
    for layer in layers:
        dev = layer.device
        if count_crossings and prev is not None and dev != prev:
            # data must hop the external bus: cost the boundary activation as COMM.
            # in1 is already in bytes (get_size folds in dbyte), so encode it as a
            # flat 1×in1 block with dbyte=1 to avoid double-counting dbyte.
            in1, _, _ = layer.get_size()
            crossing = Layer("xfer", LayerType.COMM, m=1, n=max(1, in1), dbyte=1)
            r = pim.cost(crossing)
            out.time_s += r.time_s
            _add(out.energy_j, r.energy_j)
            out.time_by_device["PIM"] = out.time_by_device.get("PIM", 0.0) + r.time_s
            out.time_by_type["COMM"] = out.time_by_type.get("COMM", 0.0) + r.time_s
            out.crossings += 1
        r = _dev_obj(dev, npu, pim).cost(layer)
        out.time_s += r.time_s
        _add(out.energy_j, r.energy_j)
        dname = "PIM" if dev == Dev.PIM else "NPU"
        out.time_by_device[dname] = out.time_by_device.get(dname, 0.0) + r.time_s
        out.time_by_type[layer.type.name] = out.time_by_type.get(layer.type.name, 0.0) + r.time_s
        prev = dev
    return out


def compose_concurrent(layers: List[Layer], npu: MobileNPU, pim: LPDDR5PIM,
                       split_attention: bool = True) -> Composed:
    """Makespan via the DAU's balanced column-wise GEMM split.

    Used by LP-Spec (split_attention=True) and by CAPIM's large-tree (μ≥μ_th)
    verify (split_attention=False).  Neither routes whole kernels: each GEMM is
    partitioned column-wise across NPU and PIM in a ratio the DAU picks so the two
    devices finish together.

    `split_attention`:
      - True  (LP-Spec): attention MATMULs are column-split like every other GEMM,
        so PIM's KV slice is gathered over the external bus each kernel.
      - False (Attn-PIM ablation, NOT CAPIM's default): attention stays PIM-pinned and
        overlaps the split FC kernels — its KV slice never crosses the bus.  Added to
        t_compute (it shares the verify critical path) but charged NO gather/crossing.

    Balance (derived from LP-Spec's "synchronize NPU and PIM" prose, §V-B; the paper
    prints per-share times T_NPU, T_PIM and T_total=min(·), not the closed form):
        r = t_p / (t_n + t_p)            # NPU/DRAM column share; (1-r) -> PIM
        t_kernel = r*t_n = t_n*t_p/(t_n+t_p)   # parallel-combination makespan
    Energy is blended E = r*E_n + (1-r)*E_p (NPU share reads weights off-chip = dear;
    PIM share reads in-bank = cheap — the asymmetry CAPIM exploits).  Nonlinear ops
    are never split -> additive on the NPU.  A column split leaves each device with a
    slice of the output, so PIM's (1-r) slice is gathered over the bus (one COMM) per
    split kernel before the next op reads the full activation.

        t_total = Σ t_kernel + Σ t_gather + Σ t_nl_npu
    """
    out = Composed()
    t_compute = 0.0
    t_comm = 0.0
    t_nl = 0.0
    for layer in layers:
        if layer.type == LayerType.MATMUL and not split_attention:
            # attention stays PIM-pinned (KV in-bank): overlaps the FC makespan on
            # the PIM side, so it is on the verify critical path (-> t_compute) but
            # pays no bus gather / crossing.
            c_p = pim.cost(layer)
            t_compute += c_p.time_s
            out.split_t_pim_bound += c_p.time_s
            _add(out.energy_j, c_p.energy_j)
            out.time_by_device["PIM"] = out.time_by_device.get("PIM", 0.0) + c_p.time_s
            out.time_by_type[layer.type.name] = out.time_by_type.get(layer.type.name, 0.0) + c_p.time_s
            continue
        if layer.type in (LayerType.FC, LayerType.MATMUL):
            c_n = npu.cost(layer)
            c_p = pim.cost(layer)
            t_n, t_p = c_n.time_s, c_p.time_s
            denom = t_n + t_p
            if denom <= 0:
                continue
            r = t_p / denom                         # NPU/DRAM column share
            t_kernel = t_n * t_p / denom            # = r*t_n = (1-r)*t_p
            t_compute += t_kernel
            out.split_t_npu_bound += t_n
            out.split_t_pim_bound += t_p
            for i in range(len(out.energy_j)):
                out.energy_j[i] += r * c_n.energy_j[i] + (1.0 - r) * c_p.energy_j[i]
            out.time_by_device["NPU"] = out.time_by_device.get("NPU", 0.0) + t_kernel
            out.time_by_device["PIM"] = out.time_by_device.get("PIM", 0.0) + t_kernel
            out.time_by_type[layer.type.name] = out.time_by_type.get(layer.type.name, 0.0) + t_kernel

            # DAU output gather: PIM's (1-r) output slice crosses to the NPU so the
            # full activation is reassembled for the next op.  Charged per split
            # kernel (FC and attention MATMUL), sized to that kernel's output.
            _, _, out_bytes = layer.get_size()
            gather_bytes = (1.0 - r) * out_bytes
            if gather_bytes > 0:
                hop = Layer("dau_gather", LayerType.COMM, m=1, n=gather_bytes, dbyte=1)
                g = pim.cost(hop)
                t_comm += g.time_s
                _add(out.energy_j, g.energy_j)
                out.time_by_device["PIM"] = out.time_by_device.get("PIM", 0.0) + g.time_s
                out.time_by_type["COMM"] = out.time_by_type.get("COMM", 0.0) + g.time_s
                out.crossings += 1
        else:  # SOFTMAX / ACT / NORM / COMM -> NPU, additive
            c = npu.cost(layer)
            t_nl += c.time_s
            _add(out.energy_j, c.energy_j)
            out.time_by_device["NPU"] = out.time_by_device.get("NPU", 0.0) + c.time_s
            out.time_by_type[layer.type.name] = out.time_by_type.get(layer.type.name, 0.0) + c.time_s
    out.time_s = t_compute + t_comm + t_nl
    return out


def cost_forward_pass(decoder_block: List[Layer], head: Layer, n_layers: int,
                      npu: MobileNPU, pim: LPDDR5PIM, exec_model: ExecModel,
                      split_attention: bool = False) -> Composed:
    """Cost one forward pass through the whole network: `n_layers` identical
    decoder blocks + one lm_head.

    The driver builds and tags `decoder_block` (one block; all n_layers blocks are
    shape- and placement-identical at batch=1) and `head` with its own router, then
    hands them here.  This function reads `layer.device` only — it holds no routing
    policy and makes no driver-specific choice.  The single block is costed once and
    scaled by `n_layers`.

    exec_model selects the composition mechanism:
      SEQUENTIAL : additive compose with PIM<->NPU crossings at device switches,
                   plus the (n_layers-1) inter-block boundary hops the single-block
                   scaling skips — added only when the block's last layer (norm2)
                   and first layer (qkv) sit on different devices.
      CONCURRENT : DAU makespan (every GEMM column-split NPU||PIM at the balance
                   ratio, nonlinear additive on the NPU).  `split_attention`:
                     True  -> attention MATMULs are column-split like every GEMM
                              (PIM's KV slice gathered over the bus each kernel);
                     False -> attention stays PIM-pinned (KV in-bank, never crosses
                              the bus) and overlaps the FC makespan.
    """
    if exec_model == ExecModel.CONCURRENT:
        total = compose_concurrent(decoder_block, npu, pim,
                                   split_attention=split_attention).scale(n_layers)
        total.merge(compose_concurrent([head], npu, pim, split_attention=split_attention))
        return total

    # SEQUENTIAL
    total = compose_sequential(decoder_block, npu, pim).scale(n_layers)

    # inter-block boundary: this block's norm2 -> next block's qkv.  If they land on
    # different devices, chaining n_layers blocks adds (n_layers-1) bus hops the
    # single-block compose never saw.  Sized to the residual-stream activation
    # (m x d_model), read straight off the boundary layer's shape — no model needed.
    if decoder_block and decoder_block[0].device != decoder_block[-1].device:
        last = decoder_block[-1]
        hop = Layer("interlayer", LayerType.COMM, m=last.m, n=last.n, dbyte=last.dbyte)
        r = pim.cost(hop)
        reps = n_layers - 1
        total.time_s += r.time_s * reps
        _add(total.energy_j, [e * reps for e in r.energy_j])
        total.time_by_device["PIM"] = total.time_by_device.get("PIM", 0.0) + r.time_s * reps
        total.time_by_type["COMM"] = total.time_by_type.get("COMM", 0.0) + r.time_s * reps
        total.crossings += reps

    # lm_head (once): a single-element list, so compose_sequential adds no crossing
    # before it (prev resets to None) — matching the old separate-compose behaviour.
    total.merge(compose_sequential([head], npu, pim))
    return total


def prefill_means(model: ModelConfig, trace: Trace,
                  npu: MobileNPU, pim: LPDDR5PIM) -> Tuple[float, float]:
    """Mean one-time prefill (time_s, energy_j) per prompt over a trace.

    Prefill is one full forward pass over the prompt on the NPU — a fixed
    placement, not a per-step routing choice: n_layers decoder blocks + one lm_head,
    costed with the shared `cost_forward_pass` primitive.  prompt_len per prompt =
    the smallest context_length over that prompt's steps (the pre-decode KV size);
    the per-prompt costs are averaged to match the reporter's per-prompt
    aggregation.  Reported only in end-to-end latency, never in the per-step
    token/s / token/J / EDP rates.
    """
    len_by_prompt: Dict[int, int] = {}
    for s in trace.steps:
        c = s.context_length
        len_by_prompt[s.prompt_id] = min(len_by_prompt.get(s.prompt_id, c), c)
    if not len_by_prompt:
        return 0.0, 0.0

    t = e = 0.0
    for L in len_by_prompt.values():
        block, head = build_prefill(model, prompt_len=L)
        tag(block, lambda layer: Dev.NPU)
        head.device = Dev.NPU
        r = cost_forward_pass(block, head, model.n_layers, npu, pim,
                              ExecModel.SEQUENTIAL)
        t += r.time_s
        e += sum(r.energy_j)
    n_prompts = len(len_by_prompt)
    return t / n_prompts, e / n_prompts


@dataclass
class StepRecord:
    prompt_id: int
    dataset: str
    step_id: int
    tokens_emitted: float
    time_s: float
    energy_j: float
    time_by_device: Dict[str, float] = field(default_factory=dict)
    energy_by_component: Dict[str, float] = field(default_factory=dict)
    time_by_type: Dict[str, float] = field(default_factory=dict)
    crossings: int = 0
    # concurrent-only: all-NPU / all-PIM bounds of the split kernels
    split_t_npu_bound: float = 0.0
    split_t_pim_bound: float = 0.0


@dataclass
class DriverResult:
    driver: str
    model: str
    steps: List[StepRecord] = field(default_factory=list)
    # One-time prefill cost (per generated sequence), kept separate From the decode
    # steps so the per-step rates (token/s, token/J, EDP) exclude it by construction.
    # Stored as a per-prompt mean (one prefill per prompt; the prompt-length variation
    # is folded in by the driver).  Used only by the end-to-end latency metric;
    # energy stays decode-only, so prefill_energy_j is carried for completeness but is
    # not Added to the token/J figure.
    prefill_time_s: float = 0.0
    prefill_energy_j: float = 0.0


# router type: (Layer) -> Device enum
Router = Callable[[Layer], Dev]


def tag(layers: List[Layer], router: Router) -> List[Layer]:
    for layer in layers:
        layer.device = router(layer)
    return layers
