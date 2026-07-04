"""
LPDDR5-PIM cost model (near-bank compute).

Analytical roofline (no external cycle-accurate PIM sim):
  FC / MATMUL:  time = max(ceil(m/N_ALU)*N_ALU * per_token_flops / PIM_INT8_GOPS,
                          traffic / PIM_INTERNAL_BW)
The N_ALU=4 ALUs are TOKEN-parallel (LP-Spec §V-B: T_PIM = N_params/BW ×
ceil(L_spec/N_ALU)), so an m-token batch is rounded up to a full ALU pass for the
COMPUTE term: m=1..4 all cost one pass.  This is the batch=1 / small-tree regime.
Only FC / MATMUL / COMM run on PIM; nonlinear ops route to the NPU (the cross-bank
reduction wall) and are asserted out in cost().

Per the LPDDR5-PIM ridge point (409.6 GOPS / 51.2 TB/s = 0.008 ops/byte), GEMV
(intensity 2 ops/byte >> ridge) is COMPUTE-bound on PIM, so this normally reduces
to flops/GOPS — but the full max() is kept so the bound tag is computed, not
assumed.  PAPI's "GEMM = reuse(=m) × GEMV" is implicit: flops already carries m.

COMM (the PIM<->NPU handoff) is costed here on the external bus (51.2 GB/s):
  time = FIXED_CROSSING_LATENCY_S + bytes / PIM_EXTERNAL_BW
The fixed per-crossing latency is the DRAM mode-switch cost (PIM-compute mode <->
host-readable mode); the NL data itself is tiny, so the crossing is dominated by
this fixed setup, not bandwidth.

Energy: internal-bank traffic × PIM_ENERGY + MACs × INT8_OP energy; external-bus
traffic (COMM) is charged at the off-chip energy rate.
"""

from __future__ import annotations

from math import ceil

from common.config import (
    MAX_COMPUTE_UTIL,
    MAX_MEM_UTIL,
    MEM_INTERNAL_PJ_PER_BIT,
    MEM_OFFCHIP_PJ_PER_BIT,
    PIM_EXTERNAL_BW,
    PIM_INT8_GOPS,
    PIM_INTERNAL_BW,
    PIM_MAC_PJ_PER_OP,
    PIM_NALU,
    pj_to_j,
)
from common.devices.base import CostResult, Device, zero_energy
from common.model import Layer
from common.type import LayerType

# Fixed latency charged once per PIM<->NPU crossing, on top of the bandwidth term
# (the activation bytes themselves).  Physically a DRAM MODE SWITCH: the PIM bank
# leaves all-bank PIM-compute mode and returns to host-readable mode for the NPU to
# read the activation (and back for the next PIM kernel).  Per Samsung's commercial
# DRAM-PIM paper (Lee et al., ISCA 2021), a mode transition is a short sequence of
# standard DRAM commands (ACT/PRE to a reserved PIM_CONF space) — not a privileged
# MRS — so its cost is governed by DRAM row-cycle timing, not a kernel launch.
#
# Magnitude from LP-Spec Table II DRAM timing (our LPDDR5-PIM baseline): one
# ACT->PRE->ACT cycle = tRC = tRAS + tRP ≈ 61 ns; a handoff is ~1–3 such cycles
# => ~60–180 ns.  Point estimate 100 ns.
FIXED_CROSSING_LATENCY_S: float = 100.0e-9   # 100 ns per crossing (DRAM mode switch)


class LPDDR5PIM(Device):
    name = "PIM"

    def __init__(
        self,
        int8_gops: float = PIM_INT8_GOPS,
        internal_bw: float = PIM_INTERNAL_BW,
        external_bw: float = PIM_EXTERNAL_BW,
        crossing_latency_s: float = FIXED_CROSSING_LATENCY_S,
        n_alu: int = PIM_NALU,
    ):
        self.int8_gops = int8_gops
        self.internal_bw = internal_bw
        self.external_bw = external_bw
        self.crossing_latency_s = crossing_latency_s
        self.n_alu = n_alu

    def cost(self, layer: Layer) -> CostResult:
        if layer.type == LayerType.COMM:
            return self._comm_cost(layer)

        # NL ops never run on PIM.
        assert layer.type in (LayerType.FC, LayerType.MATMUL), (
            f"PIM.cost got {layer.type.name}; NL must route to NPU, not PIM"
        )

        flops = layer.get_flops()
        in1, in2, out = layer.get_size()
        # Internal traffic: the stationary operand dominates (FC weights / KV-cache);
        # count all operands touched in-bank.
        traffic = in1 + in2 + out

        # N_ALU token-batching (LP-Spec §V-B): a weight pass serves up to n_alu draft
        # tokens, so verifying m tokens takes ceil(m/n_alu) passes.  Round the batch
        # up to a full pass for the COMPUTE term only -> m=1..n_alu all cost one pass.
        # Energy stays on the true `flops` below (idle lanes do no MACs); flops is
        # linear in m, so scaling by m_eff/m pads it exactly.
        m = layer.m
        if m > 0:
            m_eff = ceil(m / self.n_alu) * self.n_alu
            compute_flops = flops * (m_eff / m)
        else:
            compute_flops = flops

        compute_t = compute_flops / (self.int8_gops * MAX_COMPUTE_UTIL)
        mem_t = traffic / (self.internal_bw * MAX_MEM_UTIL)

        if compute_t >= mem_t:
            time_s, bound = compute_t, "compute"
        else:
            time_s, bound = mem_t, "memory"

        e = zero_energy()
        e[0] = pj_to_j(traffic * 8 * MEM_INTERNAL_PJ_PER_BIT)            # internal bank access
        e[2] = pj_to_j((flops / 2.0) * PIM_MAC_PJ_PER_OP)                # near-bank MAC
        layer.bound = bound
        layer.time_s = time_s
        layer.energy = e
        return CostResult(time_s, e, bound)

    def _comm_cost(self, layer: Layer) -> CostResult:
        in1, _, _ = layer.get_size()
        bytes_moved = in1
        time_s = self.crossing_latency_s + (
            bytes_moved / (self.external_bw * MAX_MEM_UTIL) if bytes_moved > 0 else 0.0
        )
        e = zero_energy()
        e[3] = pj_to_j(bytes_moved * 8 * MEM_OFFCHIP_PJ_PER_BIT)         # external-bus energy
        layer.bound = "comm"
        layer.time_s = time_s
        layer.energy = e
        return CostResult(time_s, e, "comm")
