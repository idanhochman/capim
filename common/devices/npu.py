"""
Mobile NPU cost model.

Roofline per layer: time = max(compute, mem).
  - compute: matmul on the INT8 matrix unit (FC/MATMUL); NL ops on the vector unit.
  - mem: off-chip traffic over the external bus (51.2 GB/s, shared with PIM).
Energy: off-chip traffic × OFFCHIP_ENERGY  +  MACs × NPU_INT8_OP energy.

Constants come from common.config (LP-Spec Table II + cited energy figures).
"""

from __future__ import annotations

from common.config import (
    MAX_COMPUTE_UTIL,
    MAX_MEM_UTIL,
    MEM_OFFCHIP_PJ_PER_BIT,
    NPU_INT8_TOPS,
    NPU_MAC_PJ_PER_OP,
    NPU_OFFCHIP_BW,
    NPU_VECTOR_TOPS,
    pj_to_j,
)
from common.devices.base import CostResult, Device, zero_energy
from common.model import Layer
from common.type import LayerType


class MobileNPU(Device):
    name = "NPU"

    def __init__(
        self,
        matrix_tops: float = NPU_INT8_TOPS,
        vector_tops: float = NPU_VECTOR_TOPS,
        offchip_bw: float = NPU_OFFCHIP_BW,
    ):
        self.matrix_tops = matrix_tops
        self.vector_tops = vector_tops
        self.offchip_bw = offchip_bw

    def _peak_compute(self, layer: Layer) -> float:
        if layer.type in (LayerType.FC, LayerType.MATMUL):
            return self.matrix_tops * MAX_COMPUTE_UTIL
        return self.vector_tops * MAX_COMPUTE_UTIL

    def cost(self, layer: Layer) -> CostResult:
        # PIM<->NPU crossings are costed exactly once, on the PIM side.
        assert layer.type != LayerType.COMM, (
            f"NPU.cost got a COMM layer ({layer.name}); crossings are costed on "
            f"the PIM side only, never on the NPU"
        )

        flops = layer.get_flops()
        in1, in2, out = layer.get_size()
        off_bytes = in1 + in2 + out

        compute_t = flops / self._peak_compute(layer)
        mem_t = off_bytes / (self.offchip_bw * MAX_MEM_UTIL)

        if compute_t >= mem_t:
            time_s, bound = compute_t, "compute"
        else:
            time_s, bound = mem_t, "memory"

        e = zero_energy()
        e[0] = pj_to_j(off_bytes * 8 * MEM_OFFCHIP_PJ_PER_BIT)            # off_mem
        e[2] = pj_to_j((flops / 2.0) * NPU_MAC_PJ_PER_OP)                 # alu (1 MAC = 2 flops)
        layer.bound = bound
        layer.time_s = time_s
        layer.energy = e
        return CostResult(time_s, e, bound)
