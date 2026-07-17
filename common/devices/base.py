"""
Device base class + the cost-result container.

Every device implements `cost(layer) -> CostResult` with an identical signature,
so the three drivers (AR / LP-Spec / CAPIM) share one per-op cost model and the
comparison is fair.  A device decides time via a per-layer roofline
`max(compute, mem)` and energy as a 4-component vector.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from common.model import Layer

# Energy-vector slot names (4 slots, mobile).  Structure follows PAPI's 6-vector
# [off_mem, L2, L1, reg, flop, comm] (src/devices.py:_get_energy), trimmed for a
# mobile NPU/PIM with no cache hierarchy:
#   off_mem : DRAM/memory-access energy — the dear off-chip bus on the NPU vs cheap
#             internal near-bank access on PIM (rates: config.MEM_{OFFCHIP,INTERNAL}).
#   on_chip : PAPI's L2+L1+reg tiling, collapsed to one slot; unused (=0) here
#             (no mobile cache hierarchy; second-order at batch=1 GEMV).
#   alu     : arithmetic-datapath energy = (flops/2) × config.{PIM,NPU}_MAC_PJ_PER_OP.
#   comm    : PIM<->NPU crossing over the external bus (charged on the PIM side).
E_OFF, E_ONCHIP, E_ALU, E_COMM = 0, 1, 2, 3
ENERGY_SLOTS = ("off_mem", "on_chip", "alu", "comm")


@dataclass
class CostResult:
    time_s: float
    energy_j: List[float]   # [off_mem, on_chip, alu, comm]
    bound: str              # "compute" | "memory" | "comm"

    @property
    def total_energy_j(self) -> float:
        return sum(self.energy_j)


def zero_energy() -> List[float]:
    return [0.0, 0.0, 0.0, 0.0]


class Device:
    """Abstract device.  Subclasses implement `cost`."""

    name: str = "device"

    def cost(self, layer: Layer) -> CostResult:  # pragma: no cover - interface
        raise NotImplementedError

    def cost_sequential(self, layers: List[Layer]):
        """Cost a list of layers, returning (time, energy_vec) summed."""
        t = 0.0
        e = zero_energy()
        for layer in layers:
            r = self.cost(layer)
            t += r.time_s
            for i in range(len(e)):
                e[i] += r.energy_j[i]
        return t, e
