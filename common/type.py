"""
Shared enums for the cost plane.
"""

from __future__ import annotations

from enum import Enum


class LayerType(Enum):
    FC = 0       # weight-stationary GEMM/GEMV: y = W x   (qkv, proj, ffn, lm_head, fusion, resblock)
    MATMUL = 1   # weightless batched matmul: attention score / context
    SOFTMAX = 2  # attention softmax or sampling softmax  (nonlinear)
    ACT = 3      # activation: SiLU / GELU / SwiGLU        (nonlinear)
    NORM = 4     # RMSNorm / LayerNorm                     (nonlinear)
    COMM = 5     # data movement: PIM<->NPU handoff over the external bus


# Layer types the kernel treats as "nonlinear glue" (always NPU in CAPIM/LP-Spec).
NONLINEAR = {LayerType.SOFTMAX, LayerType.ACT, LayerType.NORM}


class Device(Enum):
    NPU = 0
    PIM = 1


class ExecModel(Enum):
    """How a driver composes per-layer costs into a step latency."""
    SEQUENTIAL = 0
    CONCURRENT = 1


class Phase(Enum):
    """Which part of inference a layer list belongs to."""
    PREFILL = 0
    DECODE = 1
