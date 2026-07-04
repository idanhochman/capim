"""
Hardware + model configuration for the CAPIM simulator.

Two halves:
  1. HARDWARE — LPDDR5-PIM + mobile-NPU constants (LP-Spec Table II) and the
     four-component energy model (2 movement pJ/bit by interface + 2 compute
     pJ/MAC by device/process).
  2. MODELS — Vicuna-7B-v1.3 (the only 7B target with BOTH an official EAGLE head
     and an official MEDUSA head → apples-to-apples CAPIM vs LP-Spec) and its
     EAGLE draft head.

================================ ENERGY MODEL ================================
The four constants feed the [off_mem, on_chip, alu, comm] energy vector:
    off_mem <- MEM_INTERNAL (PIM)   or MEM_OFFCHIP (NPU)
    alu     <- PIM_MAC_PJ_PER_OP    or NPU_MAC_PJ_PER_OP
    comm    <- MEM_OFFCHIP (the PIM<->NPU crossing rides the same external bus)
    on_chip <- 0 (no mobile cache hierarchy modelled)

  MEM_OFFCHIP — off-chip transfer energy (5.47 pJ/bit):
    HBM2 memory-access energy from TPU-v4i (Jouppi et al.), quoted in SpecPIM
    (ASPLOS 2024) §7.1.  Used because LP-Spec draws its energy "from prior works
    [24],[26],[29]" (§VI) and [29] IS SpecPIM — so this keeps us on the same
    energy basis as the baseline.  It is an HBM2 stand-in (LP-Spec publishes no
    absolute LPDDR5 off-chip figure); real mobile LPDDR5 is 10–20 pJ/bit (PIM-AI,
    arXiv:2411.17309, Table 1) — carry those as a sensitivity sweep.

  MEM_INTERNAL — internal near-bank DRAM access energy (0.8 pJ/bit):
    LP-Spec §II-A: "data transfers within DRAM consume only 15% of the energy
    required for off-DRAM transfers [23]" (Samsung Hot Chips 35, 2023):
    0.15 × 5.47 ≈ 0.8.  Corroborated by PIM-AI Table 1's 0.95 pJ/bit.

  INT8 MAC energy (0.23 pJ/op for BOTH NPU and PIM):
    Horowitz (ISSCC 2014) @ 45 nm: 8-bit INT mult 0.2 + add 0.03 = 0.23 pJ/MAC.
    Kept as two constants (same nominal value) so the PIM side can be swept: the
    NPU (4 nm logic) value is a conservative upper bound; the PIM ALU (20 nm DRAM
    process) is directionally costlier but its absolute is unpublished, so the
    upside is a SWEEP, not a baked-in penalty.  Both are 2nd-order at batch=1 GEMV
    (movement dominates).
"""

from __future__ import annotations

from dataclasses import dataclass

# ===========================================================================
# HARDWARE — LPDDR5-PIM (4-die configuration, Samsung).  Source: LP-Spec Table II
# ===========================================================================

# Compute throughput: 4 dies × 102.4 GOPS each = 409.6 GOPS INT8
PIM_INT8_GOPS: float = 409.6e9          # ops/s INT8

# Internal bank bandwidth (4-die): 51.2 TB/s
PIM_INTERNAL_BW: float = 51.2e12        # bytes/s

# External I/O bandwidth (off-chip, shared with NPU): 51.2 GB/s
PIM_EXTERNAL_BW: float = 51.2e9         # bytes/s

PIM_FREQ_HZ: float = 200e6              # Hz
PIM_CAPACITY_BYTES: float = 16e9        # 3 PIM ranks + 1 DRAM rank × 4 GB

# Four 32-wide ALUs per MPU (LP-Spec Table II).  TOKEN-parallel: one weight pass
# serves up to N_ALU draft tokens, so verifying m tokens takes ceil(m/N_ALU)
# passes (LP-Spec §V-B: T_PIM = N_params/BW × ceil(L_spec/N_ALU)).
PIM_NALU: int = 4

# ---------------------------------------------------------------------------
# Mobile NPU.  Source: LP-Spec Table II
# ---------------------------------------------------------------------------
NPU_INT8_TOPS: float = 32.8e12          # matrix unit, ops/s INT8
NPU_VECTOR_TOPS: float = 8.2e12         # vector unit, ops/s
NPU_FREQ_HZ: float = 1e9                # Hz
NPU_CORES: int = 16
NPU_LOCAL_BUFFER_BYTES: float = 256e3   # 256 KB per core
NPU_SCRATCHPAD_BYTES: float = 8e6       # 8 MB total
NPU_OFFCHIP_BW: float = 51.2e9          # bytes/s (shared channel with PIM external I/O)

# ---------------------------------------------------------------------------
# Energy constants — see module docstring for full derivation/sourcing.
# 2 movement (pJ/bit, by interface) + 2 compute (pJ/MAC, by device/process)
# ---------------------------------------------------------------------------
MEM_INTERNAL_PJ_PER_BIT: float = 0.8    # internal near-bank (PIM path), 15% × 5.47
MEM_OFFCHIP_PJ_PER_BIT: float = 5.47    # off-chip bus (NPU HOST reads + PIM<->NPU comm)
PIM_MAC_PJ_PER_OP: float = 0.23         # near-bank ALU (20 nm DRAM); logic-process FLOOR
NPU_MAC_PJ_PER_OP: float = 0.23         # NPU matrix/vector unit (4 nm logic); upper bound

# Utilisation derates (PAPI SCALING_FACTOR).
MAX_COMPUTE_UTIL: float = 0.8
MAX_MEM_UTIL: float = 0.85


def pj_to_j(pj: float) -> float:
    """Convert picojoules to joules."""
    return pj * 1e-12


def bits_to_bytes(bits: int) -> float:
    return bits / 8


# ===========================================================================
# MODELS
# ===========================================================================

@dataclass(frozen=True)
class ModelConfig:
    name: str
    d_model: int            # hidden dimension
    n_heads: int            # number of attention (query) heads
    n_kv_heads: int         # KV heads (GQA; == n_heads for MHA)
    n_layers: int           # transformer layers
    intermediate_size: int  # FFN intermediate dimension
    vocab_size: int
    bytes_per_param: int    # 1 = INT8 (W8A8), 2 = FP16

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    def weight_bytes(self) -> float:
        """Approx. total linear-layer weight in bytes (attention proj + FFN; the
        embedding table is excluded — not streamed during decoding)."""
        d = self.d_model
        kv_dim = self.n_kv_heads * self.head_dim
        ffn = self.intermediate_size
        attn_params = d * d + d * kv_dim + d * kv_dim + d * d       # Wq,Wk,Wv,Wo
        ffn_params = d * ffn + d * ffn + ffn * d                    # SwiGLU gate+up+down
        return self.n_layers * (attn_params + ffn_params) * self.bytes_per_param

    def kv_cache_bytes(self, seq_len: int, batch_size: int = 1) -> float:
        """KV-cache footprint: 2 × batch × n_kv_heads × seq_len × head_dim per layer."""
        per_layer = 2 * batch_size * self.n_kv_heads * seq_len * self.head_dim
        return self.n_layers * per_layer * self.bytes_per_param


# Target: lmsys/vicuna-7b-v1.3.  Fine-tuned from LLaMA-1 but architecturally
# identical to LLaMA-2-7B (RMSNorm + SwiGLU + RoPE, 4096/32/32/11008/32000, MHA),
# so every shape/FLOP/byte count below is the standard 7B one.  The only 7B target
# with BOTH an official EAGLE head and an official MEDUSA head.
VICUNA_7B = ModelConfig(
    name="Vicuna-7B-v1.3",
    d_model=4096,
    n_heads=32,
    n_kv_heads=32,       # MHA (no GQA on the 7B model)
    n_layers=32,
    intermediate_size=11008,
    vocab_size=32000,
    bytes_per_param=1,   # INT8 quantization
)

# EAGLE draft head for Vicuna-7B (yuhuili/EAGLE-Vicuna-7B-v1.3).  A lightweight
# head — one decoder layer at the target's dims plus a fusion FC — trained to
# predict the target's hidden states; inseparable from the target.  Only the
# FC/attention weights matter for the compute roofline.
EAGLE_HEAD_VICUNA_7B = ModelConfig(
    name="EAGLE-Head-Vicuna-7B",
    d_model=4096,
    n_heads=32,
    n_kv_heads=32,       # full attention (matches the target's MHA)
    n_layers=1,
    intermediate_size=11008,
    vocab_size=32000,
    bytes_per_param=1,
)
