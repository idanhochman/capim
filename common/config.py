"""
Hardware + model configuration for the CAPIM simulator.

Two halves:
  1. Hardware — LPDDR5-PIM + mobile-NPU constants (LP-Spec Table II) and the
     four-component energy model (2 movement pJ/bit by interface + 2 compute
     pJ/MAC by device/process).
  2. Models — Vicuna-7B-v1.3 (the only 7B target with both an official EAGLE draft
     model and official MEDUSA heads, so CAPIM and LP-Spec share a backbone) and its
     EAGLE draft model.

Energy model — the four constants feed the [off_mem, on_chip, alu, comm] vector:
    off_mem <- MEM_INTERNAL (PIM)   or MEM_OFFCHIP (NPU)
    alu     <- PIM_MAC_PJ_PER_OP    or NPU_MAC_PJ_PER_OP
    comm    <- MEM_OFFCHIP (the PIM<->NPU crossing rides the same external bus)
    on_chip <- 0 (no mobile cache hierarchy modelled)

All movement/compute constants come from one source, the AttAcc simulator, which is
LP-Spec's own reference [24], so the baseline is scored on its own energy basis.  Of
LP-Spec's three cited sources it is the only complete one: SpecPIM [29] gives no in-bank
figure, McDRAM v2 [26] only system-level TOPS/W.
"""

from __future__ import annotations

from dataclasses import dataclass

# ===========================================================================
# Hardware — LPDDR5-PIM.  Source: LP-Spec Table II + §V-A/§VI-A/§VI-B
# ===========================================================================

# Compute throughput: 409.6 GOPS INT8 per die × 12 PIM dies = 4.9152 TOPS.
#   §VI-A  "we enhance the performance by 4× to 409.6 GOPS for each die"
#   §V-A + Tab.II  8 MPU × 128 MAC × 2 op × 200 MHz = 409.6 GOPS/die  (exact)
#   §VI-B  "3 PIM ranks and 1 DRAM rank" × Tab.II "# Die / Rank = 4"  ->  12 PIM dies
# All PIM ranks compute together (§V-A: all ranks switched into all-bank PIM mode).
# Cross-checked against LP-Spec's published 73.4 token/s, which needs ≥1.19 TOPS peak
# just to fit one token's MACs -- see scripts/cpu/validate_cost_model.py.
PIM_INT8_GOPS: float = 12 * 409.6e9     # ops/s INT8  (409.6 GOPS/die × 12 dies)

# Internal bank bandwidth.  LP-Spec is self-inconsistent: Table II says 51.2 TB/s,
# §II-B implies 409.6 GB/s.  We take Table II; PIM comes out compute-bound either way.
PIM_INTERNAL_BW: float = 51.2e12        # bytes/s

# External I/O bandwidth (off-chip, shared with NPU): 51.2 GB/s
PIM_EXTERNAL_BW: float = 51.2e9         # bytes/s

PIM_FREQ_HZ: float = 200e6              # Hz
PIM_CAPACITY_BYTES: float = 16e9        # 3 PIM ranks + 1 DRAM rank × 4 GB

# Four 32-wide ALUs per MPU (LP-Spec Table II).  Token-parallel: one weight pass
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
# All four are AttAcc's own figures.
MEM_INTERNAL_PJ_PER_BIT: float = 0.55   # in-bank: cell ACT/PRE 0.11 + RD/WRT 0.44
MEM_OFFCHIP_PJ_PER_BIT: float = 3.59    # off-chip: + 1.01 + 1.23 + TSV 0.5 + interposer 0.3
PIM_MAC_PJ_PER_OP: float = 0.32         # near-bank ALU
NPU_MAC_PJ_PER_OP: float = 0.32         # NPU matrix/vector unit
# Sanity: 0.55/3.59 = 15.3%, independently reproducing LP-Spec §II-A's "within-DRAM
# transfers cost 15% of off-DRAM" [23].
# AttAcc charges the same MAC energy to a DRAM-process PIM ALU and a logic-process GPU
# ALU (its DRAM-density penalty applies to area, not energy).  The two constants stay
# separate so an asymmetric PIM_MAC vs NPU_MAC sweep remains expressible.

# Utilisation derates (PAPI SCALING_FACTOR).
MAX_COMPUTE_UTIL: float = 0.8
MAX_MEM_UTIL: float = 0.85


def pj_to_j(pj: float) -> float:
    """Convert picojoules to joules."""
    return pj * 1e-12


def bits_to_bytes(bits: int) -> float:
    return bits / 8


# ===========================================================================
# Models
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
# with both an official EAGLE draft model and official MEDUSA heads.
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

# The EAGLE draft model (yuhuili/EAGLE-Vicuna-7B-v1.3) needs no config of its own: it
# is a one-layer mini-decoder at the target's dimensions, so `build_eagle_draft_step`
# emits its shape directly from VICUNA_7B.
