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

All three movement/compute constants come from ONE source: AttAcc's simulator
(`releted-repos/attacc_simulator/src/config.py`), which is LP-Spec's own reference
[24] — so we score the baseline on its own energy basis.  Its DRAM breakdown is from
MICRO'17 "Fine-Grained DRAM" (O'Connor et al.); its ALU is Verilog-synthesised at
7 nm ASAP7.  Of LP-Spec's three cited energy sources it is the only complete one:
SpecPIM [29] gives no in-bank figure (and its 1.012 pJ/MAC PE is FP16), and
McDRAM v2 [26] reports only system-level TOPS/W.

Caveat: these are HBM figures (the TSV/interposer terms don't exist on mobile).  Real
mobile LPDDR5 is dearer — 10–20 pJ/bit off-chip, 0.95 in-bank (PIM-AI arXiv:2411.17309
Tab. 1) — which only widens CAPIM's margin.  Carry it as a sensitivity, not a default.
"""

from __future__ import annotations

from dataclasses import dataclass

# ===========================================================================
# HARDWARE — LPDDR5-PIM.  Source: LP-Spec Table II + §V-A/§VI-A/§VI-B
# ===========================================================================

# Compute throughput: 409.6 GOPS INT8 PER DIE × 12 PIM dies = 4.9152 TOPS.
#
# CORRECTED 2026-07-12 (was 409.6e9, read as an aggregate over 4 dies).  The per-die
# reading is LP-Spec's own, three ways:
#   §VI-A  "we enhance the performance by 4× to 409.6 GOPS for EACH DIE."
#   Tab.II  Samsung LPDDR5-PIM baseline = 102.4 GOPS@INT8 per die; LP-Spec is 4× it.
#   §V-A   "An MPU consists of four 32-wide SIMD ALUs" = 128 INT8 MACs; Table II
#          "# MPU = 8" (a block whose "Capacity = 1 GB" is per-die):
#              8 MPU × 128 MAC × 2 op × 200 MHz = 409.6 GOPS per die   (exact)
# Die count from the config they evaluate:
#   §VI-B  "3 PIM ranks and 1 DRAM rank ... total capacity of 16 GB"  (4 ranks × 4 GB ✓)
#   Tab.II "# Die / Rank = 4"   ->  3 × 4 = 12 PIM dies.
# All PIM ranks compute together: §V-A "all PIM ranks are first switched into all bank
# mode ... then all bank PIM mode to trigger the execution"; the NMC "receives
# independent C/A signals for DRAM ranks and PIM ranks, allowing for parallel operation."
#
# Cross-check that needs no reading of Table II at all: LP-Spec's published 73.4 token/s
# (Table III) implies a 13.62 ms/token budget.  One token's forward pass through 7B INT8
# is 12.95 GOPs, so the machine MUST sustain ≥1.19 TOPS peak even if every draft token
# were accepted for free, and 1.9–5.7 TOPS peak for realistic MEDUSA (L=4–12, τ≈2.5).
# The old 409.6 GOPS could not fit ONE token's MACs into that budget (it buys 4.46 GOPs).
# See scripts/cpu/validate_cost_model.py.
PIM_INT8_GOPS: float = 12 * 409.6e9     # ops/s INT8  (409.6 GOPS/die × 12 dies)

# Internal bank bandwidth: 51.2 TB/s.
# NOTE — LP-Spec is self-inconsistent here: Table II says "On-chip Bandwidth 51.2 TB/s",
# but §II-B says a ×64 LPDDR5 chip (4 dies) reaches "409.6 GB/s" internal all-bank
# bandwidth — 125× apart.  We take Table II.  It does not bite: PIM stays compute-bound
# under either figure (the ridge at 4.9152 TOPS / 51.2 TB/s = 0.096 ops/byte is far below
# GEMV's 2 ops/byte), but it is closer to binding than it was, so flag it as a
# sensitivity if PIM ever comes out memory-bound.
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
# Revised 2026-07-12 to AttAcc's table (was 0.8 / 5.47 / 0.23 / 0.23, a mix of SpecPIM
# p.11 and Horowitz-45nm).  Line refs are into releted-repos/attacc_simulator/src/.
MEM_INTERNAL_PJ_PER_BIT: float = 0.55   # in-bank: cell ACT/PRE 0.11 + RD/WRT 0.44  [config.py:47]
MEM_OFFCHIP_PJ_PER_BIT: float = 3.59    # off-chip: + 1.01 + 1.23 + TSV 0.5 + interposer 0.3  [config.py:23]
PIM_MAC_PJ_PER_OP: float = 0.32         # near-bank ALU  [config.py:58]; see NOTE below
NPU_MAC_PJ_PER_OP: float = 0.32         # NPU matrix/vector unit  [config.py:22]
# Sanity: 0.55/3.59 = 15.3%, independently reproducing LP-Spec §II-A's "within-DRAM
# transfers cost 15% of off-DRAM" [23].
# NOTE: AttAcc charges the SAME MAC energy to a DRAM-process PIM ALU and a logic-process
# GPU ALU (its 10x DRAM-density penalty is applied to area, not energy).  The two are kept
# separate so the asymmetric PIM_MAC >> NPU_MAC sweep — the corner most hostile to CAPIM,
# and still unrun — stays expressible.

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
