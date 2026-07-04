"""
The typed-layer atom of the cost kernel + the workload builders.

A workload (a target decoder layer, an EAGLE draft sub-step, the MEDUSA heads,
a prefill pass, ...) is just an ordered list of `Layer` objects.  Cost is a pure
function of a Layer's shape (m, n, k, numOp, dbyte) and the device it runs on
(see common.devices).

`get_flops` / `get_size` follow PAPI's `src/model.py:Layer` (model-agnostic); the
layer-type set is trimmed to what mobile batch=1 inference needs (PAPI's
G2G/X2G all-reduce collapse to a single COMM type for the PIM<->NPU handoff).

Builders emit layers device-AGNOSTIC; tagging a layer with a device is the
driver/router's job.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from common.config import ModelConfig
from common.type import Device, LayerType


@dataclass
class Layer:
    """One typed operator.

    Shape semantics (match PAPI):
      FC/MATMUL:  m × k  times  k × n  ->  m × n, repeated `numOp` times.
      NL/COMM:    operate elementwise on an m × n activation, `numOp` times.

    `device` is assigned by the driver's router, not by the builder.
    `bound`, `time_s`, `energy` are filled in by Device.cost().
    """

    name: str
    type: LayerType
    m: int                               # rows of the output
    n: int                               # columns of the output
    k: int = 1                           # contraction dim
    numOp: int = 1                       # how many times the op repeats
    dbyte: int = 1                       # 1 = INT8 (W8A8), 2 = FP16
    device: Optional[Device] = None

    # filled by Device.cost()
    bound: str = ""                      # "compute" | "memory" | "comm"
    time_s: float = 0.0
    energy: list = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])

    def get_infos(self):
        return self.m, self.n, self.k, self.numOp, self.dbyte

    def get_flops(self) -> float:
        t = self.type
        if t == LayerType.SOFTMAX:
            return 5 * self.m * self.n * self.numOp
        if t == LayerType.NORM:
            return 5 * self.m * self.n * self.numOp
        if t == LayerType.ACT:
            if "relu" in self.name:
                return 1 * self.m * self.n * self.numOp
            if "glu" in self.name:            # SwiGLU/GeGLU: gate*up + activation
                return (8 + 1) * self.m * self.n * self.numOp
            return 8 * self.m * self.n * self.numOp
        if t in (LayerType.FC, LayerType.MATMUL):
            return 2 * self.m * self.n * self.k * self.numOp
        if t == LayerType.COMM:
            return 0
        raise ValueError(f"get_flops: unsupported layer type {t}")

    def get_size(self):
        """Return (in1, in2, out) traffic in BYTES.

        FC/MATMUL: in1 = activation, in2 = weight/second-operand, out = result.
        NL/COMM:   in1 = out = the activation, in2 = 0 (glu reads two inputs).
        NORM:      reads activation twice (x and the reduction), writes once.
        """
        in1 = self.numOp * self.m * self.k * self.dbyte
        in2 = self.numOp * self.n * self.k * self.dbyte
        out = self.numOp * self.m * self.n * self.dbyte

        if self.type in (LayerType.SOFTMAX, LayerType.ACT, LayerType.COMM):
            in1 = self.numOp * self.m * self.n * self.dbyte
            in2 = self.numOp * self.m * self.n * self.dbyte if "glu" in self.name else 0
            out = self.numOp * self.m * self.n * self.dbyte
        elif self.type == LayerType.NORM:
            in1 = self.numOp * self.m * self.n * self.dbyte
            in2 = in1
            out = in1
        return in1, in2, out

    def weight_bytes(self) -> float:
        """Bytes of the stationary operand (the weight matrix for FC)."""
        if self.type == LayerType.FC:
            return self.numOp * self.n * self.k * self.dbyte
        return 0.0


# ===========================================================================
# Builders — each emits one Unit of work; the driver scales per-layer cost by
# n_layers and tags devices.  Layer names follow PAPI's LLAMA branch.
# ===========================================================================

def build_decoder_layer(model: ModelConfig, m: int, ctx: int,
                        eagle_draft: bool = False) -> List[Layer]:
    """One transformer decoder layer over `m` query tokens at KV context `ctx`.

    `m`   = tokens processed this pass (prefill: prompt length; verify: tree size μ).
    `ctx` = KV-cache length the attention attends over.

    `eagle_draft=True` emits the EAGLE draft head's single decoder layer.  EAGLE's
    head layer is index 0, so its input_layernorm is never built
    (EAGLE/eagle/model/cnets1.py:399 `if self.index != 0`), and the head has no
    final norm — the LM head reads the layer output directly (cnets1.py:732).  Net:
    ONE RMSNorm, not two — here the trailing `norm2` is dropped.
    """
    d = model.d_model            # hidden / residual-stream width
    h = model.n_heads            # query heads (attention numOp)
    dh = model.head_dim          # per-head dim; attention contraction
    ff = model.intermediate_size # SwiGLU FFN hidden width
    db = model.bytes_per_param   # bytes/elem; scales traffic, not FLOPs

    # Fused q,k,v output width: PAPI uses 3*hdim/tp_dense (model.py:173); at mobile
    # batch=1 tp_dense=1, so this is 3*d.
    qkv_n = 3 * d

    layers = [
        Layer("qkv", LayerType.FC, m=m, n=qkv_n, k=d, dbyte=db),
        Layer("score", LayerType.MATMUL, m=m, n=ctx, k=dh, numOp=h, dbyte=db),
        Layer("softmax", LayerType.SOFTMAX, m=m, n=ctx, numOp=h, dbyte=db),
        Layer("context", LayerType.MATMUL, m=m, n=dh, k=ctx, numOp=h, dbyte=db),
        Layer("proj", LayerType.FC, m=m, n=d, k=d, dbyte=db),
        Layer("norm1", LayerType.NORM, m=m, n=d, dbyte=db),
        Layer("ff1", LayerType.FC, m=m, n=ff, k=d, dbyte=db),
        Layer("ff2", LayerType.FC, m=m, n=ff, k=d, dbyte=db),
        Layer("glu", LayerType.ACT, m=m, n=ff, dbyte=db),
        Layer("ff3", LayerType.FC, m=m, n=d, k=ff, dbyte=db),
    ]
    if not eagle_draft:
        layers.append(Layer("norm2", LayerType.NORM, m=m, n=d, dbyte=db))
    return layers


def build_lm_head(model: ModelConfig, m: int) -> Layer:
    """Vocabulary projection over `m` tokens (FC: d_model -> vocab).

    The LLaMA head is nn.Linear(hidden, vocab, bias=False) — a plain GEMM
    (EAGLE modeling_llama_kv.py:1212).
    """
    return Layer("lm_head", LayerType.FC, m=m, n=model.vocab_size,
                 k=model.d_model, dbyte=model.bytes_per_param)


def build_verify_pass(model: ModelConfig, m: int, ctx: int) -> List[Layer]:
    """The target's per-step verification work over a μ=`m`-node tree at context
    `ctx`: ONE decoder layer (driver scales by n_layers) + the lm_head once."""
    layers = build_decoder_layer(model, m=m, ctx=ctx)
    layers.append(build_lm_head(model, m=m))
    return layers


def build_prefill(model: ModelConfig, prompt_len: int) -> Tuple[List[Layer], Layer]:
    """One-time prefill over a `prompt_len`-token prompt, returned as the
    (decoder_block, lm_head) pair that `cost_forward_pass` consumes directly.

    Prefill is a FULL target forward pass, so — like verify and AR — the single
    decoder block must be scaled by n_layers and the lm_head added once.  Returning
    the block and head SEPARATELY (rather than one flat list) puts prefill on the
    same convention every real forward pass uses: `cost_forward_pass(block, head,
    n_layers, ...)`.  Composing the pair as a flat list would wrongly scale the head
    by n_layers too.

    First-class: the driver emits this once on the NPU (compute-bound GEMM) and
    reports it ONLY in end-to-end latency — never in the per-step token/s · token/J
    · EDP rates.  Causal prompt → attention context == prompt_len.  The lm_head is
    over the single last position (the first sampled token), m=1.
    """
    block = build_decoder_layer(model, m=prompt_len, ctx=prompt_len)
    head = build_lm_head(model, m=1)
    return block, head


def build_eagle_draft_step(model: ModelConfig, width: int, ctx: int) -> List[Layer]:
    """One EAGLE-2 draft sub-step over `width` tree nodes at KV context `ctx`.

    Grounded in EAGLE/eagle/model/cnets1.py: fusion FC (concat emb+feature, k=2d)
    -> one decoder layer (eagle_draft: one RMSNorm) -> shared lm_head -> sampling
    softmax (the confidence signal).  The driver calls this once per tree depth
    with `width` = nodes at that depth.
    """
    d = model.d_model
    db = model.bytes_per_param
    layers: List[Layer] = [
        Layer("fusion_fc", LayerType.FC, m=width, n=d, k=2 * d, dbyte=db),
    ]
    layers += build_decoder_layer(model, m=width, ctx=ctx, eagle_draft=True)
    layers.append(build_lm_head(model, m=width))
    layers.append(Layer("sample_softmax", LayerType.SOFTMAX, m=width,
                        n=model.vocab_size, dbyte=db))
    return layers


def build_medusa_draft(model: ModelConfig, medusa_num_heads: int = 5,
                       medusa_num_layers: int = 1) -> List[Layer]:
    """The MEDUSA heads from one hidden state (parallel, batch=1, the "free tail").

    Grounded in Medusa/medusa/model/medusa_model.py: ResBlock(x)=x+SiLU(Linear)
    per head, then each head's OWN lm_head (K independent vocab projections), then
    one fused sampling softmax (top-k for the static tree).  No autoregression, no
    attention, no KV.
    """
    d = model.d_model
    db = model.bytes_per_param
    layers: List[Layer] = []
    for h in range(medusa_num_heads):
        for _ in range(medusa_num_layers):
            layers.append(Layer(f"medusa{h}_resblock", LayerType.FC, m=1, n=d, k=d, dbyte=db))
            layers.append(Layer(f"medusa{h}_silu", LayerType.ACT, m=1, n=d, dbyte=db))
        layers.append(Layer(f"medusa{h}_lmhead", LayerType.FC, m=1,
                            n=model.vocab_size, k=d, dbyte=db))
    layers.append(Layer("medusa_softmax", LayerType.SOFTMAX, m=1,
                        n=model.vocab_size, numOp=medusa_num_heads, dbyte=db))
    return layers
