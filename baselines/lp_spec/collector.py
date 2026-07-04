"""
LP-Spec / MEDUSA GPU trace collector — the causal, divergence-free baseline source.

The symmetric counterpart to ``capim_ctrl/collector.py``: it runs LP-Spec's retrospective
Draft Token Pruner (DTP) inside The MEDUSA loop so the recorded ``Trace`` is the gated
trajectory, not a full static tree pruned after the fact.  Per step:

  1. SELECT   the DTP scores the full STATIC tree from the current retrospective
              per-(head,k) histogram and keeps the greedy top-L (L=4; step 0 = full-tree
              cold start).  (`baselines.lp_spec.dtp`, ported unchanged.)
  2. GATE     overwrite the pruned nodes in ``retrieve_indices`` with ``-1``
              (`common.gating.invalidate_paths`) so MEDUSA's greedy verify cannot accept
              past a pruned node and the real KV advances by the shorter prefix.
  3. RECORD   emit ONLY the kept sub-tree (+ causal ``accepted_length``).
  4. UPDATE   fold THIS step's *verified* acceptances into the histogram (causal:
              selection at step t used history < t only).

The reason MEDUSA needs care EAGLE does not: ``retrieve_indices`` is the same
STATIC buffer object that ``medusa_generate`` hands to ``generate_candidates``,
``tree_decoding`` AND ``update_inference_inputs`` in one iteration.  So we mutate that
buffer IN PLACE (``copy_``) inside the ``generate_candidates`` hook -- editing a wrapper-
local copy would leave ``update_inference_inputs`` advancing the KV by the un-gated
prefix.  We always rebuild from a pristine snapshot, and restore it on detach.

DTP identity subtlety: k_pred (the "k" in p_i^k) is the node's rank among ALL its static
siblings.  It must be derived from the FULL static tree, never from the pruned sub-tree
(where pruned siblings would shift the ranks).  So ``kp``/``pp`` are computed ONCE from
the static structure and passed to both ``select_kept`` and ``hist.update`` every step.

Histogram/step-index semantics match the CPU driver (`baselines/lp_spec/driver.py`): the
DTP histogram persists across prompts and the step index ``t`` is global, so the full-tree
cold start happens once.  Re-costing: feed the gated ``Trace`` to the driver with a large
``L_spec`` (>= full tree) -- the DTP then keeps every recorded node (pass-through, like
CAPIM's sigma=-inf), so cost depends only on the recorded gated ``mu = |nodes|``.

Recorded MEDUSA nodes carry STRUCTURE + ``accepted`` only (token_id / cumulative_log_prob
are left 0.0): the LP-Spec driver's DTP keys solely on depth / layer_idx / parent_idx /
accepted, and the cost model on the node count.

Layering mirrors the EAGLE collector: the tree math is pure torch-free functions
(unit-testable with plain lists); torch / transformers / medusa imports are LOCAL to the
model-facing methods so this module imports on a CPU box for the GPU-free dry-run.
"""

from __future__ import annotations

import functools
from typing import Any, Dict, List, Optional, Sequence, Tuple

from common.gating import invalidate_paths
from common.schema import DecodeStep, TokenNode, Trace
from baselines.lp_spec import dtp

NEG_INF = float("-inf")


# ============================ pure structure core =============================
# All inputs plain python (lists): no torch / GPU / model needed.

def build_static_structure(
    pos_ids: Sequence[int],                    # medusa_position_ids.tolist(): root=0, depth+1
    mask_rows: Sequence[Sequence[bool]],       # medusa_attn_mask[0,0].bool().tolist(): (N+1)^2
) -> Dict[int, Tuple[int, int, int]]:
    """Per static node index v in 1..N -> (depth, layer_idx, parent_node_index).

    ``parent_node_index`` is in draft-index space (0 = root).  Node index v == its
    position ``s+1`` in MEDUSA's sorted tree; ``retrieve_indices`` uses the same space.
    ``layer_idx`` is a per-depth running counter (unique within a depth and, because the
    tree is stored in (len, lex) order, ascending with sibling rank -> the DTP recovers
    k_pred correctly).
    """
    n = len(pos_ids) - 1
    depth_counter: Dict[int, int] = {}
    meta: Dict[int, Tuple[int, int, int]] = {}
    for i in range(1, n + 1):
        depth = int(pos_ids[i]) - 1
        ancestors = [j for j in range(n + 1) if mask_rows[i][j]]
        parent = ancestors[-2] if len(ancestors) >= 2 else 0   # [-1]=self, [-2]=direct parent
        li = depth_counter.get(depth, 0)
        depth_counter[depth] = li + 1
        meta[i] = (depth, li, parent)
    return meta


def _node(depth: int, layer_idx: int, parent_idx: int, accepted: bool = False) -> TokenNode:
    return TokenNode(
        depth=depth, token_id=0, log_prob=0.0, cumulative_log_prob=0.0,
        parent_idx=parent_idx, accepted=accepted, layer_idx=layer_idx,
    )


def structural_step(meta: Dict[int, Tuple[int, int, int]]) -> DecodeStep:
    """The full STATIC tree as a DecodeStep for the DTP (node list pos == index-1)."""
    nodes = []
    for v in sorted(meta):
        depth, li, parent = meta[v]
        nodes.append(_node(depth, li, parent - 1 if parent >= 1 else -1))
    return DecodeStep(step_id=0, context_length=0, nodes=nodes, accepted_length=0,
                      dataset="", prompt_id=0)


def pos_to_index(meta: Dict[int, Tuple[int, int, int]]) -> Dict[Tuple[int, int], int]:
    """(depth, layer_idx) -> static node index, to turn a DTP keep set into node ids."""
    return {(d, li): v for v, (d, li, _p) in meta.items()}


def record_kept(
    meta: Dict[int, Tuple[int, int, int]],
    kept_node_indices: Sequence[int],
) -> Tuple[List[TokenNode], Dict[int, int]]:
    """Kept sub-tree as TokenNodes (ascending node index) + {node index -> list pos}.

    ``parent_idx`` is re-indexed to the KEPT list position (-1 for children of the root);
    depth / layer_idx stay the STATIC values so Pos identity is preserved across steps.
    """
    kept = sorted(kept_node_indices)
    new_index_of = {v: k for k, v in enumerate(kept)}
    nodes = []
    for v in kept:
        depth, li, parent = meta[v]
        parent_idx = new_index_of[parent] if parent >= 1 else -1   # ancestor-closed -> kept
        nodes.append(_node(depth, li, parent_idx))
    return nodes, new_index_of


def mark_accepted(
    nodes: List[TokenNode],
    edited_rows: Sequence[Sequence[int]],
    new_index_of: Dict[int, int],
    best_candidate: int,
    accept_length: int,
) -> int:
    """Flag the accepted prefix on the kept nodes; return accepted DRAFT-token count.

    Walks the winning (gated) path over the first ``accept_length`` draft positions
    (column 0 is the root, -1 is padding) and marks the corresponding kept nodes.
    """
    if not (0 <= best_candidate < len(edited_rows)):
        return 0
    path = list(edited_rows[best_candidate])
    accepted = 0
    for j in range(1, accept_length + 1):
        if j >= len(path):
            break
        idx = path[j]
        if idx > 0:
            pos = new_index_of.get(idx)
            if pos is not None:
                nodes[pos].accepted = True
                accepted += 1
    return accepted


# ============================ model load + collector ==========================

def load_medusa_model(
    base_model_path: str,
    medusa_model_path: str,
    *,
    load_in_8bit: bool = True,
    load_in_4bit: bool = False,
):
    """Load an 8-bit ``MedusaModel`` (medusa-vicuna-7b-v1.3 head on Vicuna-7B).

    Same ``device_map={"":0}`` single-GPU pin as the EAGLE loader (the vendored tree-
    attention code is not model-parallel safe).
    """
    import torch
    from transformers import BitsAndBytesConfig

    from common.sd_repos import add_medusa_to_path
    add_medusa_to_path()
    from medusa.model.medusa_model import MedusaModel

    kwargs: Dict[str, Any] = dict(torch_dtype=torch.float16, device_map={"": 0})
    if load_in_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4",
        )
    elif load_in_8bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)

    model = MedusaModel.from_pretrained(medusa_model_path, **kwargs)
    model.eval()
    return model, model.get_tokenizer()


class Collector:
    """Attach to a ``MedusaModel`` and record a causal DTP-gated ``Trace``.

    One collector spans a whole run: the DTP histogram and the global step index ``t``
    persist across prompts (matching the CPU driver).  Call ``set_prompt`` between the
    per-prompt ``medusa_generate`` calls.
    """

    def __init__(self, L: int = 4, dataset: str = "unknown", selection: str = "greedy_headk"):
        self.L = L
        self.dataset = dataset
        self.selection = selection
        self.prompt_id = 0
        self._steps: List[DecodeStep] = []
        self._t = 0                              # GLOBAL step index (cold start at t==0)
        gran = "node" if selection == "greedy_node" else "headk"
        self._hist = dtp.DTPHist(granularity=gran)
        self._pending: Optional[Dict[str, Any]] = None
        # static structure (lazy, from model.medusa_buffers on first candidate call)
        self._ready = False
        self._meta = None
        self._struct_step = None
        self._kp = None
        self._pp = None
        self._pos_to_index = None
        self._pristine_rows = None
        # restore handles
        self._model = None
        self._mod = None
        self._orig: Dict[str, Any] = {}

    def set_prompt(self, prompt_id: int) -> None:
        self.prompt_id = prompt_id
        self._pending = None                     # defensive: no step straddles prompts

    # -- attach / detach ---------------------------------------------------
    def attach(self, medusa_model: Any) -> None:
        import medusa.model.medusa_model as mod   # loop's namespace (from .utils import *)
        self._model = medusa_model
        self._mod = mod
        for name in ("generate_candidates", "tree_decoding", "evaluate_posterior"):
            self._orig[name] = getattr(mod, name)
        mod.generate_candidates = self._wrap_candidates(mod.generate_candidates)
        mod.tree_decoding = self._wrap_tree_decoding(mod.tree_decoding)
        mod.evaluate_posterior = self._wrap_eval(mod.evaluate_posterior)

    def detach(self) -> List[DecodeStep]:
        if self._mod is not None:
            for name, fn in self._orig.items():
                setattr(self._mod, name, fn)
        # leave the model's static buffer clean (we mutated it in place)
        if self._ready and self._model is not None:
            import torch
            buf = self._model.medusa_buffers["retrieve_indices"]
            buf.copy_(torch.tensor(self._pristine_rows, dtype=buf.dtype, device=buf.device))
        return list(self._steps)

    # -- lazy static init --------------------------------------------------
    def _ensure_static(self) -> None:
        if self._ready:
            return
        buffers = self._model.medusa_buffers
        pos_ids = buffers["medusa_position_ids"].tolist()
        mask_rows = buffers["medusa_attn_mask"][0, 0].bool().tolist()
        self._pristine_rows = buffers["retrieve_indices"].tolist()
        self._meta = build_static_structure(pos_ids, mask_rows)
        n_from_rows = max(max(r) for r in self._pristine_rows)
        assert len(self._meta) == n_from_rows, (
            f"tree size mismatch: meta {len(self._meta)} vs retrieve_indices max {n_from_rows}"
        )
        self._struct_step = structural_step(self._meta)
        self._kp = dtp.k_pred_map(self._struct_step)
        self._pp = dtp.parent_pos_map(self._struct_step)
        self._pos_to_index = pos_to_index(self._meta)
        self._ready = True

    # -- wrappers ----------------------------------------------------------
    def _wrap_candidates(self, orig):
        col = self

        @functools.wraps(orig)
        def wrapper(medusa_logits, logits, tree_indices, retrieve_indices, *args, **kwargs):
            import torch
            col._ensure_static()

            kept_pos = dtp.select_kept(
                col._struct_step, col._t, col.L, col.selection, col._hist, col._kp, col._pp,
            )
            kept_idx = {col._pos_to_index[p] for p in kept_pos}
            edited_rows = invalidate_paths(col._pristine_rows, kept_idx)
            kept_nodes, new_index_of = record_kept(col._meta, kept_idx)

            # mutate the shared static buffer IN PLACE (from pristine), so
            # tree_decoding + update_inference_inputs in this same iteration see the gate.
            retrieve_indices.copy_(torch.tensor(
                edited_rows, dtype=retrieve_indices.dtype, device=retrieve_indices.device,
            ))
            col._pending = dict(
                kept_nodes=kept_nodes, edited_rows=edited_rows,
                new_index_of=new_index_of, context_length=0,
            )
            return orig(medusa_logits, logits, tree_indices, retrieve_indices, *args, **kwargs)

        return wrapper

    def _wrap_tree_decoding(self, orig):
        col = self

        @functools.wraps(orig)
        def wrapper(model, tree_candidates, past_key_values, position_ids, input_ids,
                    retrieve_indices, *args, **kwargs):
            if col._pending is not None:
                col._pending["context_length"] = int(input_ids.shape[1])
            return orig(model, tree_candidates, past_key_values, position_ids, input_ids,
                        retrieve_indices, *args, **kwargs)

        return wrapper

    def _wrap_eval(self, orig):
        col = self

        @functools.wraps(orig)
        def wrapper(logits, candidates, *args, **kwargs):
            best_candidate, accept_length = orig(logits, candidates, *args, **kwargs)
            p = col._pending
            if p is not None:
                mark_accepted(p["kept_nodes"], p["edited_rows"], p["new_index_of"],
                              int(best_candidate), int(accept_length))
                step = DecodeStep(
                    step_id=col._t,
                    context_length=p["context_length"],
                    nodes=p["kept_nodes"],
                    accepted_length=int(accept_length),   # raw draft accepts; driver adds bonus
                    dataset=col.dataset,
                    prompt_id=col.prompt_id,
                )
                col._steps.append(step)
                # causal update: fold THIS step's verified accepts in, STATIC kp/pp
                col._hist.update(step, col._kp, col._pp)
                col._t += 1
                col._pending = None
            return best_candidate, accept_length

        return wrapper


def collect(
    medusa_model: Any,
    tokenizer: Any,
    prompts: Sequence[str],
    *,
    dataset: str,
    L: int = 4,
    selection: str = "greedy_headk",
    max_new_tokens: int = 200,
    model_name: str = "Vicuna-7B-v1.3",
    draft_head: str = "FasterDecoding/medusa-vicuna-7b-v1.3",
    temperature: float = 0.0,
    medusa_choices=None,
) -> Trace:
    """Run DTP-gated MEDUSA over ``prompts`` (one persistent collector) -> a ``Trace``.

    ``L`` / ``selection`` are the collection-side pruning knobs, fed straight into the
    in-loop ``dtp.select_kept`` (the SAME DTP the CPU driver replays):
      - gated (headline)  : ``selection="greedy_headk", L=4`` -- LP-Spec at its optimum.
      - full-tree (ungated): ``selection="full"`` -- verify every node each step, real
        accepts recorded; the MEDUSA analog of the EAGLE collector's ``sigma_th=-inf``,
        the trace the driver then sweeps ``L_spec`` over.

    The histogram / step index persist across prompts (driver-matching).  Greedy
    (temperature=0) by default -- the in-loop gate is exact-in-practice there.
    ``medusa_generate`` is a generator; we drain it to run the loop to completion.
    """
    import torch

    col = Collector(L=L, dataset=dataset, selection=selection)
    col.attach(medusa_model)
    try:
        for pid, prompt in enumerate(prompts):
            col.set_prompt(pid)
            inputs = tokenizer(prompt, return_tensors="pt").to(medusa_model.base_model.device)
            with torch.no_grad():
                for _ in medusa_model.medusa_generate(
                    inputs["input_ids"], temperature=temperature,
                    max_steps=max_new_tokens, medusa_choices=medusa_choices,
                ):
                    pass                          # drain the generator
    finally:
        steps = col.detach()                      # restores patches + static buffer

    trace = Trace(
        steps=steps,
        model=model_name,
        sd_method="medusa",
        metadata=dict(
            dataset=dataset, L=L, selection=selection, temperature=temperature,
            draft_head=draft_head, n_prompts=len(prompts), max_new_tokens=max_new_tokens,
            gated=(selection != "full"),
        ),
    )
    trace.compute_summary()
    return trace
