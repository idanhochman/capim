"""
CAPIM / EAGLE-2 GPU trace collector — the divergence-free (causal) trajectory source.

Wraps a live EAGLE draft layer so the confidence gate fires INSIDE the drafting loop:
per step we read the vendored ``_capim_scores`` stash, keep the nodes whose cumulative
log-prob clears ``sigma_th``, and overwrite the pruned nodes in ``retrieve_indices``
with ``-1`` (``common.gating.invalidate_paths``).  The unchanged EAGLE verify then
cannot accept past a pruned node, and ``update_inference_inputs`` advances the real KV
by that shorter prefix -- so the recorded ``Trace`` is the gated trajectory, not a full
tree pruned after the fact (the trace-replay divergence in-loop gating avoids).

Two collection modes, one flag (``sigma_th``):
  - **finite sigma_th** (e.g. -1.5): GATED.  Invalidate inside the loop; record ONLY
    the kept sub-tree (+ the causal ``accepted_length``).  The driver then reads
    mu = |nodes| and re-costs the trajectory-invariant knobs (draft_device, mu_th) into
    the operation modes -- it must NOT re-threshold sigma (that would re-introduce the
    divergence).  This is the headline generator.
  - **sigma_th = -inf**: UNGATED / full-tree.  No invalidation, record EVERY drafted
    node with its real acceptance.  This is the full-tree recording, so the
    CPU driver can sweep sigma (CAPIM) / L (LP-Spec) off it -- the σ-characterization
    figure and the native frontier still come from here.  (Recording pruned nodes on a
    *gated* run would be wrong: their ``accepted=False`` is an artifact of our mask, not
    a real target rejection.  Only the ungated run genuinely verifies every node.)

Layering
--------
The recording/gating math is two PURE functions -- ``record_gated_step`` and
``mark_accepted`` -- that take plain python lists (already ``.tolist()``-ed) and are
unit-testable with NO torch, GPU or model (mirrors ``common.gating``).  Everything that
touches torch / transformers / the EAGLE package is confined to ``load_eagle_model`` and
the ``Collector`` wrapper methods, with the heavy imports done LOCALLY so this module
imports cleanly on a CPU box for the GPU-free dry-run.

``accepted_length`` convention: the RAW EAGLE ``accept_length`` (number of accepted
DRAFT tokens, no bonus).  The driver adds the +1 bonus itself (capim_ctrl/driver.py).
"""

from __future__ import annotations

import functools
from typing import Any, Dict, List, Optional, Sequence, Tuple

from common.gating import invalidate_paths
from common.schema import DecodeStep, TokenNode, Trace

NEG_INF = float("-inf")


# ============================ pure recording core =============================
# All inputs are plain python (lists / floats): no torch, GPU or model needed.

def _parents_from_tree_mask(tree_mask_rows: Sequence[Sequence[bool]], n: int) -> Dict[int, int]:
    """Direct parent (in draft-index space, 0 = root) of each draft node 1..n.

    ``tree_mask_rows[i][j]`` is True iff node ``j`` is an ancestor of node ``i`` (both
    in draft-index space where 0 = root/sample token).  Row ``i`` always contains ``i``
    (self) and ``0`` (root); the direct parent is the largest ancestor strictly below
    ``i`` -- i.e. the second-largest True column.  EAGLE builds the tree in depth order,
    so a parent's index is always < its child's.
    """
    parent: Dict[int, int] = {}
    for i in range(1, n + 1):
        ancestors = [j for j in range(n + 1) if tree_mask_rows[i][j]]
        # ancestors ascending; [-1] == i (self), [-2] == direct parent (0 for depth-0).
        parent[i] = ancestors[-2] if len(ancestors) >= 2 else 0
    return parent


def record_gated_step(
    draft_token_ids: Sequence[int],      # draft_tokens[0].tolist(): [root, node1..nodeN]
    retrieve_rows: Sequence[Sequence[int]],  # retrieve_indices.tolist(): -1-padded paths
    tree_mask_rows: Sequence[Sequence[bool]],  # tree_mask[0,0].bool().tolist(): (N+1)^2
    tree_pos_ids: Sequence[int],         # tree_position_ids.tolist(): root=0, node depth+1
    cum_scores: Sequence[float],         # _capim_scores.tolist(): cumulative log-prob, len N
    sigma_th: float,
) -> Tuple[List[TokenNode], List[List[int]], Dict[int, int]]:
    """Turn one drafted tree into (kept nodes, gated retrieve_indices, index remap).

    Returns:
        nodes:          TokenNode list for the KEPT sub-tree (all nodes if sigma=-inf),
                        in ascending draft-index (== depth) order.  ``parent_idx`` is the
                        GLOBAL position within THIS list (-1 for children of the root),
                        as the schema/driver require.  ``accepted`` starts False.
        edited_rows:    retrieve_indices with each path truncated at its first pruned
                        node (identical object semantics to native -1 padding).  Equals
                        ``retrieve_rows`` unchanged when sigma = -inf.
        new_index_of:   {draft_index -> position in ``nodes``} for the kept nodes, used
                        by ``mark_accepted`` to flag the accepted prefix.
    """
    n = len(cum_scores)                       # number of drafted (non-root) nodes
    parent_of = _parents_from_tree_mask(tree_mask_rows, n)

    if sigma_th == NEG_INF:
        kept_draft = list(range(1, n + 1))    # every node
        edited_rows = [list(r) for r in retrieve_rows]   # unchanged (copy for safety)
    else:
        kept_draft = [d for d in range(1, n + 1) if cum_scores[d - 1] >= sigma_th]
        edited_rows = invalidate_paths(retrieve_rows, set(kept_draft))

    new_index_of = {d: k for k, d in enumerate(kept_draft)}

    nodes: List[TokenNode] = []
    depth_layer: Dict[int, int] = {}
    for d in kept_draft:                       # ascending -> parents precede children
        cum = float(cum_scores[d - 1])
        p_draft = parent_of[d]                 # 0 == root, else another kept draft node
        if p_draft == 0:
            parent_idx = -1
            parent_cum = 0.0
        else:
            parent_idx = new_index_of[p_draft]  # ancestor-closed -> parent is kept
            parent_cum = float(cum_scores[p_draft - 1])
        depth = int(tree_pos_ids[d]) - 1        # tree_position_ids: root=0, depth-0 draft=1
        layer_idx = depth_layer.get(depth, 0)
        depth_layer[depth] = layer_idx + 1
        nodes.append(TokenNode(
            depth=depth,
            token_id=int(draft_token_ids[d]),
            log_prob=cum - parent_cum,
            cumulative_log_prob=cum,
            parent_idx=parent_idx,
            accepted=False,
            layer_idx=layer_idx,
        ))
    return nodes, edited_rows, new_index_of


def mark_accepted(
    nodes: List[TokenNode],
    edited_rows: Sequence[Sequence[int]],
    new_index_of: Dict[int, int],
    best_candidate: int,
    accept_length: int,
) -> int:
    """Flag the accepted prefix on the KEPT nodes; return accepted DRAFT-token count.

    Walks the winning path (``edited_rows[best_candidate]``) over the first
    ``accept_length`` draft positions (column 0 is the always-skipped root, padding is
    -1) and sets ``accepted=True`` on the corresponding kept nodes.  Because the path is
    the *gated* one, every accepted index is a kept node.
    """
    if not (0 <= best_candidate < len(edited_rows)):
        return 0
    path = list(edited_rows[best_candidate])
    accepted = 0
    for j in range(1, accept_length + 1):       # skip column 0 (root); include accept_length
        if j >= len(path):
            break
        idx = path[j]
        if idx > 0:                             # skip root (0) and padding (-1)
            pos = new_index_of.get(idx)
            if pos is not None:
                nodes[pos].accepted = True
                accepted += 1
    return accepted


# ============================ model load + collector ==========================

def load_eagle_model(
    base_model_path: str,
    ea_model_path: str,
    *,
    load_in_8bit: bool = True,
    load_in_4bit: bool = False,
    total_token: int = 60,
    depth: int = 5,
    top_k: int = 10,
):
    """Load an 8-bit ``EaModel`` (EAGLE-2 on Vicuna-7B) for collection.

    Applies the ``_init_weights`` quant guard as a runtime CLASS PATCH (no submodule
    edit): stock ``LlamaPreTrainedModel._init_weights`` calls ``.normal_()`` on every
    weight during ``from_pretrained``, which crashes on bitsandbytes-packed (non-float)
    tensors -- and re-initialising an already-loaded quantized weight would be wrong.  We
    wrap it to skip non-floating-point weights before delegating.
    """
    import torch  # local: keeps this module CPU/import-clean for the dry-run
    from transformers import BitsAndBytesConfig

    from common.sd_repos import add_eagle_to_path
    add_eagle_to_path()
    from eagle.model import modeling_llama_kv as mlk
    from eagle.model.ea_model import EaModel

    _orig_init = mlk.LlamaPreTrainedModel._init_weights

    @functools.wraps(_orig_init)
    def _guarded_init_weights(self, module):
        w = getattr(module, "weight", None)
        if w is not None and not w.data.is_floating_point():
            return
        return _orig_init(self, module)

    mlk.LlamaPreTrainedModel._init_weights = _guarded_init_weights

    kwargs: Dict[str, Any] = dict(
        base_model_path=base_model_path,
        ea_model_path=ea_model_path,
        total_token=total_token,
        depth=depth,
        top_k=top_k,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map={"": 0},       # pin whole model to one GPU (vendored KV code isn't MP-safe)
        use_safetensors=True,
        use_eagle3=False,          # EAGLE-2
    )
    if load_in_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4",
        )
    elif load_in_8bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)

    model = EaModel.from_pretrained(**kwargs)
    model.eval()
    return model, model.get_tokenizer()


class Collector:
    """Attach to an ``EaModel`` to record a gated (or ungated) EAGLE-2 ``Trace``.

    Usage (on the GPU box):
        col = Collector(sigma_th=-1.5, dataset="alpaca", prompt_id=0)
        col.attach(ea_model)
        ea_model.eagenerate(input_ids, temperature=0, top_p=0, top_k=0, max_new_tokens=200)
        steps = col.detach()          # list[DecodeStep] for this prompt
    """

    def __init__(self, sigma_th: float, dataset: str = "unknown", prompt_id: int = 0):
        self.sigma_th = sigma_th
        self.dataset = dataset
        self.prompt_id = prompt_id
        self._steps: List[DecodeStep] = []
        self._step_id = 0
        self._pending: Optional[Dict[str, Any]] = None
        # restore handles
        self._ea_layer = None
        self._orig_topk = None
        self._ea_model_mod = None
        self._orig_eval = None

    # -- attach / detach ---------------------------------------------------
    def attach(self, ea_model: Any) -> None:
        from capim_ctrl.eagle_topk import bind_capim_topk  # local: imports torch

        ea_layer = ea_model.ea_layer
        self._ea_layer = ea_layer
        bind_capim_topk(ea_layer)                  # install vendored topK (sets _capim_scores)
        self._orig_topk = ea_layer.topK_genrate
        ea_layer.topK_genrate = self._wrap_topk(self._orig_topk)

        # ea_model.py does `from .utils import *`, so its evaluate_posterior is a
        # separate name in that module -- patch it there (not utils).
        import eagle.model.ea_model as ea_model_mod
        self._ea_model_mod = ea_model_mod
        self._orig_eval = ea_model_mod.evaluate_posterior
        ea_model_mod.evaluate_posterior = self._wrap_eval(ea_model_mod.evaluate_posterior)

    def detach(self) -> List[DecodeStep]:
        if self._ea_layer is not None and self._orig_topk is not None:
            self._ea_layer.topK_genrate = self._orig_topk
        if self._ea_model_mod is not None and self._orig_eval is not None:
            self._ea_model_mod.evaluate_posterior = self._orig_eval
        return list(self._steps)

    # -- wrappers ----------------------------------------------------------
    def _wrap_topk(self, orig):
        col = self

        @functools.wraps(orig)
        def wrapper(hidden_states, input_ids, head, logits_processor):
            import torch
            result = orig(hidden_states, input_ids, head, logits_processor)
            draft_tokens, retrieve_indices, tree_mask, tree_position_ids = result

            draft_token_ids = draft_tokens[0].tolist()
            retrieve_rows = retrieve_indices.tolist()
            tree_mask_rows = tree_mask[0, 0].bool().tolist()
            tree_pos = tree_position_ids.tolist()
            cum_scores = col._ea_layer._capim_scores.tolist()
            assert len(cum_scores) == len(draft_token_ids) - 1, (
                f"_capim_scores len {len(cum_scores)} != draft_tokens-1 "
                f"{len(draft_token_ids) - 1}"
            )

            nodes, edited_rows, new_index_of = record_gated_step(
                draft_token_ids, retrieve_rows, tree_mask_rows, tree_pos, cum_scores,
                col.sigma_th,
            )
            col._pending = dict(
                nodes=nodes,
                edited_rows=edited_rows,
                new_index_of=new_index_of,
                context_length=int(input_ids.shape[1]),
                sample_token_id=int(input_ids[0, -1].item()),
            )

            if edited_rows == retrieve_rows:
                return result                     # ungated (or nothing pruned): untouched
            ri_new = torch.tensor(
                edited_rows, dtype=retrieve_indices.dtype, device=retrieve_indices.device,
            )
            return draft_tokens, ri_new, tree_mask, tree_position_ids

        return wrapper

    def _wrap_eval(self, orig):
        col = self

        @functools.wraps(orig)
        def wrapper(logits, candidates, logits_processor):
            best_candidate, accept_length, sample_p = orig(logits, candidates, logits_processor)
            p = col._pending
            if p is not None:
                mark_accepted(
                    p["nodes"], p["edited_rows"], p["new_index_of"],
                    int(best_candidate), int(accept_length),
                )
                col._steps.append(DecodeStep(
                    step_id=col._step_id,
                    context_length=p["context_length"],
                    nodes=p["nodes"],
                    accepted_length=int(accept_length),   # raw draft accepts; driver adds bonus
                    dataset=col.dataset,
                    prompt_id=col.prompt_id,
                    sample_token_id=p["sample_token_id"],
                ))
                col._step_id += 1
                col._pending = None
            return best_candidate, accept_length, sample_p

        return wrapper


def collect(
    ea_model: Any,
    tokenizer: Any,
    prompts: Sequence[str],
    *,
    dataset: str,
    sigma_th: float,
    max_new_tokens: int = 200,
    model_name: str = "Vicuna-7B-v1.3",
    draft_head: str = "yuhuili/EAGLE-Vicuna-7B-v1.3",
    temperature: float = 0.0,
) -> Trace:
    """Run gated EAGLE-2 over ``prompts`` and return a single ``Trace``.

    One ``Collector`` per prompt (fresh step ids / pending), steps concatenated.  Greedy
    (temperature=0) by default -- the in-loop gate is provably exact there.
    """
    import time
    import torch

    print(f"[eagle] {dataset}: {len(prompts)} prompts  "
          f"(sigma_th={sigma_th}, max_new_tokens={max_new_tokens})", flush=True)
    t0 = time.time()
    all_steps: List[DecodeStep] = []
    for pid, prompt in enumerate(prompts):
        col = Collector(sigma_th=sigma_th, dataset=dataset, prompt_id=pid)
        col.attach(ea_model)
        try:
            inputs = tokenizer(prompt, return_tensors="pt").to(ea_model.base_model.device)
            with torch.no_grad():
                ea_model.eagenerate(
                    inputs["input_ids"], temperature=temperature, top_p=0, top_k=0,
                    max_new_tokens=max_new_tokens,
                )
        finally:
            sp = col.detach()
        all_steps.extend(sp)
        mu  = sum(s.tree_size      for s in sp) / len(sp) if sp else 0.0
        acc = sum(s.accepted_length for s in sp) / len(sp) if sp else 0.0
        print(f"  [{pid + 1:>3}/{len(prompts)}] {len(sp):>4} steps  "
              f"μ={mu:4.1f}  accept={acc:4.2f}  ({time.time() - t0:6.1f}s)", flush=True)

    trace = Trace(
        steps=all_steps,
        model=model_name,
        sd_method="eagle2",
        metadata=dict(
            dataset=dataset, sigma_th=sigma_th, temperature=temperature,
            draft_head=draft_head, n_prompts=len(prompts), max_new_tokens=max_new_tokens,
            gated=(sigma_th != NEG_INF),
        ),
    )
    trace.compute_summary()
    return trace
