"""
Vendored EAGLE ``topK_genrate`` (draft-tree builder) for CAPIM's sigma-gate collector.

This is a VERBATIM copy of ``ea_layer.topK_genrate`` from the pinned EAGLE submodule
(``sd_repos/EAGLE/eagle/model/cnets1.py`` @ cb7e084), with ONE addition: a single
marked ``_capim_scores`` stash line that saves the per-node cumulative log-prob of the
selected tree so the collector can read it back and apply the confidence gate.  Nothing
else is changed -- ``tests/test_eagle_topk_parity.py`` ast-diffs this function against
the live submodule (stripping only the marked CAPIM block) and fails on any drift.

Why a vendored copy instead of editing the submodule
----------------------------------------------------
The upstream repos are unmodified git submodules (a fresh ``--recurse-submodules``
clone reproduces the exact code).  We install our variant at runtime with
``bind_capim_topk(ea_layer)``, which rebinds the method on the live draft layer -- no
source edit, no 50-file fork.  ``self`` is the EAGLE draft ``Model`` (``ea_layer``), so
every ``self.*`` attribute/method (``total_tokens``, ``depth``, ``top_k``, ``reset``,
``stable_kv``, ``logsoftmax``, ``tree_mask_init``, ``embed_tokens``, ``position_ids``,
``self(...)`` forward) resolves against that instance exactly as the original method did.

The stash (``self._capim_scores``) holds one cumulative log-prob per SELECTED draft
node, in ``draft_tokens[1:]`` order (root excluded), so ``len(_capim_scores) ==
draft_tokens.shape[1] - 1``.  The collector maps score index ``i`` to node index
``i + 1`` (node 0 is the always-kept root) when building the keep set for
``common.gating.invalidate_paths``.
"""

from __future__ import annotations

import types

import torch


@torch.no_grad()
def topK_genrate(self, hidden_states, input_ids, head, logits_processor):

    input_ids = input_ids.to(hidden_states.device)
    total_tokens = self.total_tokens
    depth = self.depth
    top_k = self.top_k

    sample_token = input_ids[:, -1]

    scores_list = []
    parents_list = []
    ss_token = []

    input_ids = input_ids[:, 1:]
    input_ids = input_ids.to(hidden_states.device)

    len_posi = input_ids.shape[1]
    self.reset()

    # with Timer("draft many"):
    if hasattr(self, "stable_kv") and self.stable_kv is not None:
        kv_len = self.stable_kv[0][0].shape[2]
        out_hidden, past_key_values = self(hidden_states, input_ids=input_ids[:, kv_len:],
                                           past_key_values=self.stable_kv, use_cache=True)
    else:
        out_hidden, past_key_values = self(hidden_states, input_ids=input_ids, use_cache=True)
    self.stable_kv = past_key_values
    last_hidden = out_hidden[:, -1]

    last_headout = head(last_hidden)

    last_p = self.logsoftmax(last_headout)
    top = torch.topk(last_p, top_k, dim=-1)
    topk_index, topk_p = top.indices, top.values
    scores = topk_p[0]
    scores_list.append(scores[None])
    parents_list.append(torch.zeros(1, dtype=torch.long, device=scores.device))
    ss_token.append(topk_index)
    input_ids = topk_index
    input_hidden = last_hidden[None].repeat(1, top_k, 1)
    tree_mask = self.tree_mask_init
    topk_cs_index = torch.arange(top_k, device=self.embed_tokens.weight.device)

    # 4
    for i in range(depth):
        self.tree_mask = tree_mask
        position_ids = len_posi + self.position_ids
        # with Timer("draft one"):
        out_hidden, past_key_values = self(input_hidden, input_ids=input_ids, past_key_values=past_key_values,
                                           position_ids=position_ids, use_cache=True)
        len_posi += 1

        # with Timer("sort1"):
        bias1 = top_k if i > 0 else 0
        bias2 = max(0, i - 1)
        bias = 1 + top_k ** 2 * bias2 + bias1
        parents = (topk_cs_index + bias)
        parents_list.append(parents)

        last_headout = head(out_hidden[0])
        last_p = self.logsoftmax(last_headout)

        top = torch.topk(last_p, top_k, dim=-1)
        topk_index, topk_p = top.indices, top.values

        cu_scores = topk_p + scores[:, None]

        topk_cs = torch.topk(cu_scores.view(-1), top_k, dim=-1)
        topk_cs_index, topk_cs_p = topk_cs.indices, topk_cs.values
        scores = topk_cs_p

        out_ids = topk_cs_index // top_k
        input_hidden = out_hidden[:, out_ids]

        input_ids = topk_index.view(-1)[topk_cs_index][None]

        ss_token.append(topk_index)
        scores_list.append(cu_scores)
        tree_mask = torch.cat((tree_mask[:, :, out_ids], self.tree_mask_init), dim=3)



    scores_list = torch.cat(scores_list, dim=0).view(-1)
    ss_token_list = torch.cat(ss_token, dim=0).view(-1)
    top_scores = torch.topk(scores_list, total_tokens, dim=-1)
    top_scores_index = top_scores.indices
    top_scores_index = torch.sort(top_scores_index).values

    draft_tokens = ss_token_list[top_scores_index]
    draft_tokens = torch.cat((sample_token, draft_tokens), dim=0)

    draft_parents = torch.cat(parents_list, dim=0)[top_scores_index // top_k].long()
    mask_index = torch.searchsorted(top_scores_index, draft_parents - 1, right=False)
    # mask_index[(top_scores_index[mask_index]!=draft_parents - 1)]=-1
    mask_index[draft_parents == 0] = -1
    mask_index = mask_index + 1
    mask_index_list = mask_index.tolist()
    # with Timer("mask"):
    tree_mask = torch.eye(total_tokens + 1).bool()
    tree_mask[:, 0] = True
    for i in range(total_tokens):
        tree_mask[i + 1].add_(tree_mask[mask_index_list[i]])


    tree_position_ids = torch.sum(tree_mask, dim=1) - 1

    tree_mask = tree_mask.float()[None, None]
    draft_tokens = draft_tokens[None]

    # ==== CAPIM STASH BEGIN (the ONLY addition vs upstream cnets1.py) ====
    # Per-node cumulative log-prob for the selected tree, aligned to
    # draft_tokens[1:] order (both index the sorted `top_scores_index`).
    # Read back by the collector's sigma-gate; len == draft_tokens - 1.
    self._capim_scores = scores_list[top_scores_index].detach().cpu()
    # ==== CAPIM STASH END ====
    del parents_list, scores_list, ss_token, ss_token_list, draft_parents

    # with Timer("retrieve"):

    max_depth = torch.max(tree_position_ids) + 1
    noleaf_index = torch.unique(mask_index).tolist()
    noleaf_num = len(noleaf_index) - 1
    leaf_num = total_tokens - noleaf_num

    retrieve_indices = torch.zeros(leaf_num, max_depth.item(), dtype=torch.long) - 1
    retrieve_indices = retrieve_indices.tolist()

    rid = 0
    position_ids_list = tree_position_ids.tolist()

    for i in range(total_tokens + 1):
        if i not in noleaf_index:
            cid = i
            depth = position_ids_list[i]
            for j in reversed(range(depth + 1)):
                retrieve_indices[rid][j] = cid
                cid = mask_index_list[cid - 1]
            rid += 1

    if logits_processor is not None:
        maxitem = total_tokens + 5

        def custom_sort(lst):
            # sort_keys=[len(list)]
            sort_keys = []
            for i in range(len(lst)):
                sort_keys.append(lst[i] if lst[i] >= 0 else maxitem)
            return sort_keys

        retrieve_indices = sorted(retrieve_indices, key=custom_sort)

    retrieve_indices = torch.tensor(retrieve_indices, dtype=torch.long)
    del mask_index, mask_index_list, noleaf_index, noleaf_num, leaf_num, max_depth, rid
    tree_position_ids = tree_position_ids.to(hidden_states.device)

    return draft_tokens, retrieve_indices, tree_mask, tree_position_ids


def bind_capim_topk(ea_layer) -> None:
    """Install the vendored ``topK_genrate`` (with the ``_capim_scores`` stash) onto a
    live EAGLE draft layer, replacing the upstream method for the rest of the run.

    Idempotent: re-binding is harmless.  Call once at collector attach, after the model
    is loaded, before generation.
    """
    ea_layer.topK_genrate = types.MethodType(topK_genrate, ea_layer)
