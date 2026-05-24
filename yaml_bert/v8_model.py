"""v8 Phase 0 model: encoder + tree aggregator + atomic Token Head.

The Token Head predicts atomic key targets (vocab ~1000) instead of compound
trigrams. It conditions on a concatenation of:
  - the per-token hidden state h_i
  - the document vector doc_vec
  - the immediate parent subtree vector s_parent(i)

This carries kind context through doc_vec instead of through compound target
vocabulary.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from yaml_bert.aggregator import TreeAggregator
from yaml_bert.config import YamlBertConfig
from yaml_bert.embedding import YamlBertEmbedding


class V8Model(nn.Module):
    """v8 Phase 0: encoder + aggregator + atomic Token Head.

    No reconstruction head, no compound output heads.
    """

    def __init__(
        self,
        config: YamlBertConfig,
        embedding: YamlBertEmbedding,
        atomic_vocab_size: int,
    ) -> None:
        super().__init__()
        self.embedding = embedding
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.num_heads,
            dim_feedforward=config.d_ff,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=config.num_layers,
        )
        self.aggregator = TreeAggregator(d_model=config.d_model)
        # Token Head input: [h_i ; doc_vec ; s_parent] = 3 * d_model
        self.token_head = nn.Linear(3 * config.d_model, atomic_vocab_size)

    def forward(
        self,
        token_ids: torch.Tensor,
        node_types: torch.Tensor,
        depths: torch.Tensor,
        sibling_indices: torch.Tensor,
        batch_info: list[dict],
        padding_mask: torch.Tensor | None = None,
        *,
        parent_of_tensor: torch.Tensor | None = None,
        top_level_key_mask: torch.Tensor | None = None,
        edges_by_depth: dict[int, torch.Tensor] | None = None,
        parents_by_depth: dict[int, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (logits, doc_vec). Vectorized path activates when the
        precomputed tensor kwargs are provided (always true at training)."""
        x = self.embedding(token_ids, node_types, depths, sibling_indices)
        x = self.encoder(x, src_key_padding_mask=padding_mask)

        # Aggregator: forwards through to its own vectorized/reference dispatch.
        subtree_vecs, doc_vec = self.aggregator(
            x, batch_info,
            parent_of_tensor=parent_of_tensor,
            top_level_key_mask=top_level_key_mask,
            edges_by_depth=edges_by_depth,
            parents_by_depth=parents_by_depth,
        )

        b, n, d = x.shape

        if parent_of_tensor is not None:
            # Vectorized s_parent. parent_of_tensor being set implies all four
            # precompute kwargs were provided (aggregator enforces all-or-none).
            safe_parent = parent_of_tensor.clamp(min=0)  # (B, N)
            s_parent = torch.gather(
                subtree_vecs, dim=1,
                index=safe_parent.unsqueeze(-1).expand(-1, -1, d),
            )  # (B, N, d)
            no_parent_mask = (parent_of_tensor == -1).unsqueeze(-1)  # (B, N, 1)
            s_parent = torch.where(
                no_parent_mask, doc_vec.unsqueeze(1), s_parent,
            )
        else:
            # Reference path: per-doc Python loop (kept for tests / fallback).
            s_parent = torch.zeros_like(x)
            for doc_idx in range(b):
                parent_of = batch_info[doc_idx]["parent_of"]
                for i in range(min(n, len(parent_of))):
                    p = parent_of[i]
                    if p >= 0:
                        s_parent[doc_idx, i] = subtree_vecs[doc_idx, p]
                    else:
                        s_parent[doc_idx, i] = doc_vec[doc_idx]

        doc_vec_broadcast = doc_vec.unsqueeze(1).expand(b, n, d)
        head_input = torch.cat([x, doc_vec_broadcast, s_parent], dim=-1)
        logits = self.token_head(head_input)

        return logits, doc_vec
