"""YAML-BERT model: encoder + tree aggregator + atomic Token Head.

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
from yaml_bert.reconstruction_head import ReconstructionHead


class YamlBertModel(nn.Module):
    """YAML-BERT encoder + aggregator + atomic Token Head.

    Predicts atomic key targets conditioned on doc_vec + parent subtree vec.
    Optionally trains a reconstruction head when recon_enabled=True.
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

        # Reconstruction Head: built unconditionally; only USED when caller
        # passes subtree_roots_flat. Cost when unused: ~0 (no forward call).
        # pos_emb = depth_embedding(root_depth) + sibling_embedding(root_sibling)
        # Each embedding maps into d_model, so d_pos = 2 * d_model.
        d_pos = 2 * config.d_model
        self.recon_head = ReconstructionHead(
            d_model=config.d_model,
            d_pos=d_pos,
            atomic_vocab_size=atomic_vocab_size,
        )

        # Recon path uses self.embedding.depth_embedding and sibling_embedding
        # directly — both must be present. Variants NO_DEPTH/NO_SIBLING/SEQUENTIAL
        # set those to None, which would crash forward. Surface the constraint
        # at init time with a clear error.
        if config.recon_enabled:
            if self.embedding.depth_embedding is None or \
               self.embedding.sibling_embedding is None:
                raise ValueError(
                    "YamlBertModel: recon_enabled=True requires tree_pos_variant=FULL "
                    f"(got variant where depth_embedding="
                    f"{self.embedding.depth_embedding} and sibling_embedding="
                    f"{self.embedding.sibling_embedding}). The reconstruction "
                    "head uses both depth and sibling embeddings for the root "
                    "position embedding."
                )

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
        subtree_mask: torch.Tensor | None = None,
        subtree_roots_flat: torch.Tensor | None = None,
    ) -> tuple:
        """Returns (logits, doc_vec) or (logits, doc_vec, recon_logits).

        recon_logits only returned when subtree_roots_flat is provided AND
        has at least one row."""
        x = self.embedding(token_ids, node_types, depths, sibling_indices)
        x = self.encoder(x, src_key_padding_mask=padding_mask)

        # Aggregator: forwards through to its own vectorized/reference dispatch.
        subtree_vecs, doc_vec = self.aggregator(
            x, batch_info,
            parent_of_tensor=parent_of_tensor,
            top_level_key_mask=top_level_key_mask,
            edges_by_depth=edges_by_depth,
            parents_by_depth=parents_by_depth,
            subtree_mask=subtree_mask,
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

        # Reconstruction path: only if caller provided subtree roots
        if subtree_roots_flat is not None and subtree_roots_flat.size(0) > 0:
            # subtree_roots_flat: (M, 2) of [batch_idx, root_pos]
            batch_idx_per_root = subtree_roots_flat[:, 0]   # (M,)
            root_pos_per_root = subtree_roots_flat[:, 1]    # (M,)

            doc_vec_per_root = doc_vec[batch_idx_per_root]  # (M, d_model)

            # Build pos_emb_per_root from the same depth/sibling embedding params
            # already used in the embedding layer — no new parameters introduced.
            root_depths = depths[batch_idx_per_root, root_pos_per_root]     # (M,)
            root_siblings = sibling_indices[batch_idx_per_root, root_pos_per_root]  # (M,)
            depth_e = self.embedding.depth_embedding(root_depths)            # (M, d_model)
            sibling_e = self.embedding.sibling_embedding(root_siblings)      # (M, d_model)
            pos_emb_per_root = torch.cat([depth_e, sibling_e], dim=-1)      # (M, 2*d_model)

            recon_logits = self.recon_head(doc_vec_per_root, pos_emb_per_root)
            return logits, doc_vec, recon_logits

        return logits, doc_vec
