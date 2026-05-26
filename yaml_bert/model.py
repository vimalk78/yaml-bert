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
        logical_ids: torch.Tensor,
        n_logical_per_doc: torch.Tensor,
        parent_of_tensor: torch.Tensor | None = None,
        top_level_key_mask: torch.Tensor | None = None,
        edges_by_depth: dict[int, torch.Tensor] | None = None,
        parents_by_depth: dict[int, torch.Tensor] | None = None,
        subtree_mask: torch.Tensor | None = None,
        subtree_roots_flat: torch.Tensor | None = None,
    ) -> tuple:
        """Returns (logits, doc_vec) or (logits, doc_vec, recon_logits).

        v9: token_ids/node_types/depths/sibling_indices/logical_ids/padding_mask
        are SUBWORD-level (B, N_sub). atomic_labels and the aggregator output
        are LOGICAL-level (B, L_max). The Token Head consumes per-logical-node
        pooled hidden state.
        """
        from yaml_bert.aggregator import _pool_subwords

        x = self.embedding(token_ids, node_types, depths, sibling_indices)
        x = self.encoder(x, src_key_padding_mask=padding_mask)

        # Aggregator pools internally; (subtree_vecs, doc_vec) are logical-level.
        subtree_vecs, doc_vec = self.aggregator(
            x, batch_info,
            logical_ids=logical_ids,
            n_logical_per_doc=n_logical_per_doc,
            parent_of_tensor=parent_of_tensor,
            top_level_key_mask=top_level_key_mask,
            edges_by_depth=edges_by_depth,
            parents_by_depth=parents_by_depth,
            subtree_mask=subtree_mask,
        )

        # Re-pool subword hiddens to logical level for the Token Head input.
        # (One extra index_add call; the plan flags this as a follow-up.)
        h_logical = _pool_subwords(x, logical_ids, n_logical_per_doc)
        b, L_max, d = h_logical.shape

        if parent_of_tensor is not None:
            safe_parent = parent_of_tensor.clamp(min=0)
            s_parent = torch.gather(
                subtree_vecs, dim=1,
                index=safe_parent.unsqueeze(-1).expand(-1, -1, d),
            )
            no_parent_mask = (parent_of_tensor == -1).unsqueeze(-1)
            s_parent = torch.where(
                no_parent_mask, doc_vec.unsqueeze(1), s_parent,
            )
        else:
            s_parent = torch.zeros_like(h_logical)
            for doc_idx in range(b):
                parent_of = batch_info[doc_idx]["parent_of"]
                for i in range(min(L_max, len(parent_of))):
                    p = parent_of[i]
                    if p >= 0:
                        s_parent[doc_idx, i] = subtree_vecs[doc_idx, p]
                    else:
                        s_parent[doc_idx, i] = doc_vec[doc_idx]

        doc_vec_broadcast = doc_vec.unsqueeze(1).expand(b, L_max, d)
        head_input = torch.cat([h_logical, doc_vec_broadcast, s_parent], dim=-1)
        logits = self.token_head(head_input)  # (B, L_max, atomic_vocab_size)

        if subtree_roots_flat is not None and subtree_roots_flat.size(0) > 0:
            batch_idx_per_root = subtree_roots_flat[:, 0]
            root_pos_per_root = subtree_roots_flat[:, 1]
            doc_vec_per_root = doc_vec[batch_idx_per_root]
            # Root positions live in LOGICAL coords. Look up depth from batch_info;
            # sibling: find any subword with logical_id == root_pos, use its sibling.
            root_depths_list = [
                batch_info[bi]["depth_of"][rp]
                for bi, rp in zip(batch_idx_per_root.tolist(), root_pos_per_root.tolist())
            ]
            root_siblings_list = []
            for bi, rp in zip(batch_idx_per_root.tolist(), root_pos_per_root.tolist()):
                positions = (logical_ids[bi] == rp).nonzero(as_tuple=True)[0]
                if len(positions) > 0:
                    root_siblings_list.append(int(sibling_indices[bi, positions[0]].item()))
                else:
                    # Shouldn't happen: a recon root must have at least one subword.
                    root_siblings_list.append(0)
            root_depths = torch.tensor(root_depths_list, device=depths.device, dtype=torch.long)
            root_siblings = torch.tensor(root_siblings_list, device=depths.device, dtype=torch.long)

            depth_e = self.embedding.depth_embedding(root_depths)
            sibling_e = self.embedding.sibling_embedding(root_siblings)
            pos_emb_per_root = torch.cat([depth_e, sibling_e], dim=-1)

            recon_logits = self.recon_head(doc_vec_per_root, pos_emb_per_root)
            return logits, doc_vec, recon_logits

        return logits, doc_vec
