from __future__ import annotations

import torch
import torch.nn as nn

from yaml_bert.config import TreePosVariant, YamlBertConfig


class YamlBertEmbedding(nn.Module):
    """Embedding layer with tree positional encoding for YAML-BERT.

    Produces input vectors by summing:
    - Token embedding (key_embedding or value_embedding, routed by node_type)
    - Tree positional encoding (composition depends on config.tree_pos_variant)

    Kind and parent awareness come from the hybrid prediction targets, not input encoding.
    """

    def __init__(
        self,
        config: YamlBertConfig,
        key_vocab_size: int,
        value_vocab_size: int,
    ) -> None:
        super().__init__()
        d: int = config.d_model
        variant: TreePosVariant = config.tree_pos_variant
        self.variant: TreePosVariant = variant

        self.key_embedding: nn.Embedding = nn.Embedding(key_vocab_size, d)
        self.value_embedding: nn.Embedding = nn.Embedding(value_vocab_size, d)
        self.node_type_embedding: nn.Embedding = nn.Embedding(4, d)

        use_depth: bool = variant in (TreePosVariant.FULL, TreePosVariant.NO_SIBLING)
        use_sibling: bool = variant in (TreePosVariant.FULL, TreePosVariant.NO_DEPTH)
        use_seq_pos: bool = variant == TreePosVariant.SEQUENTIAL

        self.depth_embedding: nn.Embedding | None = (
            nn.Embedding(config.max_depth, d) if use_depth else None
        )
        self.sibling_embedding: nn.Embedding | None = (
            nn.Embedding(config.max_sibling, d) if use_sibling else None
        )
        self.pos_embedding: nn.Embedding | None = (
            nn.Embedding(config.max_seq_len, d) if use_seq_pos else None
        )

        self.layer_norm: nn.LayerNorm = nn.LayerNorm(d)

    def forward(
        self,
        token_ids: torch.Tensor,
        node_types: torch.Tensor,
        depths: torch.Tensor,
        sibling_indices: torch.Tensor,
    ) -> torch.Tensor:
        is_key: torch.Tensor = (node_types == 0) | (node_types == 2)
        key_vocab_size: int = self.key_embedding.num_embeddings
        val_vocab_size: int = self.value_embedding.num_embeddings
        key_emb: torch.Tensor = self.key_embedding(token_ids.clamp(0, key_vocab_size - 1))
        val_emb: torch.Tensor = self.value_embedding(token_ids.clamp(0, val_vocab_size - 1))
        token_emb: torch.Tensor = torch.where(is_key.unsqueeze(-1), key_emb, val_emb)

        tree_pos: torch.Tensor = self.node_type_embedding(node_types)
        if self.depth_embedding is not None:
            tree_pos = tree_pos + self.depth_embedding(depths)
        if self.sibling_embedding is not None:
            tree_pos = tree_pos + self.sibling_embedding(sibling_indices)
        if self.pos_embedding is not None:
            seq_len: int = token_ids.size(1)
            max_pos: int = self.pos_embedding.num_embeddings
            positions: torch.Tensor = (
                torch.arange(seq_len, device=token_ids.device)
                .clamp(max=max_pos - 1)
                .unsqueeze(0)
                .expand(token_ids.size(0), seq_len)
            )
            tree_pos = tree_pos + self.pos_embedding(positions)

        return self.layer_norm(token_emb + tree_pos)
