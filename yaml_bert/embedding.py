from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from yaml_bert.config import YamlBertConfig


class YamlBertEmbedding(nn.Module):
    """Embedding layer supporting multiple positional encoding strategies for ablation."""

    def __init__(
        self,
        config: YamlBertConfig,
        key_vocab_size: int,
        value_vocab_size: int,
    ) -> None:
        super().__init__()
        d: int = config.d_model
        self.pos_encoding = config.pos_encoding
        self.key_embedding: nn.Embedding = nn.Embedding(key_vocab_size, d)
        self.value_embedding: nn.Embedding = nn.Embedding(value_vocab_size, d)

        if self.pos_encoding in ("full_tree", "depth_only", "sibling_only"):
            self.depth_embedding: nn.Embedding = nn.Embedding(config.max_depth, d)
            self.sibling_embedding: nn.Embedding = nn.Embedding(config.max_sibling, d)
            self.node_type_embedding: nn.Embedding = nn.Embedding(4, d)

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

        tree_pos = torch.zeros_like(token_emb)
        if self.pos_encoding in ("full_tree", "depth_only"):
            tree_pos = tree_pos + self.depth_embedding(depths)
        if self.pos_encoding in ("full_tree", "sibling_only"):
            tree_pos = tree_pos + self.sibling_embedding(sibling_indices)
        if self.pos_encoding == "full_tree":
            tree_pos = tree_pos + self.node_type_embedding(node_types)

        return self.layer_norm(token_emb + tree_pos)
