from __future__ import annotations

import torch
import torch.nn as nn

from yaml_bert.config import YamlBertConfig
from yaml_bert.types import NodeType


class YamlBertEmbedding(nn.Module):
    """Embedding layer with tree positional encoding for YAML-BERT.

    Produces input vectors by summing:
    - Token embedding (key_embedding or value_embedding, routed by node_type)
    - Tree positional encoding (depth + sibling + node_type + parent_key)
    """

    def __init__(
        self,
        config: YamlBertConfig,
        key_vocab_size: int,
        value_vocab_size: int,
        kind_vocab_size: int | None = None,
    ) -> None:
        super().__init__()

        d: int = config.d_model

        # Token embeddings — separate tables for keys and values
        self.key_embedding: nn.Embedding = nn.Embedding(key_vocab_size, d)
        self.value_embedding: nn.Embedding = nn.Embedding(value_vocab_size, d)

        # Tree positional encoding components
        self.depth_embedding: nn.Embedding = nn.Embedding(config.max_depth, d)
        self.sibling_embedding: nn.Embedding = nn.Embedding(config.max_sibling, d)
        self.node_type_embedding: nn.Embedding = nn.Embedding(4, d)
        self.parent_key_embedding: nn.Embedding = nn.Embedding(key_vocab_size, d)

        self.kind_embedding: nn.Embedding | None = None
        if kind_vocab_size is not None:
            self.kind_embedding = nn.Embedding(kind_vocab_size, d)

        self.layer_norm: nn.LayerNorm = nn.LayerNorm(d)

    def forward(
        self,
        token_ids: torch.Tensor,
        node_types: torch.Tensor,
        depths: torch.Tensor,
        sibling_indices: torch.Tensor,
        parent_key_ids: torch.Tensor,
        kind_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Route token embedding based on node_type
        # KEY=0, LIST_KEY=2 use key_embedding; VALUE=1, LIST_VALUE=3 use value_embedding
        is_key: torch.Tensor = (node_types == 0) | (node_types == 2)

        key_vocab_size: int = self.key_embedding.num_embeddings
        val_vocab_size: int = self.value_embedding.num_embeddings
        key_emb: torch.Tensor = self.key_embedding(token_ids.clamp(0, key_vocab_size - 1))
        val_emb: torch.Tensor = self.value_embedding(token_ids.clamp(0, val_vocab_size - 1))
        token_emb: torch.Tensor = torch.where(
            is_key.unsqueeze(-1), key_emb, val_emb
        )

        # Tree positional encoding
        tree_pos: torch.Tensor = (
            self.depth_embedding(depths)
            + self.sibling_embedding(sibling_indices)
            + self.node_type_embedding(node_types)
            + self.parent_key_embedding(parent_key_ids)
        )

        if self.kind_embedding is not None and kind_ids is not None:
            tree_pos = tree_pos + self.kind_embedding(kind_ids)

        return self.layer_norm(token_emb + tree_pos)
