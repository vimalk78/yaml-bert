from __future__ import annotations

import torch
import torch.nn as nn

from yaml_bert.config import TreePosVariant, YamlBertConfig


class YamlBertEmbedding(nn.Module):
    """v9 embedding layer: single subword table + tree positional encoding.

    Produces input vectors by summing:
    - Subword embedding (looked up by token_id; same table for KEY and VALUE
      positions — what they ARE is signalled separately via node_type_emb)
    - Tree positional encoding (composition depends on config.tree_pos_variant)
    """

    def __init__(
        self,
        config: YamlBertConfig,
        subword_vocab_size: int,
    ) -> None:
        super().__init__()
        d: int = config.d_model
        variant: TreePosVariant = config.tree_pos_variant
        self.variant: TreePosVariant = variant

        self.subword_embedding: nn.Embedding = nn.Embedding(subword_vocab_size, d)
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
        token_emb = self.subword_embedding(token_ids)

        tree_pos = self.node_type_embedding(node_types)
        if self.depth_embedding is not None:
            tree_pos = tree_pos + self.depth_embedding(depths)
        if self.sibling_embedding is not None:
            tree_pos = tree_pos + self.sibling_embedding(sibling_indices)
        if self.pos_embedding is not None:
            seq_len: int = token_ids.size(1)
            max_pos: int = self.pos_embedding.num_embeddings
            positions = (
                torch.arange(seq_len, device=token_ids.device)
                .clamp(max=max_pos - 1)
                .unsqueeze(0)
                .expand(token_ids.size(0), seq_len)
            )
            tree_pos = tree_pos + self.pos_embedding(positions)

        return self.layer_norm(token_emb + tree_pos)
