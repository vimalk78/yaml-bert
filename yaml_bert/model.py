from __future__ import annotations

import torch
import torch.nn as nn

from yaml_bert.config import YamlBertConfig
from yaml_bert.embedding import YamlBertEmbedding


class YamlBertModel(nn.Module):
    """YAML-BERT: Transformer encoder with tree positional encoding.

    Takes linearized YAML node sequences, applies tree-aware embeddings,
    processes through a transformer encoder, and predicts masked keys.
    """

    def __init__(
        self,
        config: YamlBertConfig,
        embedding: YamlBertEmbedding,
        key_vocab_size: int,
    ) -> None:
        super().__init__()

        self.embedding: YamlBertEmbedding = embedding

        encoder_layer: nn.TransformerEncoderLayer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.num_heads,
            dim_feedforward=config.d_ff,
            batch_first=True,
        )
        self.encoder: nn.TransformerEncoder = nn.TransformerEncoder(
            encoder_layer, num_layers=config.num_layers
        )

        self.key_prediction_head: nn.Linear = nn.Linear(config.d_model, key_vocab_size)
        self.loss_fn: nn.CrossEntropyLoss = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(
        self,
        token_ids: torch.Tensor,
        node_types: torch.Tensor,
        depths: torch.Tensor,
        sibling_indices: torch.Tensor,
        parent_key_ids: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        kind_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x: torch.Tensor = self.embedding(
            token_ids, node_types, depths, sibling_indices, parent_key_ids,
            kind_ids=kind_ids,
        )
        x = self.encoder(x, src_key_padding_mask=padding_mask)
        key_logits: torch.Tensor = self.key_prediction_head(x)
        return key_logits

    def compute_loss(
        self,
        key_logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        return self.loss_fn(
            key_logits.view(-1, key_logits.size(-1)),
            labels.view(-1),
        )

    @torch.no_grad()
    def get_attention_weights(
        self,
        token_ids: torch.Tensor,
        node_types: torch.Tensor,
        depths: torch.Tensor,
        sibling_indices: torch.Tensor,
        parent_key_ids: torch.Tensor,
        kind_ids: torch.Tensor | None = None,
    ) -> list[torch.Tensor]:
        """Extract attention weights from all layers.

        Returns list of tensors, one per layer,
        each of shape (batch_size, num_heads, seq_len, seq_len).
        """
        self.eval()
        x: torch.Tensor = self.embedding(
            token_ids, node_types, depths, sibling_indices, parent_key_ids,
            kind_ids=kind_ids,
        )

        attention_weights: list[torch.Tensor] = []
        for layer in self.encoder.layers:
            # PyTorch TransformerEncoderLayer uses self_attn (MultiheadAttention)
            # Call it directly with need_weights=True
            attn_output, attn_weight = layer.self_attn(
                x, x, x, need_weights=True, average_attn_weights=False
            )
            attention_weights.append(attn_weight)

            # Run the rest of the layer to get input for next layer
            x = layer(x)

        return attention_weights
