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
        kind_vocab_size: int | None = None,
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
        self.kind_classifier: nn.Linear = nn.Linear(config.d_model, kind_vocab_size or 1)
        self.parent_key_classifier: nn.Linear = nn.Linear(config.d_model, key_vocab_size)

        self.key_loss_fn: nn.CrossEntropyLoss = nn.CrossEntropyLoss(ignore_index=-100)
        self.aux_loss_fn: nn.CrossEntropyLoss = nn.CrossEntropyLoss()

    def forward(
        self,
        token_ids: torch.Tensor,
        node_types: torch.Tensor,
        depths: torch.Tensor,
        sibling_indices: torch.Tensor,
        parent_key_ids: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        kind_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x: torch.Tensor = self.embedding(
            token_ids, node_types, depths, sibling_indices, parent_key_ids,
            kind_ids=kind_ids,
        )
        x = self.encoder(x, src_key_padding_mask=padding_mask)

        key_logits: torch.Tensor = self.key_prediction_head(x)
        kind_logits: torch.Tensor = self.kind_classifier(x)
        parent_logits: torch.Tensor = self.parent_key_classifier(x)

        return key_logits, kind_logits, parent_logits

    def compute_loss(
        self,
        key_logits: torch.Tensor,
        labels: torch.Tensor,
        kind_logits: torch.Tensor | None = None,
        kind_labels: torch.Tensor | None = None,
        parent_logits: torch.Tensor | None = None,
        parent_labels: torch.Tensor | None = None,
        alpha: float = 0.0,
        beta: float = 0.0,
    ) -> torch.Tensor:
        key_loss: torch.Tensor = self.key_loss_fn(
            key_logits.view(-1, key_logits.size(-1)),
            labels.view(-1),
        )

        total_loss: torch.Tensor = key_loss

        if alpha > 0 and kind_logits is not None and kind_labels is not None:
            kind_loss: torch.Tensor = self.aux_loss_fn(
                kind_logits.view(-1, kind_logits.size(-1)),
                kind_labels.view(-1),
            )
            total_loss = total_loss + alpha * kind_loss

        if beta > 0 and parent_logits is not None and parent_labels is not None:
            parent_loss: torch.Tensor = self.aux_loss_fn(
                parent_logits.view(-1, parent_logits.size(-1)),
                parent_labels.view(-1),
            )
            total_loss = total_loss + beta * parent_loss

        return total_loss

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
        """Extract attention weights from all layers."""
        self.eval()
        x: torch.Tensor = self.embedding(
            token_ids, node_types, depths, sibling_indices, parent_key_ids,
            kind_ids=kind_ids,
        )

        attention_weights: list[torch.Tensor] = []
        for layer in self.encoder.layers:
            attn_output, attn_weight = layer.self_attn(
                x, x, x, need_weights=True, average_attn_weights=False
            )
            attention_weights.append(attn_weight)
            x = layer(x)

        return attention_weights
