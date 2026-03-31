from __future__ import annotations

import torch
import torch.nn as nn

from yaml_bert.config import YamlBertConfig
from yaml_bert.embedding import YamlBertEmbedding


class YamlBertModel(nn.Module):
    """YAML-BERT: Transformer encoder with tree positional encoding.

    Two prediction heads: simple (bigram) + kind-specific (trigram).
    """

    def __init__(
        self,
        config: YamlBertConfig,
        embedding: YamlBertEmbedding,
        simple_vocab_size: int,
        kind_vocab_size: int,
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

        self.simple_head: nn.Linear = nn.Linear(config.d_model, simple_vocab_size)
        self.kind_head: nn.Linear = nn.Linear(config.d_model, kind_vocab_size)
        self.loss_fn: nn.CrossEntropyLoss = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(
        self,
        token_ids: torch.Tensor,
        node_types: torch.Tensor,
        depths: torch.Tensor,
        sibling_indices: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x: torch.Tensor = self.embedding(token_ids, node_types, depths, sibling_indices)
        x = self.encoder(x, src_key_padding_mask=padding_mask)
        simple_logits: torch.Tensor = self.simple_head(x)
        kind_logits: torch.Tensor = self.kind_head(x)
        return simple_logits, kind_logits

    def compute_loss(
        self,
        simple_logits: torch.Tensor,
        simple_labels: torch.Tensor,
        kind_logits: torch.Tensor,
        kind_labels: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        simple_loss: torch.Tensor = self.loss_fn(
            simple_logits.view(-1, simple_logits.size(-1)),
            simple_labels.view(-1),
        )

        # Kind loss: only compute if there are kind-specific labels in this batch
        has_kind: bool = (kind_labels != -100).any().item()
        if has_kind:
            kind_loss: torch.Tensor = self.loss_fn(
                kind_logits.view(-1, kind_logits.size(-1)),
                kind_labels.view(-1),
            )
        else:
            kind_loss = torch.tensor(0.0, device=simple_logits.device)

        total: torch.Tensor = simple_loss + kind_loss
        return total, {"simple": simple_loss.item(), "kind": kind_loss.item()}
