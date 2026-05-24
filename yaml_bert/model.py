from __future__ import annotations

import torch
import torch.nn as nn

from yaml_bert.config import YamlBertConfig
from yaml_bert.embedding import YamlBertEmbedding
from yaml_bert.tree_bias import TreeBias


class YamlBertModel(nn.Module):
    """YAML-BERT: Transformer encoder with tree positional encoding.

    Two prediction heads: simple (bigram) + kind-specific (trigram).
    Optional tree-distance attention bias (v7).
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
        self.num_heads: int = config.num_heads

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

        self.tree_bias: TreeBias | None = (
            TreeBias(num_heads=config.num_heads) if config.tree_bias_enabled else None
        )

    def forward(
        self,
        token_ids: torch.Tensor,
        node_types: torch.Tensor,
        depths: torch.Tensor,
        sibling_indices: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        tree_distances: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x: torch.Tensor = self.embedding(token_ids, node_types, depths, sibling_indices)

        # If tree_bias is enabled and a distance matrix is provided, compute
        # an additive per-(batch, head, i, j) attention bias and pass it as
        # the encoder's attn_mask. nn.TransformerEncoder applies the same
        # mask at every layer — fine because the bias is structural and
        # layer-independent.
        attn_mask: torch.Tensor | None = None
        if self.tree_bias is not None and tree_distances is not None:
            attn_mask = self.tree_bias(tree_distances)  # (B*num_heads, N, N)

        x = self.encoder(x, mask=attn_mask, src_key_padding_mask=padding_mask)
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


def checkpoint_has_tree_bias(state_dict: dict) -> bool:
    """True iff a checkpoint's model_state_dict contains tree-bias weights.
    Use this at load time to decide whether to instantiate the model with
    `tree_bias_enabled=True` (v7+) or `False` (v6.1 and earlier).
    """
    return any(k.startswith("tree_bias.") for k in state_dict)
