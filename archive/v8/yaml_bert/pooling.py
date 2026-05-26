from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DocumentPooling(nn.Module):
    """Pooling by Multi-head Attention.

    The kind node queries all other nodes via cross-attention
    to produce a single document embedding.
    """

    def __init__(self, d_model: int, num_heads: int = 4) -> None:
        super().__init__()
        self.query_proj: nn.Linear = nn.Linear(d_model, d_model)
        self.cross_attn: nn.MultiheadAttention = nn.MultiheadAttention(
            d_model, num_heads, batch_first=True,
        )
        self.layer_norm: nn.LayerNorm = nn.LayerNorm(d_model)

    def forward(
        self,
        kind_hidden: torch.Tensor,
        all_hidden: torch.Tensor,
    ) -> torch.Tensor:
        query: torch.Tensor = self.query_proj(kind_hidden)
        doc_emb, _ = self.cross_attn(query, all_hidden, all_hidden)
        doc_emb = self.layer_norm(doc_emb)
        return doc_emb.squeeze(1)


def supervised_contrastive_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    embeddings = F.normalize(embeddings, dim=1)
    batch_size: int = embeddings.shape[0]

    sim: torch.Tensor = embeddings @ embeddings.T / temperature

    label_mask: torch.Tensor = labels.unsqueeze(0) == labels.unsqueeze(1)
    self_mask: torch.Tensor = torch.eye(batch_size, dtype=torch.bool, device=embeddings.device)
    label_mask = label_mask & ~self_mask

    sim_max, _ = sim.max(dim=1, keepdim=True)
    sim = sim - sim_max.detach()

    exp_sim: torch.Tensor = torch.exp(sim) * (~self_mask).float()
    log_prob: torch.Tensor = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)

    pos_count: torch.Tensor = label_mask.float().sum(dim=1).clamp(min=1)
    loss: torch.Tensor = -(label_mask.float() * log_prob).sum(dim=1) / pos_count

    return loss.mean()
