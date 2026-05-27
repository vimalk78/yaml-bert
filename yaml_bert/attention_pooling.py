"""Simple learned-attention pooling.

STATUS (as of v9, 2026-05-27): NOT integrated into the current model.
Kept as a v10+ ablation baseline against the tree aggregator. Tests if
the structural prior in TreeAggregator buys anything over a vanilla
single-head learned-attention pooling.

Has tests in tests/test_attention_pooling.py.
"""
import math

import torch
import torch.nn as nn


class AttentionPooling(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.W_doc = nn.Linear(in_features=d_model, out_features=1)  # (d_model,1)

    def forward(
        self,
        h: torch.Tensor,  # Shape(Batch, Seq_Len, d_model)
    ):
        scores: torch.Tensor = self.W_doc(h)  # (Batch, Seq_Len, 1)
        scores = scores / math.sqrt(self.d_model)
        scores = scores.view(scores.shape[0], 1, scores.shape[1])  # (Batch, 1, Seq_Len)
        weights: torch.Tensor = scores.softmax(dim=-1)  # (Batch,1, Seq_Len)
        return (
            (weights @ h).squeeze(1),
            weights,
        )  # returning weights for testing/visualization (Batch,d_model), (Batch,1,Seq_Len)
