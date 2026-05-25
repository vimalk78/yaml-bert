"""Reconstruction Head: predict bag of atomic keys in a masked subtree.

Reads [doc_vec ; pos_emb(masked_root)] and outputs BCE logits over the atomic
key vocabulary. The position embedding is intentionally only depth + sibling
(not the root's key identity) so the head must use doc_vec to disambiguate
which keys are in the subtree — this is the pretraining pressure on doc_vec.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ReconstructionHead(nn.Module):
    def __init__(self, d_model: int, d_pos: int, atomic_vocab_size: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model + d_pos, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, atomic_vocab_size),
        )

    def forward(
        self,
        doc_vec_per_subtree: torch.Tensor,
        pos_emb_per_subtree: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            doc_vec_per_subtree: (M, d_model) — doc_vec repeated per masked
                subtree in the batch (M = total subtrees across batch).
            pos_emb_per_subtree: (M, d_pos) — [depth_emb ; sibling_emb] of
                each masked subtree's root.
        Returns:
            (M, atomic_vocab_size) logits — pass to BCE-with-logits loss.
        """
        return self.mlp(torch.cat(
            [doc_vec_per_subtree, pos_emb_per_subtree], dim=-1,
        ))
