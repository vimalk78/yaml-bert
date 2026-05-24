"""Tree aggregator: bottom-up combine of token hidden states into subtree
vectors + a document vector at the root.

Phase 0 uses mean combine (no learnable params) and a per-document Python
loop (no batched scatter ops). If Phase 0 benchmarking shows this is a
bottleneck, vectorize using scatter operations in a follow-up.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class TreeAggregator(nn.Module):
    """Bottom-up tree aggregation with mean combine.

    Input: encoder hidden states + per-document children/parent info.
    Output: a subtree vector per KEY position, plus a single doc vector
    per document (mean of top-level keys' subtree vectors).

    Phase 0: mean combine — no learnable parameters.
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.d_model = d_model

    def forward(
        self,
        hidden_states: torch.Tensor,
        batch_info: list[dict],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden_states: (B, N, d_model) per-position hidden states from encoder
            batch_info: list of B dicts, each from compute_children_info

        Returns:
            subtree_vecs: (B, N, d_model) — subtree vector at each KEY
                position. Non-KEY positions hold their original hidden state
                (placeholder; not used downstream by Token Head conditioning).
            doc_vec: (B, d_model) — mean of top-level (depth-0) keys' subtree
                vectors.
        """
        b, n, d = hidden_states.shape
        subtree_vecs = hidden_states.clone()
        # Propagate dtype from hidden_states so mixed-precision (autocast)
        # downstream doesn't hit dtype mismatch in Token Head concatenation.
        doc_vec = torch.zeros(b, d, device=hidden_states.device,
                              dtype=hidden_states.dtype)

        for doc_idx in range(b):
            info = batch_info[doc_idx]
            children_of = info["children_of"]
            depth_of = info["depth_of"]
            key_positions = info["key_positions"]

            # Group key positions by depth, deepest first
            keys_by_depth: dict[int, list[int]] = {}
            for kp in key_positions:
                keys_by_depth.setdefault(depth_of[kp], []).append(kp)

            for depth in sorted(keys_by_depth.keys(), reverse=True):
                for parent_pos in keys_by_depth[depth]:
                    children = children_of[parent_pos]
                    # Mean of [self] + [child_subtree_vec for each child]
                    if children:
                        child_vecs = subtree_vecs[doc_idx, children]   # (k, d)
                        own = hidden_states[doc_idx, parent_pos:parent_pos + 1]  # (1, d)
                        combined = torch.cat([own, child_vecs], dim=0).mean(dim=0)
                    else:
                        # Leaf key: subtree = own hidden state
                        combined = hidden_states[doc_idx, parent_pos]
                    subtree_vecs[doc_idx, parent_pos] = combined

            # doc_vec = mean of top-level (depth-0) keys' subtree vecs
            top_level = keys_by_depth.get(0, [])
            if top_level:
                doc_vec[doc_idx] = subtree_vecs[doc_idx, top_level].mean(dim=0)

        return subtree_vecs, doc_vec
