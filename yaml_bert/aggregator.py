"""Tree aggregator: bottom-up combine of token hidden states into subtree
vectors + a document vector at the root.

Mean combine (no learnable params). Two execution paths share the same
output: a per-document reference loop, and a batched scatter-based
vectorized path activated when the caller passes precomputed tensors.
Numerical equivalence between the two paths is locked by
tests/test_aggregator_vectorized.py.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class TreeAggregator(nn.Module):
    """Bottom-up tree aggregation with mean combine.

    Two execution paths:
    - Reference path (default): per-doc Python loop. Used when the batch
      doesn't provide vectorized precompute tensors. Kept for tests and
      for backward compatibility.
    - Vectorized path (preferred during training): batched PyTorch scatter
      ops, processed depth-by-depth. Activated when caller passes the
      precomputed tensors as kwargs.

    Both paths produce numerically equivalent output on the same inputs
    (guaranteed by tests/test_aggregator_vectorized.py).
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.d_model = d_model

    def forward(
        self,
        hidden_states: torch.Tensor,
        batch_info: list[dict],
        *,
        parent_of_tensor: torch.Tensor | None = None,
        top_level_key_mask: torch.Tensor | None = None,
        edges_by_depth: dict[int, torch.Tensor] | None = None,
        parents_by_depth: dict[int, torch.Tensor] | None = None,
        subtree_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden_states: (B, N, d_model) per-position hidden states.
            batch_info: list of B dicts (legacy path; required even when
                vectorized — kept for fallback compat).
            parent_of_tensor / top_level_key_mask / edges_by_depth /
                parents_by_depth: when ALL provided, use vectorized path.
            subtree_mask: (B, N) bool tensor. When provided, positions where
                subtree_mask[b, i]=True are excluded from sums into doc_vec
                and from contributing to ancestor subtree_vecs. Used for the
                v8 reconstruction objective's leak-prevention.

        Returns:
            (subtree_vecs, doc_vec)
        """
        provided = (
            parent_of_tensor is not None,
            top_level_key_mask is not None,
            edges_by_depth is not None,
            parents_by_depth is not None,
        )
        if any(provided):
            if not all(provided):
                raise ValueError(
                    "TreeAggregator.forward: vectorized kwargs must be passed "
                    "all-or-none. Got: "
                    f"parent_of_tensor={'set' if provided[0] else 'None'}, "
                    f"top_level_key_mask={'set' if provided[1] else 'None'}, "
                    f"edges_by_depth={'set' if provided[2] else 'None'}, "
                    f"parents_by_depth={'set' if provided[3] else 'None'}"
                )
            return self._forward_vectorized(
                hidden_states,
                top_level_key_mask=top_level_key_mask,
                edges_by_depth=edges_by_depth,
                parents_by_depth=parents_by_depth,
                subtree_mask=subtree_mask,
            )
        return self._forward_reference(
            hidden_states, batch_info, subtree_mask=subtree_mask,
        )

    def _forward_reference(
        self,
        hidden_states: torch.Tensor,
        batch_info: list[dict],
        *,
        subtree_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-document Python loop. Original Phase 0 implementation,
        extended with leak-aware subtree_mask exclusion."""
        b, n, d = hidden_states.shape
        subtree_vecs = hidden_states.clone()
        doc_vec = torch.zeros(b, d, device=hidden_states.device,
                              dtype=hidden_states.dtype)

        def is_masked(doc_idx: int, pos: int) -> bool:
            if subtree_mask is None:
                return False
            return bool(subtree_mask[doc_idx, pos].item())

        for doc_idx in range(b):
            info = batch_info[doc_idx]
            children_of = info["children_of"]
            depth_of = info["depth_of"]
            key_positions = info["key_positions"]

            keys_by_depth: dict[int, list[int]] = {}
            for kp in key_positions:
                keys_by_depth.setdefault(depth_of[kp], []).append(kp)

            for depth in sorted(keys_by_depth.keys(), reverse=True):
                for parent_pos in keys_by_depth[depth]:
                    if is_masked(doc_idx, parent_pos):
                        # Masked root: keep its hidden state as-is, don't aggregate
                        continue
                    children = [
                        c for c in children_of[parent_pos]
                        if not is_masked(doc_idx, c)
                    ]
                    if children:
                        child_vecs = subtree_vecs[doc_idx, children]
                        own = hidden_states[doc_idx, parent_pos:parent_pos + 1]
                        combined = torch.cat(
                            [own, child_vecs], dim=0,
                        ).mean(dim=0)
                    else:
                        combined = hidden_states[doc_idx, parent_pos]
                    subtree_vecs[doc_idx, parent_pos] = combined

            top_level = [
                kp for kp in keys_by_depth.get(0, [])
                if not is_masked(doc_idx, kp)
            ]
            if top_level:
                doc_vec[doc_idx] = subtree_vecs[doc_idx, top_level].mean(dim=0)

        return subtree_vecs, doc_vec

    def _forward_vectorized(
        self,
        hidden_states: torch.Tensor,
        *,
        top_level_key_mask: torch.Tensor,
        edges_by_depth: dict[int, torch.Tensor],
        parents_by_depth: dict[int, torch.Tensor],
        subtree_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Batched scatter-based path with leak-aware subtree_mask filtering."""
        B, N, d = hidden_states.shape
        subtree_vecs = hidden_states.clone()

        # Hoist mask-on-device once; reused throughout this forward pass.
        sm: torch.Tensor | None = (
            subtree_mask.to(hidden_states.device)
            if subtree_mask is not None else None
        )

        # Process depths deepest-first.
        # parents_by_depth[d] = (P, 2) of [doc_idx, parent_pos]
        # edges_by_depth[d]   = (E, 3) of [doc_idx, child_pos, parent_pos]
        for depth in sorted(edges_by_depth.keys(), reverse=True):
            edges = edges_by_depth[depth].to(hidden_states.device)
            parents = parents_by_depth[depth].to(hidden_states.device)

            doc_idx_e = edges[:, 0]   # (E,)
            child_pos = edges[:, 1]   # (E,)
            parent_pos_e = edges[:, 2]  # (E,)

            # Filter edges where either endpoint is in a masked subtree
            if sm is not None:
                keep_edge = ~(sm[doc_idx_e, child_pos] | sm[doc_idx_e, parent_pos_e])
                doc_idx_e = doc_idx_e[keep_edge]
                child_pos = child_pos[keep_edge]
                parent_pos_e = parent_pos_e[keep_edge]

            # Gather child subtree vectors (E, d)
            child_vecs = subtree_vecs[doc_idx_e, child_pos]

            # Linear index (doc_idx, parent_pos) → flat
            parent_linear_e = doc_idx_e * N + parent_pos_e  # (E,)

            # Accumulate sum and count into per-(B*N) slots
            sum_acc = torch.zeros(
                B * N, d,
                dtype=hidden_states.dtype, device=hidden_states.device,
            )
            sum_acc.index_add_(0, parent_linear_e, child_vecs)

            # Use fp32 for count accumulation: fp16 has only 11 mantissa bits,
            # so parents with >2048 children would lose precision under autocast.
            count_acc = torch.zeros(
                B * N, dtype=torch.float32,
                device=hidden_states.device,
            )
            count_acc.index_add_(
                0, parent_linear_e,
                torch.ones_like(parent_linear_e, dtype=torch.float32),
            )

            # For each parent at this depth: mean = (sum + own) / (count + 1)
            # Filter parents: skip those that are themselves masked
            parent_doc_idx = parents[:, 0]   # (P,)
            parent_pos_p = parents[:, 1]     # (P,)
            if sm is not None:
                keep_parent = ~sm[parent_doc_idx, parent_pos_p]
                parent_doc_idx = parent_doc_idx[keep_parent]
                parent_pos_p = parent_pos_p[keep_parent]

            parent_linear_p = parent_doc_idx * N + parent_pos_p  # (P,)

            sum_at_parents = sum_acc[parent_linear_p]                              # (P, d)
            count_at_parents = count_acc[parent_linear_p].to(hidden_states.dtype)  # (P,)
            own_at_parents = hidden_states[parent_doc_idx, parent_pos_p]  # (P, d)

            mean_at_parents = (sum_at_parents + own_at_parents) / (
                count_at_parents.unsqueeze(-1) + 1.0
            )

            # Write back
            subtree_vecs[parent_doc_idx, parent_pos_p] = mean_at_parents

        # doc_vec: masked positions excluded from top-level mean
        effective_top_level = top_level_key_mask
        if sm is not None:
            effective_top_level = top_level_key_mask & ~sm

        mask_f = effective_top_level.to(hidden_states.dtype).unsqueeze(-1)  # (B, N, 1)
        weighted = subtree_vecs * mask_f                                     # (B, N, d)
        sum_per_doc = weighted.sum(dim=1)                                    # (B, d)
        count_per_doc = effective_top_level.sum(
            dim=1, dtype=torch.float32,
        ).clamp(min=1).to(hidden_states.dtype).unsqueeze(-1)
        doc_vec = sum_per_doc / count_per_doc

        # If a doc has no top-level keys (count was 0 → clamped to 1), the
        # numerator is also 0, so doc_vec is zero — matches reference path.
        return subtree_vecs, doc_vec
