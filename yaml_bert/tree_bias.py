"""Tree-distance attention bias.

STATUS (as of v9, 2026-05-27): IMPLEMENTED + WIRED + DISABLED-BY-DEFAULT.
Disabled in v8 commit ff36d9c for perf reasons — per-position `attn_mask`
forces PyTorch's nn.MultiheadAttention off the fast fused-attention
kernel, making training 3-5× slower. Never got to a quality verdict.

Worth revisiting in v10+ when:
  - We have a concrete failing test that tree_bias would plausibly fix
    (e.g., a wrong-parent-pollution probe that v9 BPE didn't fully close)
  - AND we're willing to accept the perf cost OR invest in a custom
    fused kernel / torch.compile path
  - See MEMORY.md "Tree bias in attention" for the project-level rationale.

Adds a learned bias `b(tree_distance(i, j))` to attention logits at every
layer. Per-head, per-distance learnable scalars. Inspired by T5's relative
position bias and ALiBi, adapted to use tree distance instead of sequence
distance.

Motivation: the existing model exhibits wrong-parent pollution at depth ≥ 2
(e.g., `resources::limits` predicted at 95% under `securityContext: {}`).
Tree-bias gives attention a structural prior — siblings and direct
parents/children get encouraged-attention biases; distant cousins get
discouraged ones — letting the model condition more strongly on structural
proximity.

Sizing rationale: max tree distance equals 2 * max_depth (two leaves at
max depth with LCA at root). With Lever 5's depth cap = 9, max distance
in training is 18. So `MAX_DISTANCE = 18` matches training distribution
exactly. Any OOD distance at inference (only possible if depth cap is
bypassed) gets clipped to 18.
"""
from __future__ import annotations

import torch
import torch.nn as nn

MAX_DISTANCE: int = 18  # matches 2 * Lever 5 depth cap (9)


def compute_tree_distance_matrix(
    full_paths: list[list[str]],
    max_distance: int = MAX_DISTANCE,
) -> torch.Tensor:
    """Compute the pairwise tree-distance matrix for a batch of documents.

    Args:
        full_paths: For each document, the FULL path string per node,
            including the node's own token. E.g. a node at
            `spec.containers.0.name` has full_path = "spec.containers.0.name".
            Components are "."-separated. A root node's full_path equals its
            token (e.g. "spec"). Note: this includes list indices as path
            components, so "spec.containers.0" and "spec.containers.1" are
            distinct paths. Using full paths means siblings differ by their
            terminal component (which is what we want for LCA via common
            prefix).
        max_distance: Clip distances at this value. Distances beyond max
            get this same value (lossy at extreme OOD, but bounded).

    Returns:
        Tensor of shape (B, N, N) with int distance values, where B = batch
        size and N = max sequence length in the batch. dtype = long (for
        index lookup into bias embedding). Padding positions get distance
        = max_distance.

    Algorithm:
        For nodes A and B with full paths Pa and Pb:
            len(Pa) = depth(A) (in path-component terms)
            common_prefix_len = len(longest common prefix of Pa, Pb)
            distance(A, B) = len(Pa) + len(Pb) - 2 * common_prefix_len

        Verifies:
            - Self: Pa == Pb → common = len(Pa) → distance = 0 ✓
            - Parent-child: Pa is prefix of Pb of length len(Pa) → distance = len(Pb) - len(Pa) ✓
            - Siblings: same parent path, different terminal token → common = parent_len → distance = 2 ✓
            - Cousins: distance = 4 ✓
    """
    batch_size = len(full_paths)
    max_n = max((len(p) for p in full_paths), default=0)

    # Pre-split each path into components for fast prefix comparison.
    split_paths: list[list[list[str]]] = [
        [p.split(".") if p else [] for p in doc_paths]
        for doc_paths in full_paths
    ]

    distances = torch.full((batch_size, max_n, max_n), max_distance, dtype=torch.long)

    for b in range(batch_size):
        doc_split = split_paths[b]
        n = len(doc_split)
        for i in range(n):
            parts_i = doc_split[i]
            li = len(parts_i)
            for j in range(i, n):  # upper triangle, mirror at end
                parts_j = doc_split[j]
                lj = len(parts_j)
                # Longest common prefix length
                lca = 0
                for a, b_ in zip(parts_i, parts_j):
                    if a == b_:
                        lca += 1
                    else:
                        break
                dist = li + lj - 2 * lca
                if dist > max_distance:
                    dist = max_distance
                distances[b, i, j] = dist
                distances[b, j, i] = dist

    return distances


class TreeBias(nn.Module):
    """Per-(distance, head) learnable bias added to attention scores.

    Shape: (max_distance + 1) × num_heads learnable scalars. For each
    pair (i, j) in the input sequence with tree distance d, every head h
    sees bias[d, h] added to its attention score for that pair.

    Wiring into attention: produce a (B, num_heads, N, N) bias tensor by
    looking up bias[distances]. Pass through PyTorch's attn_mask machinery
    by reshaping to (B * num_heads, N, N).
    """

    def __init__(self, num_heads: int, max_distance: int = MAX_DISTANCE) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.max_distance = max_distance
        # +1 to include distance=0 (self). Init at zero so the bias starts
        # neutral; training shapes it from there.
        self.bias = nn.Embedding(max_distance + 1, num_heads)
        nn.init.zeros_(self.bias.weight)

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        """Convert (B, N, N) int distances into (B * num_heads, N, N) bias.

        The output shape matches PyTorch's nn.MultiheadAttention attn_mask
        convention for per-head, per-batch masks.
        """
        # distances clipped during compute_tree_distance_matrix
        bias = self.bias(distances)  # (B, N, N, num_heads)
        # Move heads to second dim, then flatten batch×heads
        bias = bias.permute(0, 3, 1, 2).contiguous()  # (B, num_heads, N, N)
        b, h, n, _ = bias.shape
        return bias.view(b * h, n, n)
