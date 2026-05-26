"""Tests for tree-distance attention bias module."""
from __future__ import annotations

import torch

from yaml_bert.tree_bias import (
    MAX_DISTANCE,
    TreeBias,
    compute_tree_distance_matrix,
)


# -----------------------------------------------------------------
# Distance matrix computation
# -----------------------------------------------------------------


def test_self_distance_is_zero():
    """A node has tree-distance 0 to itself."""
    # Three nodes with their FULL paths
    paths = [["spec", "metadata", "spec.replicas"]]
    d = compute_tree_distance_matrix(paths)
    assert d[0, 0, 0].item() == 0
    assert d[0, 1, 1].item() == 0
    assert d[0, 2, 2].item() == 0


def test_siblings_have_distance_two():
    """Two nodes sharing the same parent are siblings → tree distance 2."""
    # parent at root, two siblings under it: parent.A and parent.B
    paths = [["parent", "parent.A", "parent.B"]]
    d = compute_tree_distance_matrix(paths)
    # Nodes 1 and 2 are siblings under "parent"
    assert d[0, 1, 2].item() == 2
    assert d[0, 2, 1].item() == 2


def test_parent_child_distance_is_one():
    """A key and its direct child are 1 apart."""
    # "parent" at root, "parent.child" is its direct child
    paths = [["parent", "parent.child"]]
    d = compute_tree_distance_matrix(paths)
    assert d[0, 0, 1].item() == 1
    assert d[0, 1, 0].item() == 1


def test_grandparent_grandchild_distance_is_two():
    paths = [["spec", "spec.containers", "spec.containers.0"]]
    d = compute_tree_distance_matrix(paths)
    # spec → spec.containers: 1
    # spec.containers → spec.containers.0: 1
    # spec → spec.containers.0: 2
    assert d[0, 0, 2].item() == 2


def test_cousins_distance_four():
    """Different children of different parents under a common grandparent."""
    # Common grandparent: root
    # Two parents: spec.A and spec.B (siblings)
    # Two cousins: spec.A.x and spec.B.y
    paths = [["spec.A.x", "spec.B.y"]]
    d = compute_tree_distance_matrix(paths)
    # spec.A.x → spec.A → spec → spec.B → spec.B.y = 4 edges
    assert d[0, 0, 1].item() == 4


def test_unrelated_subtrees_distance_via_root():
    """Two leaves with totally different roots."""
    paths = [["spec.containers.0.name", "metadata.name"]]
    d = compute_tree_distance_matrix(paths)
    # Common prefix = 0 (different roots: "spec" vs "metadata")
    # Distance = 4 + 2 - 0 = 6
    assert d[0, 0, 1].item() == 6


def test_distance_matrix_is_symmetric():
    paths = [[
        "spec",
        "spec.containers",
        "spec.containers.0",
        "metadata",
    ]]
    d = compute_tree_distance_matrix(paths)
    n = 4
    for i in range(n):
        for j in range(n):
            assert d[0, i, j].item() == d[0, j, i].item(), \
                f"asymmetric at ({i},{j}): {d[0, i, j]} vs {d[0, j, i]}"


def test_padding_positions_get_max_distance():
    """Empty doc slots in a batch pad to max_distance."""
    paths = [
        ["metadata", "metadata.name"],                          # 2 nodes
        ["spec", "spec.containers", "spec.containers.0"],       # 3 nodes
    ]
    d = compute_tree_distance_matrix(paths)
    # Shape should be (2, 3, 3) — max_n across batch
    assert d.shape == (2, 3, 3)
    # Position 2 in doc 0 is padding → max_distance
    assert d[0, 0, 2].item() == MAX_DISTANCE
    assert d[0, 2, 0].item() == MAX_DISTANCE


def test_distance_clipped_at_max():
    """Distances beyond MAX_DISTANCE clip to MAX_DISTANCE."""
    deep_a = ".".join(f"l{i}" for i in range(15))
    deep_b = ".".join(f"r{i}" for i in range(15))
    paths = [[deep_a, deep_b]]
    d = compute_tree_distance_matrix(paths)
    assert d[0, 0, 1].item() == MAX_DISTANCE


# -----------------------------------------------------------------
# TreeBias module
# -----------------------------------------------------------------


def test_tree_bias_output_shape():
    """TreeBias produces (B*num_heads, N, N) suitable for attn_mask."""
    num_heads = 8
    tb = TreeBias(num_heads=num_heads)
    distances = torch.zeros((2, 5, 5), dtype=torch.long)  # batch=2, seq=5
    bias = tb(distances)
    assert bias.shape == (2 * num_heads, 5, 5)


def test_tree_bias_zero_init():
    """At init, all biases are zero (neutral starting point)."""
    tb = TreeBias(num_heads=4)
    distances = torch.tensor([[[0, 1, 2], [1, 0, 1], [2, 1, 0]]], dtype=torch.long)
    bias = tb(distances)
    assert torch.all(bias == 0)


def test_tree_bias_distances_with_same_value_get_same_bias():
    """All (i,j) pairs with the same distance get the same bias per head."""
    tb = TreeBias(num_heads=4)
    # Force non-zero distinct biases
    with torch.no_grad():
        tb.bias.weight[2] = torch.tensor([0.1, 0.2, 0.3, 0.4])

    # Two pairs both with distance=2
    distances = torch.tensor([[[0, 2], [2, 0]]], dtype=torch.long)
    bias = tb(distances)
    # bias[head, 0, 1] should equal bias[head, 1, 0] (both distance=2)
    # Shape: (B*H, N, N) = (1*4, 2, 2)
    expected = [0.1, 0.2, 0.3, 0.4]
    for head in range(4):
        assert bias[head, 0, 1].item() == bias[head, 1, 0].item()
        assert abs(bias[head, 0, 1].item() - expected[head]) < 1e-5
