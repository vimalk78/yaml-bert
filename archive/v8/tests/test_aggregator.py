import torch
from yaml_bert.aggregator import TreeAggregator


def _make_simple_info(n_pos: int, key_positions: list[int],
                      children_of: list[list[int]], depths: list[int]) -> dict:
    parent_of = [-1] * n_pos
    for p, kids in enumerate(children_of):
        for k in kids:
            parent_of[k] = p
    return {
        "children_of": children_of,
        "parent_of": parent_of,
        "key_positions": key_positions,
        "depth_of": depths,
    }


def test_aggregator_output_shapes():
    """Aggregator returns (subtree_vecs, doc_vec) with correct shapes."""
    d_model = 16
    n = 5
    agg = TreeAggregator(d_model=d_model)
    hidden = torch.randn(1, n, d_model)
    info = [_make_simple_info(
        n_pos=n,
        key_positions=[0, 1, 2],   # 3 KEYs
        children_of=[[1, 2], [], []],  # 0 has children 1,2; 1 and 2 are leaves
        depths=[0, 1, 1, 1, 1],
    )]
    subtree_vecs, doc_vec = agg(hidden, info)
    assert subtree_vecs.shape == (1, n, d_model)
    assert doc_vec.shape == (1, d_model)


def test_aggregator_leaf_key_uses_self_hidden():
    """For a leaf KEY (no key children), subtree_vec equals the key's
    own hidden state (mean of [self], since no children)."""
    d_model = 4
    agg = TreeAggregator(d_model=d_model)
    hidden = torch.tensor([[
        [1., 2., 3., 4.],   # pos 0 (a key with no children)
        [9., 9., 9., 9.],   # pos 1 (a value, not a child for subtree purposes)
    ]])
    info = [_make_simple_info(
        n_pos=2,
        key_positions=[0],
        children_of=[[]],
        depths=[0, 0],
    )]
    subtree_vecs, _ = agg(hidden, info)
    # leaf key: subtree_vec = mean of [self] = self
    assert torch.allclose(subtree_vecs[0, 0], hidden[0, 0])


def test_aggregator_internal_node_means_children_and_self():
    """For an internal node, subtree_vec = mean(self, child_subtree_vecs)."""
    d_model = 2
    agg = TreeAggregator(d_model=d_model)
    hidden = torch.tensor([[
        [0., 0.],  # pos 0 (root key)
        [4., 4.],  # pos 1 (child 1, leaf key)
        [8., 8.],  # pos 2 (child 2, leaf key)
    ]])
    info = [_make_simple_info(
        n_pos=3,
        key_positions=[0, 1, 2],
        children_of=[[1, 2], [], []],
        depths=[0, 1, 1],
    )]
    subtree_vecs, doc_vec = agg(hidden, info)
    # Leaf keys: subtree = self
    # Root: subtree = mean(self=0, child1_subtree=[4,4], child2_subtree=[8,8])
    #             = mean([0,0], [4,4], [8,8]) = [4,4]
    expected_root = torch.tensor([4., 4.])
    assert torch.allclose(subtree_vecs[0, 0], expected_root)
    # doc_vec = mean of top-level keys' subtree vecs = root's subtree = [4,4]
    assert torch.allclose(doc_vec[0], expected_root)
