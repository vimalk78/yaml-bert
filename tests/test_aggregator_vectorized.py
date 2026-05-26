"""Numerical equivalence: per-doc reference path vs vectorized path.

All tests use synthetic inputs only — no Vocabulary or YamlBertDataset needed.

Tree shape used in most tests (4 logical KEY nodes, depth 0-2):

    0 (root, depth 0)
    ├── 1 (depth 1)
    │   └── 3 (depth 2)
    └── 2 (depth 1)

Logical positions map directly to a (B, L, d) pooled tensor.
"""
import torch
import pytest

from yaml_bert.aggregator import TreeAggregator


# ---------------------------------------------------------------------------
# Synthetic batch_info helpers
# ---------------------------------------------------------------------------

def _make_single_doc_info():
    """Return batch_info for one document with the 4-node tree above.

    All 4 nodes are KEY nodes (depth 0-2).  The list-of-lists layout matches
    what compute_children_info() returns in yaml_bert/dataset.py.
    """
    #          pos:  0       1       2       3
    children_of = [[1, 2], [3],   [],     []]
    parent_of   = [-1,      0,     0,      1]
    depth_of    = [0,       1,     1,      2]
    key_positions = [0, 1, 2, 3]
    full_path_of  = ["root", "root.a", "root.b", "root.a.x"]
    return {
        "children_of": children_of,
        "parent_of":   parent_of,
        "depth_of":    depth_of,
        "key_positions": key_positions,
        "full_path_of":  full_path_of,
    }


def _make_second_doc_info():
    """Slightly different 3-node tree for the second doc in a batch.

    0 (root, depth 0)
    └── 1 (depth 1)
        └── 2 (depth 2)
    """
    children_of = [[1],   [2],   []]
    parent_of   = [-1,     0,     1]
    depth_of    = [0,      1,     2]
    key_positions = [0, 1, 2]
    full_path_of  = ["r", "r.c", "r.c.d"]
    return {
        "children_of": children_of,
        "parent_of":   parent_of,
        "depth_of":    depth_of,
        "key_positions": key_positions,
        "full_path_of":  full_path_of,
    }


def _build_vectorized_tensors(batch_info: list[dict], L: int):
    """Replicate the collate_fn logic that builds the four vectorized tensors.

    Returns (parent_of_tensor, top_level_key_mask, edges_by_depth, parents_by_depth).
    L is the padded logical dimension.
    """
    B = len(batch_info)
    parent_of_tensor  = torch.full((B, L), -1, dtype=torch.long)
    top_level_key_mask = torch.zeros((B, L), dtype=torch.bool)

    for b_idx, info in enumerate(batch_info):
        parent_of = info["parent_of"]
        n_b = len(parent_of)
        if n_b > 0:
            parent_of_tensor[b_idx, :n_b] = torch.tensor(parent_of, dtype=torch.long)
        depth_of = info["depth_of"]
        depth_zero_kps = [kp for kp in info["key_positions"] if depth_of[kp] == 0]
        if depth_zero_kps:
            top_level_key_mask[b_idx, depth_zero_kps] = True

    edges_by_depth: dict[int, list[tuple[int, int, int]]] = {}
    parents_set_by_depth: dict[int, set[tuple[int, int]]] = {}
    for b_idx, info in enumerate(batch_info):
        children_of = info["children_of"]
        depth_of    = info["depth_of"]
        for parent_pos in info["key_positions"]:
            kids = children_of[parent_pos]
            if not kids:
                continue
            parent_depth = depth_of[parent_pos]
            edges_by_depth.setdefault(parent_depth, []).extend(
                (b_idx, child_pos, parent_pos) for child_pos in kids
            )
            parents_set_by_depth.setdefault(parent_depth, set()).add(
                (b_idx, parent_pos)
            )

    edges_tensors = {
        d: torch.tensor(edges, dtype=torch.long)
        for d, edges in edges_by_depth.items()
    }
    parents_tensors = {
        d: torch.tensor(sorted(parents_set), dtype=torch.long)
        for d, parents_set in parents_set_by_depth.items()
    }
    return parent_of_tensor, top_level_key_mask, edges_tensors, parents_tensors


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_partial_vectorized_kwargs_raises():
    """Passing only some of the four vectorized kwargs must raise ValueError."""
    agg = TreeAggregator(d_model=8)
    hidden = torch.zeros(1, 4, 8)
    batch_info = [_make_single_doc_info()]
    B, N = 1, 4
    logical_ids = torch.arange(N).unsqueeze(0).expand(B, N)
    n_logical_per_doc = torch.tensor([N] * B)
    with pytest.raises(ValueError, match="all-or-none"):
        agg(hidden, batch_info,
            logical_ids=logical_ids, n_logical_per_doc=n_logical_per_doc,
            parent_of_tensor=torch.full((1, 4), -1, dtype=torch.long))


def test_vectorized_aggregator_equals_per_doc_reference():
    """Vectorized aggregator produces numerically identical output to the
    per-doc reference path on a single-document synthetic batch."""
    d_model = 16
    L = 4   # 4 logical nodes
    B = 1

    torch.manual_seed(42)
    # pooled: directly (B, L, d) — skip the subword-pooling step
    pooled = torch.randn(B, L, d_model)

    info = _make_single_doc_info()
    batch_info = [info]

    parent_of_tensor, top_level_key_mask, edges_by_depth, parents_by_depth = (
        _build_vectorized_tensors(batch_info, L)
    )

    agg = TreeAggregator(d_model=d_model)

    # Reference path: use _forward_reference directly so we bypass _pool_subwords
    ref_subtree, ref_doc = agg._forward_reference(pooled, batch_info)

    # Vectorized path: use _forward_vectorized directly
    vec_subtree, vec_doc = agg._forward_vectorized(
        pooled,
        top_level_key_mask=top_level_key_mask,
        edges_by_depth=edges_by_depth,
        parents_by_depth=parents_by_depth,
    )

    assert torch.allclose(ref_subtree, vec_subtree, atol=1e-6), (
        f"subtree_vecs mismatch: max diff = "
        f"{(ref_subtree - vec_subtree).abs().max().item():.3e}"
    )
    assert torch.allclose(ref_doc, vec_doc, atol=1e-6), (
        f"doc_vec mismatch: max diff = "
        f"{(ref_doc - vec_doc).abs().max().item():.3e}"
    )


def test_vectorized_aggregator_batch_equals_reference():
    """Vectorized path matches reference path on a 2-doc batch with
    different tree shapes and sizes (padded to the longer doc)."""
    d_model = 16
    # Doc 0 has 4 logical nodes, doc 1 has 3 → pad to L=4
    L = 4
    B = 2

    torch.manual_seed(7)
    pooled = torch.randn(B, L, d_model)

    info0 = _make_single_doc_info()   # 4 nodes, tree depth 0-2
    info1 = _make_second_doc_info()   # 3 nodes, tree depth 0-2

    batch_info = [info0, info1]

    parent_of_tensor, top_level_key_mask, edges_by_depth, parents_by_depth = (
        _build_vectorized_tensors(batch_info, L)
    )

    agg = TreeAggregator(d_model=d_model)

    ref_subtree, ref_doc = agg._forward_reference(pooled, batch_info)
    vec_subtree, vec_doc = agg._forward_vectorized(
        pooled,
        top_level_key_mask=top_level_key_mask,
        edges_by_depth=edges_by_depth,
        parents_by_depth=parents_by_depth,
    )

    assert torch.allclose(ref_subtree, vec_subtree, atol=1e-6), (
        f"subtree_vecs mismatch (batch): max diff = "
        f"{(ref_subtree - vec_subtree).abs().max().item():.3e}"
    )
    assert torch.allclose(ref_doc, vec_doc, atol=1e-6), (
        f"doc_vec mismatch (batch): max diff = "
        f"{(ref_doc - vec_doc).abs().max().item():.3e}"
    )


def test_vectorized_aggregator_with_subtree_mask_equals_reference():
    """Vectorized path with subtree_mask matches reference path with same mask.

    Mask: exclude logical position 1 (depth-1 node "root.a") from doc 0.
    This removes pos 1 AND its child pos 3 from aggregation and doc_vec.
    """
    d_model = 16
    L = 4
    B = 2

    torch.manual_seed(13)
    pooled = torch.randn(B, L, d_model)

    info0 = _make_single_doc_info()
    info1 = _make_second_doc_info()
    batch_info = [info0, info1]

    parent_of_tensor, top_level_key_mask, edges_by_depth, parents_by_depth = (
        _build_vectorized_tensors(batch_info, L)
    )

    # Mask doc 0, pos 1 ("root.a") — also naturally excludes its child 3
    subtree_mask = torch.zeros(B, L, dtype=torch.bool)
    subtree_mask[0, 1] = True
    subtree_mask[0, 3] = True  # child of 1; also mask explicitly

    agg = TreeAggregator(d_model=d_model)

    ref_subtree, ref_doc = agg._forward_reference(
        pooled, batch_info, subtree_mask=subtree_mask
    )
    vec_subtree, vec_doc = agg._forward_vectorized(
        pooled,
        top_level_key_mask=top_level_key_mask,
        edges_by_depth=edges_by_depth,
        parents_by_depth=parents_by_depth,
        subtree_mask=subtree_mask,
    )

    assert torch.allclose(ref_subtree, vec_subtree, atol=1e-6), (
        f"subtree mismatch (with mask): max diff = "
        f"{(ref_subtree - vec_subtree).abs().max().item():.3e}"
    )
    assert torch.allclose(ref_doc, vec_doc, atol=1e-6), (
        f"doc_vec mismatch (with mask): max diff = "
        f"{(ref_doc - vec_doc).abs().max().item():.3e}"
    )


def test_aggregator_subtree_mask_excludes_positions_from_doc_vec():
    """A subtree_mask covering a top-level key removes it from doc_vec mean.

    Tree: 0 (root, d=0) → 1 (d=1) → 2 (d=1).
    Both 1 and 2 are top-level keys (depth 0 in their respective synthetic
    trees — we use a 3-node flat tree here so two keys are at depth 0).

    Single doc: root 0, children 1 and 2 (both depth 0 top-level keys so that
    masking key 1 visibly changes the doc_vec mean).
    """
    d_model = 8
    # Flat tree: 0 is depth-0 root; 1 and 2 are also depth-0 top-level keys
    # (siblings, no parent-child relationship). This is the easiest shape to
    # verify: doc_vec = mean(subtree_vecs[top_level_keys]).
    children_of  = [[], [], []]
    parent_of    = [-1, -1, -1]
    depth_of     = [0,   0,  0]
    key_positions = [0, 1, 2]
    full_path_of  = ["a", "b", "c"]
    info = {
        "children_of":  children_of,
        "parent_of":    parent_of,
        "depth_of":     depth_of,
        "key_positions": key_positions,
        "full_path_of":  full_path_of,
    }

    B, L = 1, 3
    batch_info = [info]
    parent_of_tensor, top_level_key_mask, edges_by_depth, parents_by_depth = (
        _build_vectorized_tensors(batch_info, L)
    )

    torch.manual_seed(99)
    pooled = torch.randn(B, L, d_model)

    agg = TreeAggregator(d_model=d_model)

    # Baseline: no mask → doc_vec is mean of all 3 positions
    _, doc_no_mask = agg._forward_vectorized(
        pooled,
        top_level_key_mask=top_level_key_mask,
        edges_by_depth=edges_by_depth,
        parents_by_depth=parents_by_depth,
    )

    # Mask position 0 ("a") — should shift the doc_vec mean
    sm = torch.zeros(B, L, dtype=torch.bool)
    sm[0, 0] = True

    _, doc_with_mask = agg._forward_vectorized(
        pooled,
        top_level_key_mask=top_level_key_mask,
        edges_by_depth=edges_by_depth,
        parents_by_depth=parents_by_depth,
        subtree_mask=sm,
    )

    assert not torch.allclose(doc_no_mask, doc_with_mask, atol=1e-5), (
        "Expected doc_vec to differ after masking a top-level key"
    )
    # Sanity check: with mask, doc_vec should equal mean of positions 1 and 2
    expected = pooled[0, [1, 2]].mean(dim=0)
    assert torch.allclose(doc_with_mask[0], expected, atol=1e-6), (
        f"doc_vec with mask != mean(pos 1, 2): diff = "
        f"{(doc_with_mask[0] - expected).abs().max().item():.3e}"
    )
