"""Numerical equivalence: per-doc reference path vs vectorized path."""
import torch

from yaml_bert.aggregator import TreeAggregator
from yaml_bert.linearizer import YamlLinearizer
from yaml_bert.v8_dataset import V8Dataset, v8_collate_fn
from yaml_bert.vocab import VocabBuilder
from yaml_bert.config import YamlBertConfig


def test_vectorized_aggregator_equals_per_doc_reference():
    """Vectorized aggregator produces numerically identical output to the
    per-doc reference path, given the same hidden states + batch_info."""
    docs = [
        YamlLinearizer().linearize(
            "apiVersion: v1\nkind: Pod\nmetadata:\n  name: a\n"
            "spec:\n  containers:\n  - name: x\n"),
        YamlLinearizer().linearize(
            "apiVersion: apps/v1\nkind: Deployment\nspec:\n  replicas: 3\n"
            "  selector:\n    matchLabels:\n      app: y\n"),
    ]
    vocab = VocabBuilder().build([n for d in docs for n in d], min_freq=1)
    config = YamlBertConfig(v8_mode=True, mask_prob=0.0, d_model=16)
    ds = V8Dataset(docs, vocab, config)
    batch = v8_collate_fn([ds[0], ds[1]])

    B, N = batch["token_ids"].shape
    d_model = 16
    torch.manual_seed(0)
    hidden = torch.randn(B, N, d_model)

    agg = TreeAggregator(d_model=d_model)

    # Reference path: legacy, no tensor kwargs
    ref_subtree, ref_doc = agg(hidden, batch["batch_info"])

    # Vectorized path: pass precomputed tensors as kwargs
    vec_subtree, vec_doc = agg(
        hidden, batch["batch_info"],
        parent_of_tensor=batch["parent_of_tensor"],
        top_level_key_mask=batch["top_level_key_mask"],
        edges_by_depth=batch["edges_by_depth"],
        parents_by_depth=batch["parents_by_depth"],
    )

    assert torch.allclose(ref_subtree, vec_subtree, atol=1e-6), (
        f"subtree_vecs mismatch: max diff = "
        f"{(ref_subtree - vec_subtree).abs().max().item()}"
    )
    assert torch.allclose(ref_doc, vec_doc, atol=1e-6), (
        f"doc_vec mismatch: max diff = "
        f"{(ref_doc - vec_doc).abs().max().item()}"
    )


def test_partial_vectorized_kwargs_raises():
    """Passing only some of the four vectorized kwargs is a silent-bug
    footgun; aggregator must reject it explicitly."""
    import pytest
    agg = TreeAggregator(d_model=8)
    hidden = torch.zeros(1, 4, 8)
    batch_info = [{"children_of": {}, "depth_of": {}, "key_positions": [],
                   "parent_of": [], "full_path_of": {}}]
    with pytest.raises(ValueError, match="all-or-none"):
        agg(hidden, batch_info,
            parent_of_tensor=torch.full((1, 4), -1, dtype=torch.long))


def test_vectorized_aggregator_with_subtree_mask_equals_reference():
    """Vectorized path with subtree_mask matches reference path with same mask."""
    from yaml_bert.aggregator import TreeAggregator
    from yaml_bert.linearizer import YamlLinearizer
    from yaml_bert.v8_dataset import V8Dataset, v8_collate_fn
    from yaml_bert.vocab import VocabBuilder
    from yaml_bert.config import YamlBertConfig

    docs = [
        YamlLinearizer().linearize(
            "apiVersion: apps/v1\nkind: Deployment\nspec:\n"
            "  replicas: 3\n  template:\n    spec:\n      containers:\n"
            "      - name: x\n        image: nginx\n"),
        YamlLinearizer().linearize(
            "apiVersion: v1\nkind: Pod\nmetadata:\n  name: y\n"
            "spec:\n  containers:\n  - name: z\n"),
    ]
    vocab = VocabBuilder().build([n for d in docs for n in d], min_freq=1)
    config = YamlBertConfig(v8_mode=True, mask_prob=0.0, d_model=16,
                            recon_enabled=True)
    ds = V8Dataset(docs, vocab, config)
    batch = v8_collate_fn([ds[0], ds[1]])
    # Synthesize a subtree_mask manually so the test doesn't depend on
    # random picker state. Mask all depth>=2 positions in doc 0.
    sm = batch["subtree_mask"].clone()
    # Force a known mask: cover position 3 in doc 0 (some inner position)
    if sm.shape[1] > 3:
        sm[0, 3] = True

    B, N = batch["token_ids"].shape
    d_model = 16
    torch.manual_seed(0)
    hidden = torch.randn(B, N, d_model)

    agg = TreeAggregator(d_model=d_model)

    # Reference path with subtree_mask
    ref_subtree, ref_doc = agg(hidden, batch["batch_info"], subtree_mask=sm)

    # Vectorized path with subtree_mask
    vec_subtree, vec_doc = agg(
        hidden, batch["batch_info"],
        parent_of_tensor=batch["parent_of_tensor"],
        top_level_key_mask=batch["top_level_key_mask"],
        edges_by_depth=batch["edges_by_depth"],
        parents_by_depth=batch["parents_by_depth"],
        subtree_mask=sm,
    )

    assert torch.allclose(ref_subtree, vec_subtree, atol=1e-6), (
        f"subtree mismatch: max diff = "
        f"{(ref_subtree - vec_subtree).abs().max().item()}"
    )
    assert torch.allclose(ref_doc, vec_doc, atol=1e-6)


def test_aggregator_subtree_mask_excludes_positions_from_doc_vec():
    """A subtree_mask covering a top-level key removes it from doc_vec mean."""
    from yaml_bert.aggregator import TreeAggregator
    from yaml_bert.linearizer import YamlLinearizer
    from yaml_bert.v8_dataset import V8Dataset, v8_collate_fn
    from yaml_bert.vocab import VocabBuilder
    from yaml_bert.config import YamlBertConfig

    docs = [
        YamlLinearizer().linearize(
            "apiVersion: v1\nkind: Pod\nspec:\n  x: 1\n  y: 2\n"),
    ]
    vocab = VocabBuilder().build([n for d in docs for n in d], min_freq=1)
    config = YamlBertConfig(v8_mode=True, mask_prob=0.0, d_model=8,
                            recon_enabled=True)
    ds = V8Dataset(docs, vocab, config)
    batch = v8_collate_fn([ds[0]])

    # Find the position of "spec" (depth-0 KEY)
    info = batch["batch_info"][0]
    spec_pos = next(
        kp for kp in info["key_positions"]
        if info["depth_of"][kp] == 0 and info["full_path_of"][kp] == "spec"
    )

    B, N = batch["token_ids"].shape
    torch.manual_seed(0)
    hidden = torch.randn(B, N, 8)

    # No-mask baseline — use SAME vectorized path as the masked run so a
    # divergence is attributable to the mask, not to a cross-path difference.
    agg = TreeAggregator(d_model=8)
    _, doc_no_mask = agg(
        hidden, batch["batch_info"],
        parent_of_tensor=batch["parent_of_tensor"],
        top_level_key_mask=batch["top_level_key_mask"],
        edges_by_depth=batch["edges_by_depth"],
        parents_by_depth=batch["parents_by_depth"],
    )

    # Mask the entire spec subtree (spec + x + y)
    sm = torch.zeros((B, N), dtype=torch.bool)
    descendants = {spec_pos}
    for child in info["children_of"][spec_pos]:
        descendants.add(child)
    for pos in descendants:
        sm[0, pos] = True

    _, doc_with_mask = agg(
        hidden, batch["batch_info"],
        parent_of_tensor=batch["parent_of_tensor"],
        top_level_key_mask=batch["top_level_key_mask"],
        edges_by_depth=batch["edges_by_depth"],
        parents_by_depth=batch["parents_by_depth"],
        subtree_mask=sm,
    )
    # The two doc_vecs should differ — masking out spec changes the mean
    assert not torch.allclose(doc_no_mask, doc_with_mask, atol=1e-5)
