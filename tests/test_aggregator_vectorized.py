"""Numerical equivalence: per-doc reference path vs vectorized path."""
import torch

from yaml_bert.aggregator import TreeAggregator
from yaml_bert.linearizer import YamlLinearizer
from yaml_bert.v8_dataset import (
    V8Dataset, compute_children_info, v8_collate_fn,
)
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
