"""Tests for empty-parent probing in suggest.py.

Two layers:
1. Unit tests on `_find_empty_mapping_paths` — pure YAML walking, no model.
2. Integration test that exercises full `suggest_missing_fields` on
   variants with empty parents. Skipped if the v6.1 checkpoint isn't
   present.
"""
from __future__ import annotations

import os

import pytest

from yaml_bert.suggest import _find_empty_mapping_paths


def test_empty_mapping_value():
    """`securityContext: {}` is detected as an empty parent."""
    paths = _find_empty_mapping_paths("securityContext: {}\n")
    assert paths == [("securityContext", "securityContext", 1)]


def test_none_value_treated_as_empty_parent():
    """`resources:` (no value) parses as None in PyYAML; should still probe.
    This is the bug fix discovered when testing live on the HF Space."""
    paths = _find_empty_mapping_paths("resources:\n")
    assert paths == [("resources", "resources", 1)]


def test_scalar_value_is_not_an_empty_parent():
    """Real scalar values (`image: nginx`) must NOT be treated as
    empty parents — probing under them makes no sense."""
    paths = _find_empty_mapping_paths("image: nginx:1.25\n")
    assert paths == []


def test_nested_empty_mapping():
    """Empty mappings at depth report the correct child depth."""
    yaml_str = """spec:
  securityContext: {}
"""
    paths = _find_empty_mapping_paths(yaml_str)
    assert paths == [("spec.securityContext", "securityContext", 2)]


def test_nested_none_value():
    """`spec.securityContext:` (None) reports the correct child depth."""
    yaml_str = """spec:
  securityContext:
"""
    paths = _find_empty_mapping_paths(yaml_str)
    assert paths == [("spec.securityContext", "securityContext", 2)]


def test_inside_list_item():
    """An empty mapping inside a list item under a key — depth follows
    the linearizer convention (list items don't increment depth)."""
    yaml_str = """spec:
  containers:
  - name: nginx
    securityContext: {}
"""
    paths = _find_empty_mapping_paths(yaml_str)
    # securityContext sits under containers[0]; linearizer puts its
    # depth at 2 (containers is at 1, list item dict at 1, keys at 2),
    # so children would be at depth 3.
    assert ("spec.containers.0.securityContext", "securityContext", 3) in paths


def test_multiple_empties_in_one_doc():
    """A typical incomplete Pod can have several empty parents at once."""
    yaml_str = """apiVersion: v1
kind: Pod
metadata:
  name: nginx
spec:
  containers:
  - name: nginx
    image: nginx:1.25
    resources:
    livenessProbe: {}
    securityContext:
"""
    paths = _find_empty_mapping_paths(yaml_str)
    found_paths = {p[0] for p in paths}
    assert "spec.containers.0.resources" in found_paths
    assert "spec.containers.0.livenessProbe" in found_paths
    assert "spec.containers.0.securityContext" in found_paths


def test_empty_yaml_returns_nothing():
    """Edge case: empty input parses to None and returns []."""
    assert _find_empty_mapping_paths("") == []
    assert _find_empty_mapping_paths("\n") == []


# ----------------------------------------------------------------------------
# Integration test — full suggest pipeline. Skipped if the v6.1 checkpoint
# isn't on disk (CI environments without model artifacts will skip cleanly).
# ----------------------------------------------------------------------------

CHECKPOINT_PATH = os.path.join(
    os.path.dirname(__file__), "..",
    "output_v6.1_lever1_only_seed42", "checkpoints", "yaml_bert_v4_epoch_30.pt",
)
VOCAB_PATH = os.path.join(
    os.path.dirname(__file__), "..",
    "output_v6.1_lever1_only_seed42", "vocab.json",
)


@pytest.fixture(scope="module")
def model_and_vocab():
    """Load v6.1 model and vocab once per module. Skips if missing.

    v6.1 was trained without tree_bias (v7 addition), so we construct the
    model with tree_bias_enabled=False to match the checkpoint's
    architecture exactly.
    """
    if not (os.path.exists(CHECKPOINT_PATH) and os.path.exists(VOCAB_PATH)):
        pytest.skip("v6.1 checkpoint not available")
    import torch
    from yaml_bert.config import YamlBertConfig
    from yaml_bert.embedding import YamlBertEmbedding
    from yaml_bert.model import YamlBertModel
    from yaml_bert.vocab import Vocabulary

    vocab = Vocabulary.load(VOCAB_PATH)
    config = YamlBertConfig(tree_bias_enabled=False)  # v6.1 didn't have tree_bias
    emb = YamlBertEmbedding(
        config=config,
        key_vocab_size=vocab.key_vocab_size,
        value_vocab_size=vocab.value_vocab_size,
    )
    model = YamlBertModel(
        config=config, embedding=emb,
        simple_vocab_size=vocab.simple_target_vocab_size,
        kind_vocab_size=vocab.kind_target_vocab_size,
    )
    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, vocab


def _surfaces_under(suggestions, parent_path, expected_key):
    return any(
        s["parent_path"] == parent_path and s["missing_key"] == expected_key
        for s in suggestions
    )


def test_integration_resources_empty(model_and_vocab):
    """Empty `resources:` under a container should surface limits + requests."""
    from yaml_bert.suggest import suggest_missing_fields
    model, vocab = model_and_vocab
    yaml_str = """apiVersion: v1
kind: Pod
metadata:
  name: nginx
spec:
  containers:
  - name: nginx
    image: nginx:1.25
    resources:
"""
    suggestions, _ = suggest_missing_fields(model, vocab, yaml_str, threshold=0.1)
    parent = "spec.containers.0.resources"
    assert _surfaces_under(suggestions, parent, "limits") or \
           _surfaces_under(suggestions, parent, "requests"), \
           f"expected limits or requests under {parent}, got {suggestions}"


def test_integration_volume_claim_template_spec_empty(model_and_vocab):
    """Empty `volumeClaimTemplates[0].spec: {}` should surface accessModes."""
    from yaml_bert.suggest import suggest_missing_fields
    model, vocab = model_and_vocab
    yaml_str = """apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: db
spec:
  serviceName: db
  replicas: 3
  selector:
    matchLabels: {app: db}
  template:
    metadata: {labels: {app: db}}
    spec:
      containers:
      - name: db
        image: postgres:15
  volumeClaimTemplates:
  - metadata: {name: data}
    spec: {}
"""
    suggestions, _ = suggest_missing_fields(model, vocab, yaml_str, threshold=0.1)
    parent = "spec.volumeClaimTemplates.0.spec"
    assert _surfaces_under(suggestions, parent, "accessModes"), \
           f"expected accessModes under {parent}, got {suggestions}"
