"""Test CRD generalization: unseen kinds should still get universal structure right.

The model was trained on known K8s kinds. A made-up CRD kind tests whether:
1. Universal structure (apiVersion, kind, metadata, name, labels) still works
2. Kind-specific head gracefully handles unknown kinds
3. Deeper structure (common patterns) still applies

Usage:
    PYTHONPATH=. python scripts/test_crd_generalization.py output_v4/checkpoints/yaml_bert_v4_epoch_15.pt --vocab output_v4/vocab.json
"""
from __future__ import annotations
import _setup_path  # noqa: F401

import argparse

import torch
import torch.nn.functional as F

from yaml_bert.config import YamlBertConfig
from yaml_bert.embedding import YamlBertEmbedding
from yaml_bert.linearizer import YamlLinearizer
from yaml_bert.annotator import DomainAnnotator
from yaml_bert.model import YamlBertModel
from yaml_bert.vocab import Vocabulary, UNIVERSAL_ROOT_KEYS
from yaml_bert.types import NodeType


CRD_YAMLS: list[tuple[str, str, list[tuple[str, str, list[str]]]]] = [
    # (title, yaml_text, [(mask_token, description, expected_in_top5)])

    ("Unknown CRD — universal structure", """\
apiVersion: example.com/v1
kind: MyCustomWidget
metadata:
  name: my-widget
  labels:
    app: widget
spec:
  replicas: 3
  selector:
    matchLabels:
      app: widget
""", [
        ("kind", "root key 'kind'", ["kind"]),
        ("metadata", "root key 'metadata'", ["metadata"]),
        ("name", "metadata.name — labels already present, should predict name", ["name"]),
        ("labels", "metadata.labels — name already present, should predict labels", ["labels"]),
    ]),

    ("Unknown CRD — with namespace", """\
apiVersion: acme.io/v1alpha1
kind: RocketLauncher
metadata:
  name: falcon9
  namespace: space
  labels:
    mission: mars
spec:
  stages: 2
  payload:
    weight: 5000
""", [
        ("name", "metadata.name — namespace+labels present", ["name"]),
        ("namespace", "metadata.namespace — name+labels present", ["namespace", "annotations"]),
        ("labels", "metadata.labels — name+namespace present", ["labels", "annotations"]),
    ]),

    ("Kind head on unknown CRD — expected to fail gracefully", """\
apiVersion: example.com/v1
kind: MyCustomWidget
metadata:
  name: my-widget
spec:
  replicas: 3
  selector:
    matchLabels:
      app: widget
""", [
        ("replicas", "spec.replicas via kind head — unknown kind, expect [UNK]", ["replicas", "[UNK]"]),
        ("selector", "spec.selector via kind head — unknown kind", ["selector", "[UNK]"]),
    ]),

    ("Kind head on known kind — should work", """\
apiVersion: apps/v1
kind: Deployment
metadata:
  name: test
spec:
  replicas: 3
  selector:
    matchLabels:
      app: test
  template:
    metadata:
      labels:
        app: test
    spec:
      containers:
      - name: app
        image: app:latest
""", [
        ("replicas", "Deployment spec.replicas via kind head", ["replicas"]),
        ("selector", "Deployment spec.selector via kind head", ["selector"]),
        ("template", "Deployment spec.template via kind head", ["template"]),
    ]),

    ("Unknown CRD with containers pattern", """\
apiVersion: custom.io/v1
kind: WorkloadRunner
metadata:
  name: runner
spec:
  template:
    spec:
      containers:
      - name: worker
        image: worker:latest
        resources:
          limits:
            cpu: "1"
""", [
        ("metadata", "root 'metadata'", ["metadata"]),
        ("name", "container name — image already present", ["name"]),
        ("image", "container image — name already present", ["image"]),
        ("limits", "resources.limits — should predict requests as sibling", ["limits", "requests"]),
    ]),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=str)
    parser.add_argument("--vocab", type=str, default="output_v4/vocab.json")
    args = parser.parse_args()

    torch.manual_seed(42)
    vocab = Vocabulary.load(args.vocab)
    config = YamlBertConfig()
    emb = YamlBertEmbedding(config=config, key_vocab_size=vocab.key_vocab_size,
                            value_vocab_size=vocab.value_vocab_size)
    model = YamlBertModel(config=config, embedding=emb,
                          simple_vocab_size=vocab.simple_target_vocab_size,
                          kind_vocab_size=vocab.kind_target_vocab_size)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Loaded epoch {checkpoint['epoch']}\n")

    linearizer = YamlLinearizer()
    annotator = DomainAnnotator()
    type_map = {NodeType.KEY: 0, NodeType.VALUE: 1, NodeType.LIST_KEY: 2, NodeType.LIST_VALUE: 3}

    total_tests = 0
    passed_tests = 0

    for title, yaml_text, test_cases in CRD_YAMLS:
        print(f"{'=' * 60}")
        print(f"  {title}")
        print(f"{'=' * 60}")

        nodes = linearizer.linearize(yaml_text)
        annotator.annotate(nodes)

        token_ids, node_types, depths, siblings = [], [], [], []
        for node in nodes:
            if node.node_type in (NodeType.KEY, NodeType.LIST_KEY):
                token_ids.append(vocab.encode_key(node.token))
            else:
                token_ids.append(vocab.encode_value(node.token))
            node_types.append(type_map[node.node_type])
            depths.append(min(node.depth, 15))
            siblings.append(min(node.sibling_index, 31))

        for mask_token, description, expected in test_cases:
            # Find mask position
            mask_pos = -1
            for i, node in enumerate(nodes):
                if node.token == mask_token and mask_pos == -1:
                    mask_pos = i

            if mask_pos == -1:
                print(f"\n  SKIP: '{mask_token}' not found")
                continue

            masked_node = nodes[mask_pos]
            parent_key = Vocabulary.extract_parent_key(masked_node.parent_path)
            use_kind_head = (
                masked_node.depth == 1
                and parent_key not in UNIVERSAL_ROOT_KEYS
                and parent_key != ""
            )

            # Mask and predict
            masked_ids = list(token_ids)
            masked_ids[mask_pos] = vocab.special_tokens["[MASK]"]
            t = lambda x: torch.tensor([x])

            with torch.no_grad():
                simple_logits, kind_logits = model(t(masked_ids), t(node_types), t(depths), t(siblings))

            head_name = "kind" if use_kind_head else "structure"
            if use_kind_head:
                logits = kind_logits
                id_to_target = {v: k for k, v in vocab.kind_target_vocab.items()}
            else:
                logits = simple_logits
                id_to_target = {v: k for k, v in vocab.simple_target_vocab.items()}
            for tok, tok_id in vocab.special_tokens.items():
                id_to_target[tok_id] = tok

            probs = F.softmax(logits[0, mask_pos], dim=-1)
            topk = probs.topk(5)
            predictions = []
            for i in range(5):
                idx = topk.indices[i].item()
                target_str = id_to_target.get(idx, f"[ID:{idx}]")
                key_name = target_str.rsplit("::", 1)[-1]
                predictions.append((key_name, topk.values[i].item()))

            top5_keys = [k for k, _ in predictions]
            hit = any(e in top5_keys for e in expected)
            total_tests += 1
            if hit:
                passed_tests += 1

            status = "PASS" if hit else "FAIL"
            print(f"\n  [{status}] {description} (head={head_name})")
            for i, (key, prob) in enumerate(predictions):
                marker = " <--" if key in expected else ""
                print(f"    {i+1}. '{key}' ({prob:.2%}){marker}")

    print(f"\n{'=' * 60}")
    print(f"  CRD GENERALIZATION: {passed_tests}/{total_tests} tests passed")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
