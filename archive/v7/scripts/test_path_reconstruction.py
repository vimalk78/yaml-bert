"""Test that the model reconstructs full tree paths through bigram/trigram chains.

For a deeply nested key, mask at each depth and verify the model predicts
the correct compound target — confirming it knows each edge in the path.

Usage:
    PYTHONPATH=. python scripts/test_path_reconstruction.py output_v4/checkpoints/yaml_bert_v4_epoch_15.pt --vocab output_v4/vocab.json
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


TEST_YAMLS: list[tuple[str, str]] = [
    ("Deployment — full depth path", """\
apiVersion: apps/v1
kind: Deployment
metadata:
  name: web
spec:
  replicas: 3
  template:
    metadata:
      labels:
        app: web
    spec:
      containers:
      - name: nginx
        image: nginx:latest
        resources:
          limits:
            cpu: "1"
            memory: 512Mi
          requests:
            cpu: 500m
            memory: 256Mi
        ports:
        - containerPort: 80
          protocol: TCP
"""),

    ("Service — shallow path", """\
apiVersion: v1
kind: Service
metadata:
  name: web-svc
  labels:
    app: web
spec:
  type: ClusterIP
  selector:
    app: web
  ports:
  - port: 80
    targetPort: 8080
    protocol: TCP
"""),
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

    total = 0
    passed = 0

    for title, yaml_text in TEST_YAMLS:
        print(f"{'=' * 70}")
        print(f"  {title}")
        print(f"{'=' * 70}")

        nodes = linearizer.linearize(yaml_text)
        annotator.annotate(nodes)

        # Extract kind
        kind = ""
        for i, n in enumerate(nodes):
            if n.token == "kind" and n.depth == 0 and i + 1 < len(nodes):
                kind = nodes[i + 1].token
                break

        # Encode
        token_ids, node_types, depths, siblings = [], [], [], []
        for node in nodes:
            if node.node_type in (NodeType.KEY, NodeType.LIST_KEY):
                token_ids.append(vocab.encode_key(node.token))
            else:
                token_ids.append(vocab.encode_value(node.token))
            node_types.append(type_map[node.node_type])
            depths.append(min(node.depth, 15))
            siblings.append(min(node.sibling_index, 31))

        # Test each KEY position
        for pos, node in enumerate(nodes):
            if node.node_type not in (NodeType.KEY, NodeType.LIST_KEY):
                continue

            # Determine expected target and head
            parent_key = Vocabulary.extract_parent_key(node.parent_path)
            use_kind_head = (
                node.depth == 1
                and parent_key not in UNIVERSAL_ROOT_KEYS
                and parent_key != ""
            )

            if node.depth == 0:
                expected_target = node.token
            elif use_kind_head:
                expected_target = f"{kind}::{parent_key}::{node.token}"
            else:
                expected_target = f"{parent_key}::{node.token}" if parent_key else node.token

            # Check if target exists in vocab
            if use_kind_head:
                target_id = vocab.encode_kind_target(expected_target)
                id_to_target = {v: k for k, v in vocab.kind_target_vocab.items()}
            else:
                target_id = vocab.encode_simple_target(expected_target)
                id_to_target = {v: k for k, v in vocab.simple_target_vocab.items()}
            for tok, tok_id in vocab.special_tokens.items():
                id_to_target[tok_id] = tok

            # Skip if target not in vocab (rare tokens filtered by min_freq)
            if target_id == vocab.special_tokens["[UNK]"]:
                continue

            # Mask and predict
            masked_ids = list(token_ids)
            masked_ids[pos] = vocab.special_tokens["[MASK]"]
            t = lambda x: torch.tensor([x])

            with torch.no_grad():
                simple_logits, kind_logits = model(t(masked_ids), t(node_types), t(depths), t(siblings))

            logits = kind_logits if use_kind_head else simple_logits
            probs = F.softmax(logits[0, pos], dim=-1)
            top1_id = probs.argmax().item()
            top1_target = id_to_target.get(top1_id, f"[ID:{top1_id}]")
            top1_prob = probs[top1_id].item()

            hit = (top1_id == target_id)
            total += 1
            if hit:
                passed += 1

            head = "kind" if use_kind_head else "structure"
            status = "PASS" if hit else "FAIL"
            path_display = node.parent_path if node.parent_path else "(root)"

            if hit:
                print(f"  [{status}] depth={node.depth} path={path_display} → {expected_target} ({top1_prob:.1%}) [{head}]")
            else:
                print(f"  [{status}] depth={node.depth} path={path_display}")
                print(f"         expected: {expected_target}")
                print(f"         got:      {top1_target} ({top1_prob:.1%}) [{head}]")

    print(f"\n{'=' * 70}")
    print(f"  Path Reconstruction: {passed}/{total} edges correct")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
