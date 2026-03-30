"""Test how much value embeddings contribute to key prediction.

Replaces all value token IDs with [UNK] and measures accuracy drop.

Usage:
    PYTHONPATH=. python scripts/test_value_ablation.py output_v1/yaml_bert_v1_final.pt --vocab output_v1/vocab.json
"""
from __future__ import annotations
import _setup_path  # noqa: F401

import argparse
import random

import torch
import torch.nn.functional as F

from yaml_bert.config import YamlBertConfig
from yaml_bert.embedding import YamlBertEmbedding
from yaml_bert.model import YamlBertModel
from yaml_bert.linearizer import YamlLinearizer
from yaml_bert.annotator import DomainAnnotator
from yaml_bert.dataset import _extract_kind
from yaml_bert.vocab import Vocabulary
from yaml_bert.types import NodeType, YamlNode


TEST_YAMLS: list[str] = [
    """\
apiVersion: apps/v1
kind: Deployment
metadata:
  name: web
  labels:
    app: web
spec:
  replicas: 3
  selector:
    matchLabels:
      app: web
  template:
    metadata:
      labels:
        app: web
    spec:
      containers:
      - name: web
        image: nginx:1.21
        ports:
        - containerPort: 80
        resources:
          limits:
            memory: 128Mi
""",
    """\
apiVersion: v1
kind: Service
metadata:
  name: web-service
  labels:
    app: web
spec:
  selector:
    app: web
  ports:
  - port: 80
    targetPort: 8080
    protocol: TCP
  type: ClusterIP
""",
    """\
apiVersion: v1
kind: Pod
metadata:
  name: debug
  namespace: default
spec:
  containers:
  - name: debug
    image: busybox
    command: ["sh", "-c", "sleep 3600"]
    resources:
      limits:
        memory: 64Mi
""",
    """\
apiVersion: v1
kind: ConfigMap
metadata:
  name: app-config
  namespace: production
data:
  DB_HOST: postgres
  DB_PORT: "5432"
  LOG_LEVEL: info
""",
    """\
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: db
  labels:
    app: postgres
spec:
  serviceName: db-headless
  replicas: 3
  selector:
    matchLabels:
      app: postgres
  template:
    metadata:
      labels:
        app: postgres
    spec:
      containers:
      - name: postgres
        image: postgres:14
        ports:
        - containerPort: 5432
""",
]


def evaluate(
    model: YamlBertModel,
    vocab: Vocabulary,
    yaml_texts: list[str],
    ablate_values: bool = False,
) -> dict[str, float]:
    """Run key prediction on test YAMLs, optionally ablating values."""
    linearizer = YamlLinearizer()
    annotator = DomainAnnotator()
    model.eval()

    type_map = {NodeType.KEY: 0, NodeType.VALUE: 1, NodeType.LIST_KEY: 2, NodeType.LIST_VALUE: 3}
    unk_id: int = vocab.special_tokens["[UNK]"]
    mask_id: int = vocab.special_tokens["[MASK]"]

    total_masked: int = 0
    top1_correct: int = 0
    top5_correct: int = 0

    random.seed(42)

    for yaml_text in yaml_texts:
        nodes = linearizer.linearize(yaml_text)
        if not nodes:
            continue
        annotator.annotate(nodes)

        token_ids: list[int] = []
        node_types: list[int] = []
        depths: list[int] = []
        siblings: list[int] = []
        parent_keys: list[int] = []

        for node in nodes:
            if node.node_type in (NodeType.KEY, NodeType.LIST_KEY):
                token_ids.append(vocab.encode_key(node.token))
            else:
                if ablate_values:
                    token_ids.append(unk_id)  # Replace all values with [UNK]
                else:
                    token_ids.append(vocab.encode_value(node.token))
            node_types.append(type_map[node.node_type])
            depths.append(min(node.depth, 15))
            siblings.append(min(node.sibling_index, 31))
            parent_keys.append(vocab.encode_key(Vocabulary.extract_parent_key(node.parent_path)))

        kind = _extract_kind(nodes)
        kind_id = vocab.encode_kind(kind)
        kind_ids = [kind_id] * len(nodes)

        # Mask each key node one at a time and predict
        for i, node in enumerate(nodes):
            if node.node_type not in (NodeType.KEY, NodeType.LIST_KEY):
                continue

            original_id: int = token_ids[i]
            masked_ids: list[int] = token_ids.copy()
            masked_ids[i] = mask_id

            t = lambda x: torch.tensor([x])
            with torch.no_grad():
                key_logits, _, _ = model(
                    t(masked_ids), t(node_types), t(depths), t(siblings), t(parent_keys),
                    kind_ids=t(kind_ids),
                )

            probs = F.softmax(key_logits[0, i], dim=-1)
            top5 = probs.topk(5).indices.tolist()

            if top5[0] == original_id:
                top1_correct += 1
            if original_id in top5:
                top5_correct += 1
            total_masked += 1

    return {
        "top1": top1_correct / max(total_masked, 1),
        "top5": top5_correct / max(total_masked, 1),
        "total": total_masked,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=str)
    parser.add_argument("--vocab", type=str, default="output_v1/vocab.json")
    args = parser.parse_args()

    torch.manual_seed(42)
    vocab = Vocabulary.load(args.vocab)
    config = YamlBertConfig()
    emb = YamlBertEmbedding(
        config=config,
        key_vocab_size=vocab.key_vocab_size,
        value_vocab_size=vocab.value_vocab_size,
        kind_vocab_size=vocab.kind_vocab_size,
    )
    model = YamlBertModel(
        config=config, embedding=emb,
        key_vocab_size=vocab.key_vocab_size,
        kind_vocab_size=vocab.kind_vocab_size,
    )
    cp = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(cp["model_state_dict"], strict=False)
    print(f"Loaded epoch {cp.get('epoch', '?')}")

    print("\n=== Normal (values present) ===")
    normal = evaluate(model, vocab, TEST_YAMLS, ablate_values=False)
    print(f"  Top-1: {normal['top1']:.2%}")
    print(f"  Top-5: {normal['top5']:.2%}")
    print(f"  Total masked: {normal['total']}")

    print("\n=== Ablated (all values replaced with [UNK]) ===")
    ablated = evaluate(model, vocab, TEST_YAMLS, ablate_values=True)
    print(f"  Top-1: {ablated['top1']:.2%}")
    print(f"  Top-5: {ablated['top5']:.2%}")
    print(f"  Total masked: {ablated['total']}")

    print("\n=== Impact ===")
    drop1: float = normal["top1"] - ablated["top1"]
    drop5: float = normal["top5"] - ablated["top5"]
    print(f"  Top-1 drop: {drop1:.2%}")
    print(f"  Top-5 drop: {drop5:.2%}")

    if drop1 < 0.02:
        print(f"\n  Verdict: Values contribute MINIMALLY (<2% drop)")
    elif drop1 < 0.10:
        print(f"\n  Verdict: Values contribute MODERATELY ({drop1:.1%} drop)")
    else:
        print(f"\n  Verdict: Values contribute SIGNIFICANTLY ({drop1:.1%} drop)")


if __name__ == "__main__":
    main()
