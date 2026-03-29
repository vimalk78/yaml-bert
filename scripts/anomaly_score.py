"""YAML-BERT Anomaly Scorer.

Scores each key in a YAML document by how "surprising" it is at its tree position.
Syntactically valid YAML can be semantically wrong — this tool catches that.

For each key node:
- Mask it and ask the model to predict what key should be there
- If the actual key has low probability → anomalous (structurally unusual)
- If the actual key has high probability → normal (expected at this position)

Usage:
    python anomaly_score.py output_v1/checkpoints/yaml_bert_epoch_10.pt --yaml-file my_manifest.yaml
    python anomaly_score.py output_v1/checkpoints/yaml_bert_epoch_10.pt --yaml-text "..."
    python anomaly_score.py output_v1/checkpoints/yaml_bert_epoch_10.pt --run-examples
"""
from __future__ import annotations
import _setup_path  # noqa: F401

import argparse
import os

import torch
import torch.nn.functional as F

from yaml_bert.config import YamlBertConfig
from yaml_bert.annotator import DomainAnnotator
from yaml_bert.dataset import _extract_kind
from yaml_bert.embedding import YamlBertEmbedding
from yaml_bert.linearizer import YamlLinearizer
from yaml_bert.model import YamlBertModel
from yaml_bert.vocab import Vocabulary
from yaml_bert.types import NodeType


def score_yaml(
    model: YamlBertModel,
    vocab: Vocabulary,
    yaml_text: str,
    threshold: float = 0.01,
) -> list[dict]:
    """Score each key in a YAML document for anomalies.

    Args:
        model: Trained YAML-BERT model
        vocab: Vocabulary
        yaml_text: Raw YAML text
        threshold: Keys with probability below this are flagged as anomalous

    Returns:
        List of dicts with key info and anomaly scores, sorted by score (most anomalous first)
    """
    linearizer: YamlLinearizer = YamlLinearizer()
    annotator: DomainAnnotator = DomainAnnotator()
    nodes = linearizer.linearize(yaml_text)
    if not nodes:
        return []
    annotator.annotate(nodes)

    type_map = {NodeType.KEY: 0, NodeType.VALUE: 1, NodeType.LIST_KEY: 2, NodeType.LIST_VALUE: 3}
    token_ids: list[int] = []
    node_types: list[int] = []
    depths: list[int] = []
    siblings: list[int] = []
    parent_keys: list[int] = []

    for node in nodes:
        if node.node_type in (NodeType.KEY, NodeType.LIST_KEY):
            token_ids.append(vocab.encode_key(node.token))
        else:
            token_ids.append(vocab.encode_value(node.token))
        node_types.append(type_map[node.node_type])
        depths.append(min(node.depth, 15))
        siblings.append(min(node.sibling_index, 31))
        parent_keys.append(vocab.encode_key(Vocabulary.extract_parent_key(node.parent_path)))

    kind: str = _extract_kind(nodes)
    kind_id: int = vocab.encode_kind(kind)
    kind_ids: list[int] = [kind_id] * len(nodes)

    results: list[dict] = []
    mask_id: int = vocab.special_tokens["[MASK]"]

    for i, node in enumerate(nodes):
        if node.node_type not in (NodeType.KEY, NodeType.LIST_KEY):
            continue

        # Mask this key and predict
        masked_ids: list[int] = token_ids.copy()
        original_id: int = masked_ids[i]
        masked_ids[i] = mask_id

        t = lambda x: torch.tensor([x])
        with torch.no_grad():
            logits, _, _ = model(
                t(masked_ids), t(node_types), t(depths), t(siblings), t(parent_keys),
                kind_ids=t(kind_ids),
            )

        probs = F.softmax(logits[0, i], dim=-1)
        actual_prob: float = probs[original_id].item()

        topk = probs.topk(5)
        top_predictions: list[tuple[str, float]] = [
            (vocab.decode_key(topk.indices[j].item()), topk.values[j].item())
            for j in range(5)
        ]

        parent_key: str = Vocabulary.extract_parent_key(node.parent_path)
        is_anomalous: bool = actual_prob < threshold

        results.append({
            "position": i,
            "key": node.token,
            "depth": node.depth,
            "parent_key": parent_key,
            "parent_path": node.parent_path,
            "actual_probability": actual_prob,
            "is_anomalous": is_anomalous,
            "top_predictions": top_predictions,
        })

    # Sort by probability (most anomalous first)
    results.sort(key=lambda x: x["actual_probability"])
    return results


def print_report(
    results: list[dict],
    title: str = "",
    show_all: bool = False,
) -> None:
    """Print anomaly report."""
    if title:
        print(f"\n{'=' * 70}")
        print(f"  {title}")
        print(f"{'=' * 70}")

    anomalies: list[dict] = [r for r in results if r["is_anomalous"]]
    normal: list[dict] = [r for r in results if not r["is_anomalous"]]

    if anomalies:
        print(f"\n  ANOMALIES FOUND: {len(anomalies)}")
        for r in anomalies:
            print(f"\n    [{r['actual_probability']:.2%}] '{r['key']}' at depth={r['depth']}, parent='{r['parent_key']}'")
            print(f"         path: {r['parent_path']}")
            print(f"         Model expected:")
            for j, (key, prob) in enumerate(r["top_predictions"]):
                marker: str = " <-- actual" if key == r["key"] else ""
                print(f"           {j+1}. '{key}' ({prob:.2%}){marker}")
    else:
        print(f"\n  No anomalies detected.")

    if show_all and normal:
        print(f"\n  Normal keys ({len(normal)}):")
        for r in sorted(normal, key=lambda x: x["actual_probability"]):
            print(f"    [{r['actual_probability']:.2%}] '{r['key']}' (parent='{r['parent_key']}')")

    total: int = len(results)
    avg_prob: float = sum(r["actual_probability"] for r in results) / max(total, 1)
    min_prob: float = min(r["actual_probability"] for r in results) if results else 0
    print(f"\n  Summary: {len(anomalies)} anomalies / {total} keys, avg confidence: {avg_prob:.2%}, min: {min_prob:.2%}")


EXAMPLES: list[tuple[str, str]] = [
    # 1. Valid Deployment — should have no anomalies
    ("Valid Deployment (should be clean)", """\
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
"""),

    # 2. Deployment with replicas under metadata — semantically wrong
    ("Deployment: replicas under metadata (WRONG)", """\
apiVersion: apps/v1
kind: Deployment
metadata:
  name: web
  replicas: 3
spec:
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
        image: nginx
"""),

    # 3. Service with containers — Services don't have containers
    ("Service with containers (WRONG)", """\
apiVersion: v1
kind: Service
metadata:
  name: svc
spec:
  containers:
  - name: app
    image: nginx
  ports:
  - port: 80
"""),

    # 4. Pod with replicas — Pods don't have replicas
    ("Pod with replicas (WRONG)", """\
apiVersion: v1
kind: Pod
metadata:
  name: pod
spec:
  replicas: 3
  containers:
  - name: app
    image: nginx
"""),

    # 5. Deployment with containerPort in Service-style position
    ("Deployment: port instead of containerPort (WRONG)", """\
apiVersion: apps/v1
kind: Deployment
metadata:
  name: web
spec:
  replicas: 1
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
        image: nginx
        ports:
        - port: 80
          targetPort: 8080
"""),

    # 6. ConfigMap with spec — ConfigMaps use data, not spec
    ("ConfigMap with spec (WRONG)", """\
apiVersion: v1
kind: ConfigMap
metadata:
  name: config
spec:
  key1: value1
  key2: value2
"""),

    # 7. Deployment missing template — has raw containers under spec
    ("Deployment: containers directly under spec (WRONG)", """\
apiVersion: apps/v1
kind: Deployment
metadata:
  name: web
spec:
  replicas: 1
  containers:
  - name: web
    image: nginx
"""),

    # 8. Secret with spec instead of data
    ("Secret with spec instead of data (WRONG)", """\
apiVersion: v1
kind: Secret
metadata:
  name: mysecret
type: Opaque
spec:
  username: admin
  password: secret123
"""),

    # 9. Valid Service — should be clean
    ("Valid Service (should be clean)", """\
apiVersion: v1
kind: Service
metadata:
  name: my-service
spec:
  selector:
    app: web
  ports:
  - port: 80
    targetPort: 8080
    protocol: TCP
  type: ClusterIP
"""),

    # 10. Ingress with containerPort — Ingress doesn't have containerPort
    ("Ingress with containerPort (WRONG)", """\
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: ingress
spec:
  containerPort: 80
  rules:
  - host: example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: svc
            port:
              number: 80
"""),

    # 11. Job with replicas — Jobs don't use replicas
    ("Job with replicas (WRONG)", """\
apiVersion: batch/v1
kind: Job
metadata:
  name: job
spec:
  replicas: 3
  template:
    spec:
      containers:
      - name: worker
        image: busybox
      restartPolicy: Never
"""),

    # 12. Role with spec — Roles use rules, not spec
    ("Role with spec instead of rules (WRONG)", """\
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: reader
  namespace: default
spec:
- apiGroups: [""]
  resources: ["pods"]
  verbs: ["get"]
"""),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="YAML-BERT Anomaly Scorer")
    parser.add_argument("checkpoint", type=str)
    parser.add_argument("--vocab", type=str, default="output_v1/vocab.json")
    parser.add_argument("--yaml-file", type=str, default=None)
    parser.add_argument("--yaml-text", type=str, default=None)
    parser.add_argument("--threshold", type=float, default=0.01,
                        help="Probability threshold for anomaly (default: 1%%)")
    parser.add_argument("--show-all", action="store_true", help="Show normal keys too")
    parser.add_argument("--run-examples", action="store_true", help="Run built-in examples")
    args = parser.parse_args()

    print("Loading model...")
    vocab: Vocabulary = Vocabulary.load(args.vocab)
    config: YamlBertConfig = YamlBertConfig()
    emb = YamlBertEmbedding(config=config, key_vocab_size=vocab.key_vocab_size, value_vocab_size=vocab.value_vocab_size, kind_vocab_size=vocab.kind_vocab_size)
    model = YamlBertModel(config=config, embedding=emb, key_vocab_size=vocab.key_vocab_size, kind_vocab_size=vocab.kind_vocab_size)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.eval()
    print(f"Loaded epoch {checkpoint['epoch']}")

    if args.yaml_file:
        with open(args.yaml_file) as f:
            yaml_text: str = f.read()
        results = score_yaml(model, vocab, yaml_text, threshold=args.threshold)
        print_report(results, title=args.yaml_file, show_all=args.show_all)

    elif args.yaml_text:
        results = score_yaml(model, vocab, args.yaml_text, threshold=args.threshold)
        print_report(results, show_all=args.show_all)

    elif args.run_examples:
        detected: int = 0
        total_wrong: int = 0

        for title, yaml_text in EXAMPLES:
            results = score_yaml(model, vocab, yaml_text, threshold=args.threshold)
            print_report(results, title=title)

            is_wrong: bool = "WRONG" in title
            has_anomalies: bool = any(r["is_anomalous"] for r in results)

            if is_wrong:
                total_wrong += 1
                if has_anomalies:
                    detected += 1

        print(f"\n{'=' * 70}")
        print(f"DETECTION SUMMARY")
        print(f"{'=' * 70}")
        print(f"Semantically wrong YAMLs detected: {detected}/{total_wrong}")
        print(f"{'=' * 70}")

    else:
        print("Provide --yaml-file, --yaml-text, or --run-examples")


if __name__ == "__main__":
    main()
