"""Test: does the model know that readinessProbe is expected when livenessProbe exists?

Usage:
    CUDA_VISIBLE_DEVICES="" PYTHONPATH=. python scripts/test_probe_convention.py output_v3_quick/checkpoints/yaml_bert_epoch_5.pt --vocab output_v3_quick/vocab.json
"""
from __future__ import annotations
import _setup_path  # noqa: F401

import argparse

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


def predict_at_position(
    model: YamlBertModel,
    vocab: Vocabulary,
    yaml_text: str,
    mask_token: str,
    k: int = 10,
) -> list[tuple[str, float]]:
    linearizer = YamlLinearizer()
    annotator = DomainAnnotator()
    nodes = linearizer.linearize(yaml_text)
    annotator.annotate(nodes)

    type_map = {NodeType.KEY: 0, NodeType.VALUE: 1, NodeType.LIST_KEY: 2, NodeType.LIST_VALUE: 3}
    token_ids, node_types, depths, siblings, parent_keys = [], [], [], [], []
    mask_pos = -1

    for i, node in enumerate(nodes):
        if node.node_type in (NodeType.KEY, NodeType.LIST_KEY):
            token_ids.append(vocab.encode_key(node.token))
        else:
            token_ids.append(vocab.encode_value(node.token))
        node_types.append(type_map[node.node_type])
        depths.append(min(node.depth, 15))
        siblings.append(min(node.sibling_index, 31))
        parent_keys.append(vocab.encode_key(Vocabulary.extract_parent_key(node.parent_path)))

        if node.token == mask_token and mask_pos == -1:
            mask_pos = i

    if mask_pos == -1:
        return []

    kind = _extract_kind(nodes)
    kind_id = vocab.encode_kind(kind)
    kind_ids = [kind_id] * len(nodes)

    token_ids[mask_pos] = vocab.special_tokens["[MASK]"]

    t = lambda x: torch.tensor([x])
    with torch.no_grad():
        key_logits, _, _ = model(
            t(token_ids), t(node_types), t(depths), t(siblings), t(parent_keys),
            kind_ids=t(kind_ids),
        )

    probs = F.softmax(key_logits[0, mask_pos], dim=-1)
    topk = probs.topk(k)
    return [(vocab.decode_key(topk.indices[i].item()), topk.values[i].item()) for i in range(k)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=str)
    parser.add_argument("--vocab", type=str, default="output_v1/vocab.json")
    args = parser.parse_args()

    vocab = Vocabulary.load(args.vocab)
    config = YamlBertConfig()
    emb = YamlBertEmbedding(config=config, key_vocab_size=vocab.key_vocab_size,
                            value_vocab_size=vocab.value_vocab_size, kind_vocab_size=vocab.kind_vocab_size)
    model = YamlBertModel(config=config, embedding=emb, key_vocab_size=vocab.key_vocab_size,
                          kind_vocab_size=vocab.kind_vocab_size)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.eval()
    print(f"Loaded epoch {checkpoint['epoch']}")

    # ==========================================================
    print("\n" + "=" * 60)
    print("TEST 1: Container with livenessProbe — does model expect readinessProbe?")
    print("=" * 60)

    yaml_with_both = """\
apiVersion: apps/v1
kind: Deployment
metadata:
  name: app
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
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
"""
    preds = predict_at_position(model, vocab, yaml_with_both, "readinessProbe")
    print("  Mask 'readinessProbe' (livenessProbe exists as sibling):")
    for i, (key, prob) in enumerate(preds[:5]):
        marker = " <--" if key == "readinessProbe" else ""
        print(f"    {i+1}. '{key}' ({prob:.2%}){marker}")

    # ==========================================================
    print("\n" + "=" * 60)
    print("TEST 2: Container WITHOUT livenessProbe — does model still expect readinessProbe?")
    print("=" * 60)

    yaml_without_liveness = """\
apiVersion: apps/v1
kind: Deployment
metadata:
  name: app
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
        - containerPort: 80
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
"""
    preds = predict_at_position(model, vocab, yaml_without_liveness, "readinessProbe")
    print("  Mask 'readinessProbe' (no livenessProbe present):")
    for i, (key, prob) in enumerate(preds[:5]):
        marker = " <--" if key == "readinessProbe" else ""
        print(f"    {i+1}. '{key}' ({prob:.2%}){marker}")

    # ==========================================================
    print("\n" + "=" * 60)
    print("TEST 3: Container with resources.requests — does model expect resources.limits?")
    print("=" * 60)

    yaml_resources = """\
apiVersion: apps/v1
kind: Deployment
metadata:
  name: app
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
        resources:
          limits:
            memory: 128Mi
            cpu: 500m
          requests:
            memory: 64Mi
"""
    preds = predict_at_position(model, vocab, yaml_resources, "limits")
    print("  Mask 'limits' (requests exists as sibling):")
    for i, (key, prob) in enumerate(preds[:5]):
        marker = " <--" if key == "limits" else ""
        print(f"    {i+1}. '{key}' ({prob:.2%}){marker}")

    # ==========================================================
    print("\n" + "=" * 60)
    print("TEST 4: Deployment without resources — what does model predict after image?")
    print("=" * 60)

    yaml_no_resources = """\
apiVersion: apps/v1
kind: Deployment
metadata:
  name: app
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
        resources:
          limits:
            memory: 128Mi
"""
    preds = predict_at_position(model, vocab, yaml_no_resources, "resources")
    print("  Mask 'resources' — what does model think should be here?")
    for i, (key, prob) in enumerate(preds[:5]):
        marker = " <--" if key == "resources" else ""
        print(f"    {i+1}. '{key}' ({prob:.2%}){marker}")

    # ==========================================================
    print("\n" + "=" * 60)
    print("TEST 5: Pod without securityContext — does model expect it?")
    print("=" * 60)

    yaml_security = """\
apiVersion: v1
kind: Pod
metadata:
  name: pod
spec:
  securityContext:
    runAsNonRoot: true
  containers:
  - name: app
    image: nginx
"""
    preds = predict_at_position(model, vocab, yaml_security, "securityContext")
    print("  Mask 'securityContext' under Pod spec:")
    for i, (key, prob) in enumerate(preds[:5]):
        marker = " <--" if key == "securityContext" else ""
        print(f"    {i+1}. '{key}' ({prob:.2%}){marker}")

    # ==========================================================
    print("\n" + "=" * 60)
    print("TEST 6: Convention — namespace usually present in metadata")
    print("=" * 60)

    yaml_ns = """\
apiVersion: apps/v1
kind: Deployment
metadata:
  name: app
  namespace: production
  labels:
    app: web
spec:
  replicas: 3
"""
    preds = predict_at_position(model, vocab, yaml_ns, "namespace")
    print("  Mask 'namespace' under metadata:")
    for i, (key, prob) in enumerate(preds[:5]):
        marker = " <--" if key == "namespace" else ""
        print(f"    {i+1}. '{key}' ({prob:.2%}){marker}")


if __name__ == "__main__":
    main()
