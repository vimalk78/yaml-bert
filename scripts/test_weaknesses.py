"""Stress-test the model to find weaknesses.

Tests:
1. CRD with known structure — unknown kind but Deployment-like spec
2. Deep nesting — predictions at depth 5+
3. Confidence distribution — histogram of prediction confidences
4. Rare kind performance — less common K8s resources

Usage:
    PYTHONPATH=. python scripts/test_weaknesses.py <checkpoint> --vocab <vocab>
"""
from __future__ import annotations
import _setup_path  # noqa: F401

import argparse

import torch
import torch.nn.functional as F

from yaml_bert.config import YamlBertConfig
from yaml_bert.embedding import YamlBertEmbedding
from yaml_bert.linearizer import YamlLinearizer
from yaml_bert.model import YamlBertModel
from yaml_bert.vocab import Vocabulary, UNIVERSAL_ROOT_KEYS, compute_target
from yaml_bert.dataset import _extract_kind
from yaml_bert.types import NodeType, YamlNode


def load_model(checkpoint_path: str, vocab_path: str):
    torch.manual_seed(42)
    vocab = Vocabulary.load(vocab_path)
    config = YamlBertConfig()
    emb = YamlBertEmbedding(config=config, key_vocab_size=vocab.key_vocab_size,
                            value_vocab_size=vocab.value_vocab_size)
    model = YamlBertModel(config=config, embedding=emb,
                          simple_vocab_size=vocab.simple_target_vocab_size,
                          kind_vocab_size=vocab.kind_target_vocab_size)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, vocab


def predict_at_position(model, vocab, yaml_text, mask_token, top_k=5):
    """Mask a key and return top-k predictions with confidence."""
    linearizer = YamlLinearizer()
    nodes = linearizer.linearize(yaml_text)
    if not nodes:
        return None, None, None

    type_map = {NodeType.KEY: 0, NodeType.VALUE: 1, NodeType.LIST_KEY: 2, NodeType.LIST_VALUE: 3}
    token_ids, node_types, depths, siblings = [], [], [], []
    mask_pos = -1

    for i, node in enumerate(nodes):
        if node.node_type in (NodeType.KEY, NodeType.LIST_KEY):
            token_ids.append(vocab.encode_key(node.token))
        else:
            token_ids.append(vocab.encode_value(node.token))
        node_types.append(type_map[node.node_type])
        depths.append(min(node.depth, 15))
        siblings.append(min(node.sibling_index, 31))

        if node.token == mask_token and mask_pos == -1:
            mask_pos = i

    if mask_pos == -1:
        return None, None, None

    masked_node = nodes[mask_pos]
    kind = _extract_kind(nodes)
    parent_key = Vocabulary.extract_parent_key(masked_node.parent_path)
    use_kind_head = (
        masked_node.depth == 1
        and parent_key not in UNIVERSAL_ROOT_KEYS
        and parent_key != ""
    )

    token_ids[mask_pos] = vocab.special_tokens["[MASK]"]

    t = lambda x: torch.tensor([x])
    with torch.no_grad():
        simple_logits, kind_logits = model(t(token_ids), t(node_types), t(depths), t(siblings))

    if use_kind_head:
        logits = kind_logits
        id_to_target = {v: k for k, v in vocab.kind_target_vocab.items()}
    else:
        logits = simple_logits
        id_to_target = {v: k for k, v in vocab.simple_target_vocab.items()}
    for tok, tok_id in vocab.special_tokens.items():
        id_to_target[tok_id] = tok

    probs = F.softmax(logits[0, mask_pos], dim=-1)
    topk = probs.topk(top_k)
    predictions = []
    for i in range(top_k):
        idx = topk.indices[i].item()
        target_str = id_to_target.get(idx, f"[ID:{idx}]")
        key_name = target_str.rsplit("::", 1)[-1]
        predictions.append((key_name, topk.values[i].item(), target_str))

    return predictions, masked_node.depth, use_kind_head


def test_crd_with_known_structure(model, vocab):
    """Unknown kind but structure identical to Deployment."""
    print("=" * 60)
    print("  Test: CRD with Deployment-like structure")
    print("  Unknown kind 'MyApp' but spec has replicas, selector,")
    print("  template — same as Deployment. Does the model predict")
    print("  Deployment-appropriate keys or fall back to generic?")
    print("=" * 60)

    crd_deployment = """\
apiVersion: apps.example.com/v1
kind: MyApp
metadata:
  name: my-app
spec:
  replicas: 3
  selector:
    matchLabels:
      app: my-app
  template:
    metadata:
      labels:
        app: my-app
    spec:
      containers:
      - name: app
        image: my-app:latest
        ports:
        - containerPort: 8080
"""

    crd_service = """\
apiVersion: networking.example.com/v1
kind: MyService
metadata:
  name: my-svc
spec:
  type: ClusterIP
  ports:
  - port: 80
    targetPort: 8080
  selector:
    app: my-app
"""

    tests = [
        ("MyApp — mask 'replicas' under spec", crd_deployment, "replicas",
         ["replicas", "[UNK]"], "unknown kind → expect [UNK] or closest match"),
        ("MyApp — mask 'image' under containers", crd_deployment, "image",
         ["image"], "universal structure should generalize"),
        ("MyApp — mask 'name' under metadata", crd_deployment, "name",
         ["name"], "universal metadata field"),
        ("MyApp — mask 'labels' under template.metadata", crd_deployment, "labels",
         ["labels"], "universal metadata field"),
        ("MyService — mask 'type' under spec", crd_service, "type",
         ["type", "[UNK]"], "unknown kind → expect [UNK] or closest match"),
        ("MyService — mask 'port' under ports", crd_service, "port",
         ["port", "targetPort"], "universal port structure"),
    ]

    passed = 0
    for name, yaml_text, mask_token, expected, note in tests:
        preds, depth, kind_head = predict_at_position(model, vocab, yaml_text, mask_token)
        if preds is None:
            print(f"\n  [SKIP] {name} — token not found")
            continue

        top1_key = preds[0][0]
        top1_conf = preds[0][1]
        hit = top1_key in expected

        status = "PASS" if hit else "FAIL"
        if hit:
            passed += 1
        print(f"\n  [{status}] {name}")
        print(f"         {note}")
        for key, conf, target in preds[:3]:
            print(f"         {key} ({conf:.1%}) [{target}]")

    print(f"\n  Result: {passed}/{len(tests)}")


def test_deep_nesting(model, vocab):
    """Test predictions at depth 5+."""
    print(f"\n{'=' * 60}")
    print("  Test: Deep nesting (depth 5+)")
    print("  Predictions deep in the tree where training data is sparse.")
    print("=" * 60)

    deep_yaml = """\
apiVersion: apps/v1
kind: Deployment
metadata:
  name: deep-test
spec:
  template:
    spec:
      containers:
      - name: app
        image: app:latest
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
        resources:
          limits:
            memory: 128Mi
            cpu: 250m
          requests:
            memory: 64Mi
            cpu: 100m
        volumeMounts:
        - name: config
          mountPath: /etc/config
          readOnly: true
"""

    tests = [
        ("depth 4: mask 'httpGet' under livenessProbe", "httpGet",
         ["httpGet", "exec", "tcpSocket", "initialDelaySeconds"]),
        ("depth 5: mask 'path' under httpGet", "path",
         ["path", "port", "scheme", "httpHeaders"]),
        ("depth 5: mask 'port' under httpGet", "port",
         ["port", "path", "scheme"]),
        ("depth 4: mask 'memory' under limits", "memory",
         ["memory", "cpu"]),
        ("depth 4: mask 'mountPath' under volumeMounts", "mountPath",
         ["mountPath", "name", "readOnly", "subPath"]),
        ("depth 4: mask 'readOnly' under volumeMounts", "readOnly",
         ["readOnly", "mountPath", "subPath"]),
    ]

    passed = 0
    for name, mask_token, expected in tests:
        preds, depth, kind_head = predict_at_position(model, vocab, deep_yaml, mask_token)
        if preds is None:
            print(f"\n  [SKIP] {name} — token not found")
            continue

        top1_key = preds[0][0]
        top1_conf = preds[0][1]
        hit = any(top1_key == e for e in expected)

        status = "PASS" if hit else "FAIL"
        if hit:
            passed += 1
        print(f"\n  [{status}] {name} (actual depth: {depth})")
        for key, conf, target in preds[:3]:
            print(f"         {key} ({conf:.1%}) [{target}]")

    print(f"\n  Result: {passed}/{len(tests)}")


def test_rare_kinds(model, vocab):
    """Test less common resource types."""
    print(f"\n{'=' * 60}")
    print("  Test: Rare kind performance")
    print("  Resources that appear less often in training data.")
    print("=" * 60)

    tests = [
        ("NetworkPolicy — mask 'podSelector'", """\
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: deny-all
spec:
  podSelector:
    matchLabels:
      app: web
  policyTypes:
  - Ingress
""", "podSelector", ["podSelector", "ingress", "egress", "policyTypes"]),

        ("PodDisruptionBudget — mask 'minAvailable'", """\
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: pdb
spec:
  minAvailable: 1
  selector:
    matchLabels:
      app: web
""", "minAvailable", ["minAvailable", "maxUnavailable", "selector"]),

        ("ResourceQuota — mask 'hard'", """\
apiVersion: v1
kind: ResourceQuota
metadata:
  name: quota
spec:
  hard:
    pods: "10"
    requests.cpu: "4"
""", "hard", ["hard", "scopeSelector", "scopes"]),

        ("LimitRange — mask 'limits'", """\
apiVersion: v1
kind: LimitRange
metadata:
  name: limits
spec:
  limits:
  - default:
      cpu: 500m
    type: Container
""", "limits", ["limits"]),

        ("StorageClass — mask 'provisioner'", """\
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: fast
provisioner: kubernetes.io/gce-pd
parameters:
  type: pd-ssd
""", "provisioner", ["provisioner", "parameters", "reclaimPolicy"]),
    ]

    passed = 0
    for name, yaml_text, mask_token, expected in tests:
        preds, depth, kind_head = predict_at_position(model, vocab, yaml_text, mask_token)
        if preds is None:
            print(f"\n  [SKIP] {name} — token not found")
            continue

        top1_key = preds[0][0]
        top1_conf = preds[0][1]
        hit = any(top1_key == e for e in expected)

        status = "PASS" if hit else "FAIL"
        if hit:
            passed += 1
        print(f"\n  [{status}] {name}")
        for key, conf, target in preds[:3]:
            print(f"         {key} ({conf:.1%}) [{target}]")

    print(f"\n  Result: {passed}/{len(tests)}")


def test_confidence_distribution(model, vocab):
    """Analyze confidence distribution across many predictions."""
    print(f"\n{'=' * 60}")
    print("  Test: Confidence distribution")
    print("  How confident is the model across diverse predictions?")
    print("=" * 60)

    yamls_and_tokens = [
        # High confidence expected (common, unambiguous)
        ("""\
apiVersion: apps/v1
kind: Deployment
metadata:
  name: test
spec:
  replicas: 3
""", "kind", "root key"),
        ("""\
apiVersion: apps/v1
kind: Deployment
metadata:
  name: test
spec:
  replicas: 3
""", "name", "metadata child"),
        ("""\
apiVersion: apps/v1
kind: Deployment
metadata:
  name: test
spec:
  replicas: 3
  template:
    spec:
      containers:
      - name: app
        image: app:latest
""", "image", "container field"),
        # Medium confidence expected (some ambiguity)
        ("""\
apiVersion: apps/v1
kind: Deployment
metadata:
  name: test
  labels:
    app: test
    version: v1
""", "version", "label key"),
        ("""\
apiVersion: v1
kind: Pod
metadata:
  name: test
spec:
  containers:
  - name: app
    image: app:latest
    resources:
      limits:
        cpu: 500m
""", "limits", "resources child"),
        # Low confidence expected (ambiguous position)
        ("""\
apiVersion: apps/v1
kind: Deployment
metadata:
  name: test
  labels:
    app: test
  annotations:
    note: test
""", "annotations", "metadata sibling of labels"),
        ("""\
apiVersion: v1
kind: Service
metadata:
  name: svc
spec:
  selector:
    app: test
  ports:
  - port: 80
    targetPort: 8080
""", "targetPort", "port field"),
    ]

    buckets = {"99-100%": 0, "90-99%": 0, "70-90%": 0, "50-70%": 0, "30-50%": 0, "<30%": 0}
    all_confs = []

    print(f"\n  {'Description':<30} {'Predicted':<20} {'Confidence':>10}")
    print(f"  {'-'*30} {'-'*20} {'-'*10}")

    for yaml_text, mask_token, desc in yamls_and_tokens:
        preds, depth, kind_head = predict_at_position(model, vocab, yaml_text, mask_token)
        if preds is None:
            continue
        top1_key, top1_conf, _ = preds[0]
        all_confs.append(top1_conf)
        print(f"  {desc:<30} {top1_key:<20} {top1_conf:>10.1%}")

        if top1_conf >= 0.99:
            buckets["99-100%"] += 1
        elif top1_conf >= 0.90:
            buckets["90-99%"] += 1
        elif top1_conf >= 0.70:
            buckets["70-90%"] += 1
        elif top1_conf >= 0.50:
            buckets["50-70%"] += 1
        elif top1_conf >= 0.30:
            buckets["30-50%"] += 1
        else:
            buckets["<30%"] += 1

    print(f"\n  Distribution:")
    for bucket, count in buckets.items():
        bar = "#" * (count * 4)
        print(f"    {bucket:>8}: {count} {bar}")

    if all_confs:
        avg = sum(all_confs) / len(all_confs)
        print(f"\n  Average confidence: {avg:.1%}")
        print(f"  Min: {min(all_confs):.1%}, Max: {max(all_confs):.1%}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=str)
    parser.add_argument("--vocab", type=str, default="output_v5/vocab.json")
    args = parser.parse_args()

    model, vocab = load_model(args.checkpoint, args.vocab)

    test_crd_with_known_structure(model, vocab)
    test_deep_nesting(model, vocab)
    test_rare_kinds(model, vocab)
    test_confidence_distribution(model, vocab)


if __name__ == "__main__":
    main()
