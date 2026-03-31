"""Hard structural tests for YAML-BERT.

Tests whether the model learned real K8s structure vs just frequency patterns.

Usage:
    python test_structural.py output_v1/checkpoints/yaml_bert_epoch_10.pt
"""
from __future__ import annotations
import _setup_path  # noqa: F401

import argparse

import torch

from yaml_bert.config import YamlBertConfig
from yaml_bert.annotator import DomainAnnotator
from yaml_bert.embedding import YamlBertEmbedding
from yaml_bert.linearizer import YamlLinearizer
from yaml_bert.model import YamlBertModel
from yaml_bert.vocab import Vocabulary, UNIVERSAL_ROOT_KEYS
from yaml_bert.types import NodeType
import torch.nn.functional as F


def load_model(checkpoint_path: str, vocab_path: str = "output_v1/vocab.json") -> tuple[YamlBertModel, Vocabulary]:
    vocab: Vocabulary = Vocabulary.load(vocab_path)
    config: YamlBertConfig = YamlBertConfig()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    torch.manual_seed(42)
    emb = YamlBertEmbedding(config=config, key_vocab_size=vocab.key_vocab_size, value_vocab_size=vocab.value_vocab_size)
    model = YamlBertModel(config=config, embedding=emb, simple_vocab_size=vocab.simple_target_vocab_size, kind_vocab_size=vocab.kind_target_vocab_size)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, vocab


def _extract_key_from_target(target: str) -> str:
    """Extract the raw key name from a compound target."""
    return target.rsplit("::", 1)[-1]


def predict_masked_key(
    model: YamlBertModel,
    vocab: Vocabulary,
    yaml_text: str,
    mask_position: int,
    k: int = 10,
) -> list[tuple[str, float]]:
    """Mask a key at the given position and return top-k predictions."""
    linearizer = YamlLinearizer()
    annotator = DomainAnnotator()
    nodes = linearizer.linearize(yaml_text)
    annotator.annotate(nodes)

    type_map = {NodeType.KEY: 0, NodeType.VALUE: 1, NodeType.LIST_KEY: 2, NodeType.LIST_VALUE: 3}
    token_ids, node_types, depths, siblings = [], [], [], []

    for node in nodes:
        if node.node_type in (NodeType.KEY, NodeType.LIST_KEY):
            token_ids.append(vocab.encode_key(node.token))
        else:
            token_ids.append(vocab.encode_value(node.token))
        node_types.append(type_map[node.node_type])
        depths.append(min(node.depth, 15))
        siblings.append(min(node.sibling_index, 31))

    token_ids[mask_position] = vocab.special_tokens["[MASK]"]

    masked_node = nodes[mask_position]
    parent_key = Vocabulary.extract_parent_key(masked_node.parent_path)
    use_kind_head = (
        masked_node.depth == 1
        and parent_key not in UNIVERSAL_ROOT_KEYS
        and parent_key != ""
    )

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

    probs = F.softmax(logits[0, mask_position], dim=-1)
    topk = probs.topk(k)
    return [(_extract_key_from_target(id_to_target.get(topk.indices[i].item(), f"[ID:{topk.indices[i].item()}]")), topk.values[i].item()) for i in range(k)]


def print_predictions(
    label: str,
    predictions: list[tuple[str, float]],
    expected: str | None = None,
    should_not_predict: str | None = None,
) -> bool:
    """Print predictions and return True if test passes."""
    print(f"\n  {label}")
    passed: bool = True
    for i, (key, prob) in enumerate(predictions[:5]):
        marker: str = ""
        if expected and key == expected:
            marker = " <-- EXPECTED"
        if should_not_predict and key == should_not_predict:
            marker = " <-- SHOULD NOT APPEAR"
            passed = False
        print(f"    {i+1}. '{key}' ({prob:.2%}){marker}")

    if expected:
        top_keys = [k for k, _ in predictions[:5]]
        if expected in top_keys:
            print(f"    PASS: '{expected}' in top 5")
        else:
            print(f"    FAIL: '{expected}' not in top 5")
            passed = False

    if should_not_predict:
        top1 = predictions[0][0]
        if top1 == should_not_predict:
            print(f"    FAIL: '{should_not_predict}' is top prediction (should not be)")
            passed = False
        else:
            print(f"    PASS: '{should_not_predict}' is not top prediction")

    return passed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=str)
    parser.add_argument("--vocab", type=str, default="output_v1/vocab.json")
    args = parser.parse_args()

    model, vocab = load_model(args.checkpoint, args.vocab)
    print(f"Model loaded.\n")
    total_tests: int = 0
    passed_tests: int = 0

    # ========================================================
    print("=" * 70)
    print("TEST 1: Kind conditioning")
    print("  Does masking a key under spec give different predictions")
    print("  depending on whether kind=Deployment or kind=Service?")
    print("=" * 70)

    deployment_yaml: str = """\
apiVersion: apps/v1
kind: Deployment
metadata:
  name: test
spec:
  replicas: 3
  selector:
    matchLabels:
      app: test
"""
    nodes = YamlLinearizer().linearize(deployment_yaml)
    replicas_pos: int = next(i for i, n in enumerate(nodes) if n.token == "replicas")
    preds = predict_masked_key(model, vocab, deployment_yaml, mask_position=replicas_pos)
    total_tests += 1
    if print_predictions("Deployment: mask 'replicas' under spec (expected: replicas)", preds, expected="replicas"):
        passed_tests += 1

    service_yaml: str = """\
apiVersion: v1
kind: Service
metadata:
  name: test
spec:
  type: ClusterIP
  ports:
  - port: 80
  selector:
    app: test
"""
    nodes = YamlLinearizer().linearize(service_yaml)
    type_pos: int = next(i for i, n in enumerate(nodes) if n.token == "type")
    preds = predict_masked_key(model, vocab, service_yaml, mask_position=type_pos)
    total_tests += 1
    if print_predictions("Service: mask 'type' under spec (expected: type)", preds, expected="type"):
        passed_tests += 1

    # ========================================================
    print("\n" + "=" * 70)
    print("TEST 2: Wrong parent")
    print("  Put 'containers' under metadata. Model should predict")
    print("  metadata-appropriate keys, not 'containers'.")
    print("=" * 70)

    wrong_parent_yaml: str = """\
apiVersion: v1
kind: Pod
metadata:
  name: test
  containers:
  - name: nginx
spec:
  containers:
  - name: nginx
"""
    # Mask 'containers' under metadata (position 3)
    # The model should predict metadata-appropriate keys like 'labels', 'namespace', 'annotations'
    nodes = YamlLinearizer().linearize(wrong_parent_yaml)
    # Find the position of first 'containers'
    containers_pos: int = next(i for i, n in enumerate(nodes) if n.token == "containers")
    preds = predict_masked_key(model, vocab, wrong_parent_yaml, mask_position=containers_pos)
    total_tests += 1
    if print_predictions(
        "Mask 'containers' under metadata (should predict metadata children, not containers)",
        preds,
        should_not_predict="containers",
    ):
        passed_tests += 1

    # ========================================================
    print("\n" + "=" * 70)
    print("TEST 3: Depth awareness")
    print("  Mask a key at depth 0. Model should predict top-level keys.")
    print("  Mask a key at depth 3+. Model should predict nested keys.")
    print("=" * 70)

    depth_yaml: str = """\
apiVersion: apps/v1
kind: Deployment
metadata:
  name: test
  labels:
    app: test
spec:
  replicas: 3
  template:
    metadata:
      labels:
        app: test
    spec:
      containers:
      - name: nginx
        image: nginx:1.21
"""
    # Mask 'kind' at depth 0 (position 2)
    preds = predict_masked_key(model, vocab, depth_yaml, mask_position=2)
    total_tests += 1
    if print_predictions("Depth 0: mask 'kind' (expected: kind)", preds, expected="kind"):
        passed_tests += 1

    # Mask 'image' deep in containers (find its position)
    nodes = YamlLinearizer().linearize(depth_yaml)
    image_pos: int = next(i for i, n in enumerate(nodes) if n.token == "image")
    preds = predict_masked_key(model, vocab, depth_yaml, mask_position=image_pos)
    total_tests += 1
    if print_predictions(
        f"Depth {nodes[image_pos].depth}: mask 'image' under containers (expected: image)",
        preds,
        expected="image",
    ):
        passed_tests += 1

    # ========================================================
    print("\n" + "=" * 70)
    print("TEST 4: spec vs status distinction")
    print("  'replicas' under spec (desired) vs status (actual).")
    print("  Model should predict 'replicas' in both but from different context.")
    print("=" * 70)

    spec_status_yaml: str = """\
apiVersion: apps/v1
kind: Deployment
metadata:
  name: test
spec:
  replicas: 3
status:
  replicas: 2
  availableReplicas: 2
"""
    nodes = YamlLinearizer().linearize(spec_status_yaml)

    # Find replicas positions
    replicas_positions = [i for i, n in enumerate(nodes) if n.token == "replicas"]

    for pos in replicas_positions:
        parent = nodes[pos].parent_path
        preds = predict_masked_key(model, vocab, spec_status_yaml, mask_position=pos)
        total_tests += 1
        if print_predictions(
            f"Mask 'replicas' under {parent} (expected: replicas)",
            preds,
            expected="replicas",
        ):
            passed_tests += 1

    # ========================================================
    print("\n" + "=" * 70)
    print("TEST 5: Nonsense YAML — confidence drop")
    print("  Made-up structure should produce low-confidence predictions.")
    print("=" * 70)

    valid_yaml: str = """\
apiVersion: v1
kind: Pod
metadata:
  name: test
spec:
  containers:
  - name: nginx
    image: nginx:1.21
"""

    nonsense_yaml: str = """\
apiVersion: v1
kind: Pod
spec:
  metadata:
    containers:
      replicas:
        kind:
          apiVersion: wrong
"""

    # Compare confidence on valid vs nonsense
    nodes_valid = YamlLinearizer().linearize(valid_yaml)
    nodes_nonsense = YamlLinearizer().linearize(nonsense_yaml)

    # Mask 'containers' in both
    valid_pos: int = next(i for i, n in enumerate(nodes_valid) if n.token == "containers")
    nonsense_pos: int = next(i for i, n in enumerate(nodes_nonsense) if n.token == "containers")

    preds_valid = predict_masked_key(model, vocab, valid_yaml, mask_position=valid_pos)
    preds_nonsense = predict_masked_key(model, vocab, nonsense_yaml, mask_position=nonsense_pos)

    valid_conf: float = preds_valid[0][1]
    nonsense_conf: float = preds_nonsense[0][1]

    print(f"\n  Valid YAML: top prediction '{preds_valid[0][0]}' confidence: {valid_conf:.2%}")
    print(f"  Nonsense YAML: top prediction '{preds_nonsense[0][0]}' confidence: {nonsense_conf:.2%}")

    total_tests += 1
    if valid_conf > nonsense_conf:
        print(f"    PASS: Valid YAML has higher confidence ({valid_conf:.2%} > {nonsense_conf:.2%})")
        passed_tests += 1
    else:
        print(f"    FAIL: Nonsense YAML has equal or higher confidence")

    # ========================================================
    print("\n" + "=" * 70)
    print("TEST 6: Missing required field")
    print("  Remove metadata entirely. Mask position where it should be.")
    print("  Model should predict 'metadata'.")
    print("=" * 70)

    no_metadata_yaml: str = """\
apiVersion: apps/v1
kind: Deployment
spec:
  replicas: 3
"""
    # Position 2 is 'spec', but in a normal YAML it would be 'metadata'
    # The model should predict 'metadata' because it knows the structure
    preds = predict_masked_key(model, vocab, no_metadata_yaml, mask_position=2)
    total_tests += 1
    if print_predictions(
        "Mask 'spec' at position where metadata should be (expected: metadata)",
        preds,
        expected="metadata",
    ):
        passed_tests += 1

    # ========================================================
    print("\n" + "=" * 70)
    print(f"RESULTS: {passed_tests}/{total_tests} tests passed")
    print("=" * 70)


if __name__ == "__main__":
    main()
