"""Shared structural test runner for v7 and v8 tests.

Extracted from model_tests/test_structural.py so v8 tests can import
run_tests without pulling in v7-specific model classes.
"""
from __future__ import annotations

from yaml_bert.linearizer import YamlLinearizer


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


def run_tests(predict_fn) -> tuple[int, int]:
    """Run all 9 structural tests using predict_fn(yaml_text, mask_position) -> list[(key, prob)].

    Returns (passed_tests, total_tests).
    """
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
    preds = predict_fn(deployment_yaml, replicas_pos)
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
    preds = predict_fn(service_yaml, type_pos)
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
    preds = predict_fn(wrong_parent_yaml, containers_pos)
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
    preds = predict_fn(depth_yaml, 2)
    total_tests += 1
    if print_predictions("Depth 0: mask 'kind' (expected: kind)", preds, expected="kind"):
        passed_tests += 1

    # Mask 'image' deep in containers (find its position)
    nodes = YamlLinearizer().linearize(depth_yaml)
    image_pos: int = next(i for i, n in enumerate(nodes) if n.token == "image")
    preds = predict_fn(depth_yaml, image_pos)
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
        preds = predict_fn(spec_status_yaml, pos)
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

    preds_valid = predict_fn(valid_yaml, valid_pos)
    preds_nonsense = predict_fn(nonsense_yaml, nonsense_pos)

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
    preds = predict_fn(no_metadata_yaml, 2)
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

    return passed_tests, total_tests
