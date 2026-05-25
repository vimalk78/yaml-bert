"""Bigger-boat tests — designed to fail v5's saturated capability suite.

The pretrain capability suite saturates at ~92/93 across all ablation
variants — it can no longer differentiate trained models. These tests
target specific failure modes the existing suite does NOT probe, informed
by:

  1. v5's known structural-test failures (spec/status [UNK] at 99%,
     missing-metadata predicting 'kind' at 100%, etc.)
  2. The CRD-pollution finding (CRDs are 46% of training tokens)
  3. Long-tail annotation/label keys dropped at min_freq=100

Categories:

  vocab_gap         Positions where (parent_key, token) bigrams were
                    likely below min_freq=100 in training. Model should
                    NOT confidently predict [UNK].

  crd_pollution     Deep positions in real manifests where CRD-schema
                    keys (properties/items/description/type) might leak in.

  annotation_keys   Long-form keys with domain prefixes
                    (app.kubernetes.io/*, prometheus.io/*). Many fall
                    below min_freq=100; model should predict reasonable
                    completions, not [UNK].

  confidence_calib  Probe whether the model is appropriately uncertain
                    in ambiguous contexts and confident in clear ones.

Each test shows top-5 predictions in the output regardless of pass/fail
so the model's actual behavior is always visible.
"""
from __future__ import annotations
import _setup_path  # noqa: F401

import argparse

import torch

from test_capabilities import (  # type: ignore
    Capability,
    TestCase,
    TestResult,
    run_test,
)

from yaml_bert.config import YamlBertConfig
from yaml_bert.embedding import YamlBertEmbedding
from yaml_bert.model import YamlBertModel
from yaml_bert.vocab import Vocabulary


CRD_KEYS = ["properties", "items", "description", "type"]
NEVER_TOP1 = ["[UNK]", "[PAD]", "[MASK]"]


def build_bigger_boat() -> list[Capability]:
    capabilities: list[Capability] = []

    # ============================================================
    # vocab_gap
    # ============================================================
    capabilities.append(Capability(
        name="vocab_gap",
        phase="pretrain",
        description=(
            "Positions where the (parent_key, child_key) bigram was likely "
            "below min_freq=100 in training. Model should not collapse to [UNK]."
        ),
        tests=[
            TestCase(
                name="status.replicas in a Deployment (known v5 failure)",
                yaml_text="""\
apiVersion: apps/v1
kind: Deployment
metadata:
  name: app
spec:
  replicas: 3
  template:
    spec:
      containers:
      - name: web
        image: nginx
status:
  replicas: 3
  availableReplicas: 3
""",
                # The 2nd 'replicas' is under status. test_capabilities masks first
                # occurrence — so we mask 'availableReplicas' instead, which only
                # exists under status.
                mask_token="availableReplicas",
                expect_not_top1=NEVER_TOP1,
                expect_in_top5=["availableReplicas", "readyReplicas",
                                "updatedReplicas", "replicas"],
            ),
            TestCase(
                name="status.conditions in a Pod",
                yaml_text="""\
apiVersion: v1
kind: Pod
metadata:
  name: app
spec:
  containers:
  - name: web
    image: nginx
status:
  phase: Running
  conditions:
  - type: Ready
    status: "True"
""",
                mask_token="conditions",
                expect_not_top1=NEVER_TOP1,
                expect_in_top5=["conditions", "phase", "containerStatuses",
                                "podIP", "hostIP"],
            ),
            TestCase(
                name="HPA status currentMetrics",
                yaml_text="""\
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: app-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: app
  minReplicas: 2
  maxReplicas: 10
status:
  currentReplicas: 5
  desiredReplicas: 6
  conditions:
  - type: ScalingActive
    status: "True"
""",
                mask_token="desiredReplicas",
                expect_not_top1=NEVER_TOP1,
                expect_in_top5=["desiredReplicas", "currentReplicas",
                                "conditions", "observedGeneration"],
            ),
            TestCase(
                name="Service status.loadBalancer.ingress",
                yaml_text="""\
apiVersion: v1
kind: Service
metadata:
  name: web
spec:
  type: LoadBalancer
  selector:
    app: web
  ports:
  - port: 80
status:
  loadBalancer:
    ingress:
    - ip: 10.0.0.1
      hostname: web.example.com
""",
                mask_token="hostname",
                expect_not_top1=NEVER_TOP1,
                expect_in_top5=["hostname", "ip", "ports"],
            ),
        ],
    ))

    # ============================================================
    # crd_pollution
    # ============================================================
    capabilities.append(Capability(
        name="crd_pollution",
        phase="pretrain",
        description=(
            "Deep positions in real manifests where CRD-schema keys "
            "(properties/items/description/type) should NOT leak into top-5."
        ),
        tests=[
            TestCase(
                name="configMapKeyRef.key deep in env (mask key, not name)",
                yaml_text="""\
apiVersion: apps/v1
kind: Deployment
metadata:
  name: app
spec:
  replicas: 3
  template:
    spec:
      containers:
      - name: web
        image: nginx:1.21
        env:
        - name: API_KEY
          valueFrom:
            configMapKeyRef:
              name: cfg
              key: api-key
""",
                # The first 'key' is under configMapKeyRef (deep). No earlier 'key'.
                mask_token="key",
                expect_not_top1=NEVER_TOP1 + CRD_KEYS,
                expect_in_top5=["key", "name", "optional"],
            ),
            TestCase(
                name="httpHeaders value in a readinessProbe",
                yaml_text="""\
apiVersion: v1
kind: Pod
metadata:
  name: app
spec:
  containers:
  - name: web
    image: nginx
    readinessProbe:
      httpGet:
        path: /healthz
        port: 8080
        httpHeaders:
        - name: Authorization
          value: Bearer token
""",
                mask_token="value",
                expect_not_top1=NEVER_TOP1 + CRD_KEYS,
                expect_in_top5=["value", "name"],
            ),
            TestCase(
                name="volumeMounts subPath",
                yaml_text="""\
apiVersion: apps/v1
kind: Deployment
metadata:
  name: app
spec:
  template:
    spec:
      containers:
      - name: web
        image: nginx
        volumeMounts:
        - name: config
          mountPath: /etc/config
          subPath: app.conf
""",
                mask_token="subPath",
                expect_not_top1=NEVER_TOP1 + CRD_KEYS,
                expect_in_top5=["subPath", "readOnly", "mountPropagation"],
            ),
            TestCase(
                name="secretKeyRef.optional (rare flag, deep position)",
                yaml_text="""\
apiVersion: apps/v1
kind: Deployment
metadata:
  name: app
spec:
  template:
    spec:
      containers:
      - name: web
        image: nginx
        env:
        - name: DB_PASS
          valueFrom:
            secretKeyRef:
              name: db-secret
              key: password
              optional: false
""",
                mask_token="optional",
                expect_not_top1=NEVER_TOP1 + CRD_KEYS,
                expect_in_top5=["optional", "key", "name"],
            ),
        ],
    ))

    # ============================================================
    # annotation_keys
    # ============================================================
    capabilities.append(Capability(
        name="annotation_keys",
        phase="pretrain",
        description=(
            "Long-form keys with domain prefixes (app.kubernetes.io/*, "
            "prometheus.io/*). Many fall below min_freq=100; model should "
            "predict reasonable continuations, not [UNK]."
        ),
        tests=[
            TestCase(
                name="app.kubernetes.io/name in labels",
                yaml_text="""\
apiVersion: apps/v1
kind: Deployment
metadata:
  name: app
  labels:
    app.kubernetes.io/name: my-app
    app.kubernetes.io/version: "1.0"
    app.kubernetes.io/component: web
spec:
  template:
    spec:
      containers:
      - name: web
        image: nginx
""",
                mask_token="app.kubernetes.io/version",
                expect_not_top1=NEVER_TOP1,
                # Top-5 expectation is open: any label-key continuation is OK.
                # The pass condition is just 'not [UNK]'.
                expect_in_top5=["app.kubernetes.io/version",
                                "app.kubernetes.io/name",
                                "app.kubernetes.io/component",
                                "app.kubernetes.io/instance",
                                "app.kubernetes.io/part-of",
                                "app.kubernetes.io/managed-by"],
            ),
            TestCase(
                name="prometheus.io/scrape in annotations",
                yaml_text="""\
apiVersion: v1
kind: Service
metadata:
  name: web
  annotations:
    prometheus.io/scrape: "true"
    prometheus.io/port: "9090"
    prometheus.io/path: /metrics
spec:
  selector:
    app: web
  ports:
  - port: 80
""",
                mask_token="prometheus.io/port",
                expect_not_top1=NEVER_TOP1,
                expect_in_top5=["prometheus.io/port", "prometheus.io/scrape",
                                "prometheus.io/path", "prometheus.io/scheme"],
            ),
        ],
    ))

    # ============================================================
    # confidence_calib
    # ============================================================
    capabilities.append(Capability(
        name="confidence_calib",
        phase="pretrain",
        description=(
            "Probes whether the model's confidence is appropriately "
            "calibrated: high on clear contexts, lower on ambiguous ones."
        ),
        tests=[
            TestCase(
                name="High-confidence: containers under spec.template.spec",
                yaml_text="""\
apiVersion: apps/v1
kind: Deployment
metadata:
  name: app
spec:
  template:
    spec:
      containers:
      - name: web
        image: nginx
""",
                mask_token="containers",
                expect_in_top5=["containers"],
                expect_confidence_above=0.80,
            ),
            TestCase(
                name="Low-confidence: ambiguous position with multiple valid keys",
                yaml_text="""\
apiVersion: v1
kind: Pod
metadata:
  name: app
spec:
  containers:
  - name: web
    image: nginx
    securityContext:
      allowPrivilegeEscalation: false
""",
                mask_token="allowPrivilegeEscalation",
                # Many securityContext fields exist; model shouldn't be 99%
                # confident this specific one is the answer
                expect_not_top1=NEVER_TOP1,
                expect_confidence_below=0.80,
            ),
            TestCase(
                name="High-confidence: replicas under Deployment.spec",
                yaml_text="""\
apiVersion: apps/v1
kind: Deployment
metadata:
  name: app
spec:
  replicas: 3
  template:
    spec:
      containers:
      - name: web
        image: nginx
""",
                mask_token="replicas",
                expect_in_top5=["replicas"],
                expect_confidence_above=0.50,
            ),
        ],
    ))

    return capabilities


def run_bigger_boat_tests(
    run_test_fn,
    show_passes: bool = False,
    header: str = "YAML-BERT Bigger Boat",
) -> tuple[int, int]:
    """Run all 13 bigger-boat tests using run_test_fn(test) -> TestResult.

    Returns (grand_passed, grand_total).
    """
    print(header)
    print("=" * 75)

    capabilities = build_bigger_boat()
    grand_total = grand_passed = 0
    for cap in capabilities:
        per_test: list[TestResult] = []
        for t in cap.tests:
            per_test.append(run_test_fn(t))
        passed = sum(1 for r in per_test if r.passed)
        total = len(per_test)
        marker = "PASS" if passed == total else "FAIL"
        print(f"\n[{marker}] {cap.name}: {passed}/{total}")
        for r in per_test:
            show = (not r.passed) or show_passes
            m = "  ✓" if r.passed else "  ✗"
            if show:
                top5 = ", ".join(f"{k} ({c:.1%})" for k, c in r.predictions[:5])
                print(f"{m} {r.test_name}")
                print(f"      top-5: {top5}")
                if not r.passed:
                    print(f"      reason: {r.details}")
            else:
                print(f"{m} {r.test_name}")
        grand_total += total
        grand_passed += passed

    print()
    print("=" * 75)
    print(f"OVERALL: {grand_passed}/{grand_total} ({grand_passed / grand_total * 100:.1f}%)")
    print("=" * 75)
    return grand_passed, grand_total


def main() -> None:
    parser = argparse.ArgumentParser(description="Bigger-boat tests for YAML-BERT")
    parser.add_argument("checkpoint", type=str)
    parser.add_argument("--vocab", type=str, required=True)
    parser.add_argument("--show-passes", action="store_true",
                        help="Show top-5 for passing tests too (default: show only failures)")
    args = parser.parse_args()

    vocab: Vocabulary = Vocabulary.load(args.vocab)
    config: YamlBertConfig = YamlBertConfig()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    torch.manual_seed(42)
    emb = YamlBertEmbedding(config=config, key_vocab_size=vocab.key_vocab_size,
                            value_vocab_size=vocab.value_vocab_size)
    model = YamlBertModel(config=config, embedding=emb,
                          simple_vocab_size=vocab.simple_target_vocab_size,
                          kind_vocab_size=vocab.kind_target_vocab_size)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    header = f"YAML-BERT Bigger Boat — checkpoint epoch {checkpoint['epoch']}"

    def run_test_fn(t):
        return run_test(model, vocab, t)

    run_bigger_boat_tests(run_test_fn, show_passes=args.show_passes, header=header)


if __name__ == "__main__":
    main()
