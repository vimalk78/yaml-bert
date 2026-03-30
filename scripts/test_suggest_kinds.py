"""Test the suggest tool across many resource kinds — including obscure ones.

Usage:
    PYTHONPATH=. python scripts/test_suggest_kinds.py output_v1/yaml_bert_v1_final.pt --vocab output_v1/vocab.json
"""
from __future__ import annotations
import _setup_path  # noqa: F401

import argparse

import torch

from yaml_bert.config import YamlBertConfig
from yaml_bert.embedding import YamlBertEmbedding
from yaml_bert.model import YamlBertModel
from yaml_bert.suggest import suggest_missing_fields
from yaml_bert.vocab import Vocabulary


TEST_YAMLS: list[tuple[str, str]] = [

    ("PodDisruptionBudget — bare minimum", """\
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: my-pdb
  labels:
    app: web
spec:
  minAvailable: 1
  selector:
    matchLabels:
      app: web
"""),

    ("ResourceQuota — just hard limits", """\
apiVersion: v1
kind: ResourceQuota
metadata:
  name: compute-quota
  namespace: production
  labels:
    app: infra
spec:
  hard:
    requests.cpu: "4"
    requests.memory: 8Gi
    limits.cpu: "8"
    limits.memory: 16Gi
"""),

    ("LimitRange — only default limits", """\
apiVersion: v1
kind: LimitRange
metadata:
  name: mem-limit
  namespace: default
  labels:
    app: infra
spec:
  limits:
  - default:
      memory: 512Mi
    defaultRequest:
      memory: 256Mi
    type: Container
"""),

    ("StorageClass — minimal", """\
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: fast
  labels:
    tier: storage
provisioner: kubernetes.io/gce-pd
parameters:
  type: pd-ssd
"""),

    ("PriorityClass — basic", """\
apiVersion: scheduling.k8s.io/v1
kind: PriorityClass
metadata:
  name: high-priority
  labels:
    tier: system
value: 1000000
globalDefault: false
description: High priority workloads
"""),

    ("Endpoints — minimal", """\
apiVersion: v1
kind: Endpoints
metadata:
  name: my-service
  namespace: default
  labels:
    app: web
subsets:
- addresses:
  - ip: 10.0.0.1
  ports:
  - port: 8080
    protocol: TCP
"""),

    ("ServiceAccount — bare", """\
apiVersion: v1
kind: ServiceAccount
metadata:
  name: my-sa
  namespace: default
  labels:
    app: backend
"""),

    ("PersistentVolume — hostPath", """\
apiVersion: v1
kind: PersistentVolume
metadata:
  name: pv-data
  labels:
    tier: storage
spec:
  capacity:
    storage: 10Gi
  accessModes:
  - ReadWriteOnce
  hostPath:
    path: /data
"""),

    ("PersistentVolumeClaim — basic", """\
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: pvc-data
  namespace: default
  labels:
    app: db
spec:
  accessModes:
  - ReadWriteOnce
  resources:
    requests:
      storage: 5Gi
"""),

    ("Ingress — single rule, no TLS", """\
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: my-ingress
  labels:
    app: web
spec:
  rules:
  - host: example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: web
            port:
              number: 80
"""),

    ("CronJob — minimal", """\
apiVersion: batch/v1
kind: CronJob
metadata:
  name: cleanup
  labels:
    app: maintenance
spec:
  schedule: "0 3 * * *"
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: cleanup
            image: busybox
            command: ["sh", "-c", "echo cleanup"]
          restartPolicy: Never
"""),

    ("DaemonSet — bare monitoring agent", """\
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: node-monitor
  labels:
    app: monitor
spec:
  selector:
    matchLabels:
      app: monitor
  template:
    metadata:
      labels:
        app: monitor
    spec:
      containers:
      - name: agent
        image: monitor-agent:latest
"""),

    ("ValidatingWebhookConfiguration — minimal", """\
apiVersion: admissionregistration.k8s.io/v1
kind: ValidatingWebhookConfiguration
metadata:
  name: validate-pods
  labels:
    app: admission
webhooks:
- name: validate.pods.example.com
  clientConfig:
    service:
      name: webhook-service
      namespace: default
      path: /validate
  rules:
  - apiGroups: [""]
    apiVersions: ["v1"]
    operations: ["CREATE"]
    resources: ["pods"]
  admissionReviewVersions: ["v1"]
  sideEffects: None
"""),

    ("CustomResourceDefinition — skeleton", """\
apiVersion: apiextensions.k8s.io/v1
kind: CustomResourceDefinition
metadata:
  name: myresources.example.com
  labels:
    app: platform
spec:
  group: example.com
  names:
    kind: MyResource
    plural: myresources
    singular: myresource
  scope: Namespaced
  versions:
  - name: v1
    served: true
    storage: true
"""),

    ("MutatingWebhookConfiguration — minimal", """\
apiVersion: admissionregistration.k8s.io/v1
kind: MutatingWebhookConfiguration
metadata:
  name: mutate-pods
  labels:
    app: admission
webhooks:
- name: mutate.pods.example.com
  clientConfig:
    service:
      name: webhook-service
      namespace: default
      path: /mutate
  rules:
  - apiGroups: [""]
    apiVersions: ["v1"]
    operations: ["CREATE"]
    resources: ["pods"]
  admissionReviewVersions: ["v1"]
  sideEffects: None
"""),

    ("HorizontalPodAutoscaler — CPU only", """\
apiVersion: autoscaling/v1
kind: HorizontalPodAutoscaler
metadata:
  name: web-hpa
  namespace: default
  labels:
    app: web
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: web
  minReplicas: 2
  maxReplicas: 10
  targetCPUUtilizationPercentage: 80
"""),

    ("StatefulSet — minimal without volumeClaimTemplates", """\
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: db
  labels:
    app: db
spec:
  serviceName: db-headless
  replicas: 3
  selector:
    matchLabels:
      app: db
  template:
    metadata:
      labels:
        app: db
    spec:
      containers:
      - name: db
        image: postgres:14
"""),

]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=str)
    parser.add_argument("--vocab", type=str, default="output_v1/vocab.json")
    parser.add_argument("--threshold", type=float, default=0.3)
    args = parser.parse_args()

    torch.manual_seed(42)
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

    total_suggestions: int = 0
    kinds_tested: int = 0

    for title, yaml_text in TEST_YAMLS:
        suggestions = suggest_missing_fields(model, vocab, yaml_text, threshold=args.threshold)
        kinds_tested += 1
        total_suggestions += len(suggestions)

        print(f"\n{'=' * 60}")
        print(f"  {title}")
        print(f"{'=' * 60}")

        if not suggestions:
            print("  No suggestions.")
            continue

        by_parent: dict[str, list] = {}
        for s in suggestions:
            by_parent.setdefault(s["parent_path"], []).append(s)

        for parent, items in by_parent.items():
            path_display = parent if parent else "(root)"
            print(f"  {path_display}:")
            for item in items:
                conf = item["confidence"]
                strength = "STRONG" if conf > 0.8 else "MODERATE" if conf > 0.5 else "WEAK"
                print(f"    [{conf:5.1%}] {item['missing_key']} ({strength})")

    print(f"\n{'=' * 60}")
    print(f"  SUMMARY: {total_suggestions} suggestions across {kinds_tested} resource kinds")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
