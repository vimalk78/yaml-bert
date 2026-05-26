"""Local microbenchmark: vectorized aggregator must be substantially
faster than the per-doc reference path on a representative batch.

CPU-only. Regression gate: ≥2.5× speedup (measured median ~3.0×). GPU benchmark (Task 6) is the
real acceptance gate (≥7 it/s training throughput).
"""
import time

import torch

from yaml_bert.aggregator import TreeAggregator
from yaml_bert.linearizer import YamlLinearizer
from yaml_bert.dataset import YamlBertDataset, collate_fn
from yaml_bert.vocab import VocabBuilder
from yaml_bert.config import YamlBertConfig


def _make_batch(batch_size: int = 32, d_model: int = 256):
    """Build a synthetic batch with realistic K8s manifest sizes.

    Goal: tree depth >=6, ~50-100 nodes per doc — comparable to median
    real-corpus documents. Smaller docs make the reference Python loop
    cheap enough that vectorization overhead dominates."""
    yamls = [
        # Deployment with multiple containers, nested labels, volumes
        "apiVersion: apps/v1\nkind: Deployment\nmetadata:\n"
        "  name: web\n  namespace: production\n  labels:\n"
        "    app: nginx\n    tier: frontend\n    env: prod\n"
        "    version: v1\n  annotations:\n"
        "    description: web frontend\n"
        "spec:\n  replicas: 3\n  strategy:\n    type: RollingUpdate\n"
        "    rollingUpdate:\n      maxSurge: 1\n      maxUnavailable: 0\n"
        "  selector:\n    matchLabels:\n      app: nginx\n      tier: frontend\n"
        "  template:\n    metadata:\n      labels:\n        app: nginx\n"
        "        tier: frontend\n        env: prod\n"
        "    spec:\n      containers:\n"
        "      - name: nginx\n        image: nginx:1.25\n"
        "        ports:\n        - containerPort: 80\n          name: http\n"
        "        - containerPort: 443\n          name: https\n"
        "        env:\n        - name: LOG_LEVEL\n          value: info\n"
        "        resources:\n          requests:\n            cpu: 100m\n"
        "            memory: 128Mi\n          limits:\n            cpu: 500m\n"
        "            memory: 512Mi\n"
        "      - name: sidecar\n        image: envoy:1.28\n"
        "        ports:\n        - containerPort: 9090\n",

        # Service with multiple ports + annotations
        "apiVersion: v1\nkind: Service\nmetadata:\n"
        "  name: web\n  namespace: production\n  labels:\n"
        "    app: nginx\n    tier: frontend\n  annotations:\n"
        "    service.kubernetes.io/topology-aware-hints: auto\n"
        "spec:\n  type: ClusterIP\n  selector:\n    app: nginx\n"
        "    tier: frontend\n  ports:\n"
        "  - name: http\n    port: 80\n    targetPort: 8080\n    protocol: TCP\n"
        "  - name: https\n    port: 443\n    targetPort: 8443\n    protocol: TCP\n"
        "  - name: metrics\n    port: 9090\n    targetPort: 9090\n    protocol: TCP\n",

        # Pod with init + main containers and volume mounts
        "apiVersion: v1\nkind: Pod\nmetadata:\n  name: app\n"
        "  namespace: production\n  labels:\n    app: backend\n    tier: api\n"
        "spec:\n  serviceAccountName: backend\n"
        "  initContainers:\n  - name: init\n    image: busybox:1.36\n"
        "    command:\n    - sh\n    - -c\n    - sleep 5\n"
        "  containers:\n  - name: main\n    image: backend:1.0\n"
        "    ports:\n    - containerPort: 8080\n      name: http\n"
        "    env:\n    - name: DB_HOST\n      value: postgres\n"
        "    - name: DB_PORT\n      value: '5432'\n"
        "    volumeMounts:\n    - name: data\n      mountPath: /data\n"
        "    - name: config\n      mountPath: /etc/app\n"
        "  volumes:\n  - name: data\n    emptyDir: {}\n"
        "  - name: config\n    configMap:\n      name: app-config\n",

        # ConfigMap with nested structured data
        "apiVersion: v1\nkind: ConfigMap\nmetadata:\n  name: cfg\n"
        "  namespace: production\n  labels:\n    app: web\n    tier: frontend\n"
        "data:\n  config.yaml: |\n    server:\n      host: 0.0.0.0\n"
        "      port: 8080\n      tls:\n        enabled: true\n        cert: /etc/tls/cert\n"
        "    database:\n      host: postgres\n      port: 5432\n      pool:\n"
        "        min: 5\n        max: 20\n  feature_flags: |\n"
        "    enable_new_ui: true\n    enable_metrics: true\n",
    ]
    yamls = (yamls * ((batch_size // len(yamls)) + 1))[:batch_size]
    docs = [YamlLinearizer().linearize(y) for y in yamls]
    flat = [n for d in docs for n in d]
    vocab = VocabBuilder().build(flat, min_freq=1)
    config = YamlBertConfig(mask_prob=0.0, d_model=d_model)
    ds = YamlBertDataset(docs, vocab, config)
    batch = collate_fn([ds[i] for i in range(len(ds))])
    B, N = batch["token_ids"].shape
    torch.manual_seed(0)
    hidden = torch.randn(B, N, d_model)
    return hidden, batch


def test_vectorized_aggregator_is_at_least_2_5x_faster_on_cpu():
    """Vectorized path on a synthetic 32-doc batch should be ≥2.5× faster
    than the per-doc reference path on CPU. Measured stable median: ~3.0×."""
    d_model = 256
    hidden, batch = _make_batch(batch_size=32, d_model=d_model)
    agg = TreeAggregator(d_model=d_model)

    # Warmup once for each path (catches first-call compilation overhead)
    _ = agg(hidden, batch["batch_info"])
    _ = agg(
        hidden, batch["batch_info"],
        parent_of_tensor=batch["parent_of_tensor"],
        top_level_key_mask=batch["top_level_key_mask"],
        edges_by_depth=batch["edges_by_depth"],
        parents_by_depth=batch["parents_by_depth"],
    )

    # Use enough iterations to stabilize CPU timing (measured median: ~3.0x
    # at n_iters=100; n_iters=30 has ±15% jitter that causes spurious failures).
    n_iters = 100

    # Reference path
    t0 = time.perf_counter()
    for _ in range(n_iters):
        agg(hidden, batch["batch_info"])
    ref_time = time.perf_counter() - t0

    # Vectorized path
    t0 = time.perf_counter()
    for _ in range(n_iters):
        agg(
            hidden, batch["batch_info"],
            parent_of_tensor=batch["parent_of_tensor"],
            top_level_key_mask=batch["top_level_key_mask"],
            edges_by_depth=batch["edges_by_depth"],
            parents_by_depth=batch["parents_by_depth"],
        )
    vec_time = time.perf_counter() - t0

    speedup = ref_time / vec_time
    print(f"\nreference: {ref_time:.3f}s for {n_iters} iters "
          f"({ref_time / n_iters * 1000:.1f}ms/iter)")
    print(f"vectorized: {vec_time:.3f}s for {n_iters} iters "
          f"({vec_time / n_iters * 1000:.1f}ms/iter)")
    print(f"speedup: {speedup:.1f}x")
    # 2.5x CPU threshold catches regressions (a broken vectorization drops to ~1x).
    # Measured stable median on these docs: ~3.0x at n_iters=100.
    # GPU benchmark in Task 6 is the real acceptance gate (≥7 it/s training).
    assert speedup >= 2.5, (
        f"Vectorized path only {speedup:.1f}× faster — expected ≥2.5× on CPU. "
        f"A regression to ~1× indicates vectorization is broken; "
        f"a value between 1× and 2.5× warrants investigation but may be acceptable. "
        f"Final acceptance is GPU training speed in Task 6."
    )
