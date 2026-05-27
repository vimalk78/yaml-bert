"""Run the HF-Space structural probes + the C/E collision case against
a trained v9 checkpoint.

Probes (same as the HF Space's Structural Probes tab):
  1. Pod ± initContainers
  2. Service type (ClusterIP / NodePort / LoadBalancer)
  3. Pods in same namespace vs different namespace
  4. Pod vs Deployment wrapping the same Pod
  5. C/E collision (web-1 vs web-3 staging Pods — both [UNK] in v8, distinguishable in v9)

Run:
    PYTHONPATH=. python scripts/v9_structural_probes.py \
        --checkpoint output_v9_276K_recon_seed42/v9_checkpoint.pt \
        --vocab     output_v9_276K_recon_seed42/vocab.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from yaml_bert.annotator import DomainAnnotator
from yaml_bert.config import YamlBertConfig
from yaml_bert.dataset import YamlBertDataset, collate_fn
from yaml_bert.embedding import YamlBertEmbedding
from yaml_bert.linearizer import YamlLinearizer
from yaml_bert.model import YamlBertModel
from yaml_bert.vocab import Vocabulary


# ---- Probe manifests (mirrors hf-space/app.py) ----

_POD_NGINX = """apiVersion: v1
kind: Pod
metadata:
  name: nginx-app
spec:
  containers:
  - name: app
    image: nginx
    ports:
    - containerPort: 80
"""

_POD_REDIS = """apiVersion: v1
kind: Pod
metadata:
  name: redis-app
spec:
  containers:
  - name: app
    image: redis
    ports:
    - containerPort: 6379
"""

_POD_NGINX_INIT = """apiVersion: v1
kind: Pod
metadata:
  name: nginx-with-init
spec:
  initContainers:
  - name: setup
    image: busybox
    command: ["sh", "-c", "echo init"]
  containers:
  - name: app
    image: nginx
    ports:
    - containerPort: 80
"""

_POD_REDIS_INIT = """apiVersion: v1
kind: Pod
metadata:
  name: redis-with-init
spec:
  initContainers:
  - name: setup
    image: busybox
    command: ["sh", "-c", "echo init"]
  containers:
  - name: app
    image: redis
    ports:
    - containerPort: 6379
"""

_SVC_CLUSTERIP_WEB = """apiVersion: v1
kind: Service
metadata:
  name: web-clusterip
spec:
  type: ClusterIP
  selector: {app: web}
  ports:
  - port: 80
    targetPort: 8080
"""

_SVC_CLUSTERIP_API = """apiVersion: v1
kind: Service
metadata:
  name: api-clusterip
spec:
  type: ClusterIP
  selector: {app: api}
  ports:
  - port: 443
    targetPort: 8443
"""

_SVC_NODEPORT = """apiVersion: v1
kind: Service
metadata:
  name: web-nodeport
spec:
  type: NodePort
  selector: {app: web}
  ports:
  - port: 80
    targetPort: 8080
    nodePort: 30080
"""

_SVC_LOADBALANCER = """apiVersion: v1
kind: Service
metadata:
  name: web-lb
spec:
  type: LoadBalancer
  selector: {app: web}
  externalTrafficPolicy: Local
  ports:
  - port: 80
    targetPort: 8080
"""

_POD_NS_PROD_1 = """apiVersion: v1
kind: Pod
metadata: {name: web-1, namespace: production}
spec:
  containers:
  - name: app
    image: nginx
    ports:
    - containerPort: 80
"""

_POD_NS_PROD_2 = """apiVersion: v1
kind: Pod
metadata: {name: web-2, namespace: production}
spec:
  containers:
  - name: app
    image: nginx
    ports:
    - containerPort: 80
"""

_POD_NS_STAGING_1 = """apiVersion: v1
kind: Pod
metadata: {name: web-1, namespace: staging}
spec:
  containers:
  - name: app
    image: nginx
    ports:
    - containerPort: 80
"""

_POD_NS_STAGING_2 = """apiVersion: v1
kind: Pod
metadata: {name: web-2, namespace: staging}
spec:
  containers:
  - name: app
    image: nginx
    ports:
    - containerPort: 80
"""

_DEPLOY_NGINX = """apiVersion: apps/v1
kind: Deployment
metadata: {name: nginx}
spec:
  replicas: 3
  selector: {matchLabels: {app: nginx}}
  template:
    metadata: {labels: {app: nginx}}
    spec:
      containers:
      - name: app
        image: nginx
        ports:
        - containerPort: 80
"""

_CONFIGMAP_APP = """apiVersion: v1
kind: ConfigMap
metadata: {name: app-config}
data:
  config.yaml: |
    debug: true
  app.properties: |
    key1=value1
"""

# C/E collision case (the v8 failure that motivated v9)
_POD_WEB3_STAGING = """apiVersion: v1
kind: Pod
metadata: {name: web-3, namespace: staging}
spec:
  containers:
  - name: app
    image: nginx
    ports:
    - containerPort: 80
"""


PROBES = [
    {
        "id": "init",
        "title": "Pod ± initContainers",
        "manifests": [
            ("A: nginx (no init)", _POD_NGINX),
            ("B: redis (no init)", _POD_REDIS),
            ("C: nginx + init",    _POD_NGINX_INIT),
            ("D: redis + init",    _POD_REDIS_INIT),
        ],
        "verdict": lambda cos: (
            float(cos[2][3]) > max(float(cos[0][2]), float(cos[1][3])),
            f"cos(C,D)={cos[2][3]:.3f} (both init) vs "
            f"max(cos(A,C)={cos[0][2]:.3f}, cos(B,D)={cos[1][3]:.3f}) (mixed)",
        ),
    },
    {
        "id": "service-type",
        "title": "Service type (ClusterIP / NodePort / LoadBalancer)",
        "manifests": [
            ("A: ClusterIP web", _SVC_CLUSTERIP_WEB),
            ("B: ClusterIP api", _SVC_CLUSTERIP_API),
            ("C: NodePort",      _SVC_NODEPORT),
            ("D: LoadBalancer",  _SVC_LOADBALANCER),
        ],
        "verdict": lambda cos: (
            float(cos[0][1]) > max(float(cos[0][2]), float(cos[0][3]),
                                   float(cos[1][2]), float(cos[1][3]),
                                   float(cos[2][3])),
            f"cos(A,B)={cos[0][1]:.3f} (same type) vs "
            f"max cross-type={max(cos[0][2], cos[0][3], cos[1][2], cos[1][3], cos[2][3]):.3f}",
        ),
    },
    {
        "id": "namespace",
        "title": "Pods in same namespace vs different namespace",
        "manifests": [
            ("A: prod/web-1", _POD_NS_PROD_1),
            ("B: prod/web-2", _POD_NS_PROD_2),
            ("C: stg/web-1",  _POD_NS_STAGING_1),
            ("D: stg/web-2",  _POD_NS_STAGING_2),
        ],
        "verdict": lambda cos: (
            min(float(cos[0][1]), float(cos[2][3])) >
            max(float(cos[0][2]), float(cos[0][3]),
                float(cos[1][2]), float(cos[1][3])),
            f"min(same-ns)={min(cos[0][1], cos[2][3]):.3f} vs "
            f"max(cross-ns)={max(cos[0][2], cos[0][3], cos[1][2], cos[1][3]):.3f}",
        ),
    },
    {
        "id": "cross-kind",
        "title": "Pod vs Deployment wrapping the same Pod",
        "manifests": [
            ("A: Pod nginx",          _POD_NGINX),
            ("B: Pod redis",          _POD_REDIS),
            ("C: Deployment->nginx",  _DEPLOY_NGINX),
            ("D: ConfigMap (unrel.)", _CONFIGMAP_APP),
        ],
        "verdict": lambda cos: (
            float(cos[0][2]) > max(float(cos[0][3]), float(cos[2][3])),
            f"cos(Pod, Deployment-wrapping-it)={cos[0][2]:.3f} vs "
            f"max(Pod-vs-ConfigMap={cos[0][3]:.3f}, Deployment-vs-ConfigMap={cos[2][3]:.3f})",
        ),
    },
]


def load_v9(checkpoint_path: str, vocab_path: str):
    """Load v9 model + vocab from checkpoint."""
    print(f"Loading vocab from {vocab_path}")
    vocab = Vocabulary.load(vocab_path)
    print(f"  subword vocab: {vocab.subword_vocab_size}, "
          f"atomic target: {vocab.atomic_target_vocab_size}")

    print(f"Loading checkpoint from {checkpoint_path}")
    cp = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = cp.get("config")
    if cfg is None:
        # Fallback if the saved config field is missing — use defaults consistent with training
        cfg = YamlBertConfig(recon_enabled=True)
    # Reset mask_prob and recon_enabled for inference: we don't want masking
    # to happen for probe encodings.
    cfg.mask_prob = 0.0
    cfg.recon_enabled = False

    emb = YamlBertEmbedding(config=cfg, subword_vocab_size=vocab.subword_vocab_size)
    model = YamlBertModel(
        config=cfg, embedding=emb,
        atomic_vocab_size=vocab.atomic_target_vocab_size,
    )
    model.load_state_dict(cp["model_state_dict"])
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  loaded; {n_params:,} params")
    return model, vocab, cfg


def encode_yamls(model, vocab, cfg, yamls):
    """Encode a list of YAML strings to doc_vecs (one per input)."""
    lin = YamlLinearizer()
    ann = DomainAnnotator()
    docs = []
    for y in yamls:
        nodes = lin.linearize(y)
        ann.annotate(nodes)
        docs.append(nodes)
    ds = YamlBertDataset(docs, vocab, cfg)
    batch = collate_fn([ds[i] for i in range(len(docs))])
    with torch.no_grad():
        out = model(
            token_ids=batch["token_ids"],
            node_types=batch["node_types"],
            depths=batch["depths"],
            sibling_indices=batch["sibling_indices"],
            batch_info=batch["batch_info"],
            padding_mask=batch["padding_mask"],
            logical_ids=batch["logical_ids"],
            n_logical_per_doc=batch["n_logical_per_doc"],
            parent_of_tensor=batch["parent_of_tensor"],
            top_level_key_mask=batch["top_level_key_mask"],
            edges_by_depth=batch["edges_by_depth"],
            parents_by_depth=batch["parents_by_depth"],
        )
    return out[1]  # doc_vec, shape (N, d)


def cos_matrix(vecs):
    n = vecs / vecs.norm(dim=1, keepdim=True)
    return (n @ n.t()).cpu().numpy()


def run_probes(model, vocab, cfg):
    print("\n" + "=" * 70)
    print("STRUCTURAL PROBES")
    print("=" * 70)
    results = []
    for probe in PROBES:
        labels = [m[0] for m in probe["manifests"]]
        yamls = [m[1] for m in probe["manifests"]]
        vecs = encode_yamls(model, vocab, cfg, yamls)
        cos = cos_matrix(vecs)
        passed, msg = probe["verdict"](cos)
        emoji = "✅" if passed else "❌"
        print(f"\n{emoji} {probe['title']}")
        print(f"   {msg}")
        # Print matrix
        n = len(labels)
        print(f"   {'':14s}" + "".join(f"{l[:12]:>13s}" for l in labels))
        for i in range(n):
            print(f"   {labels[i][:12]:14s}" +
                  "".join(f"{cos[i][j]:>13.4f}" for j in range(n)))
        results.append({"title": probe["title"], "passed": passed,
                        "msg": msg, "cos": cos.tolist()})
    return results


def run_collision_check(model, vocab, cfg):
    print("\n" + "=" * 70)
    print("C/E COLLISION CASE")
    print("=" * 70)
    print("v8 saw cos(C, E) = 1.0000 because both `web-1` and `web-3` mapped to [UNK].")
    print("v9 should produce distinct vectors (BPE decomposes them).\n")
    vecs = encode_yamls(model, vocab, cfg, [_POD_NS_STAGING_1, _POD_WEB3_STAGING])
    cos = F.cosine_similarity(vecs[0], vecs[1], dim=0).item()
    print(f"   cos(C: staging/web-1, E: staging/web-3) = {cos:.4f}")
    if cos < 0.999:
        print(f"   ✅ FIXED — model distinguishes the two pods (cos < 0.999)")
    else:
        print(f"   ❌ Still identical at the model level")
    return cos


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--vocab", required=True)
    args = parser.parse_args()

    model, vocab, cfg = load_v9(args.checkpoint, args.vocab)
    probe_results = run_probes(model, vocab, cfg)
    collision_cos = run_collision_check(model, vocab, cfg)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for r in probe_results:
        print(f"  {'✅' if r['passed'] else '❌'} {r['title']}")
    print(f"  C/E collision: cos = {collision_cos:.4f}")


if __name__ == "__main__":
    main()
