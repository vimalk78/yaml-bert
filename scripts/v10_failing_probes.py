"""Probes designed to FAIL on v9, to motivate the v10 content-aware doc_vec work.

Each probe tests a kind of structural-or-semantic understanding that v9
currently does not deliver well. Failures here become the success
criteria for v10: a v10 architecture that closes any of these gaps is
defensible; one that doesn't is not.

Run:
    PYTHONPATH=. python scripts/v10_failing_probes.py \\
        --checkpoint output_v9_276K_recon_seed42/v9_checkpoint.pt \\
        --vocab     output_v9_276K_recon_seed42/vocab.json

Three probes:

  1. Version ordering — does the model see nginx:1.25 and nginx:1.24
     as closer than nginx:1.25 and nginx:0.9? BPE shares the `nginx`
     and `1.` subwords; if v9 leverages that, the embedding should
     reflect version distance.

  2. Foreign-key consistency — a Pod with matching `volumeMounts[*].name`
     and `volumes[*].name` is structurally coherent; one with mismatched
     names is broken. Does v9 distinguish them?

  3. Image-content retrieval — given a Pod with `image: nginx`, do the
     top-K nearest Pods (by cosine) share the image? Tests whether
     VALUE content shapes the embedding meaningfully or only structural
     features do.
"""
from __future__ import annotations

import argparse
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


# ============================================================
# Probe 1: Version ordering
# ============================================================

_POD_NGINX = """apiVersion: v1
kind: Pod
metadata:
  name: web
spec:
  containers:
  - name: app
    image: {image}
    ports:
    - containerPort: 80
"""


VERSION_PROBE = {
    "title": "Version ordering",
    "hypothesis": (
        "If BPE leaks version semantics via shared subwords (`nginx | : | "
        "1 | . | 25` vs `nginx | : | 1 | . | 24`), then close versions "
        "should be closer in doc_vec than far versions. Strong test: "
        "cos(1.25, 1.24) > cos(1.25, 1.20) > cos(1.25, 0.9)."
    ),
    "anchor": "nginx:1.25",
    "candidates": [
        ("nginx:1.24",      "1 version step back"),
        ("nginx:1.20",      "5 version steps back"),
        ("nginx:0.9",       "very far back (major version)"),
        ("redis:1.25",      "different image, same version (control)"),
    ],
}


# ============================================================
# Probe 2: Foreign-key consistency
# ============================================================

_POD_MATCHED_VOLS = """apiVersion: v1
kind: Pod
metadata:
  name: web
spec:
  volumes:
  - name: data
    emptyDir: {}
  - name: config
    configMap:
      name: app-config
  containers:
  - name: app
    image: nginx
    volumeMounts:
    - name: data
      mountPath: /var/data
    - name: config
      mountPath: /etc/config
"""

_POD_MISMATCHED_VOLS = """apiVersion: v1
kind: Pod
metadata:
  name: web
spec:
  volumes:
  - name: storage
    emptyDir: {}
  - name: settings
    configMap:
      name: app-config
  containers:
  - name: app
    image: nginx
    volumeMounts:
    - name: data
      mountPath: /var/data
    - name: config
      mountPath: /etc/config
"""

_POD_NO_VOLS = """apiVersion: v1
kind: Pod
metadata:
  name: web
spec:
  containers:
  - name: app
    image: nginx
"""

FK_PROBE = {
    "title": "Foreign-key consistency (volumeMounts ↔ volumes)",
    "hypothesis": (
        "A Pod where `volumeMounts[*].name` references undefined volumes "
        "is broken K8s — kubelet would reject it. If the model encodes "
        "this cross-position consistency, cos(matched, mismatched) should "
        "be LOWER than cos(matched, matched-similar-structure). "
        "Specifically: cos(matched, mismatched) < cos(matched, no-vols)? "
        "Honestly, vanilla self-attention rarely learns equality checks "
        "across positions without supervision — expect this to fail."
    ),
    "manifests": [
        ("A: matched names (data/config)",       _POD_MATCHED_VOLS),
        ("B: mismatched (storage/settings vs data/config)", _POD_MISMATCHED_VOLS),
        ("C: no volumes (simpler structure)",    _POD_NO_VOLS),
    ],
}


# ============================================================
# Probe 3: Image-content retrieval
# ============================================================

# A query Pod and a small corpus of candidates with various images.
# We test whether the query's top-K nearest neighbors share its image.

CONTENT_QUERY = ("nginx", """apiVersion: v1
kind: Pod
metadata:
  name: query
spec:
  containers:
  - name: app
    image: nginx
    ports:
    - containerPort: 80
""")


def _pod_with_image(name: str, image: str, port: int = 80) -> str:
    return f"""apiVersion: v1
kind: Pod
metadata:
  name: {name}
spec:
  containers:
  - name: app
    image: {image}
    ports:
    - containerPort: {port}
"""


CONTENT_CORPUS = [
    # 3 nginx pods (should be the top-3 if content-aware)
    ("nginx-1",   "nginx",         80),
    ("nginx-2",   "nginx:1.25",    80),
    ("nginx-3",   "nginx:alpine",  80),
    # 3 redis pods (control, should be further)
    ("redis-1",   "redis",         6379),
    ("redis-2",   "redis:7",       6379),
    ("redis-3",   "redis:alpine",  6379),
    # 3 postgres pods
    ("pg-1",      "postgres",      5432),
    ("pg-2",      "postgres:15",   5432),
    ("pg-3",      "postgres:alpine", 5432),
    # 3 mysql pods
    ("mysql-1",   "mysql",         3306),
    ("mysql-2",   "mysql:8",       3306),
    ("mysql-3",   "mysql:debian",  3306),
]

CONTENT_PROBE = {
    "title": "Image-content retrieval",
    "hypothesis": (
        "Given a query Pod with `image: nginx`, the top-3 nearest "
        "neighbors in the corpus should be the 3 other nginx-flavored "
        "Pods (`nginx`, `nginx:1.25`, `nginx:alpine`). Tests whether "
        "VALUE content (image string) organizes the embedding meaningfully "
        "or whether structural identity dominates so much that all Pods "
        "with the same shape (1 container + 1 port) cluster identically. "
        "Honest prediction: top-3 precision will be < 0.5 — v9's "
        "structural-doc_vec design weights structure over content."
    ),
}


# ============================================================
# Encoding
# ============================================================

def load_v9(checkpoint_path: str, vocab_path: str):
    vocab = Vocabulary.load(vocab_path)
    cp = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = cp.get("config") or YamlBertConfig(recon_enabled=True)
    cfg.mask_prob = 0.0
    cfg.recon_enabled = False
    emb = YamlBertEmbedding(config=cfg, subword_vocab_size=vocab.subword_vocab_size)
    model = YamlBertModel(
        config=cfg, embedding=emb,
        atomic_vocab_size=vocab.atomic_target_vocab_size,
    )
    model.load_state_dict(cp["model_state_dict"])
    model.eval()
    return model, vocab, cfg


def encode_yamls(model, vocab, cfg, yamls):
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


def cos(a, b):
    """Single-pair cosine."""
    return F.cosine_similarity(a, b, dim=0).item()


# ============================================================
# Probe runners
# ============================================================

def run_version_probe(model, vocab, cfg):
    print("\n" + "=" * 70)
    print("PROBE 1: VERSION ORDERING")
    print("=" * 70)
    print(VERSION_PROBE["hypothesis"])
    print()
    anchor_yaml = _POD_NGINX.format(image=VERSION_PROBE["anchor"])
    yamls = [anchor_yaml] + [
        _POD_NGINX.format(image=img) for img, _desc in VERSION_PROBE["candidates"]
    ]
    vecs = encode_yamls(model, vocab, cfg, yamls)
    anchor_vec = vecs[0]
    print(f"Anchor: {VERSION_PROBE['anchor']}")
    cosines = []
    for i, (img, desc) in enumerate(VERSION_PROBE["candidates"], start=1):
        c = cos(anchor_vec, vecs[i])
        cosines.append(c)
        print(f"  cos(anchor, {img:20s}) = {c:.4f}   ({desc})")

    # Verdict: cos(1.24) > cos(1.20) > cos(0.9)?
    same_v_diff_img = cosines[3]  # redis:1.25
    one_back = cosines[0]
    five_back = cosines[1]
    major_back = cosines[2]
    monotone = one_back > five_back > major_back
    image_dominates = (one_back > same_v_diff_img and five_back > same_v_diff_img)

    print()
    print(f"Monotonic version distance (1.24 > 1.20 > 0.9)? "
          f"{'✅' if monotone else '❌'}  ({one_back:.3f} > {five_back:.3f} > {major_back:.3f})")
    print(f"Image identity dominates version (redis:1.25 cos {same_v_diff_img:.3f} "
          f"< nginx:* cosines {one_back:.3f}/{five_back:.3f})? "
          f"{'✅' if image_dominates else '❌'}")
    print()
    if monotone:
        print("→ Model has SOME version semantics from BPE subword sharing.")
    else:
        print("→ Model does NOT cleanly rank versions by distance. Expected — "
              "BPE shares prefix subwords but doesn't encode numeric magnitude.")
    return {"passed": monotone, "cosines": cosines}


def run_fk_probe(model, vocab, cfg):
    print("\n" + "=" * 70)
    print("PROBE 2: FOREIGN-KEY CONSISTENCY (volumeMounts ↔ volumes)")
    print("=" * 70)
    print(FK_PROBE["hypothesis"])
    print()
    labels = [m[0] for m in FK_PROBE["manifests"]]
    yamls = [m[1] for m in FK_PROBE["manifests"]]
    vecs = encode_yamls(model, vocab, cfg, yamls)
    n = len(yamls)
    print("Cosine matrix:")
    print(f"  {'':30s}" + "".join(f"{l[:12]:>14s}" for l in labels))
    for i in range(n):
        row = [cos(vecs[i], vecs[j]) for j in range(n)]
        print(f"  {labels[i]:30s}" + "".join(f"{r:>14.4f}" for r in row))

    cos_AB = cos(vecs[0], vecs[1])  # matched vs mismatched
    cos_AC = cos(vecs[0], vecs[2])  # matched vs no-vols
    # Expectation: if FK consistency matters, A↔B should be DIFFERENT (lower
    # cos), even though they share full structure. A↔C is a control: very
    # different structure → low cos.
    passes = cos_AB < cos_AC
    print()
    print(f"cos(matched, mismatched) = {cos_AB:.4f}")
    print(f"cos(matched, no-vols)    = {cos_AC:.4f}")
    print()
    if passes:
        print(f"✅ Model encodes mismatch (cos {cos_AB:.3f} < cos {cos_AC:.3f}) — "
              f"surprising, would refute the expected failure.")
    else:
        print(f"❌ Model does NOT distinguish matched vs mismatched volume names. "
              f"cos(A,B)={cos_AB:.3f} is essentially as close as identical "
              f"structure could be — attention isn't comparing the literal "
              f"name VALUES across positions.")
    return {"passed": passes, "cos_AB": cos_AB, "cos_AC": cos_AC}


def run_content_retrieval_probe(model, vocab, cfg):
    print("\n" + "=" * 70)
    print("PROBE 3: IMAGE-CONTENT RETRIEVAL")
    print("=" * 70)
    print(CONTENT_PROBE["hypothesis"])
    print()
    query_image, query_yaml = CONTENT_QUERY
    corpus_yamls = [_pod_with_image(name, img, port) for name, img, port in CONTENT_CORPUS]
    all_yamls = [query_yaml] + corpus_yamls
    vecs = encode_yamls(model, vocab, cfg, all_yamls)
    qvec, corpus_vecs = vecs[0], vecs[1:]

    # Compute cosines from query to all corpus pods
    qnorm = qvec / qvec.norm()
    cnorm = corpus_vecs / corpus_vecs.norm(dim=1, keepdim=True)
    cosines = (cnorm @ qnorm).cpu().tolist()

    # Sort corpus by cosine (descending)
    ranked = sorted(
        zip(cosines, CONTENT_CORPUS), key=lambda x: -x[0],
    )
    print(f"Query image: '{query_image}'\n")
    print(f"  {'rank':>4}  {'cosine':>8}  {'image':<25s} {'port':>6}  {'is_nginx':>8}")
    n_nginx_correct = 0
    for rank, (c, (name, img, port)) in enumerate(ranked, start=1):
        is_nginx = img.startswith("nginx")
        marker = "✓" if is_nginx else " "
        print(f"  {rank:>4}  {c:>8.4f}  {img:<25s} {port:>6}  {marker:>8}")
        if rank <= 3 and is_nginx:
            n_nginx_correct += 1

    top3_precision = n_nginx_correct / 3
    print()
    print(f"Top-3 precision (nginx pods in top 3): {n_nginx_correct}/3 = {top3_precision:.0%}")
    passes = top3_precision >= 0.67  # at least 2 of top 3
    if passes:
        print(f"✅ Content-aware retrieval — image string DOES organize the "
              f"embedding meaningfully.")
    else:
        print(f"❌ Content-blind retrieval — at top-3 precision {top3_precision:.0%}, "
              f"the embedding is dominated by structural identity (all Pods have "
              f"1 container + 1 port). Image content reaches the embedding via "
              f"attention but doesn't dominate retrieval-style queries.")
    return {"passed": passes, "top3_precision": top3_precision, "ranked": ranked}


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--vocab", required=True)
    args = parser.parse_args()

    print(f"Loading v9 checkpoint from {args.checkpoint}...")
    model, vocab, cfg = load_v9(args.checkpoint, args.vocab)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  {n_params:,} params, subword vocab {vocab.subword_vocab_size}, "
          f"atomic target {vocab.atomic_target_vocab_size}")

    results = {
        "version":   run_version_probe(model, vocab, cfg),
        "foreign_key": run_fk_probe(model, vocab, cfg),
        "content_retrieval": run_content_retrieval_probe(model, vocab, cfg),
    }

    print("\n" + "=" * 70)
    print("SUMMARY (v9 baseline; failures motivate v10)")
    print("=" * 70)
    for name, r in results.items():
        emoji = "✅" if r["passed"] else "❌"
        print(f"  {emoji} {name}")
    n_failing = sum(1 for r in results.values() if not r["passed"])
    print()
    print(f"{n_failing}/3 probes fail on v9 — those are the v10 targets.")


if __name__ == "__main__":
    main()
