"""Position-level probe: does v9 distinguish "containers" across three
wrapping depths — Pod (depth 1), Deployment (depth 3), CronJob (depth 5)?

Builds 6 manifests (3 wrappings × 2 contents), encodes them, captures
the Key Head input [h_logical ; doc_vec ; s_parent] at the `containers`
logical position in each. Reports cosine matrices and verdicts at four
vector levels:

  - h_logical[containers]   per-position hidden state
  - s_parent[containers]    parent subtree vec (parent + its descendants)
  - head_input              [h_logical ; doc_vec ; s_parent] concat
  - doc_vec                 whole-document vector

Verdict (per vector level):
  strict: min(within-wrapping cosines) > max(cross-wrapping cosines)
  soft:   avg(within-wrapping cosines) > avg(cross-wrapping cosines)

Run:
    PYTHONPATH=. python scripts/wrapping_distinction_probe.py \\
        --checkpoint output_v9_276K_recon_seed42/v9_checkpoint.pt \\
        --vocab     output_v9_276K_recon_seed42/vocab.json
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
from yaml_bert.types import NodeType
from yaml_bert.vocab import Vocabulary


# Anchors grouped by wrapping kind (Pod block, Deploy block, CronJob block).
# Same container content per pair within each block.

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

_POD_POSTGRES = """apiVersion: v1
kind: Pod
metadata:
  name: postgres-app
spec:
  containers:
  - name: app
    image: postgres
    ports:
    - containerPort: 5432
"""

_DEPLOY_NGINX = """apiVersion: apps/v1
kind: Deployment
metadata:
  name: nginx
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

_DEPLOY_POSTGRES = """apiVersion: apps/v1
kind: Deployment
metadata:
  name: postgres
spec:
  replicas: 3
  selector: {matchLabels: {app: postgres}}
  template:
    metadata: {labels: {app: postgres}}
    spec:
      containers:
      - name: app
        image: postgres
        ports:
        - containerPort: 5432
"""

_CRONJOB_NGINX = """apiVersion: batch/v1
kind: CronJob
metadata:
  name: nginx-cron
spec:
  schedule: "*/5 * * * *"
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: app
            image: nginx
            ports:
            - containerPort: 80
          restartPolicy: OnFailure
"""

_CRONJOB_POSTGRES = """apiVersion: batch/v1
kind: CronJob
metadata:
  name: postgres-cron
spec:
  schedule: "*/5 * * * *"
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: app
            image: postgres
            ports:
            - containerPort: 5432
          restartPolicy: OnFailure
"""


# Ordering: same-wrapping pairs adjacent so the cos matrix shows
# block-diagonal clustering when wrapping is the dominant feature.
ANCHORS = [
    ("Pod nginx",          "Pod",     _POD_NGINX),
    ("Pod postgres",       "Pod",     _POD_POSTGRES),
    ("Deploy nginx",       "Deploy",  _DEPLOY_NGINX),
    ("Deploy postgres",    "Deploy",  _DEPLOY_POSTGRES),
    ("CronJob nginx",      "CronJob", _CRONJOB_NGINX),
    ("CronJob postgres",   "CronJob", _CRONJOB_POSTGRES),
]


def load_v9(checkpoint_path: str, vocab_path: str):
    print(f"Loading vocab from {vocab_path}")
    vocab = Vocabulary.load(vocab_path)
    print(f"  subword vocab: {vocab.subword_vocab_size}, "
          f"atomic target: {vocab.atomic_target_vocab_size}")

    print(f"Loading checkpoint from {checkpoint_path}")
    cp = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = cp.get("config")
    if cfg is None:
        cfg = YamlBertConfig(recon_enabled=True)
    cfg.mask_prob = 0.0
    cfg.recon_enabled = False

    emb = YamlBertEmbedding(config=cfg, subword_vocab_size=vocab.subword_vocab_size)
    model = YamlBertModel(
        config=cfg, embedding=emb,
        atomic_vocab_size=vocab.atomic_target_vocab_size,
    )
    model.load_state_dict(cp["model_state_dict"])
    model.eval()
    print(f"  loaded; {sum(p.numel() for p in model.parameters()):,} params")
    return model, vocab, cfg


def find_containers_logical_pos(nodes):
    """Return the unique logical position whose token=='containers'
    and node_type==KEY. Errors if not exactly one."""
    hits = [
        i for i, n in enumerate(nodes)
        if n.token == "containers" and n.node_type == NodeType.KEY
    ]
    if len(hits) != 1:
        raise ValueError(
            f"expected exactly one 'containers' KEY position, found {len(hits)}"
        )
    return hits[0], nodes[hits[0]].depth


def encode_with_head_input_capture(model, vocab, cfg, yamls):
    """Run forward and capture the concatenated Key Head input
    [h_logical ; doc_vec ; s_parent] for every position.

    Returns (head_input, doc_vec, docs):
      - head_input: (B, L_max, 3*d_model)
      - doc_vec:    (B, d_model)
      - docs:       list of nodes per input (for position lookup)
    """
    lin = YamlLinearizer()
    ann = DomainAnnotator()
    docs = []
    for y in yamls:
        nodes = lin.linearize(y)
        ann.annotate(nodes)
        docs.append(nodes)
    ds = YamlBertDataset(docs, vocab, cfg)
    batch = collate_fn([ds[i] for i in range(len(docs))])

    captured = {}

    def hook(module, inputs, output):
        captured["head_input"] = inputs[0].detach()

    handle = model.token_head.register_forward_hook(hook)
    try:
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
    finally:
        handle.remove()

    doc_vec = out[1].detach()
    return captured["head_input"], doc_vec, docs


def cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return F.cosine_similarity(a, b, dim=0).item()


def classify_pairs(wrappings: list[str]) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Return (within_wrapping_pairs, cross_wrapping_pairs) for upper-triangle
    indices of the manifest list."""
    within, cross = [], []
    n = len(wrappings)
    for i in range(n):
        for j in range(i + 1, n):
            if wrappings[i] == wrappings[j]:
                within.append((i, j))
            else:
                cross.append((i, j))
    return within, cross


def report(name: str, vecs: list[torch.Tensor], labels: list[str],
           wrappings: list[str]) -> dict:
    n = len(vecs)
    # Build cos matrix
    mat = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            mat[i][j] = cos(vecs[i], vecs[j])

    within_pairs, cross_pairs = classify_pairs(wrappings)
    within_cosines = [mat[i][j] for i, j in within_pairs]
    cross_cosines = [mat[i][j] for i, j in cross_pairs]

    strict = min(within_cosines) > max(cross_cosines)
    soft = (sum(within_cosines) / len(within_cosines)) > \
           (sum(cross_cosines) / len(cross_cosines))

    print(f"--- {name} ---")
    # Print cos matrix
    short_labels = [lbl[:15] for lbl in labels]
    print(f"   {'':18s}" + "".join(f"{l:>17s}" for l in short_labels))
    for i in range(n):
        print(f"   {short_labels[i]:18s}" +
              "".join(f"{mat[i][j]:>17.4f}" for j in range(n)))
    print()
    print(f"   within-wrapping cosines ({len(within_pairs)}): "
          f"min={min(within_cosines):.4f}, "
          f"max={max(within_cosines):.4f}, "
          f"avg={sum(within_cosines)/len(within_cosines):.4f}")
    print(f"   cross-wrapping  cosines ({len(cross_pairs)}): "
          f"min={min(cross_cosines):.4f}, "
          f"max={max(cross_cosines):.4f}, "
          f"avg={sum(cross_cosines)/len(cross_cosines):.4f}")
    print(f"   strict (min within > max cross): "
          f"{min(within_cosines):.4f} vs {max(cross_cosines):.4f}  "
          f"{'PASS' if strict else 'FAIL'}")
    print(f"   soft   (avg within > avg cross): "
          f"{sum(within_cosines)/len(within_cosines):.4f} vs "
          f"{sum(cross_cosines)/len(cross_cosines):.4f}  "
          f"{'PASS' if soft else 'FAIL'}")
    print()
    return {"strict": strict, "soft": soft, "mat": mat}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--vocab", required=True)
    args = parser.parse_args()

    model, vocab, cfg = load_v9(args.checkpoint, args.vocab)
    d_model = cfg.d_model

    labels = [a[0] for a in ANCHORS]
    wrappings = [a[1] for a in ANCHORS]
    yamls = [a[2] for a in ANCHORS]

    head_input, doc_vec, docs = encode_with_head_input_capture(
        model, vocab, cfg, yamls,
    )

    # head_input layout: [h_logical (d) ; doc_vec_broadcast (d) ; s_parent (d)]
    h_logical_all = head_input[..., :d_model]
    s_parent_all = head_input[..., 2 * d_model: 3 * d_model]

    h_vec, s_vec, head_vec, depths_used = [], [], [], []
    for i, nodes in enumerate(docs):
        pos, depth = find_containers_logical_pos(nodes)
        h_vec.append(h_logical_all[i, pos])
        s_vec.append(s_parent_all[i, pos])
        head_vec.append(head_input[i, pos])
        depths_used.append(depth)

    print("\n" + "=" * 110)
    print("WRAPPING DISTINCTION PROBE — 6 manifests, 3 wrapping depths")
    print("=" * 110)
    print("Test: at the 'containers' logical position, does v9 distinguish")
    print("Pod / Deployment / CronJob wrapping context?")
    print()
    print("Anchors (grouped by wrapping):")
    for lbl, wrap, depth in zip(labels, wrappings, depths_used):
        print(f"  {lbl:22s}  wrap={wrap:8s}  'containers' at depth={depth}")
    print()
    print("Verdict: min(within-wrapping cosines) > max(cross-wrapping cosines).")
    print()

    results = {
        "h_logical":  report("h_logical[containers]",                       h_vec,    labels, wrappings),
        "s_parent":   report("s_parent[containers] (parent subtree vec)",   s_vec,    labels, wrappings),
        "head_input": report("[h_logical ; doc_vec ; s_parent] (Key Head)", head_vec, labels, wrappings),
        "doc_vec":    report("doc_vec (whole-document vector)",
                             [doc_vec[i] for i in range(len(labels))],      labels, wrappings),
    }

    print("=" * 110)
    print("SUMMARY")
    print("=" * 110)
    for name, r in results.items():
        s = "PASS" if r["strict"] else "FAIL"
        f = "PASS" if r["soft"] else "FAIL"
        print(f"  {name:30s}  strict: {s}   soft: {f}")


if __name__ == "__main__":
    main()
