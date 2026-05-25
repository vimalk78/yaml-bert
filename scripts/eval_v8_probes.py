"""Run 4 smoke-test probes on doc_vec dumps from train_v8_phase1_recon.py.

Reads doc_vecs_epoch_<N>.pt files + the raw doc_cache.pkl, builds labels
from parsed YAML nodes, fits sklearn LogisticRegression per probe, prints a
trajectory table.

Labels are derived directly from the cached YamlNode lists — no raw YAML text
or HF re-fetch required.
"""
from __future__ import annotations
import _setup_path  # noqa: F401

import argparse
import os
import pickle
from collections import Counter

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from yaml_bert.dataset import _extract_kind
from yaml_bert.types import YamlNode


def _has_token(nodes: list[YamlNode], token: str) -> bool:
    """Return True if any node in the document has the given token."""
    return any(n.token == token for n in nodes)


def _build_labels(cached_docs: list[list[YamlNode]], top_k_kinds: int = 10) -> dict:
    """Build label arrays for the 4 probes from cached YamlNode lists.

    Returns dict with:
        kind_labels: int array (-1 for docs outside top-K kinds)
        kind_names:  list of kind strings (index → name)
        has_containers, has_init_containers, has_volume_mounts: int (0/1) arrays
    """
    kinds = [_extract_kind(doc) for doc in cached_docs]
    counter = Counter(k for k in kinds if k)
    top_kinds = [k for k, _ in counter.most_common(top_k_kinds)]
    kind_to_idx = {k: i for i, k in enumerate(top_kinds)}
    kind_labels = np.array(
        [kind_to_idx.get(k, -1) for k in kinds], dtype=int,
    )
    has_containers = np.array(
        [int(_has_token(doc, "containers")) for doc in cached_docs], dtype=int,
    )
    has_init_containers = np.array(
        [int(_has_token(doc, "initContainers")) for doc in cached_docs], dtype=int,
    )
    has_volume_mounts = np.array(
        [int(_has_token(doc, "volumeMounts")) for doc in cached_docs], dtype=int,
    )
    return {
        "kind_labels": kind_labels,
        "kind_names": top_kinds,
        "has_containers": has_containers,
        "has_init_containers": has_init_containers,
        "has_volume_mounts": has_volume_mounts,
    }


def _probe_accuracy(
    X: np.ndarray,
    y: np.ndarray,
    label_filter: np.ndarray | None = None,
) -> float:
    """Fit LogisticRegression on 80% of (X, y), report accuracy on 20%."""
    if label_filter is not None:
        X = X[label_filter]
        y = y[label_filter]
    if len(np.unique(y)) < 2:
        return float("nan")
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y,
    )
    clf = LogisticRegression(max_iter=2000)
    clf.fit(X_tr, y_tr)
    return float(clf.score(X_te, y_te))


def _eval_one_dump(doc_vecs_path: str, labels: dict) -> dict:
    """Run all 4 probes on one doc_vec dump file."""
    data = torch.load(doc_vecs_path, map_location="cpu", weights_only=False)
    X = data["doc_vecs"].numpy()  # (D, d_model)
    n = X.shape[0]

    kind_mask = labels["kind_labels"][:n] >= 0
    return {
        "kind": _probe_accuracy(
            X, labels["kind_labels"][:n], label_filter=kind_mask,
        ),
        "has_containers": _probe_accuracy(X, labels["has_containers"][:n]),
        "has_init_containers": _probe_accuracy(
            X, labels["has_init_containers"][:n]),
        "has_volume_mounts": _probe_accuracy(
            X, labels["has_volume_mounts"][:n]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True,
                        help="dir containing doc_vecs_epoch_*.pt and doc_cache.pkl")
    parser.add_argument("--top-k-kinds", type=int, default=10)
    args = parser.parse_args()

    cache_path = os.path.join(args.output_dir, "doc_cache.pkl")
    print(f"Loading doc_cache from {cache_path}")
    with open(cache_path, "rb") as f:
        cached: list[list[YamlNode]] = pickle.load(f)
    print(f"Loaded {len(cached):,} cached documents")

    print(f"Building labels for {len(cached)} docs...")
    labels = _build_labels(cached, top_k_kinds=args.top_k_kinds)
    print(f"Top kinds: {labels['kind_names']}")
    print(
        f"Counts: containers={labels['has_containers'].sum()}, "
        f"init={labels['has_init_containers'].sum()}, "
        f"vol_mounts={labels['has_volume_mounts'].sum()}"
    )

    # Find all per-epoch dumps
    dumps = sorted(
        [
            f for f in os.listdir(args.output_dir)
            if f.startswith("doc_vecs_epoch_") and f.endswith(".pt")
        ],
        key=lambda f: int(f.split("_")[3].split(".")[0]),
    )
    if not dumps:
        print(f"No per-epoch dumps in {args.output_dir}; trying doc_vecs.pt")
        candidate = os.path.join(args.output_dir, "doc_vecs.pt")
        if os.path.exists(candidate):
            dumps = ["doc_vecs.pt"]
        else:
            print(f"ERROR: no dump files found in {args.output_dir}")
            return

    print(
        f"\n{'epoch':>6} | {'kind':>8} | {'containers':>10} "
        f"| {'init':>8} | {'vol_mounts':>10}"
    )
    print("-" * 60)
    for fn in dumps:
        epoch_label: str = (
            str(int(fn.split("_")[3].split(".")[0]))
            if "epoch_" in fn
            else "final"
        )
        results = _eval_one_dump(os.path.join(args.output_dir, fn), labels)
        print(
            f"{epoch_label:>6} | "
            f"{results['kind'] * 100:>7.2f}% | "
            f"{results['has_containers'] * 100:>9.2f}% | "
            f"{results['has_init_containers'] * 100:>7.2f}% | "
            f"{results['has_volume_mounts'] * 100:>9.2f}%"
        )


if __name__ == "__main__":
    main()
