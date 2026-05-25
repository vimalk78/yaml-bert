"""Phase 0 kind-discrimination probe.

Loads the doc_vecs dumped by train_v8_phase0.py, fits a linear probe to
predict kind, reports top-K classification accuracy. Pass criterion:
>70% accuracy on >=10 common kinds.
"""
from __future__ import annotations
import _setup_path  # noqa: F401

import argparse
from collections import Counter

import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--doc-vecs", required=True,
                        help="Path to doc_vecs.pt produced by training.")
    parser.add_argument("--top-k-kinds", type=int, default=10,
                        help="Use the top-K most common kinds for the probe.")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data = torch.load(args.doc_vecs, weights_only=False)
    doc_vecs: torch.Tensor = data["doc_vecs"]
    kinds: list[str] = data["kinds"]

    print(f"Loaded {doc_vecs.shape[0]} doc vectors (d={doc_vecs.shape[1]})")

    # Restrict to top-K most common kinds (skip empty kinds)
    valid_indices = [i for i, k in enumerate(kinds) if k]
    if len(valid_indices) < doc_vecs.shape[0]:
        print(f"  filtered out {doc_vecs.shape[0] - len(valid_indices)} "
              f"docs with no kind")
    filtered_vecs = doc_vecs[valid_indices]
    filtered_kinds = [kinds[i] for i in valid_indices]

    kind_counts = Counter(filtered_kinds)
    top_kinds = [k for k, _ in kind_counts.most_common(args.top_k_kinds)]
    print(f"  top {len(top_kinds)} kinds: {top_kinds}")

    mask = [k in top_kinds for k in filtered_kinds]
    X = filtered_vecs[mask].numpy()
    y = [k for k, m in zip(filtered_kinds, mask) if m]

    if len(X) < 100:
        print(f"WARNING: only {len(X)} samples for probe - results unreliable")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y,
    )

    clf = LogisticRegression(max_iter=500)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nLinear probe accuracy on {len(top_kinds)} kinds: {acc:.4f}")
    print(f"Phase 0 pass criterion: > 0.70 ? {'PASS' if acc > 0.70 else 'FAIL'}")


if __name__ == "__main__":
    main()
