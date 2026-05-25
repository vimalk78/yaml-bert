"""Analyze compound target sparsity from cached linearized documents.

Usage:
    PYTHONPATH=. python scripts/analyze_compound_targets.py output_v3_full/doc_cache.pkl
"""
from __future__ import annotations
import _setup_path  # noqa: F401

import argparse
import pickle
import sys
from collections import Counter

from yaml_bert.types import NodeType, YamlNode
from yaml_bert.dataset import _extract_kind


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("cache", type=str, help="Path to doc_cache.pkl")
    args = parser.parse_args()

    print(f"Loading cache: {args.cache}")
    with open(args.cache, "rb") as f:
        documents: list[list[YamlNode]] = pickle.load(f)
    print(f"Loaded {len(documents):,} documents")

    compound_counts: Counter = Counter()
    simple_counts: Counter = Counter()
    path_depth_counts: Counter = Counter()

    for doc_idx, nodes in enumerate(documents):
        kind: str = _extract_kind(nodes)
        if not kind:
            continue

        for node in nodes:
            if node.node_type not in (NodeType.KEY, NodeType.LIST_KEY):
                continue

            # Determine if under spec (or other kind-specific subtree)
            is_under_spec: bool = (
                node.parent_path.startswith("spec")
                or node.parent_path.startswith("rules")
                or node.parent_path.startswith("data")
                or node.parent_path.startswith("webhooks")
                or node.parent_path.startswith("subjects")
                or node.parent_path.startswith("roleRef")
            )

            if node.depth == 0 or node.parent_path.startswith("metadata") or not is_under_spec:
                simple_counts[node.token] += 1
            else:
                # Compound: kind.parent_path.key
                full_path: str = f"{node.parent_path}.{node.token}" if node.parent_path else node.token
                compound_key: str = f"{kind}.{full_path}"
                compound_counts[compound_key] += 1
                path_depth_counts[full_path.count(".") + 1] += 1

        if (doc_idx + 1) % 50000 == 0:
            print(f"  {doc_idx + 1:,}/{len(documents):,} processed")

    print(f"\nSimple vocabulary: {len(simple_counts):,} unique keys")
    print(f"Compound vocabulary: {len(compound_counts):,} unique targets")

    print("\n=== Compound target frequency distribution ===")
    for lo, hi in [(1,1), (2,5), (6,10), (11,50), (51,100), (101,500), (501,1000), (1001,5000), (5001,50000), (50001,500000)]:
        n = sum(1 for c in compound_counts.values() if lo <= c <= hi)
        print(f"  freq {lo:>6}-{hi:<6}: {n:>6} targets")

    print("\n=== Path depth in compound targets ===")
    for depth in sorted(path_depth_counts.keys()):
        print(f"  depth {depth}: {path_depth_counts[depth]:>10,}")

    print("\n=== Top 30 compound targets ===")
    for target, count in compound_counts.most_common(30):
        print(f"  {count:>8,}  {target}")

    print("\n=== Compound targets at different min_freq ===")
    for mf in [1, 5, 10, 50, 100, 500]:
        n = sum(1 for c in compound_counts.values() if c >= mf)
        print(f"  min_freq={mf:>4}: {n:>6} targets")


if __name__ == "__main__":
    main()
