"""Analyze bigram/trigram target vocabulary size.

Rule:
- Root keys (apiVersion, kind, metadata, spec, status): unigram → "metadata"
- Under metadata.*: bigram → "metadata.name"
- First level under spec: trigram → "Deployment.spec.replicas"
- Deeper under spec: bigram → "containers.image"

Usage:
    PYTHONPATH=. python scripts/analyze_bigram_targets.py output_v3_full/doc_cache.pkl
"""
from __future__ import annotations
import _setup_path  # noqa: F401

import argparse
import pickle
from collections import Counter

from yaml_bert.types import NodeType, YamlNode
from yaml_bert.dataset import _extract_kind
from yaml_bert.vocab import Vocabulary


def compute_target(node: YamlNode, kind: str) -> tuple[str, str]:
    """Compute the bigram/trigram target for a node.

    Returns:
        (target_string, gram_type) where gram_type is "uni", "bi", or "tri"
    Uses '::' as separator to avoid confusion with dots in key names.
    """
    # Root keys: unigram
    if node.depth == 0:
        return node.token, "uni"

    parent_key: str = Vocabulary.extract_parent_key(node.parent_path)

    # Under metadata: bigram (parent::key)
    if node.parent_path.startswith("metadata") or node.parent_path == "":
        target: str = f"{parent_key}::{node.token}" if parent_key else node.token
        return target, "bi"

    # First level under spec (depth=1, parent=spec): trigram (kind::spec::key)
    if node.depth == 1 and parent_key == "spec":
        return f"{kind}::spec::{node.token}", "tri"

    # First level under status (depth=1, parent=status): trigram (kind::status::key)
    if node.depth == 1 and parent_key == "status":
        return f"{kind}::status::{node.token}", "tri"

    # Everything else: bigram (parent::key)
    return f"{parent_key}::{node.token}", "bi"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("cache", type=str)
    args = parser.parse_args()

    print(f"Loading cache: {args.cache}")
    with open(args.cache, "rb") as f:
        docs: list[list[YamlNode]] = pickle.load(f)
    print(f"Loaded {len(docs):,} documents")

    target_counts: Counter = Counter()
    unigram_counts: Counter = Counter()
    bigram_counts: Counter = Counter()
    trigram_counts: Counter = Counter()

    for nodes in docs:
        kind: str = _extract_kind(nodes)
        if not kind:
            continue
        for node in nodes:
            if node.node_type not in (NodeType.KEY, NodeType.LIST_KEY):
                continue
            target, gram_type = compute_target(node, kind)
            target_counts[target] += 1

            if gram_type == "uni":
                unigram_counts[target] += 1
            elif gram_type == "bi":
                bigram_counts[target] += 1
            else:
                trigram_counts[target] += 1

    print(f"\nTotal unique targets: {len(target_counts):,}")
    print(f"  Unigrams: {len(unigram_counts):,}")
    print(f"  Bigrams:  {len(bigram_counts):,}")
    print(f"  Trigrams: {len(trigram_counts):,}")

    print(f"\n=== Frequency distribution ===")
    for lo, hi in [(1,1), (2,5), (6,10), (11,50), (51,100), (101,500), (501,1000), (1001,5000), (5001,50000), (50001,500000)]:
        n = sum(1 for c in target_counts.values() if lo <= c <= hi)
        print(f"  freq {lo:>6}-{hi:<6}: {n:>6} targets")

    print(f"\n=== At different min_freq ===")
    for mf in [1, 5, 10, 50, 100, 500]:
        n = sum(1 for c in target_counts.values() if c >= mf)
        coverage = sum(c for c in target_counts.values() if c >= mf) / sum(target_counts.values()) * 100
        print(f"  min_freq={mf:>4}: {n:>6} targets, {coverage:.1f}% coverage")

    print(f"\n=== Top 30 targets ===")
    for target, count in target_counts.most_common(30):
        if target in unigram_counts:
            gram = "uni"
        elif target in bigram_counts:
            gram = "bi"
        else:
            gram = "tri"
        print(f"  {count:>8,}  [{gram:>3}] {target}")

    print(f"\n=== Top 20 trigrams (kind-specific) ===")
    for target, count in trigram_counts.most_common(20):
        print(f"  {count:>8,}  {target}")

    print(f"\n=== Sample bigrams ===")
    for target, count in bigram_counts.most_common(30):
        print(f"  {count:>8,}  {target}")


if __name__ == "__main__":
    main()
