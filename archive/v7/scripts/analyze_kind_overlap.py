"""Analyze which spec keys appear under multiple kinds.

Shows which compound targets actually benefit from kind-specificity.

Usage:
    PYTHONPATH=. python scripts/analyze_kind_overlap.py output_v3_full/doc_cache.pkl
"""
from __future__ import annotations
import _setup_path  # noqa: F401

import argparse
import pickle
from collections import Counter, defaultdict

from yaml_bert.types import NodeType, YamlNode
from yaml_bert.dataset import _extract_kind


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("cache", type=str)
    parser.add_argument("--min-freq", type=int, default=500)
    args = parser.parse_args()

    print(f"Loading cache: {args.cache}")
    with open(args.cache, "rb") as f:
        docs: list[list[YamlNode]] = pickle.load(f)
    print(f"Loaded {len(docs):,} documents")

    # Count compound targets
    compound: Counter = Counter()
    for nodes in docs:
        kind: str = _extract_kind(nodes)
        if not kind:
            continue
        for n in nodes:
            if n.node_type not in (NodeType.KEY, NodeType.LIST_KEY):
                continue
            if n.parent_path.startswith("spec") and n.depth > 0:
                compound[f"{kind}.{n.parent_path}.{n.token}"] += 1

    # Group by path.key, find which kinds share the same path.key
    by_path_key: defaultdict[str, dict[str, int]] = defaultdict(dict)
    for target, count in compound.items():
        if count >= args.min_freq:
            parts = target.split(".", 1)
            kind_prefix: str = parts[0]
            rest: str = parts[1]
            by_path_key[rest][kind_prefix] = count

    multi_kind = {pk: kinds for pk, kinds in by_path_key.items() if len(kinds) > 1}
    single_kind = {pk: kinds for pk, kinds in by_path_key.items() if len(kinds) == 1}

    total_targets: int = sum(1 for c in compound.values() if c >= args.min_freq)
    print(f"\nCompound targets (min_freq={args.min_freq}): {total_targets}")
    print(f"Unique path.key combos: {len(by_path_key)}")
    print(f"  Appearing in 1 kind only: {len(single_kind)}")
    print(f"  Appearing in multiple kinds: {len(multi_kind)}")

    print(f"\n=== Path.keys shared across multiple kinds (most shared first) ===")
    for pk, kinds in sorted(multi_kind.items(), key=lambda x: -len(x[1])):
        kinds_str: str = ", ".join(f"{k}({c})" for k, c in sorted(kinds.items(), key=lambda x: -x[1]))
        print(f"  {pk}")
        print(f"    {kinds_str}")

    print(f"\n=== Kind-exclusive path.keys (top 30 by frequency) ===")
    exclusive_sorted = sorted(
        single_kind.items(),
        key=lambda x: -list(x[1].values())[0]
    )
    for pk, kinds in exclusive_sorted[:30]:
        kind_name: str = list(kinds.keys())[0]
        count: int = list(kinds.values())[0]
        print(f"  {count:>8,}  {kind_name}: {pk}")


if __name__ == "__main__":
    main()
