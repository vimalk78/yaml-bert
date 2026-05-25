"""Count how many status-side trigrams would be kept vs dropped at various min_freq.

Question: if we exempt status trigrams from min_freq filtering, how many
extra targets would we be adding to the kind_target_vocab?

Usage:
    PYTHONPATH=. python scripts/count_status_trigrams.py output_v4/doc_cache.pkl
"""
from __future__ import annotations
import _setup_path  # noqa: F401

import argparse
import pickle
from collections import Counter

from yaml_bert.dataset import _extract_kind
from yaml_bert.types import NodeType, YamlNode
from yaml_bert.vocab import compute_target


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("cache", type=str, help="Path to doc_cache.pkl")
    args = parser.parse_args()

    print(f"Loading cache: {args.cache}")
    with open(args.cache, "rb") as f:
        documents: list[list[YamlNode]] = pickle.load(f)
    print(f"Loaded {len(documents):,} documents\n")

    kind_target_counts: Counter = Counter()

    for nodes in documents:
        kind: str = _extract_kind(nodes)
        for node in nodes:
            if node.node_type not in (NodeType.KEY, NodeType.LIST_KEY):
                continue
            target, head_type = compute_target(node, kind)
            if head_type == "kind_specific":
                kind_target_counts[target] += 1

    total_unique_trigrams = len(kind_target_counts)
    total_occurrences = sum(kind_target_counts.values())

    status_trigrams = {t: c for t, c in kind_target_counts.items()
                       if t.split("::", 2)[1] == "status"}
    nonstatus_trigrams = {t: c for t, c in kind_target_counts.items()
                          if t.split("::", 2)[1] != "status"}

    print(f"== Total kind_specific trigrams ==")
    print(f"  Unique trigrams:    {total_unique_trigrams:>8,}")
    print(f"  Total occurrences:  {total_occurrences:>8,}")
    print()
    print(f"== Split by parent ==")
    print(f"  status:: trigrams (unique):    {len(status_trigrams):>8,}  ({len(status_trigrams)/total_unique_trigrams:.1%})")
    print(f"  status:: trigrams (occurrences): {sum(status_trigrams.values()):>6,}  ({sum(status_trigrams.values())/total_occurrences:.1%})")
    print(f"  non-status (unique):           {len(nonstatus_trigrams):>8,}")
    print()

    print(f"== Effect of min_freq threshold ==")
    print(f"{'min_freq':>10}  {'status kept':>12}  {'status drop':>12}  {'nonstatus kept':>14}  {'nonstatus drop':>14}")
    for thresh in [1, 10, 50, 100, 500, 1000]:
        status_kept = sum(1 for c in status_trigrams.values() if c >= thresh)
        status_dropped = len(status_trigrams) - status_kept
        nonstatus_kept = sum(1 for c in nonstatus_trigrams.values() if c >= thresh)
        nonstatus_dropped = len(nonstatus_trigrams) - nonstatus_kept
        print(f"{thresh:>10}  {status_kept:>12,}  {status_dropped:>12,}  {nonstatus_kept:>14,}  {nonstatus_dropped:>14,}")
    print()

    print(f"== Top 15 status trigrams (most frequent) ==")
    for target, count in sorted(status_trigrams.items(), key=lambda x: -x[1])[:15]:
        print(f"  {count:>6,}  {target}")
    print()

    print(f"== Bottom 10 status trigrams (least frequent) ==")
    for target, count in sorted(status_trigrams.items(), key=lambda x: x[1])[:10]:
        print(f"  {count:>6,}  {target}")
    print()

    # What we'd get if we exempted status from min_freq=100:
    extra_if_exempt = sum(1 for c in status_trigrams.values() if c < 100)
    current_at_100 = sum(1 for c in kind_target_counts.values() if c >= 100)
    print(f"== Impact of exempting status:: from min_freq=100 ==")
    print(f"  Current kind_target_vocab at min_freq=100: {current_at_100:,}")
    print(f"  Extra status trigrams added if exempted:   {extra_if_exempt:,}")
    print(f"  New total kind_target_vocab:               {current_at_100 + extra_if_exempt:,}  "
          f"(+{extra_if_exempt/current_at_100:.1%})")


if __name__ == "__main__":
    main()
