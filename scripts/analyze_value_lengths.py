"""Analyze the distribution of VALUE token lengths in the v8 training corpus.

Answers: how bad is the long-value problem for sub-tokenization?

For every VALUE / LIST_VALUE node across all 276K docs:
  - length in chars
  - whether it's in the current value_vocab (i.e. NOT [UNK])
  - parent path
  - keep a sample of long values for visual inspection

Outputs printed to stdout. Pipe to a file if you want.
"""

from __future__ import annotations

import json
import pickle
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from yaml_bert.types import NodeType


CACHE_PATH = "output_v8_276K_recon_seed42/doc_cache.pkl"
VOCAB_PATH = "output_v8_276K_recon_seed42/vocab.json"
LONG_THRESHOLD = 100        # chars — values above this are "long"
SAMPLE_LONG_VALUES = 30     # how many long values to print
SAMPLE_SEED = 42


def main() -> None:
    print(f"Loading vocab from {VOCAB_PATH}...")
    with open(VOCAB_PATH) as f:
        vocab = json.load(f)
    value_vocab = set(vocab["value_vocab"].keys())
    print(f"  {len(value_vocab):,} values in vocab")

    print(f"Loading doc cache from {CACHE_PATH}...")
    with open(CACHE_PATH, "rb") as f:
        cache = pickle.load(f)
    print(f"  {len(cache):,} docs")

    # Counters
    length_buckets = [0, 10, 20, 50, 100, 200, 500, 1000, 5000, 10**9]
    length_labels = [
        "<10", "10-19", "20-49", "50-99", "100-199",
        "200-499", "500-999", "1000-4999", "5000+",
    ]
    bucket_counts = [0] * (len(length_buckets) - 1)
    bucket_counts_unk = [0] * (len(length_buckets) - 1)
    bucket_counts_invocab = [0] * (len(length_buckets) - 1)

    # Path distribution for long values
    long_path_counter: Counter[str] = Counter()

    # Sample of long values for visual inspection
    random.seed(SAMPLE_SEED)
    long_samples: list[tuple[int, str, str]] = []  # (length, path, value)

    total_values = 0
    total_unk_values = 0

    for doc_idx, nodes in enumerate(cache):
        if (doc_idx + 1) % 50000 == 0:
            print(f"  {doc_idx + 1:,} / {len(cache):,} docs scanned")

        for node in nodes:
            if node.node_type not in (NodeType.VALUE, NodeType.LIST_VALUE):
                continue
            total_values += 1
            length = len(node.token)
            is_unk = node.token not in value_vocab
            if is_unk:
                total_unk_values += 1

            # Bucket
            for b in range(len(length_buckets) - 1):
                if length < length_buckets[b + 1]:
                    bucket_counts[b] += 1
                    if is_unk:
                        bucket_counts_unk[b] += 1
                    else:
                        bucket_counts_invocab[b] += 1
                    break

            # Long-value path tracking
            if length >= LONG_THRESHOLD:
                long_path_counter[node.parent_path] += 1
                # Reservoir sampling for inspection
                if len(long_samples) < SAMPLE_LONG_VALUES:
                    long_samples.append((length, node.parent_path, node.token))
                elif random.random() < SAMPLE_LONG_VALUES / (
                    long_path_counter.total()
                ):
                    long_samples[random.randrange(SAMPLE_LONG_VALUES)] = (
                        length, node.parent_path, node.token,
                    )

    print()
    print("=" * 70)
    print(f"TOTAL VALUES: {total_values:,}")
    print(f"  in vocab:  {total_values - total_unk_values:,} "
          f"({100 * (total_values - total_unk_values) / total_values:.1f}%)")
    print(f"  [UNK]:     {total_unk_values:,} "
          f"({100 * total_unk_values / total_values:.1f}%)")
    print()

    print("LENGTH DISTRIBUTION (chars):")
    print(f"  {'bucket':<12} {'count':>12} {'pct':>8} | "
          f"{'in-vocab':>12} {'[UNK]':>12} {'unk_rate':>9}")
    for i, label in enumerate(length_labels):
        cnt = bucket_counts[i]
        pct = 100 * cnt / total_values if total_values else 0
        iv = bucket_counts_invocab[i]
        uk = bucket_counts_unk[i]
        ur = 100 * uk / cnt if cnt else 0
        print(f"  {label:<12} {cnt:>12,} {pct:>7.2f}% | "
              f"{iv:>12,} {uk:>12,} {ur:>8.1f}%")
    print()

    print(f"LONG VALUES (>={LONG_THRESHOLD} chars): "
          f"{long_path_counter.total():,} total at "
          f"{len(long_path_counter):,} distinct paths")
    print()
    print("Top 25 paths where long values appear:")
    for path, count in long_path_counter.most_common(25):
        display_path = path if path else "(root)"
        if len(display_path) > 60:
            display_path = display_path[:57] + "..."
        print(f"  {count:>8,}  {display_path}")
    print()

    print(f"SAMPLE OF {len(long_samples)} LONG VALUES:")
    print("-" * 70)
    for length, path, value in sorted(long_samples)[:SAMPLE_LONG_VALUES]:
        preview = value[:120].replace("\n", "\\n")
        if len(value) > 120:
            preview += "..."
        print(f"  [{length:>5} chars] path={path or '(root)'!r}")
        print(f"    {preview}")
        print()


if __name__ == "__main__":
    main()
