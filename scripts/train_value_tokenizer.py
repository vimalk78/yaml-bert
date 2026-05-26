"""Train a byte-level BPE tokenizer on the YAML-BERT value corpus.

Why byte-level: guarantees every string is encodable (no [UNK] for novel
inputs). Why BPE: maximizes shared subwords across the long-tail of
identifiers, paths, URLs, and image refs in the K8s YAML domain.

Source data: VALUE / LIST_VALUE tokens from the 276K-doc training cache,
truncated to MAX_CHARS to keep training tractable and to model how we
plan to handle long values at encode time.

Output: a single HF tokenizers JSON at OUT_PATH.
"""

from __future__ import annotations

import pickle
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from yaml_bert.types import NodeType

from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as BL_Decoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel as BL_PreTokenizer
from tokenizers.trainers import BpeTrainer


CACHE_PATH = "output_v8_276K_recon_seed42/doc_cache.pkl"
OUT_PATH = "output_v8_276K_recon_seed42/value_bpe_4k.json"
VOCAB_SIZE = 4096
MAX_CHARS = 64           # truncate values longer than this for training
LONG_THRESHOLD = 256     # values longer than this map to [LONG_VALUE] entirely
MIN_FREQUENCY = 2        # ignore pairs that only appear once

SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[MASK]", "[LONG_VALUE]"]


def value_iter(cache):
    """Yield (truncated) value strings from the corpus.

    Values >= LONG_THRESHOLD chars are skipped — they'll be mapped to
    [LONG_VALUE] at encode time, so they don't contribute to BPE training.
    Values between MAX_CHARS and LONG_THRESHOLD are truncated to MAX_CHARS.
    """
    total = 0
    skipped_long = 0
    truncated = 0
    t0 = time.time()
    for doc_idx, nodes in enumerate(cache):
        for node in nodes:
            if node.node_type not in (NodeType.VALUE, NodeType.LIST_VALUE):
                continue
            tok = node.token
            length = len(tok)
            if length >= LONG_THRESHOLD:
                skipped_long += 1
                continue
            if length > MAX_CHARS:
                tok = tok[:MAX_CHARS]
                truncated += 1
            total += 1
            yield tok
        if (doc_idx + 1) % 50000 == 0:
            elapsed = time.time() - t0
            print(f"  {doc_idx + 1:,} docs · {total:,} values · "
                  f"{truncated:,} truncated · {skipped_long:,} skipped "
                  f"(too long) · {elapsed:.1f}s")
    print(f"Total: {total:,} values yielded, {truncated:,} truncated, "
          f"{skipped_long:,} skipped as [LONG_VALUE]")


def main() -> None:
    print(f"Loading doc cache from {CACHE_PATH}...")
    with open(CACHE_PATH, "rb") as f:
        cache = pickle.load(f)
    print(f"  {len(cache):,} docs")

    print("Setting up byte-level BPE tokenizer...")
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = BL_PreTokenizer(add_prefix_space=False)
    tokenizer.decoder = BL_Decoder()

    trainer = BpeTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=BL_PreTokenizer.alphabet(),
        min_frequency=MIN_FREQUENCY,
        show_progress=True,
    )

    print(f"Training BPE (target vocab={VOCAB_SIZE}, "
          f"max_chars={MAX_CHARS}, min_freq={MIN_FREQUENCY})...")
    t0 = time.time()
    tokenizer.train_from_iterator(value_iter(cache), trainer=trainer)
    print(f"  trained in {time.time() - t0:.1f}s")
    print(f"  final vocab size: {tokenizer.get_vocab_size()}")

    Path(OUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(OUT_PATH)
    size_kb = Path(OUT_PATH).stat().st_size / 1024
    print(f"Saved to {OUT_PATH} ({size_kb:.0f} KB)")

    # ----- Inspection -----
    print()
    print("=" * 70)
    print("SAMPLE TOKENIZATIONS")
    print("=" * 70)

    probes = [
        ("web-1", "the collision case"),
        ("web-2", "in old vocab"),
        ("web-3", "user-added"),
        ("web-99", "extrapolation"),
        ("nginx", "common image"),
        ("nginx:1.25", "image + tag"),
        ("nginx:1.20", "different tag"),
        ("nginx:1.25.3-alpine", "patched + variant"),
        ("ml-pipeline-ui-artifact", "kebab identifier"),
        ("kube-system", "common namespace"),
        ("kubectl.kubernetes.io/last-applied-configuration", "long annotation key"),
        ("/etc/ssl/certs/ca-certificates.crt", "path"),
        ("https://github.com/kubernetes/kubernetes", "URL"),
        ("revisions.serving.knative.dev", "API group"),
        ("--namespace=production", "CLI flag"),
        ("apps/v1", "apiVersion"),
        ("8080", "port number"),
        ("ClusterIP", "service type"),
        ("RollingUpdate", "deployment strategy"),
        ("Pod", "kind"),
    ]
    print(f"{'value':<55} → {'tokens':<60} (n)")
    print("-" * 130)
    for value, desc in probes:
        enc = tokenizer.encode(value)
        tokens_repr = " | ".join(t for t in enc.tokens)
        if len(tokens_repr) > 56:
            tokens_repr = tokens_repr[:53] + "..."
        print(f"  {value[:50]!r:<50} → {tokens_repr:<60} ({len(enc.ids)})")

    # ----- Distribution stats on a sample -----
    print()
    print("=" * 70)
    print("STATS")
    print("=" * 70)

    sample_lengths: list[int] = []
    sample_count = 0
    for doc_idx, nodes in enumerate(cache):
        if doc_idx >= 5000:
            break
        for node in nodes:
            if node.node_type not in (NodeType.VALUE, NodeType.LIST_VALUE):
                continue
            tok = node.token
            if len(tok) >= LONG_THRESHOLD:
                sample_lengths.append(1)  # [LONG_VALUE] is 1 token
                continue
            if len(tok) > MAX_CHARS:
                tok = tok[:MAX_CHARS]
            enc = tokenizer.encode(tok)
            sample_lengths.append(len(enc.ids))
            sample_count += 1
        if sample_count >= 200000:
            break

    import statistics
    print(f"Subwords per value (sample of {len(sample_lengths):,} values):")
    print(f"  mean: {statistics.mean(sample_lengths):.2f}")
    print(f"  median: {statistics.median(sample_lengths)}")
    print(f"  p90: {sorted(sample_lengths)[int(len(sample_lengths)*0.9)]}")
    print(f"  p99: {sorted(sample_lengths)[int(len(sample_lengths)*0.99)]}")
    print(f"  max: {max(sample_lengths)}")

    bucket_edges = [1, 2, 3, 5, 8, 13, 21, 10**9]
    bucket_labels = ["1", "2", "3-4", "5-7", "8-12", "13-20", "21+"]
    bucket_counts = [0] * (len(bucket_edges) - 1)
    for n in sample_lengths:
        for b in range(len(bucket_edges) - 1):
            if n < bucket_edges[b + 1]:
                bucket_counts[b] += 1
                break
    print("Distribution of subwords-per-value:")
    for lab, c in zip(bucket_labels, bucket_counts):
        pct = 100 * c / len(sample_lengths)
        bar = "#" * int(pct / 2)
        print(f"  {lab:<8} {c:>8,}  {pct:>5.1f}%  {bar}")


if __name__ == "__main__":
    main()
