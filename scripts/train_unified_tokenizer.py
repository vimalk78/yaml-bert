"""Train a unified byte-level BPE tokenizer on KEY + VALUE strings.

Why unified: keys and values share substance in this domain
(`kube`, `controller`, `service`, `.k8s.io`, `nginx`, etc. appear in both).
A single BPE vocab makes shared subwords reusable across both, and the
KEY/VALUE asymmetry in the model (only KEYs aggregate into doc_vec) is
preserved via `node_type`, not via separate tokenizers.

Why this matters now: the key vocab in v8 turned out to be ~70-90%
user-defined keys (annotations, labels, env var names, ConfigMap keys),
which suffer the same long-tail / [UNK] problem as values. Sub-tokenizing
keys is therefore as important as sub-tokenizing values.
"""

from __future__ import annotations

import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from yaml_bert.types import NodeType

from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as BL_Decoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel as BL_PreTokenizer
from tokenizers.trainers import BpeTrainer


CACHE_PATH = "output_v8_276K_recon_seed42/doc_cache.pkl"
OUT_PATH = "output_v8_276K_recon_seed42/unified_bpe_8k.json"

VOCAB_SIZE = 8192
# Truncation at training time, matching what we'll do at encode time.
# Values can be longer than keys, but we cap keys for safety too.
MAX_CHARS_VALUE = 64
MAX_CHARS_KEY = 128
LONG_VALUE_THRESHOLD = 256   # value-only — keys always tokenized
MIN_FREQUENCY = 2

# Note: keys never get [LONG_VALUE] — a long key (e.g. a CRD annotation
# path) is still a meaningful identifier, even if rare. Only opaque user
# payload (long file contents in values) gets collapsed to a single token.
SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[MASK]", "[LONG_VALUE]"]

KEY_TYPES = (NodeType.KEY, NodeType.LIST_KEY)
VALUE_TYPES = (NodeType.VALUE, NodeType.LIST_VALUE)


def token_iter(cache):
    """Yield key and value strings from the corpus for BPE training."""
    total_keys = total_values = trunc_keys = trunc_values = skipped_long = 0
    t0 = time.time()
    for doc_idx, nodes in enumerate(cache):
        for node in nodes:
            tok = node.token
            n = len(tok)
            if node.node_type in KEY_TYPES:
                if n > MAX_CHARS_KEY:
                    tok = tok[:MAX_CHARS_KEY]
                    trunc_keys += 1
                total_keys += 1
                yield tok
            elif node.node_type in VALUE_TYPES:
                if n >= LONG_VALUE_THRESHOLD:
                    skipped_long += 1
                    continue
                if n > MAX_CHARS_VALUE:
                    tok = tok[:MAX_CHARS_VALUE]
                    trunc_values += 1
                total_values += 1
                yield tok
        if (doc_idx + 1) % 50000 == 0:
            elapsed = time.time() - t0
            print(f"  {doc_idx + 1:,} docs · "
                  f"{total_keys:,} keys · {total_values:,} values · "
                  f"{trunc_keys:,}+{trunc_values:,} truncated · "
                  f"{skipped_long:,} skipped · {elapsed:.1f}s")
    print(f"Total: {total_keys:,} keys + {total_values:,} values "
          f"yielded · {trunc_keys + trunc_values:,} truncated · "
          f"{skipped_long:,} long-values skipped")


def main() -> None:
    print(f"Loading doc cache from {CACHE_PATH}...")
    with open(CACHE_PATH, "rb") as f:
        cache = pickle.load(f)
    print(f"  {len(cache):,} docs")

    print("Setting up unified byte-level BPE tokenizer...")
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

    print(f"Training unified BPE (vocab={VOCAB_SIZE}, "
          f"max_key={MAX_CHARS_KEY}, max_value={MAX_CHARS_VALUE}, "
          f"long_value={LONG_VALUE_THRESHOLD})...")
    t0 = time.time()
    tokenizer.train_from_iterator(token_iter(cache), trainer=trainer)
    print(f"  trained in {time.time() - t0:.1f}s")
    print(f"  final vocab size: {tokenizer.get_vocab_size()}")

    Path(OUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(OUT_PATH)
    size_kb = Path(OUT_PATH).stat().st_size / 1024
    print(f"Saved to {OUT_PATH} ({size_kb:.0f} KB)")

    # ----- Inspection -----
    print()
    print("=" * 70)
    print("KEY TOKENIZATIONS")
    print("=" * 70)
    key_probes = [
        ("apiVersion", "core schema key"),
        ("kind", "core schema key"),
        ("metadata", "core schema key"),
        ("containerPort", "compound schema key"),
        ("nodeSelector", "compound schema key"),
        ("restartPolicy", "compound schema key"),
        ("imagePullPolicy", "compound schema key"),
        ("matchLabels", "compound schema key"),
        ("app.kubernetes.io/name", "common annotation"),
        ("kubectl.kubernetes.io/last-applied-configuration", "long annotation"),
        ("service.beta.kubernetes.io/aws-load-balancer-cross-zone-load-balancing-enabled", "very long annotation"),
        ("DATABASE_URL", "env var name"),
        ("JAVA_OPTS", "env var name"),
        ("config.yaml", "configmap data key"),
        ("test.sh", "configmap data key"),
        ("argocd.argoproj.io/sync-wave", "operator annotation"),
        ("custom-label-i-just-made-up", "fresh user key"),
    ]
    _print_probes(tokenizer, key_probes)

    print()
    print("=" * 70)
    print("VALUE TOKENIZATIONS")
    print("=" * 70)
    value_probes = [
        ("web-1", "the collision case"),
        ("web-2", "in old vocab"),
        ("web-3", "user-added"),
        ("nginx", "common image"),
        ("nginx:1.25", "image + tag"),
        ("nginx:1.20", "different tag"),
        ("Pod", "kind value"),
        ("Deployment", "kind value"),
        ("ClusterIP", "service type"),
        ("RollingUpdate", "deployment strategy"),
        ("apps/v1", "apiVersion"),
        ("v1", "apiVersion"),
        ("8080", "port"),
        ("ml-pipeline-ui-artifact", "kebab name"),
        ("kube-system", "namespace"),
        ("https://github.com/kubernetes/kubernetes", "URL"),
    ]
    _print_probes(tokenizer, value_probes)

    # ----- Stats -----
    print()
    print("=" * 70)
    print("STATS")
    print("=" * 70)
    key_lengths = []
    value_lengths = []
    sample_docs = 5000
    for doc_idx, nodes in enumerate(cache):
        if doc_idx >= sample_docs:
            break
        for node in nodes:
            tok = node.token
            if node.node_type in KEY_TYPES:
                tok = tok[:MAX_CHARS_KEY]
                key_lengths.append(len(tokenizer.encode(tok).ids))
            elif node.node_type in VALUE_TYPES:
                if len(tok) >= LONG_VALUE_THRESHOLD:
                    value_lengths.append(1)
                    continue
                tok = tok[:MAX_CHARS_VALUE]
                value_lengths.append(len(tokenizer.encode(tok).ids))

    import statistics
    print(f"Sample: {sample_docs:,} docs, "
          f"{len(key_lengths):,} keys, {len(value_lengths):,} values")
    print()
    print(f"KEY subwords per token:")
    print(f"  mean: {statistics.mean(key_lengths):.2f} · "
          f"median: {statistics.median(key_lengths)} · "
          f"p90: {sorted(key_lengths)[int(len(key_lengths)*0.9)]} · "
          f"p99: {sorted(key_lengths)[int(len(key_lengths)*0.99)]} · "
          f"max: {max(key_lengths)}")
    print(f"  {sum(1 for n in key_lengths if n == 1):,} keys = 1 subword "
          f"({100*sum(1 for n in key_lengths if n == 1)/len(key_lengths):.1f}%)")
    print()
    print(f"VALUE subwords per token:")
    print(f"  mean: {statistics.mean(value_lengths):.2f} · "
          f"median: {statistics.median(value_lengths)} · "
          f"p90: {sorted(value_lengths)[int(len(value_lengths)*0.9)]} · "
          f"p99: {sorted(value_lengths)[int(len(value_lengths)*0.99)]} · "
          f"max: {max(value_lengths)}")
    print(f"  {sum(1 for n in value_lengths if n == 1):,} values = 1 subword "
          f"({100*sum(1 for n in value_lengths if n == 1)/len(value_lengths):.1f}%)")
    print()
    avg_doc_tokens = (sum(key_lengths) + sum(value_lengths)) / sample_docs
    print(f"Average tokens per doc (after sub-tokenization): {avg_doc_tokens:.1f}")
    print(f"  (vs current atomic: ~{(len(key_lengths) + len(value_lengths)) / sample_docs:.1f})")


def _print_probes(tokenizer, probes):
    print(f"{'token':<60} → {'subwords':<60} (n)")
    print("-" * 135)
    for token, desc in probes:
        enc = tokenizer.encode(token)
        tokens_repr = " | ".join(t for t in enc.tokens)
        if len(tokens_repr) > 56:
            tokens_repr = tokens_repr[:53] + "..."
        token_repr = token if len(token) <= 55 else token[:52] + "..."
        print(f"  {token_repr!r:<58} → {tokens_repr:<60} ({len(enc.ids)})")


if __name__ == "__main__":
    main()
