"""Four checks to validate the subword tokenization redesign before
committing to a model retrain:

  1. Dataset wiring — take a real manifest, linearize it, BPE-tokenize each
     node, build the expanded subword sequence, and inspect the structure.
  2. Sequence-length distribution over the full 276K corpus after
     sub-tokenization. How many docs blow past max_seq_len=512?
  3. Tokenizer round-trip — encode → decode must equal the input.
  4. Architectural param-count impact.
"""

from __future__ import annotations

import pickle
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from yaml_bert.annotator import DomainAnnotator
from yaml_bert.linearizer import YamlLinearizer
from yaml_bert.types import NodeType

from tokenizers import Tokenizer


CACHE_PATH = "output_v8_276K_recon_seed42/doc_cache.pkl"
TOKENIZER_PATH = "output_v8_276K_recon_seed42/unified_bpe_8k.json"
LONG_VALUE_THRESHOLD = 256
MAX_CHARS_VALUE = 64
MAX_CHARS_KEY = 128

LONG_VALUE_TOKEN = "[LONG_VALUE]"

SAMPLE_MANIFEST = """apiVersion: apps/v1
kind: Deployment
metadata:
  name: web
  namespace: production
  annotations:
    kubectl.kubernetes.io/last-applied-configuration: "irrelevant payload"
    app.kubernetes.io/managed-by: argocd
spec:
  replicas: 3
  selector:
    matchLabels:
      app: web
  template:
    metadata:
      labels:
        app: web
    spec:
      containers:
      - name: app
        image: nginx:1.25
        ports:
        - containerPort: 80
"""


def encode_token_subwords(tokenizer, node):
    """Encode one node's token to a list of subword IDs.

    Mirrors what the future dataset would do:
      - VALUE >= LONG_THRESHOLD chars  → single [LONG_VALUE] token
      - VALUE > MAX_CHARS_VALUE        → truncate to MAX_CHARS_VALUE, encode
      - KEY   > MAX_CHARS_KEY          → truncate to MAX_CHARS_KEY, encode
      - everything else                → encode as-is
    """
    is_value = node.node_type in (NodeType.VALUE, NodeType.LIST_VALUE)
    tok = node.token
    if is_value and len(tok) >= LONG_VALUE_THRESHOLD:
        return [tokenizer.token_to_id(LONG_VALUE_TOKEN)]
    if is_value and len(tok) > MAX_CHARS_VALUE:
        tok = tok[:MAX_CHARS_VALUE]
    elif (not is_value) and len(tok) > MAX_CHARS_KEY:
        tok = tok[:MAX_CHARS_KEY]
    enc = tokenizer.encode(tok)
    return enc.ids


# ---------- CHECK 1: dataset wiring ----------

def check1_wiring(tokenizer):
    print("=" * 70)
    print("CHECK 1 — DATASET WIRING (one real manifest, end-to-end)")
    print("=" * 70)

    linearizer = YamlLinearizer()
    annotator = DomainAnnotator()
    nodes = linearizer.linearize(SAMPLE_MANIFEST)
    annotator.annotate(nodes)

    print(f"Linearized to {len(nodes)} logical nodes.\n")

    # Build expanded subword sequence
    expanded = []  # (subword_token, subword_id, node_type, depth, sib, path, parent_idx)
    for i, n in enumerate(nodes):
        ids = encode_token_subwords(tokenizer, n)
        tokens_repr = [tokenizer.id_to_token(t) for t in ids]
        for sub_id, sub_tok in zip(ids, tokens_repr):
            expanded.append({
                "subword": sub_tok,
                "id": sub_id,
                "node_type": n.node_type.value,
                "depth": n.depth,
                "sibling": n.sibling_index,
                "path": n.parent_path,
                "logical_node_idx": i,
            })

    print(f"Expanded to {len(expanded)} subword positions  "
          f"(ratio {len(expanded)/len(nodes):.2f}x).\n")

    print("Position-by-position (first 50):")
    print(f"  {'#':>3} {'subword':<25} {'type':<10} "
          f"{'d':>2} {'s':>2} {'logical#':>9}  path")
    print("-" * 100)
    for k, e in enumerate(expanded[:50]):
        path = e["path"]
        if len(path) > 30:
            path = "…" + path[-29:]
        print(f"  {k:>3} {e['subword'][:24]!r:<25} {e['node_type']:<10} "
              f"{e['depth']:>2} {e['sibling']:>2} {e['logical_node_idx']:>9}  {path}")
    if len(expanded) > 50:
        print(f"  ... ({len(expanded) - 50} more)")

    # Invariants
    print()
    print("Invariants:")
    by_type = {}
    for e in expanded:
        by_type[e["node_type"]] = by_type.get(e["node_type"], 0) + 1
    print(f"  positions by node_type: {by_type}")
    print(f"  unique logical_node_idx: "
          f"{len(set(e['logical_node_idx'] for e in expanded))} == {len(nodes)}")
    # Group by logical_node_idx — verify every position has the same depth/sib/path within a group
    bad = 0
    for i, n in enumerate(nodes):
        group = [e for e in expanded if e["logical_node_idx"] == i]
        if len({e["depth"] for e in group}) != 1:
            bad += 1
        if len({e["sibling"] for e in group}) != 1:
            bad += 1
        if len({e["path"] for e in group}) != 1:
            bad += 1
    print(f"  intra-group (depth/sibling/path) consistency violations: {bad}")


# ---------- CHECK 2: corpus sequence-length distribution ----------

def check2_seqlen(tokenizer):
    print()
    print("=" * 70)
    print("CHECK 2 — SEQUENCE-LENGTH DISTRIBUTION (full 276K corpus)")
    print("=" * 70)

    print(f"Loading doc cache from {CACHE_PATH}...")
    with open(CACHE_PATH, "rb") as f:
        cache = pickle.load(f)
    print(f"  {len(cache):,} docs")

    long_value_id = tokenizer.token_to_id(LONG_VALUE_TOKEN)

    lengths_atomic: list[int] = []
    lengths_subword: list[int] = []
    t0 = time.time()
    BATCH = 1024
    for doc_idx, nodes in enumerate(cache):
        if not nodes:
            continue
        atomic = len(nodes)
        lengths_atomic.append(atomic)

        # Build the list of strings to encode (with truncation) plus a mask
        # for which positions are [LONG_VALUE] shortcuts
        strs = []
        long_value_count = 0
        for n in nodes:
            is_value = n.node_type in (NodeType.VALUE, NodeType.LIST_VALUE)
            tok = n.token
            if is_value and len(tok) >= LONG_VALUE_THRESHOLD:
                long_value_count += 1
                continue
            if is_value and len(tok) > MAX_CHARS_VALUE:
                tok = tok[:MAX_CHARS_VALUE]
            elif (not is_value) and len(tok) > MAX_CHARS_KEY:
                tok = tok[:MAX_CHARS_KEY]
            strs.append(tok)
        encs = tokenizer.encode_batch(strs) if strs else []
        subword_total = sum(len(e.ids) for e in encs) + long_value_count
        lengths_subword.append(subword_total)

        if (doc_idx + 1) % 25000 == 0:
            elapsed = time.time() - t0
            print(f"  {doc_idx + 1:,} / {len(cache):,} docs  ({elapsed:.1f}s)")

    print(f"  done in {time.time() - t0:.1f}s")
    print()

    import statistics as st
    print(f"ATOMIC sequence lengths (per-node, today's design):")
    _print_stats(lengths_atomic)
    print(f"SUBWORD sequence lengths (proposed):")
    _print_stats(lengths_subword)

    expansion = [s / a for s, a in zip(lengths_subword, lengths_atomic) if a > 0]
    print(f"  expansion ratio: mean={st.mean(expansion):.2f}, "
          f"median={st.median(expansion):.2f}")
    print()

    thresholds = [128, 256, 384, 512, 768, 1024, 2048]
    print(f"Fraction of docs that fit at each max_seq_len:")
    print(f"  {'max_len':>8}  atomic %     subword %")
    for t in thresholds:
        a = sum(1 for x in lengths_atomic if x <= t) / len(lengths_atomic) * 100
        s = sum(1 for x in lengths_subword if x <= t) / len(lengths_subword) * 100
        print(f"  {t:>8}  {a:>7.2f}%    {s:>7.2f}%")


def _print_stats(lengths):
    import statistics as st
    print(f"  count: {len(lengths):,}")
    print(f"  mean:  {st.mean(lengths):.1f}")
    print(f"  median: {st.median(lengths):.0f}")
    sl = sorted(lengths)
    for pc in (50, 75, 90, 95, 99, 99.9):
        i = min(len(sl) - 1, int(len(sl) * pc / 100))
        print(f"  p{pc}: {sl[i]:>6,}")
    print(f"  max:    {max(lengths):,}")


# ---------- CHECK 3: round-trip ----------

def check3_roundtrip(tokenizer):
    print()
    print("=" * 70)
    print("CHECK 3 — ROUND-TRIP (encode → decode == original)")
    print("=" * 70)

    test_strings = [
        # short, exotic identifiers
        "apiVersion", "kind", "metadata", "web-1", "web-99",
        # values with special chars
        "nginx:1.25", "apps/v1", "8080", "ClusterIP",
        # long-ish keys
        "app.kubernetes.io/name",
        "service.beta.kubernetes.io/aws-load-balancer-cross-zone-load-balancing-enabled",
        # spaces and punctuation
        "Hello World",
        "/etc/ssl/certs/ca-certificates.crt",
        "https://github.com/kubernetes/kubernetes",
        # unicode
        "résumé.txt",
        "日本語",
        # empty-ish edge cases
        "a", " ", "  ",
    ]
    print(f"Testing {len(test_strings)} strings...")
    failures = []
    for s in test_strings:
        enc = tokenizer.encode(s)
        decoded = tokenizer.decode(enc.ids)
        if decoded != s:
            failures.append((s, decoded))
            print(f"  ✗ {s!r}  →  {enc.tokens}  →  {decoded!r}")
        else:
            print(f"  ✓ {s!r}  ({len(enc.ids)} subwords)")
    if not failures:
        print("All round-trips passed.")
    else:
        print(f"\n{len(failures)} round-trip failures.")


# ---------- CHECK 4: param-count impact ----------

def check4_params(tokenizer):
    print()
    print("=" * 70)
    print("CHECK 4 — ARCHITECTURAL PARAM IMPACT")
    print("=" * 70)

    import json
    with open("output_v8_276K_recon_seed42/vocab.json") as f:
        vocab = json.load(f)
    n_keys = len(vocab["key_vocab"]) + len(vocab["special_tokens"])
    n_values = len(vocab["value_vocab"]) + len(vocab["special_tokens"])
    n_unified = tokenizer.get_vocab_size()
    d_model = 256

    print(f"d_model = {d_model}")
    print()
    print(f"v8 today (separate key + value tables):")
    print(f"  key_embedding:   {n_keys:>6} × {d_model} = {n_keys * d_model:>10,} params")
    print(f"  value_embedding: {n_values:>6} × {d_model} = {n_values * d_model:>10,} params")
    print(f"  total:                           = {(n_keys + n_values) * d_model:>10,} params")
    print()
    print(f"v9 proposed (one unified subword table):")
    print(f"  subword_embedding: {n_unified:>4} × {d_model} = {n_unified * d_model:>10,} params")
    print()
    saving = (n_keys + n_values - n_unified) * d_model
    print(f"Embedding-table net change: {-saving:+,} params "
          f"({100 * -saving / ((n_keys + n_values) * d_model):+.1f}%)")
    print()
    # v8 total was 22.5M params
    v8_total = 22_525_506
    new_total_estimate = v8_total - saving
    print(f"Estimated total params:")
    print(f"  v8 today:     {v8_total:>11,}")
    print(f"  v9 estimate:  {new_total_estimate:>11,}  "
          f"({100 * (new_total_estimate - v8_total) / v8_total:+.1f}%)")
    print()
    print("Caveat: this assumes the rest of the architecture is unchanged.")
    print("In practice we might want to bump d_model or add a small")
    print("node_type embedding (4 types × d_model = trivial).")


# ---------- main ----------

def main() -> None:
    print(f"Loading unified tokenizer from {TOKENIZER_PATH}...")
    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    print(f"  vocab size: {tokenizer.get_vocab_size()}")
    print()

    check1_wiring(tokenizer)
    check3_roundtrip(tokenizer)
    check4_params(tokenizer)
    # Check 2 is slowest — run it last so the others' output is visible first
    check2_seqlen(tokenizer)


if __name__ == "__main__":
    main()
