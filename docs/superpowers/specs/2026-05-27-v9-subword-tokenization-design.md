# YAML-BERT v9 — Sub-tokenization Design Spec

## Goals & Non-Goals

### Primary goal

Replace v8's atomic key + atomic value vocabularies with a **single unified
byte-level BPE subword vocabulary**, so that the model can:

1. **Decompose long-tail tokens** that were `[UNK]` in v8 (`web-1`, `web-3`,
   `mycorp/internal-tool`, novel CRD annotations) into meaningful sub-pieces.
2. **Share substance across related tokens** (`nginx:1.25` and `nginx:1.20`
   share `nginx | : | 1 | .`; only the trailing digits differ).
3. **Eliminate the `[UNK]` collision class** where two distinct manifests
   collapse to the same `doc_vec` because their differing values both map
   to `[UNK]`. This was demonstrated as a live failure in the HF Space
   structural-probes tab.

### Secondary goal

**Drop the model's parameter count by ~41% as a side effect.** v8's two
embedding tables (6K keys + 38K values = 11.4M params) replace with one
8K subword table (2.1M params). Saving = 9.3M params. Possible future
investment of that budget into `d_model` or layers, deferred to a later
version.

### Non-goals (scope discipline)

This version makes **exactly one architectural change**: sub-tokenization.
Specifically NOT in scope for v9:

- **Letting values contribute to `doc_vec`.** Discussed and intriguing, but
  deferred. v9's `doc_vec` is structural-only (KEY subtrees), same as v8.
- **Predicting value subwords in MLM.** v9 masks KEYs only, same as v8.
- **Aggregating values into key subtree vectors.** Same v8 KEY-only
  aggregation logic.
- **Restricting cross-position attention to values.** Pure attention
  unchanged.
- **A second structural-vs-full embedding output.** v9 produces one
  `doc_vec`, same shape as v8.
- **Seq2seq prediction heads, autoregressive decoding, etc.**
- **Cleanup of `atomic_target_vocab` redundancy in `vocab.py`.** A good
  thing to do at some point, but not blocking v9. Tracked separately.

The principle: **one change at a time**. Sub-tokenization is a large enough
intervention that mixing it with other changes would make any
post-training analysis ambiguous about what caused what.

### Architecture family (fixed)

- Encoder-only transformer (BERT-style) — unchanged from v8
- MLM-style self-supervised pretraining — unchanged from v8
- Tree positional encoding (`depth` + `sibling_index` + `node_type`) —
  unchanged from v8
- Bottom-up tree aggregator with structural `doc_vec` — unchanged from v8
- Reconstruction objective (subtree masking + bag-of-keys) — kept,
  targeting the structural `doc_vec`

## Motivation: What v8 Gets Wrong

The HF Space structural-probes tab demonstrates the failure directly. The
"Pods in same namespace vs different namespace" preset includes:

| Letter | Manifest |
|---|---|
| A | `name: web-1`, `namespace: production` |
| B | `name: web-2`, `namespace: production` |
| C | `name: web-1`, `namespace: staging` |
| D | `name: web-2`, `namespace: staging` |

After a user adds `name: web-3, namespace: staging`, the v8 model reports
`cos(C, E) = 1.0000` — exactly identical. Reason:

```
'web-1' in value_vocab: False  → [UNK]
'web-2' in value_vocab: True   → its own embedding
'web-3' in value_vocab: False  → [UNK]
```

Both `web-1` and `web-3` map to `[UNK]`, so positions C and E produce
**byte-for-byte identical inputs** to the model. The model is right —
it sees the same input. The vocabulary is what's wrong.

This isn't a one-off quirk. The same problem affects:
- Image versioning (`nginx:1.25` vs `nginx:1.20` — different atoms, no
  relation visible to the model)
- Custom CRD annotations (`mycorp.com/internal-tool`, `dev.example.io/foo`
  — all `[UNK]`)
- User-defined ConfigMap keys, env var names, label keys

Corpus measurements (see `scripts/analyze_value_lengths.py` output):

- 5.49M total VALUE tokens across 276K docs
- 85.6% in vocab, **14.4% are `[UNK]`**
- `[UNK]` rate climbs with token length: 3.6% at <10 chars → 32.5% at
  20-49 chars → 53.2% at 50-99 chars

The key vocabulary suffers a parallel problem. Of 6,046 entries:
- ~70-90% are user-defined (annotation FQDNs, env var names, ConfigMap
  data keys, RBAC aggregation labels)
- Only ~500-1,500 are genuine K8s schema keys

Both halves of v8's vocabulary need the same fix: subword decomposition.

## Architecture Sketch

The shape doesn't change. Only the **input embedding** and **per-position
node tracking** change. All marked **NEW** below.

```
┌──────────────────────────────────────────────────────────────────────┐
│  Input: linearized YAML nodes, then BPE-expanded                     │
│                                                                      │
│  For each logical node (one per linearizer output position):         │
│    BPE-encode the token → 1..K subword positions                     │
│    Each subword position carries:                                    │
│      - subword_id   (the BPE token id)                       NEW     │
│      - node_type    (KEY / VALUE / LIST_KEY / LIST_VALUE)            │
│      - depth        (replicated from logical node)                   │
│      - sibling      (replicated from logical node)                   │
│      - parent_path  (replicated from logical node)                   │
│      - logical_id   (which original linearizer position)     NEW     │
└──────────────────────────────────────────────────────────────────────┘
                            ↓
┌──────────────────────────────────────────────────────────────────────┐
│  Embedding (unified subword table replaces v8's key + value)         │
│                                                                      │
│  emb[pos] =                                                          │
│    subword_emb[subword_id]            ← NEW: replaces v8 key + value │
│    + node_type_emb[node_type]         ← unchanged from v8            │
│    + depth_emb[depth]                 ← unchanged from v8            │
│    + sibling_emb[sibling]             ← unchanged from v8            │
└──────────────────────────────────────────────────────────────────────┘
                            ↓
┌──────────────────────────────────────────────────────────────────────┐
│  Encoder (unchanged from v8)                                         │
│  Transformer blocks. Output: hidden state h_i per subword position.  │
└──────────────────────────────────────────────────────────────────────┘
                            ↓
┌──────────────────────────────────────────────────────────────────────┐
│  Subword Pooling (NEW)                                               │
│                                                                      │
│  For each logical node l:                                            │
│    h_l = mean(h_i for i s.t. logical_id[i] == l)                     │
│                                                                      │
│  Result: one hidden vector per logical node, shape (n_logical, d).   │
│  From here, everything looks like v8 to downstream modules.          │
└──────────────────────────────────────────────────────────────────────┘
                            ↓
┌──────────────────────────────────────────────────────────────────────┐
│  Tree Aggregator (unchanged from v8)                                 │
│  Bottom-up over KEY logical nodes:                                   │
│    s_v = combine(h_v, children_of_v)                                 │
│  Output: doc_vec (root structural aggregation, same semantics as v8) │
└──────────────────────────────────────────────────────────────────────┘
                            ↓
          ┌─────────────────┴─────────────────────────┐
          ↓                                           ↓
┌──────────────────────────┐         ┌────────────────────────────────┐
│  Token Head (unchanged)  │         │  Reconstruction Head           │
│  Input: pooled hidden of │         │  (unchanged from v8)           │
│    a logical KEY +       │         │  Target: doc_vec_structural    │
│    doc_vec + s_parent    │         │                                │
│  Predict: atomic key id  │         │                                │
│    (from v8's            │         │                                │
│    atomic_target_vocab)  │         │                                │
└──────────────────────────┘         └────────────────────────────────┘
```

## Tokenizer

### Choice: byte-level BPE, vocab=8192

- **Byte-level** — guarantees every string is encodable. No `[UNK]` ever
  emitted by the tokenizer itself (vs the v8 vocab which had ~14.4% [UNK]
  values at the value level). At worst, an unseen byte sequence
  decomposes to its bytes.
- **BPE** — maximizes shared subwords across the long-tail of
  identifiers, paths, URLs, image refs, and CRD annotations in our domain.
- **Unified (one tokenizer for both keys and values)** — keys and values
  share substance in this domain (`kube`, `controller`, `service`,
  `.k8s.io`, `nginx` all appear in both). Single vocab is more efficient.
  The KEY/VALUE asymmetry stays at the `node_type` level.
- **8192 vocab size** — validated by training: common K8s schema keys
  (`apiVersion`, `kind`, `containerPort`, `nodeSelector`,
  `restartPolicy`, `imagePullPolicy`, `matchLabels`) all remain single
  subwords. User-defined keys decompose. Per-doc sequence-length impact
  is acceptable (see "Sequence length" below).

### Special tokens

The tokenizer's special tokens (reserved at training time):

| Token | Purpose |
|---|---|
| `[PAD]` | Padding (existing semantics) |
| `[UNK]` | Required by HF BPE but should never be emitted given byte-level |
| `[MASK]` | MLM masking (existing semantics) |
| `[LONG_VALUE]` | Single token replacing values ≥ 256 chars (see below) |

### Long values

Per the corpus analysis:

- 96% of values are < 100 chars (short identifiers, image refs, URLs)
- 0.95% of values are ≥ 256 chars (embedded file contents — full ConfigMap
  YAML/JSON, certificates, Grafana dashboards)

The long tail is pure user payload: file contents, CRD documentation
strings. No structural meaning. Encoding the actual bytes would blow the
sequence-length budget for no benefit.

**Rule (matches tokenizer training):**

| Value length | Encoding |
|---|---|
| < 256 chars but > 64 chars | truncate to 64 chars, then BPE-encode |
| ≥ 256 chars | replace with single `[LONG_VALUE]` token |

Keys are not subject to this rule (keys are always meaningful identifiers,
never opaque payload). Keys longer than 128 chars are truncated to 128
chars and BPE-encoded.

### Trained artifact

`output_v8_276K_recon_seed42/unified_bpe_8k.json` (526 KB). Will be moved
to `output_v9_*/value_bpe_unified.json` or similar at training time.

## Dataset Changes

### What v8 emits per position

Today's `YamlBertDataset.__getitem__` produces these tensors (per doc):

```python
{
  "token_ids":      LongTensor (N,)    # atomic key or value id
  "node_types":     LongTensor (N,)    # KEY / VALUE / LIST_KEY / LIST_VALUE
  "depths":         LongTensor (N,)    # tree depth
  "sibling_indices":LongTensor (N,)    # sibling rank under parent
  "parent_of":      LongTensor (N,)    # per-node parent position (for aggregator)
  # ... + MLM labels, MASK positions, recon stuff
}
```

Each position corresponds to one linearizer node.

### What v9 emits per position

After BPE-expansion, N grows (avg ~2.3× per doc). Same tensors, plus:

```python
{
  "token_ids":      LongTensor (N',)   # BPE subword id (was key/value id)
  "node_types":     LongTensor (N',)   # replicated from logical node
  "depths":         LongTensor (N',)   # replicated from logical node
  "sibling_indices":LongTensor (N',)   # replicated from logical node
  "logical_ids":    LongTensor (N',)   # NEW: index into the logical-node list
  "parent_of":      LongTensor (n_logical,)   # now indexed by logical, not subword
  # ... + MLM labels (whole-key masking), recon stuff
}
```

`logical_ids[i] = j` means subword `i` came from logical node `j`. This is
the contract that lets the pooling step group subwords back into their
logical nodes.

### Whole-key MLM masking (the only MLM change)

v8 picks ~15% of KEY positions and replaces each picked position's
`token_id` with `[MASK]`, recording the original atomic key as the target.

v9 still picks ~15% of **logical KEY positions** for masking, but:

1. For each picked logical KEY, **all of its subword positions** get
   replaced with the `[MASK]` subword id. The model cannot peek at K-1
   subwords to recover the Kth.
2. The MLM target is the same as v8: the **atomic key id** from
   `atomic_target_vocab` (which the dataset already builds from the
   training corpus). One target per masked logical KEY.
3. The Token Head consumes the **pooled hidden state** for the masked
   logical KEY's subwords and outputs over `atomic_target_vocab`. Same
   head architecture as v8.

The atomic_target_vocab survives unchanged because it's the right output
vocabulary even with BPE inputs: we want to predict whole-key targets
(`containers`, `replicas`, `restartPolicy`), not subwords.

If a masked KEY's atomic form is not in `atomic_target_vocab` (rare with
the v8-trained vocab — most user-defined keys above min_freq made it in,
but some won't), the position contributes no MLM loss. Same fallback as
v8.

### Reconstruction objective

Kept verbatim from v8. Bag-of-keys prediction over a masked subtree,
conditioned on `doc_vec`. With v9, "bag of keys" means the bag of atomic
key targets for the KEYs within the masked subtree — same atomic target
vocab as MLM. No change to the recon machinery.

## Model Changes

### Embedding module (`yaml_bert/embedding.py`)

Today: separate `key_embedding` and `value_embedding` tables, selection
gated by `node_type`.

New: single `subword_embedding` of shape `(vocab_size=8192, d_model=256)`.
Indexed directly by `token_ids` regardless of `node_type`. The
`node_type_embedding`, `depth_embedding`, `sibling_embedding` are
unchanged and summed in as today.

### Encoder

No changes.

### Aggregator (`yaml_bert/aggregator.py`)

Today: operates over per-node hidden states from the encoder, indexed
into KEY positions for tree aggregation.

New: adds a **subword pooling step** at the front. Inputs are hidden
states per subword + `logical_ids`. For each logical node, mean-pool its
subwords' hiddens → one vector per logical node. From that point on, the
aggregator logic is identical to v8 (build child→parent edges over KEY
logical nodes, aggregate bottom-up, produce `doc_vec` and per-key subtree
vecs).

This isolates the BPE-aware code to one place. The model below the
pooling step sees a logical-node-level world that matches v8.

### Token Head

No changes to the head's architecture. Its input is now the pooled
hidden state for the masked logical KEY (computed at the same pooling
step). Output is over the same `atomic_target_vocab` as v8.

### Reconstruction Head

No changes. Operates on `doc_vec` and subtree positional encoding as
today. Targets `atomic_target_vocab` ids.

## Hyperparameters

| Hyperparameter | v8 | v9 | Why change |
|---|---|---|---|
| `d_model` | 256 | **256** | Change one thing at a time. Could bump in v10. |
| `num_layers` | 6 | 6 | unchanged |
| `num_heads` | 8 | 8 | unchanged |
| `d_ff` | 1024 | 1024 | unchanged |
| `max_seq_len` | 512 | **768** | Covers 99.1% of corpus post-BPE vs 97.5% at 512 |
| `vocab_size` (input) | 6K + 38K | **8192** (unified) | The whole point of v9 |
| `mask_prob` | 0.15 | 0.15 | unchanged |
| `lr` | 1e-4 | 1e-4 | unchanged |
| `batch_size` | 32 | 32 (validate; see risk) | unchanged unless OOM |
| `num_epochs` | 30 | 30 | unchanged |
| `recon_loss_weight` | 0.5 | 0.5 | unchanged |

### Sequence length: why 768

Measured over the full 276K corpus (`scripts/validate_subword_pipeline.py`):

| `max_seq_len` | Atomic coverage (v8) | Subword coverage (v9) |
|---|---|---|
| 256 | 98.7% | 90.9% |
| 384 | 99.6% | 95.2% |
| 512 | 99.8% | 97.5% |
| **768** | 99.9% | **99.1%** |
| 1024 | 99.96% | 99.55% |

At 512, 2.3% of docs get truncated post-BPE — losing real structural
content. 768 recovers that without going overboard. The remaining 0.9%
are pathological CRDs / ConfigMap-content-heavy manifests where some
truncation is fine.

Cost: attention is O(n²), so growing `max_seq_len` from 512 → 768 is
~2.25× attention compute per padded position. Combined with the 2.3×
average sequence growth from BPE, per-doc cost rises ~5× vs v8. Offset
by the 41% smaller model.

## Parameter Budget

| Component | v8 params | v9 params | Δ |
|---|---|---|---|
| Key embedding | 1.55M | — | -1.55M |
| Value embedding | 9.82M | — | -9.82M |
| Subword embedding | — | **2.10M** | +2.10M |
| Encoder + aggregator + heads | 11.16M | 11.16M | 0 |
| **Total** | **22.53M** | **~13.25M** | **-41.2%** |

(Subject to small adjustments from changes in the input projection and
output head shapes; should be within 0.1M of estimate.)

## Training Setup

| Setting | Value |
|---|---|
| Corpus | `substratusai/the-stack-yaml-k8s`, same 276K docs used for v8 |
| Hardware | JarvisLabs L4 GPU (per existing playbook) |
| Initialization | **From scratch.** Encoder weights could be warm-started from v8, but the embedding tables are entirely different vocabularies. Cleaner to train fresh. Warm-start could be tried later as an ablation. |
| Loss | MLM + reconstruction, same weighting as v8 |
| Checkpoint cadence | Same as v8 (every 5 epochs + final) |

## Validation Plan

### Before training (local)

1. **Subword-pipeline validation script** (`scripts/validate_subword_pipeline.py`)
   already verifies tokenizer round-trips, end-to-end dataset wiring,
   corpus-wide sequence length distribution. Re-run after dataset code
   change to confirm no regressions.

2. **Per-batch shape audit.** A short script that runs one batch through
   the new dataset + collate + model forward, confirms all tensors have
   expected shapes, no NaN losses on initialization.

3. **5K quick-mode training run** (~30 minutes on L4): does MLM loss
   decrease, does recon loss decrease, no NaN/Inf, no OOM at batch_size=32.
   If OOM at 768, retry at batch_size=16 (with gradient accumulation to
   match effective batch size).

### After full training

4. **Recover all v8 capability test suites** (capabilities, structural,
   bigger-boat). v9 is allowed to score differently from v8 (different
   tokenization makes some tasks easier and others harder), but must
   pass overall structural understanding tests.

5. **Re-run the HF Space structural probes:**
   - `Pod ± initContainers` should still PASS
   - `Service type` should still PASS
   - **`Pods same/different namespace` may now PASS** (the namespace
     value is no longer [UNK], the model has the chance to use it). Either
     outcome is interesting — pass means namespace gets encoded; fail
     means our doc_vec design (KEYs only) successfully ignores value
     differences as intended.
   - `Pod vs Deployment wrapping it` should still PASS

6. **Re-run the C/E collision probe.** Adding `name: web-3, namespace:
   staging` should now produce a distinct doc_vec (not `cos = 1.0000`).
   If this still collides, the BPE is failing or the dataset wiring is
   broken.

7. **Acceptance gate (go/no-go for replacing v8 in the HF Space):**
   - All 4 structural probes give honest, defensible results
   - The collision case no longer collides
   - Capability test pass rate within ±10% of v8 (allowing for the BPE
     transition)
   - Doc_vec retrieval quality (kind k-NN purity on a held-out 5K) no
     worse than v8

## Risks & Mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| OOM at `max_seq_len=768` with `batch_size=32` | Medium | Fall back to batch_size=16 + 2-step gradient accumulation. |
| Whole-key MLM is too easy (model just learns "what subword is `[MASK]`?") | Low | The Token Head outputs over `atomic_target_vocab`, not over BPE subwords, so the task remains "predict the missing whole key". |
| Aggregator pooling step introduces a bug that's hard to debug | Medium | Tasks explicitly include a per-batch shape audit; pooling is a single isolated function with unit tests. |
| Subword expansion makes some long manifests untrainable at 768 | Low | 0.9% of corpus exceeds 768; for those, the truncation loses non-essential tail content (long descriptions / ConfigMap data). Same truncation policy as v8. |
| Whole-word masking sometimes masks a key whose atomic form isn't in `atomic_target_vocab` | Medium (already a v8 issue) | Same as v8: no loss contribution from that position. Logged but not fatal. |
| Tokenizer doesn't behave well at training time vs validation (e.g., trains on a different doc subset than encodes) | Low | Same corpus, same cache. Tokenizer is deterministic. |

## Open Questions Explicitly NOT Resolved

These are real questions but answered with "later":

- **Should values contribute to `doc_vec`?** Discussed at length.
  Deferred. Tracked in [project memory: tokenization → attention article].
- **Should we predict value subwords in MLM?** Discussed. Deferred.
- **Should the aggregator absorb child VALUE subtree vectors into a KEY's
  subtree vec?** Discussed. Deferred. Possibly a clean v10 change.
- **Should we restrict cross-position attention from VALUEs?** A radical
  idea worth exploring later.
- **Cleanup of `atomic_target_vocab` redundancy.** Good idea, not blocking.
- **Head+tail truncation instead of `[LONG_VALUE]`** for ≥256-char values.
  For certificates, shell commands, and DNS-style annotations the tail
  carries real signal (`-----END CERTIFICATE-----`, output redirection,
  `_URL`/`_PASSWORD` suffixes). Worth revisiting in v10 if/when values
  contribute to `doc_vec` — at that point the content signal matters.

The pattern: **v9 is a measurement, not a redesign**. Sub-tokenization is
big enough on its own. We ship it, measure what changes, then make
informed decisions about the next architectural increment.

## References

- `docs/superpowers/specs/2026-05-25-v8-design.md` — v8 spec this builds on
- `docs/key-value-design-rationale.md` — the KEY/VALUE asymmetry doc
  (will need updating in v10 if/when we move values into `doc_vec`)
- `scripts/train_unified_tokenizer.py` — tokenizer training script
- `scripts/analyze_value_lengths.py` — corpus length analysis
- `scripts/validate_subword_pipeline.py` — pre-flight validation checks
- `output_v8_276K_recon_seed42/unified_bpe_8k.json` — trained tokenizer
