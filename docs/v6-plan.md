# YAML-BERT v6 — Proposed Improvements

## Context

v5 passes 93/93 pretrain capability tests, but bigger-boat probing
(`model_tests/test_bigger_boat.py`) and training-corpus analysis surfaced
three distinct failure modes the saturated capability suite cannot catch:

1. **Status-side vocab gap** — model predicts `[UNK]` at 99% confidence for
   status fields like `Deployment.status.replicas` or
   `Pod.status.conditions`. Root cause: those (parent, child) bigrams were
   below `min_freq=100` and got dropped from the target vocab.
2. **CRD training-token dominance** — CRDs are **3.0% of documents** but
   **46.0% of training tokens** (the average CRD is 1,219 tokens vs an
   average Deployment's 80). The model has been trained roughly half on
   schema-definition content rather than actual K8s state.
3. **Long-tail annotation/label keys** — uniformly applying `min_freq=100`
   to keys under `metadata.labels` and `metadata.annotations` is the wrong
   policy. These slots are user-extensible namespaces; their semantics
   doesn't depend on global frequency.

We measured each: 5.5% of all training key positions (718,725 of 13.1M)
currently have `[UNK]` targets and produce spurious [UNK] supervision. Of
those, 88K are status-side positions inside CRDs.

This document catalogs proposed v6 interventions, with cost estimates and
known caveats. Each can be implemented and evaluated independently.

## Goals

- Eliminate the status blind spot without losing CRD-instance generalization
- Rebalance training-token pressure so real-manifest patterns aren't drowned
- Handle long-tail annotation/label keys more honestly

## Non-goals (defer to v7)

- Sub-tokenization for novel-key OOV resilience
- Multi-document handling (one YAML per file assumed)
- Attention-mechanism architecture changes (tree-bias, RoPE, etc.)

## Proposed levers

### Lever 1 — Selective masking (skip [UNK] targets)

**What.** In `yaml_bert/dataset.py::_getitem_v4`, before masking a position,
compute the target. If the encoded target id is `[UNK]`, skip masking that
position. The token stays as input context but is not used as a label.

**Why.** Currently 5.5% of training key positions train the model to
predict `[UNK]` confidently. This is the proximate cause of the 99%
`[UNK]` predictions in the status blind spot.

**Cost.** ~5 lines of code. No retraining infrastructure changes. Same
data, same model architecture.

**Expected gain.** Status-position predictions become "wrong but not [UNK]"
— model picks the best of the valid alternatives instead of confidently
emitting [UNK]. Per-token loss decreases ~5% (fewer noisy positions).

**Caveats.**
- Rare keys never appear as targets. Their *input* embeddings still get
  trained, but never via masked-prediction at their own position.
- The model loses any (small) signal it currently learns about when [UNK]
  is appropriate. In practice [UNK] should rarely be a meaningful answer.

**Rollout.** Pure dataset change. Re-train v6 from scratch (or do a 5K
quick-mode sanity check first against v5 to confirm convergence behavior).

### Lever 2 — Loss reweighting by document size

**What.** In `yaml_bert/trainer.py`, weight each token's loss by
`1 / doc_size` (or `1 / sqrt(doc_size)` for a softer rebalance). A
1,219-token CRD then contributes the same *total* loss as an 80-token
Deployment.

**Why.** CRDs are 3% of documents but get 46% of gradient pressure today.
Reweighting brings their effective training-pressure share down to their
document-count share (3%) — still represented in vocab and structure, just
not dominant.

**Cost.** ~15 lines (compute per-doc weight, pass through collate_fn, apply
in `model.compute_loss`). No data changes.

**Expected gain.** Real-manifest patterns (which today get drowned out by
CRD volume) get proportionally more training pressure. Token embeddings
for `name`, `properties`, `items`, etc. should re-balance toward their
manifest meanings.

**Caveats.**
- Softer rebalance (`1 / sqrt`) is probably better than hard `1 / size` —
  the latter may underweight CRDs to the point that the model forgets
  schema patterns.
- Need to verify the kind/parent-aware target heads still see enough
  examples per epoch.

### Lever 3 — Per-parent-path min_freq

**What.** During vocab construction in `yaml_bert/vocab.py`, apply a
context-aware threshold:
- Tokens under `metadata.labels.*` or `metadata.annotations.*` → `min_freq=5`
  (or lower)
- Tokens elsewhere → `min_freq=100` (default)

**Why.** Annotation/label keys are user-extensible by design. Filtering
them by global frequency conflates "rare in the corpus" with "not
meaningful" — they're not the same thing.

**Cost.** ~30 lines in `VocabBuilder.build` plus ability to track
parent-context during token counting.

**Expected gain.** Long-form annotation keys like `helm.sh/chart`,
`argocd.argoproj.io/sync-wave` get distinct embeddings instead of all
collapsing to [UNK]. Model can predict specific annotations at appropriate
positions.

**Caveats.**
- Vocab grows substantially — possibly 3–5× on the key side. Embedding
  parameters scale with vocab size.
- Most rare annotations have tiny counts; their learned embeddings will be
  noisy.
- Dilutes the simple_head's output distribution if simple_target_vocab
  also expands. Mitigated by Lever 4.

### Lever 4 — Separate `annotation_head`

**What.** Add a third prediction head specifically for positions whose
parent_path indicates `labels` or `annotations`. The dataset routes
positions to one of three heads based on parent context:

```
simple_head      → structural keys                  (existing)
kind_head        → kind-specific structural keys    (existing)
annotation_head  → label/annotation keys            (new)
```

Each head has its own target vocab sized appropriately for its slot.

**Why.** Annotations/labels are a distinct modality from structural keys.
Sharing the simple_head dilutes structural predictions (more competitors
in the softmax) and undertrains annotations (lost in the noise of common
structural keys).

**Cost.** ~100 lines: new head in `model.py`, new routing in
`dataset._getitem_v4`, new vocab in `VocabBuilder`. Architecture change.

**Expected gain.** Structural predictions stay sharp; annotation
predictions become first-class. Composes well with Lever 3 (annotation
head consumes the lenient-min_freq vocab).

**Caveats.**
- 3rd head adds ~1.3M params (≈16% bump on the 8M v5 model).
- Annotation positions are sparse — head may be undertrained without loss
  reweighting.
- More moving parts in training. Multi-task balance becomes a tuning
  surface.

### Lever 6 — apiVersion-aware kind head

**What.** Currently `kind_head` predicts compound targets like
`Deployment::spec::replicas` — conditioned on `kind` only. Extend the
trigram target encoding to include `apiVersion`, producing targets like
`apps/v1::Deployment::spec::replicas`. The dataset constructs the trigram
from `(apiVersion, kind, parent, child)` at vocab-build time; the
kind_head's output vocab grows to enumerate the versioned trigrams.

**Why.** `next-training-improvements.md` mentioned deepening trigrams.
The bigger-boat test design exposes a separate but related gap: the model
has no test coverage for distinguishing equivalent kinds across API
versions (`apps/v1` Deployment vs deprecated `apps/v1beta1`,
`networking.k8s.io/v1` Ingress with required `pathType` vs the v1beta1
variant without it, `autoscaling/v2` HPA with `metrics[]` vs
`autoscaling/v1` with `targetCPUUtilizationPercentage`). Real corpora
include all of these; v5 likely mixes them.

**Cost.** ~20 lines: extend `Vocabulary.encode_kind_target` to include
apiVersion as a prefix; widen `kind_target_vocab` accordingly; no model
architecture changes.

**Expected gain.** Closes Gap 2 in `test-design-bigger-boat.md`. The
kind_head's output vocab doubles or so (versioned trigrams) — manageable.

**Caveats.**
- Larger kind_target_vocab. Each version-kind pair gets its own targets;
  some will be sparse if a version is rare in training.
- Shared backbone may still confuse `apps/v1::Deployment` and
  `apps/v1beta1::Deployment` via similar tokens; the head decoding gets
  them right but internal representations may stay entangled.

### Lever 5 — CRD doc-size cap (optional, more aggressive)

**What.** During cache build, truncate each CRD document to the first N
tokens (e.g., N=200). This is *separate* from `max_seq_len=512` which
applies to all docs uniformly.

**Why.** Even after `max_seq_len` truncation, CRDs contribute ~4M training
tokens. A more aggressive CRD-specific cap drops that further.

**Cost.** ~10 lines in `yaml_bert/cache.py` build path.

**Expected gain.** Token-distribution rebalancing similar to Lever 2 but
via deletion rather than weighting.

**Caveats.**
- **Destructive.** Lose CRD structural patterns that might be useful.
- Lever 2 (loss reweighting) achieves a similar outcome without throwing
  away data. Prefer Lever 2 unless reweighting proves insufficient.

## Phase 1 — the active plan

The three levers below address two of the three failure modes identified
in Context. They are independent and compose cleanly (see Open Questions
for why). All three are dataset/vocab changes — no model architecture
changes — so they can be implemented and trained as one v6.1 sweep.

| Lever | Targeted gap (per [`test-design-bigger-boat.md`](./test-design-bigger-boat.md)) | Cost |
|---|---|---|
| **1. Selective masking** | Gap 1 (status completion) | ~5 lines in dataset.py |
| **2. Loss reweighting** (`1/sqrt(doc_size)`) | CRD token dominance (Context #2) | ~15 lines across trainer.py + collate_fn |
| **6. apiVersion-aware kind head** | Gap 2 (API version awareness) | ~20 lines in vocab.py + dataset.py |

### v6.1 implementation checklist

1. **Implement Lever 1** — `dataset._getitem_v4`: encode target before
   masking; skip position if target id is `[UNK]`. Add unit test.
2. **Implement Lever 6** — `Vocabulary.encode_kind_target`: prepend
   apiVersion to the trigram; widen `kind_target_vocab`. Rebuild vocab
   from the existing cache. Add unit test.
3. **Implement Lever 2** — `trainer.YamlBertTrainer.train`: compute
   per-doc weight `1/sqrt(len(doc))` in `collate_fn`, multiply into
   `loss_fn` per-position before reduction. Add unit test.
4. **Smoke run** — 5K quick mode, one seed. Confirms convergence
   behavior and that no NaNs appear from the new weighting.
5. **Full run** — 276K full corpus, 30 epochs on L4 (or scale-down
   match the v5 hyperparameters).

### v6.1 evaluation rubric

Compare on:
- **Existing capability tests** (`test_capabilities.py`) — must stay at
  93/93. Any regression below 90/93 is a hard fail.
- **Existing structural tests** (`test_structural.py`) — v5 was 6/9.
  Target: 8+/9 (status failures closed).
- **Expanded bigger boat** (per test-design doc, ~25 tests across 6
  gaps). Per-gap thresholds:
  - Gap 1 (status): 0/8 → **≥ 6/8**
  - Gap 2 (API version): unmeasured → **≥ 50%**
  - Gap 4 (CRD-instance): unmeasured → **≥ 50%** (OOV; lower bar)
  - Gap 5 (OOD calibration): partial → **≥ 75%**
  - crd_pollution: **stay at 4/4** (no regression)
- **suggest_fields wrong-level rate** — track as a diagnostic; not
  expected to improve from these levers (motivates v7 tree-bias work).

## Deferred — revisit only if Phase 1 leaves gaps

These were earlier proposals that we've chosen not to pursue actively in
Phase 1.

### Lever 3 — Per-parent-path min_freq
### Lever 4 — Separate `annotation_head`

Both target annotation/label key handling. Deferred because:
- Existing bigger-boat tests on annotation keys (`app.kubernetes.io/*`,
  `prometheus.io/*`) pass on v5. The model already handles common
  annotations reasonably.
- The long-tail of truly novel annotations needs sub-tokenization to
  fix properly — a v7 concern.
- Adding a third head increases architectural complexity for a failure
  mode that didn't make the top-6 gap list.

If v6.1 evaluation shows annotation handling has become a top gap, we
revisit. Otherwise these levers remain documented but unimplemented.

### Lever 5 — CRD doc-size cap

Hard truncation of CRDs to N tokens at cache build. Deferred as
**fallback only**: Lever 2 (loss reweighting) achieves similar
rebalancing without destroying CRD structural content. Use this only if
Lever 2 proves insufficient on the CRD-instance test corpus.

## Open questions — resolved

1. **Lever 1 + Lever 2 composition.** *Resolved: clean compose.* They
   operate at different stages — Lever 1 selects masked positions, Lever
   2 weights resulting per-token loss. Side effect: effective masking
   rate drops from 15% to ~14.2% (5.5% of positions skipped). One extra
   epoch can compensate if convergence becomes a concern.
2. **`1/size` vs `1/sqrt(size)`.** *Resolved: start with `1/sqrt(size)`.*
   Rebalances CRD:Deployment ratio from 15:1 to ~4:1 — substantial
   without going all the way to 1:1 (which might be too aggressive).
   Flip to `1/size` in a follow-up only if `1/sqrt` is insufficient.
3. **Annotation handling specifics.** *Moot.* Levers 3 and 4 are
   deferred from active Phase 1.
4. **CRD-instance test corpus.** *Resolved: hand-write 4–6 manifests.*
   No need to scrape. Common Operator patterns are well-documented:
   `Prometheus`, `ServiceMonitor`, `Certificate`/`Issuer`,
   `Application`, `Pipeline`. Cost: ~half day to draft and verify they
   linearize cleanly.

## Remaining unknowns (to discover during v6.1 evaluation)

- Whether `1/sqrt(doc_size)` actually rebalances enough on the CRD-
  instance test set, or whether `1/size` (or Lever 5 fallback) is needed.
- Whether the smaller per-epoch supervision from Lever 1 requires
  bumping epochs from 30 to e.g. 32 to match v5's convergence point.
- Whether the wider `kind_target_vocab` (from Lever 6) adds noise that
  hurts other categories — most likely no, but worth confirming.

## How we'll know if v6 worked

See [`test-design-bigger-boat.md`](./test-design-bigger-boat.md) for the
6 gaps and per-gap pass thresholds. Briefly:

1. **Existing capability tests** should remain at 93/93 — v6 should not
   regress what v5 already does well.
2. **Structural tests** — current v5 is 6/9. Expect v6 to close the
   status failures: 8+/9.
3. **Bigger boat (expanded to ~25 tests)** — per-gap targets:
   - Gap 1 (status): 0/8 → ≥ 6/8
   - Gap 2 (API version): unmeasured → ≥ 50%
   - Gap 4 (CRD-instance): unmeasured → ≥ 50%
   - Gap 5 (OOD calibration): partial → ≥ 75%
   - crd_pollution: unchanged at 4/4
4. **Suggest_fields wrong-level prediction rate** — track as a metric
   even though it's not directly improved by these levers (it motivates
   v7 tree-attention work).

## Decision — committed scope

**Build Phase 1 (Levers 1, 2, 6) together as v6.1.** Implement, train at
5K-quick-mode first for a sanity pass, then full 276K × 30 epochs.

Evaluate against the rubric above. If Phase 1 succeeds (most gaps hit
their thresholds), v6.1 ships and we plan v7. If specific gaps remain
(e.g., annotation handling becomes the dominant remaining weakness),
revisit Levers 3 and/or 4.
