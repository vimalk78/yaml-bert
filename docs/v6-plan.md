# YAML-BERT v6 — Phase 1 Plan

> **Status update (2026-05-23):** Lever 1 shipped as v6.1 — full corpus,
> 30 epochs, evaluated. Closes the structural-test status failures
> (6/9 → 9/9) and eliminates 99% [UNK] over-confidence. Capability
> tests within noise of v5. Bigger-boat vocab_gap still 0/4 because
> status targets are absent from the vocab — Lever 1 alone can't add
> them. See [`evaluation-results.md` Section 7](./evaluation-results.md)
> for full numbers. Levers 5 and 6 below remain unimplemented.

## Context

v5 passes 93/93 pretrain capability tests, but probing surfaced three
failure modes the saturated capability suite cannot catch:

1. **Status-side vocab gap.** Model predicts `[UNK]` at 99% confidence
   for status fields like `Deployment.status.replicas` or
   `Pod.status.conditions`. Root cause: those (parent, child) bigrams
   were below `min_freq=100` and got dropped from the target vocab. We
   measured 5.5% of all training key positions (718,725 of 13.1M) have
   `[UNK]` targets and currently train the model to predict `[UNK]`
   confidently.

2. **CRD content dominates training tokens.** CRDs are 3.0% of documents
   but 46.0% of training tokens. The average CRD is 1,219 tokens vs an
   average Deployment's 80. The model has been trained roughly half on
   schema-definition content rather than actual K8s state. Critically,
   the corpus depth distribution shows that **depth 11+ is exclusively
   CRD content** (≥99.997%), and even depth 8–10 is dominated by CRDs.

3. **No API-version awareness.** The kind_head conditions on `kind`
   alone, not on `apiVersion`. Real corpora include deprecated API
   versions (`apps/v1beta1`, `extensions/v1beta1`, `autoscaling/v1` vs
   `autoscaling/v2`) with materially different schemas.

This phase targets all three with minimal, surgical interventions. No
model architecture changes — only data and target-vocab adjustments.

## Phase 1 levers

### Lever 1 — Selective masking (skip [UNK] targets)

**What.** In `yaml_bert/dataset.py::_getitem_v4`, before masking a
position, compute the target. If the encoded target id is `[UNK]`, skip
masking that position. The token stays as input context but is not used
as a label.

**Why.** Currently 5.5% of training key positions train the model to
predict `[UNK]` confidently. This is the proximate cause of the 99%
`[UNK]` predictions in the status blind spot.

**Cost.** ~5 lines of code. No retraining infrastructure changes. Same
data, same model architecture.

**Expected gain.** Status-position predictions become "wrong but not
[UNK]" — model picks the best of the valid alternatives instead of
confidently emitting [UNK]. Per-token loss decreases ~5% (fewer noisy
positions).

**Caveats.**
- Rare keys never appear as targets. Their *input* embeddings still get
  trained, but never via masked-prediction at their own position.
- Effective masking rate drops from 15% to ~14.2% (5.5% of positions are
  now skipped). One extra epoch can compensate if convergence becomes a
  concern.

### Lever 5 — Universal depth cap at linearization

**What.** In `yaml_bert/linearizer.py`, add a `max_depth_cap` parameter
(default 9) to `_walk`. When `depth > max_depth_cap`, stop recursing —
no nodes are emitted below that depth. Wire it into `YamlBertConfig`.

**Why.** The depth distribution shows:

| Depth band | Non-CRD content | CRD content |
|---|---|---|
| 0–7 | 99.0% of non-CRD tokens | 8.5% of CRD tokens |
| 8–10 | 1.0% of non-CRD tokens | 12.4% of CRD tokens |
| 11+ | **0.003%** of non-CRD | **79.1%** of CRD |

Below depth 10, content is essentially exclusively CRD schema
definition. A universal cap at `max_depth_cap=9` drops 85% of CRD
content while preserving 99.98% of non-CRD content (loses only ~2,700
tokens out of 11.8M — tail of complex StatefulSets/CronJobs).

**Cost.** ~5 lines in `_walk`: early-return when depth exceeds the cap.

**Expected gain.** CRD share of training tokens drops from 46% to ~13%
— close to their doc-count share of 3%. Token embeddings for `name`,
`properties`, `items`, etc. should re-balance toward their
manifest-context meanings rather than schema-definition meanings.

**Caveats.**
- Truncates 0.02% of non-CRD content (the tail of complex
  StatefulSets/CronJobs/Jobs). Negligible.
- Reduces total training tokens by ~40% (since CRDs are 46% of tokens
  and we cut 85% of CRDs). Effective dataset is smaller; may need to
  monitor convergence.
- Choice of D=9 is based on a single distributional snapshot. If we
  ever change the training corpus, D should be re-derived.

**Alternative considered and rejected:** loss reweighting by
`1/sqrt(doc_size)`. Rejected because it penalizes legitimately larger
documents (DaemonSet at 129 tokens vs ConfigMap at 18) — bigger docs
have more legitimate structural signal; making them "pay a price" is
the wrong framing. Depth cap is surgical: it removes only the CRD-deep
schema content, leaves everything else untouched.

### Lever 6 — apiVersion-aware kind head

**What.** Currently `kind_head` predicts compound targets like
`Deployment::spec::replicas`. Extend the trigram encoding to include
`apiVersion`, producing targets like
`apps/v1::Deployment::spec::replicas`. The dataset constructs the
trigram from `(apiVersion, kind, parent, child)` at vocab-build time;
the kind_head's output vocab grows to enumerate the versioned trigrams.

**Why.** `next-training-improvements.md` mentioned deepening trigrams;
the bigger-boat test design (Gap 2) exposes a separate but related gap:
no test coverage for distinguishing equivalent kinds across API
versions. Real corpora include all of:
- `apps/v1` Deployment vs deprecated `apps/v1beta1`
- `networking.k8s.io/v1` Ingress (requires `pathType`) vs v1beta1 (no
  `pathType`)
- `autoscaling/v2` HPA (`metrics[]`) vs `autoscaling/v1`
  (`targetCPUUtilizationPercentage`)

v5 likely mixes these.

**Cost.** ~20 lines: extend `Vocabulary.encode_kind_target` to include
apiVersion as a prefix; widen `kind_target_vocab` accordingly. No model
architecture changes.

**Expected gain.** Closes Gap 2 in `test-design-bigger-boat.md`. The
kind_head's output vocab roughly doubles (versioned trigrams) —
manageable.

**Caveats.**
- Larger `kind_target_vocab`. Some version-kind pairs will be sparse if
  a version is rare in training.
- Lever 1 (selective masking) will skip the targets for rare versions,
  preventing them from training the head to predict [UNK].

## Goals

- Close the status blind spot (Lever 1 directly targets this).
- Rebalance training-token pressure so real-manifest patterns dominate
  over CRD schema content (Lever 5).
- Add API-version awareness so version-specific schemas are
  distinguishable (Lever 6).

## Non-goals (defer)

- **Loss reweighting** — too blunt; penalizes legitimately bigger docs.
  Lever 5 (depth cap) achieves the rebalancing surgically instead.
- **Annotation-specific handling** — existing v5 bigger-boat tests
  on annotation keys (`app.kubernetes.io/*`, `prometheus.io/*`) pass.
  Long-tail novel annotations need sub-tokenization (v7).
- **Sub-tokenization for OOV resilience** — v7.
- **Attention-mechanism architecture changes** (tree-bias, RoPE) — v7.
- **Multi-document handling** — one YAML per file assumed.

## Implementation checklist

1. **Lever 5 (depth cap)** — easiest, do first. Add `max_depth_cap`
   parameter to `_walk`. Default 9. Add to `YamlBertConfig`. Add unit
   test that verifies depth-> N nodes are dropped.
2. **Lever 1 (selective masking)** — in `_getitem_v4`, compute target
   before deciding to mask; skip if target is `[UNK]`. Add unit test.
3. **Lever 6 (apiVersion-aware kind head)** — in `Vocabulary`, add
   `apiVersion` to the kind trigram encoding. Rebuild vocab. Add unit
   test that verifies trigram lookups distinguish versions.
4. **Cache rebuild** — re-linearize the training corpus with
   `max_depth_cap=9`. Expect ~40% smaller cache.
5. **Smoke run** — 5K quick mode, one seed. Confirms convergence
   behavior and that no NaNs appear from the new vocab.
6. **Full run** — 276K full corpus, 30 epochs on L4 (or scale-down
   match the v5 hyperparameters).

## Evaluation rubric

See [`test-design-bigger-boat.md`](./test-design-bigger-boat.md) for
the 6 gaps and per-gap pass thresholds. Briefly:

1. **Existing capability tests** (`test_capabilities.py`) — must stay
   at 93/93. Any regression below 90/93 is a hard fail.
2. **Existing structural tests** (`test_structural.py`) — current v5 is
   6/9. Target: 8+/9 (status failures closed).
3. **Expanded bigger boat** (~25 tests across 6 gaps). Per-gap targets:
   - Gap 1 (status): 0/8 → **≥ 6/8**
   - Gap 2 (API version): unmeasured → **≥ 50%**
   - Gap 4 (CRD-instance): unmeasured → **≥ 50%** (OOV; lower bar)
   - Gap 5 (OOD calibration): partial → **≥ 75%**
   - crd_pollution: **stay at 4/4** (no regression)
4. **suggest_fields wrong-level rate** — track as a diagnostic; not
   expected to improve from these levers (motivates v7 tree-bias work).

## Open questions — resolved

1. **Lever 1 + Lever 5 composition.** Independent: Lever 5 affects what
   nodes exist; Lever 1 affects which existing nodes get masked. Apply
   Lever 5 at cache-build time, Lever 1 at training time.

2. **Why D=9 and not 8 or 10?** Based on the depth distribution:
   - D=8: drops 87% of CRD content but also 0.5% of non-CRD tokens
   - D=9: drops 85% of CRD content, 0.02% of non-CRD
   - D=10: drops 79% of CRD content, 0.01% of non-CRD
   
   D=9 is the sweet spot where the marginal CRD reduction stops
   exceeding the marginal non-CRD loss.

3. **Universal vs CRD-specific cap?** Universal. Saves ~5 lines of code
   (no kind detection in walker), costs 2,700 non-CRD tokens (0.02%).
   Philosophically cleaner — no document is treated as a special case.

4. **CRD-instance test corpus.** Hand-write 4–6 manifests based on
   common Operator patterns: `Prometheus`, `ServiceMonitor`,
   `Certificate`/`Issuer`, `Application`, `Pipeline`. Cost: ~half day.

## Remaining unknowns (to discover during v6.1 evaluation)

- Whether D=9 is right empirically or whether D=10 (slightly more CRD
  content kept) is better for downstream CRD-instance generalization.
- Whether the reduced effective masking from Lever 1 requires bumping
  epochs from 30 to e.g. 32 to match v5's convergence point.
- Whether the wider `kind_target_vocab` (from Lever 6) adds noise that
  hurts other categories.

## Followups (TODO, not blocking v6.1)

- **HF Model repo.** v6.1 lives only in the Space (`vimalk78/yaml-bert`,
  the Gradio demo). For discoverability + a citable model page, create a
  separate Model repo (e.g. `vimalk78/yaml-bert-v6.2` once we have a
  better checkpoint to publish), with:
    - Model card README (frontmatter: tags, license, datasets, library)
    - Loading snippet (the model is custom-arch, not `from_pretrained`-compatible)
    - Architecture summary + evaluation table + citation
  Defer until we have a v6.2+ checkpoint that's worth publishing as the
  canonical version. The current v6.1 is a partial fix; publishing it
  feels premature.
- **Refactor Space app to download model from Model repo** instead of
  bundling the .pt in the Space LFS. Cleaner separation, smaller Space
  footprint. Depends on the Model repo above.

## Decision — committed scope

**Build Phase 1 (Levers 1, 5, 6) together as v6.1.** Implement, train
at 5K-quick-mode first for a sanity pass, then full 276K × 30 epochs.

Evaluate against the rubric above. If Phase 1 succeeds, v6.1 ships and
we plan v7. If specific gaps remain, revisit selectively.
