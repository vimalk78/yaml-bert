# v8 Phase 1 — Problem Statement for Brainstorming

## Where we are

Phase 0 ran and produced a clean go/no-go decision: **GO for Phase 1, with conditions**. Full results in `docs/v8-phase0-results.md`. The architecture (encoder + bottom-up tree aggregator + atomic Token Head conditioned on `[h_i ; doc_vec ; s_parent]`) was validated — a linear probe on the resulting doc_vec achieves 99.87% accuracy distinguishing the top 10 K8s kinds, decisively fixing the v4-era atomic-only failure mode (which had 0.84-0.92 cross-kind similarity).

Phase 0's failure mode was speed: 3.6 it/s vs v7's ~9 it/s (2.5× slower), caused by per-doc Python loops in the aggregator and in `s_parent` construction. This was a known risk we explicitly accepted for Phase 0 simplicity.

## What's confirmed by Phase 0

- The architecture (encoder → aggregator → atomic conditioned head) **trains stably and learns kind-discriminative representations**.
- Mean combine for the aggregator is sufficient for kind-level discrimination — no immediate need for attention combine on this front.
- Atomic prediction (vocab ~427) is a viable alternative to compound targets (vocab ~28K) when conditioned on doc_vec.
- The 5K-doc / 10-epoch quick-mode benchmark is workable for iteration.

## What's open for Phase 1

The v8 design spec (`docs/superpowers/specs/2026-05-25-v8-design.md`) explicitly deferred these decisions:

1. **Aggregator vectorization** — Phase 1's gating prerequisite. Current per-doc Python loop must be replaced with batched scatter ops (or PyG/DGL equivalent) before any other Phase 1 work. Target: ≥ 7 it/s.
2. **Combine function** — mean was used in Phase 0 and worked for kind discrimination. Open: do we need attention combine or Tree-LSTM for finer-grained tasks (e.g., within-kind retrieval)?
3. **Reconstruction objective** — design and implement subtree-masking reconstruction. Doc_vec + a small decoder predicts the bag of keys present in masked subtree.
4. **Joint loss weighting** — `α·L_mlm + β·L_reconstruction`. Initial weights, tuning strategy, warmup schedule.
5. **Evaluation framework** — multi-task benchmark suite (YAML-MTEB analog). What tasks, how to construct labels, what's the headline metric.
6. **Scale-up** — Phase 0 was 5K docs / 10 epochs. Phase 1 needs to validate at full 276K corpus / 30 epochs (assuming vectorized aggregator).
7. **Optional: refined token-prediction conditioning.** Currently uses `[h_i ; doc_vec ; s_parent]`. Open: is `s_parent` (immediate parent subtree) the right level of locality, or should we condition on multiple subtree levels?

## Architecture commitments (fixed, do not revisit)

Inherited from the v8 spec:

- Encoder-only transformer (BERT-style)
- MLM-style self-supervised pretraining at token level
- Tree positional encoding retained
- Atomic prediction (vocab ~1K) at Token Head, NOT compound (vocab ~28K)
- Bottom-up tree aggregator producing per-key subtree vectors + doc_vec
- Token Head input: concatenation of per-token hidden state + doc_vec + parent subtree vector

These are settled. Phase 1 brainstorm should NOT revisit "should we use CLS instead of aggregation" or "should we keep compound targets."

## What's been ruled out (with Phase 0 evidence)

- **Per-doc Python loops at training scale.** Phase 0 measured 2.5× slowdown vs v7. Must vectorize.
- **Tree-bias attention with current PyTorch nn.TransformerEncoder** (carried over from v7 work — disabled because non-None attn_mask forces slow path).

## Open design questions

1. **What's the right vectorization for the aggregator?** PyG / DGL primitives vs custom scatter ops vs a fully fused CUDA kernel. Trade-off: external dependency vs implementation complexity vs perf.
2. **What's the right reconstruction objective?** Bag-of-keys is simplest and was sketched in the spec. Alternatives: predict tree shape (more structured), predict token sequences (more elaborate). Trade-off: signal richness vs implementation complexity.
3. **How to weight the losses?** Naive sum, learned weights, gradient balancing (e.g., GradNorm). For Phase 1, can probably start naive and iterate.
4. **What tasks for the eval suite?** Inherited candidates from the spec: kind classification probe, has-feature multi-probes, within-kind retrieval, drift sensitivity, reconstruction quality. Need to pick the minimum viable set.
5. **Schedule:** train all three losses (MLM + reconstruction + ...) jointly from epoch 1? Or warm up with MLM only, then add reconstruction? Or stages?
6. **When to scale to full corpus?** Run all Phase 1 design iterations on 5K-doc quick mode, then scale up once for a "final" v8 checkpoint? Or scale up earlier and iterate at full scale?

## Non-goals (explicit)

- Multi-document modeling. Deferred to a later version.
- OpenShift-specific training data. v8 trains on the same general K8s corpus; OpenShift specialization is a separate downstream concern.
- Beating frontier LLMs.
- Replacing v7 (it remains the deployed model on HF Space until v8 demonstrably outperforms it).

## What we'd like out of this brainstorm

For each open design question above, surface design alternatives, surface implementation gotchas, and pick a direction. Output is a Phase 1 spec that produces v8 at full corpus with a complete evaluation comparing against `mean-pool(v7)` baseline.

The Phase 1 spec should support a similar two-stage execution: minimum-viable implementation first, validate at quick-mode, then scale to full corpus.
