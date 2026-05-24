# v8 Phase 1 — Aggregator Vectorization Results

## Setup

- **Date:** 2026-05-25
- **Hardware:** JarvisLabs L4 GPU (instance `415256`, fresh — v7 full-corpus training continues on separate instance `415123`)
- **Training subset:** 5,000 docs from `substratusai/the-stack-yaml-k8s` (same as Phase 0)
- **Epochs:** 10
- **Batch size:** 32
- **Model params:** 5,449,387 (unchanged from Phase 0)
- **Changes since Phase 0:** `TreeAggregator` + `V8Model.s_parent` vectorized (batched PyTorch scatter ops; `v8_collate_fn` now precomputes `parent_of_tensor`, `top_level_key_mask`, `edges_by_depth`, `parents_by_depth`). No behavior change — numerical equivalence locked by `tests/test_aggregator_vectorized.py` and `tests/test_v8_model_e2e.py::test_v8_smoke_e2e_vectorized_path`.

## Measured Metrics

| Metric | Phase 0 | Phase 1 (vectorized) | Target | Verdict |
|---|---|---|---|---|
| Training step time (it/s) | 3.6 | **~14** (13.14–14.66, median ~13.9) | ≥ 7 | ✅✅ PASS (2× target, exceeds v7's ~9 it/s) |
| Total training time (10 epochs) | 7.3 min | **1.87 min** (111.9s) | ≤ 4 min | ✅✅ PASS (2× headroom) |
| Kind probe accuracy (top 10 kinds) | 99.87% | **99.87%** | ≥ 95% | ✅ PASS (identical to Phase 0) |
| Loss at epoch 10 | 0.8182 | **0.8181** | trending down | ✅ PASS (matches Phase 0 to 4 decimals) |
| Training stability | clean | **clean** | NaN-free | ✅ PASS |
| Equivalence tests | n/a | **14/14 pass** | all pass | ✅ PASS |

## Loss trajectory (per epoch)

```
Epoch  1: 3.2985  (157 batches, 12.0s, 13.14 it/s)
Epoch  2: 2.0835  (157 batches, 11.4s, 13.73 it/s)
Epoch  3: 1.6582  (157 batches, 10.8s, 14.54 it/s)
Epoch  4: 1.4312  (157 batches, 11.0s, 14.32 it/s)
Epoch  5: 1.2683  (157 batches, 11.2s, 14.08 it/s)
Epoch  6: 1.1304  (157 batches, 10.9s, 14.35 it/s)
Epoch  7: 1.0280  (157 batches, 11.3s, 13.94 it/s)
Epoch  8: 0.9522  (157 batches, 10.7s, 14.66 it/s)
Epoch  9: 0.8846  (157 batches, 11.4s, 13.82 it/s)
Epoch 10: 0.8181  (157 batches, 11.3s, 13.88 it/s)
```

Total: 111.9s (vs Phase 0's 436.4s — a 3.9× wall-clock speedup).

Side-by-side per-epoch loss vs Phase 0:

```
Epoch  Phase 0   Phase 1   Δ
   1   3.2985    3.2985    0.0000
   2   2.0835    2.0835    0.0000
   3   1.6581    1.6582   +0.0001
   4   1.4313    1.4312   -0.0001
   5   1.2682    1.2683   +0.0001
   6   1.1303    1.1304   +0.0001
   7   1.0280    1.0280    0.0000
   8   0.9523    0.9522   -0.0001
   9   0.8847    0.8846   -0.0001
  10   0.8182    0.8181   -0.0001
```

Max per-epoch delta: 0.0001. The drift is at fp32 round-off scale, consistent with the equivalence test's `atol=1e-6` guarantee compounding across 1,570 optimizer steps.

## Observations

- **Speed beat the v7 baseline outright.** Target was within 25% of v7's ~9 it/s; achieved ~14 it/s — roughly 55% faster than v7. The v8 model is half v7's parameter count (5.4M vs 13.4M) and the vectorized aggregator removes the CPU/GPU sync per document; the combined effect overshot the goal.
- **Behavior identical to Phase 0 down to 0.0001 per epoch.** Side-by-side loss trace matches to 4 decimal places. Same kind probe accuracy (99.87%). Optimizer trajectory is — within float round-off — the same trajectory. The numerical-equivalence tests held end-to-end.
- **All 14 local tests stayed green throughout the implementation** (3 aggregator + 2 aggregator_vectorized + 1 aggregator_perf_smoke + 9 v8_dataset + 4 v8_model_e2e). The backward-compat fallback paths in `TreeAggregator` and `V8Model` mean older test fixtures continue to exercise the reference path while production code takes the vectorized path.
- **GPU utilization is now compute-bound, not Python-bound.** Phase 0's bottleneck was the per-doc loop blocking the GPU; with that gone, ~14 it/s on an L4 for a 5.4M-param model is in the right ballpark for a transformer of this size and batch.

## Decision

**GO for the next Phase 1 mini-cycle: reconstruction objective design.**

### Why GO

Every acceptance gate passed by a wide margin. Speed exceeded v7's baseline (not just within 25% of it). Kind probe is unchanged at 99.87%. Loss trajectory matches Phase 0 to 4 decimals. There is no signal that the vectorization sacrificed anything — it strictly removed CPU-side overhead.

The architectural validation from Phase 0 transfers cleanly. We now have a fast foundation to layer the reconstruction objective on. The next mini-cycle can focus on reconstruction design without paying for slow training during iteration.

### Cost

- JarvisLabs L4 (instance `415256`): ~7 minutes wall time including setup → ~$0.10 (well under the $0.55 budget).
- Instance destroyed after benchmark.

### Files

- Checkpoint: `output_v8_phase1_vec_seed42/v8_phase0.pt` (5.4M params)
- Document vectors: `output_v8_phase1_vec_seed42/doc_vecs.pt` (5000 × 256)
- Vocab: `output_v8_phase1_vec_seed42/vocab.json` (key=427, atomic=427)
- Doc cache: `output_v8_phase1_vec_seed42/doc_cache.pkl` (18.4 MB)
- All downloaded locally; instance destroyed.

### What to take into the next mini-cycle

- The vectorized aggregator + s_parent are the new baseline. Reconstruction work plugs on top of them.
- The numerical-equivalence test pattern (reference path vs vectorized path with `torch.allclose`) is a reusable template for future optimizations — keep it in mind when adding the reconstruction head.
- v7's training (instance `415123`) is still ongoing in parallel; v7 remains the deployed model until v8 demonstrably outperforms it across the still-to-be-built evaluation suite.
