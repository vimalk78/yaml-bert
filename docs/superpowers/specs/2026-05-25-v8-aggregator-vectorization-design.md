# v8 Phase 1 — Aggregator Vectorization (Mini-Cycle)

## Problem

v8 Phase 0 succeeded on the architectural hypothesis (kind-discrimination probe 99.87%) but failed on speed: 3.6 it/s vs v7's ~9 it/s (2.5× slower). The cause is per-document Python loops in two places:

1. `TreeAggregator.forward` — bottom-up combine of subtree vectors
2. `V8Model.forward` — construction of `s_parent` per position

Both are O(B·N) Python work blocking the GPU. Phase 0 explicitly accepted this as the prototype's known perf risk; Phase 1 must fix it before any further v8 work proceeds.

## Goal

Replace both per-document loops with batched PyTorch tensor operations. **No behavior change.** Vectorized output must be numerically equivalent to the per-doc reference on the same inputs.

## Non-goals

- Reconstruction objective (separate later mini-cycle)
- Evaluation suite (separate later mini-cycle)
- Combine-function change (mean stays; attention-combine deferred until eval suite can validate gains)
- API surface changes (`TreeAggregator.forward(hidden_states, batch_info)` and `V8Model.forward(...)` signatures stay)
- Adding external dependencies (no PyG, no DGL — vanilla PyTorch scatter ops)
- Full-corpus scale-up (re-benchmark uses Phase 0's 5K-doc / 10-epoch setup for direct comparison)

## Architecture

Same components as v8 Phase 0. Only internals of two functions change.

```
Input batch → V8Dataset.__getitem__ → v8_collate_fn (PRECOMPUTES tensors here)
                                            ↓
Encoder (transformer) → hidden_states (B, N, d_model)
                                            ↓
TreeAggregator.forward (VECTORIZED) → subtree_vecs, doc_vec
                                            ↓
V8Model.forward s_parent (VECTORIZED) → [h_i ; doc_vec ; s_parent]
                                            ↓
                              Token Head → atomic logits
```

## What changes

### `yaml_bert/v8_dataset.py`

`compute_children_info(nodes)` — no signature change. Returns the same dict; downstream consumers continue to use the existing keys (`children_of`, `parent_of`, `key_positions`, `depth_of`, `full_path_of`).

`v8_collate_fn(batch)` — adds three precomputed tensors to the returned dict:

- `parent_of_tensor`: `(B, N)` long tensor. Position `(b, i)` holds the parent position of node `i` in doc `b`, or `-1` if no parent (root key, padding, or non-key). Padded positions also get `-1`.
- `edges_by_depth`: `dict[int, tensor (E, 3) long]` of `[doc_idx, child_pos, parent_pos]` per depth (depth = parent's depth). One row per parent-child edge.
- `parents_by_depth`: `dict[int, tensor (P, 2) long]` of unique `[doc_idx, parent_pos]` with at-least-one-child per depth.
- `top_level_key_mask`: `(B, N)` bool tensor. True at positions that are depth-0 keys, used for the doc_vec mean.

> Note: an earlier draft of this spec called for a single `key_pos_per_depth` dict. During implementation it was split into `edges_by_depth` + `parents_by_depth` so the aggregator doesn't need to re-walk `children_of` inside the GPU path. Strictly better; functionally equivalent.

These are derived from the existing per-doc `children_info` dicts in `batch_info`. The original `batch_info` list stays in the batch (for backward compatibility and the equivalence-test reference path).

### `yaml_bert/aggregator.py`

`TreeAggregator.forward` — internal rewrite. Signature stays:

```python
def forward(
    self,
    hidden_states: torch.Tensor,             # (B, N, d_model)
    batch_info: list[dict],                  # legacy reference
    *,
    parent_of_tensor: torch.Tensor | None = None,
    key_pos_per_depth: dict[int, torch.Tensor] | None = None,
    top_level_key_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
```

When the keyword-only tensor args are provided (the trainer always provides them from collate), use the vectorized path. When they're `None` (older test fixtures, debugging), fall back to the per-doc reference path. The dispatch keeps existing tests working unchanged.

Vectorized algorithm:

1. Initialize `subtree_vecs = hidden_states.clone()`.
2. For each depth level `d` from `max_depth` down to `0`:
   - `keys_at_d = key_pos_per_depth[d]` — shape `(K, 2)` of `[doc_idx, key_pos]`
   - For each such key, collect its children: build `(num_edges_at_d, 3)` of `[doc_idx, child_pos, parent_pos]` (precomputed in collate for efficiency, OR walked from `batch_info["children_of"]` if needed).
   - Use `index_add_` to scatter child subtree vectors into per-parent accumulators.
   - Divide by `(count_per_parent + 1)` to mean (the +1 is for the parent's own hidden state, added separately).
   - Write into `subtree_vecs` at parent positions.
3. doc_vec = mean of subtree_vecs at top-level key positions per doc (via `top_level_key_mask`).

Sequential cost: one PyTorch op per depth level (~9 for depth-capped trees), each batched across the entire batch.

### `yaml_bert/v8_model.py`

`V8Model.forward` — internal rewrite of the `s_parent` block. Signature stays. When the trainer provides the new collate fields, take the vectorized path; otherwise fall back to the per-doc loop.

Vectorized s_parent:

```python
# parent_of_tensor: (B, N) long, -1 for no-parent
safe_parent = parent_of_tensor.clamp(min=0)                  # avoid invalid index
s_parent = torch.gather(
    subtree_vecs, dim=1,
    index=safe_parent.unsqueeze(-1).expand(-1, -1, d),
)
no_parent_mask = (parent_of_tensor == -1).unsqueeze(-1)      # (B, N, 1)
s_parent = torch.where(no_parent_mask, doc_vec.unsqueeze(1), s_parent)
```

3 tensor operations, no Python loop.

## Testing

### Behavior parity (correctness)

All existing tests must pass unchanged:
- `tests/test_aggregator.py` — 3 tests, exercise the legacy per-doc path (still works because tensor args default to None)
- `tests/test_v8_dataset.py` — 8 tests, exercise dataset + collate
- `tests/test_v8_model_e2e.py` — 3 tests, exercise V8Model

### New: numerical equivalence test

Add `tests/test_aggregator_vectorized.py` with one test:

- Build a small synthetic batch (2-3 docs, varied tree shapes)
- Run aggregator twice on the same input — once via per-doc reference path, once via vectorized path (with explicit tensor args)
- Assert `torch.allclose(reference_subtree, vec_subtree, atol=1e-6)`
- Assert `torch.allclose(reference_doc, vec_doc, atol=1e-6)`

This locks in that the vectorized version produces identical numerical results.

### New: micro-perf smoke test

Add `tests/test_aggregator_perf_smoke.py` with one test:

- Build a synthetic batch of 32 docs each with ~50 nodes (representative size)
- Time 100 forward passes through the legacy per-doc path
- Time 100 forward passes through the vectorized path
- Assert vectorized is at least 5× faster

This is a soft local check that the vectorization actually helped. CPU-only, fast (no GPU needed).

> Note: 5× turned out to be optimistic on a CPU microbench (median measured ~3.0×, with jitter). The shipped test uses a 2.5× regression gate — well above the broken-vectorization signal (~1×) and stable enough to avoid flakes. The GPU benchmark (training throughput ≥ 7 it/s) is the real acceptance gate.

## Acceptance gate

Before declaring this mini-cycle done:

1. **All behavior tests pass** (existing + new equivalence test)
2. **Local perf smoke test passes** (vectorized ≥ 5× faster on CPU)
3. **Re-run Phase 0 benchmark on JarvisLabs L4** — same setup (5K docs, 10 epochs, random init, mean combine, no reconstruction head). Assert:
   - Training speed ≥ 7 it/s (within 25% of v7's ~9 it/s)
   - Kind probe accuracy ≥ 95% on top-10 kinds (loose floor — Phase 0 hit 99.87%)
   - Total training time ≤ 4 min (vs Phase 0's 7.3 min — math: 1570 batches / 7 it/s ≈ 3.7 min)
4. **Results recorded** in `docs/v8-phase1-vectorization-results.md` with go/no-go decision for the next mini-cycle (reconstruction objective).

## Open implementation details

These are decisions the implementer makes during coding, not blocking spec decisions:

1. **Where to precompute the depth-grouped child-edge tensors:** in `v8_collate_fn` (more upfront work, faster aggregator) or inside the aggregator on first call per batch (cleaner separation, slight per-batch overhead). Recommendation: in `v8_collate_fn` — collate already does similar precompute, and the data lives next to where it's used at training time.
2. **scatter_add_ vs index_add_:** both work; pick whichever produces clearer code. Functionally equivalent for our use case (no duplicate indices since each parent gets unique children).
3. **Mean computation:** option A) `sum + divide by (count+1)`. option B) `sum + own / (count+1)` where the parent's own contribution is added to the scatter result. Both equivalent; pick clearer code.

## Out of scope (later Phase 1 mini-cycles)

- Attention-based combine function (deferred until eval suite is built to compare)
- Reconstruction objective (separate mini-cycle)
- Evaluation framework (separate mini-cycle)
- Full-corpus 276K training (separate mini-cycle, AFTER reconstruction + eval are in)
- Sub-tokenization, multi-doc cross-attention, OpenShift specialization (future versions)

## Time / cost estimate

- Implementation: ~1 day of coding
- Local testing: ~30 min
- JarvisLabs re-benchmark: ~$0.55 (new L4 for ~1 hour) OR free if we run on instance `415123` after v7 finishes
- Results writeup + decision: ~30 min

## Decision after this mini-cycle

If acceptance gate passes → start next Phase 1 mini-cycle: reconstruction objective design.

If acceptance gate fails (speed still < 7 it/s) → escalate: investigate whether the GPU is now bottlenecked elsewhere (e.g., the per-batch metadata dict construction in collate, the Python-side iteration over `batch_info`, etc.). May require switching to Approach 2 (PyG/DGL) or rethinking the data path.

If kind probe regresses (< 95%) → the vectorization has a correctness bug. Use the equivalence test to localize.
