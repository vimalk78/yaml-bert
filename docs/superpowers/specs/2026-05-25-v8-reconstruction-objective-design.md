# v8 Phase 1 — Reconstruction Objective (Mini-Cycle)

## Problem

Phase 0 + aggregator-vectorization confirmed: `doc_vec` from bottom-up tree aggregation carries kind information (99.87% probe accuracy on 5K-doc subset). But that signal is shallow — every doc has "kind: X" tokens near the top; any model can attend to them. Whether `doc_vec` captures finer-grained *structural* information about a document's body content is untested.

The MLM-only objective trains local-context prediction (predict the masked token given surrounding tokens). It doesn't directly pressure `doc_vec` to encode whole-document structure. To do that, we need an objective whose loss flows back through `doc_vec` and forces it to carry information beyond the encoder's local context.

The v8 design spec sketched this as **subtree reconstruction**: mask a tree subtree; predict the bag of keys present in that subtree from `doc_vec`. This mini-cycle builds and validates that objective.

One refinement over the v8 spec's framing: a single doc can have 1-3 masked subtrees, all sharing the same `doc_vec`. To let the head know *which* subtree it's predicting, the head also reads the position embedding of the masked subtree's root. Concretely the input is `[doc_vec ; pos_emb(masked_root)]` (see Architecture below). `doc_vec` carries the document-level information; the root position embedding disambiguates which subtree the head should predict.

## Goal

Implement the reconstruction objective + 4 cheap smoke-test probes. Run a controlled comparison: MLM-only baseline vs MLM+reconstruction. Decide go/no-go for the objective based on probe deltas + reconstruction loss trajectory.

## Non-goals (later mini-cycles)

- A proper evaluation framework (this cycle uses *smoke-test* probes only — yes/no signal)
- Attention-based aggregator combine function
- Full-corpus 276K training
- Hyperparameter tuning beyond the documented defaults
- OpenShift-specific specialization

## Architecture

Inherits v8 Phase 1 (vectorized aggregator + V8Model). Two additions:

```
hidden_states (B, N, d)
  ├→ atomic Token Head (unchanged) → L_mlm
  └→ Aggregator (leak-aware: excludes positions inside any masked subtree
                 from sums into doc_vec and ancestor subtree_vecs)
       → doc_vec, subtree_vecs
            └→ Reconstruction Head [doc_vec ; pos_emb(masked_root)]
                 → small MLP → BCE logits over atomic vocab → L_recon

L_total = 1.0·L_mlm + 0.5·L_recon
```

### What's new

1. **Reconstruction Head** (new module, ~205K params): small MLP reading `[doc_vec ; pos_emb(masked_subtree_root)]`, output BCE logits over the atomic key vocabulary (~427 classes).
2. **Subtree masking** in `V8Dataset.__getitem__`: per-doc, pick 1-3 mutually disjoint subtrees, mark their positions with the existing `[MASK]` sentinel.
3. **Leak-aware aggregator path**: a new `subtree_mask` precompute tensor flowing through collate → aggregator, used to exclude masked-subtree positions from aggregation. Doc_vec is computed only from genuinely-unmasked context.
4. **In-training monitoring**: per-epoch separate-loss logging + per-epoch probe evaluation + held-out validation loss.

### What stays

- Encoder, tree positional encoding, MLM objective on atomic Token Head, vectorized aggregator architecture, vocab.
- The `[MASK]` sentinel — no new vocab token. Encoder treats subtree-masked positions identically to MLM-masked positions; the distinction (and the leak exclusion) happens at the aggregator level via the `subtree_mask` tensor.

### Reconstruction Head architecture

```python
class ReconstructionHead(nn.Module):
    """Predict bag of atomic keys in a masked subtree, from doc_vec + root pos."""
    def __init__(self, d_model: int, d_pos: int, atomic_vocab_size: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model + d_pos, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, atomic_vocab_size),
        )

    def forward(self, doc_vec_per_subtree, pos_emb_per_subtree):
        # doc_vec_per_subtree: (M, d_model)         — repeated per subtree in batch
        # pos_emb_per_subtree: (M, d_pos)           — [depth_emb ; sibling_emb] of root
        #                                             (d_pos = d_depth + d_sibling)
        # Returns logits (M, V_atomic)
        return self.mlp(torch.cat([doc_vec_per_subtree, pos_emb_per_subtree], dim=-1))
```

The position embedding is `[depth_emb(root) ; sibling_emb(root)]` — concat of the existing learned depth and sibling embeddings. No new learnable embeddings.

**Deliberately excluded:**
- `node_type_emb(root)` — always KEY for masked subtree roots (we only pick KEY positions with children); constant, no signal.
- The root's KEY identity (e.g., "containers" vs "metadata") — including it would let the head memorize a per-key lookup table ("containers subtrees have keys {name, image, ports, env}"), making doc_vec unnecessary for prediction and *removing the pretraining pressure on doc_vec*. The head is intentionally under-informed about the root key so doc_vec becomes load-bearing.
- The encoder's hidden state at the root position (`h_root`) — would be a soft leak (encoder's MLM-style guess of what the root key was), same logic as the aggregator leak we eliminated.

`M = Σ_b len(subtree_roots[b])` — total number of masked subtrees in a batch (1-3 per doc × 32 docs ≈ 32-96). Trivial compute vs encoder.

## Data path

### When masking runs

Per `V8Dataset.__getitem__` call (during training, in DataLoader workers, CPU-side, parallel with GPU work). Fresh random masks every epoch — same rationale as BERT-style MLM: prevent the model from memorizing specific masked-position patterns.

**NOT baked into the corpus cache.** The descendant sets per KEY position depend only on tree shape (static per doc) and ARE cached at `V8Dataset.__init__` time — but the picking itself is per-call.

### Complexity

- **One-time at `V8Dataset.__init__`** (per doc): compute `descendants_of` for every KEY-with-children position via DFS. Cost per doc: `O(N · D)` where N = nodes and D = max tree depth (capped at 9). For 5K docs × ~500 ops each ≈ 2.5M ops ≈ <1s.
- **Per `__getitem__` call**: only candidate filter + random pick. Cost: `O(K)` (K = number of KEY positions with children) plus tiny set intersections for the disjoint check. ~30-100 ops per doc. Trivial vs the existing `compute_children_info` work.

### V8Dataset.__getitem__ — subtree masking algorithm

```python
def pick_subtrees(N, key_positions, depth_of, children_of, mlm_masked_positions,
                  rng) -> list[int]:
    """Return 1-3 mutually-disjoint subtree root positions (or [] if none qualify)."""
    if N < MIN_DOC_NODES:
        return []

    def descendants_of(pos):
        out = {pos}
        stack = list(children_of.get(pos, []))
        while stack:
            p = stack.pop()
            out.add(p)
            stack.extend(children_of.get(p, []))
        return out

    candidates = []
    for kp in key_positions:
        if depth_of[kp] < 1:
            continue
        if not children_of.get(kp):
            continue
        descs = descendants_of(kp)
        if len(descs) > MAX_SUBTREE_FRACTION * N:
            continue
        if descs & mlm_masked_positions:
            continue
        candidates.append((kp, descs))

    if not candidates:
        return []

    rng.shuffle(candidates)
    num_to_pick = rng.randint(1, min(3, len(candidates)))
    picked = []
    picked_positions = set()
    for kp, descs in candidates:
        if descs & picked_positions:
            continue
        if len(picked_positions | descs) > MAX_TOTAL_SUBTREE_FRACTION * N:
            continue
        picked.append(kp)
        picked_positions.update(descs)
        if len(picked) >= num_to_pick:
            break
    return picked
```

**Constants:**
- `MIN_DOC_NODES = 10` — docs smaller than this skip reconstruction (still do MLM)
- `MAX_SUBTREE_FRACTION = 0.30` — no single subtree larger than 30% of doc
- `MAX_TOTAL_SUBTREE_FRACTION = 0.05` — total subtree-masked positions ≤ 5% of doc

After picking, replace tokens in `picked_positions` with `[MASK]` sentinel.

### Outputs added by V8Dataset.__getitem__

- `subtree_mask`: `(N,)` bool — True at any position inside any masked subtree
- `subtree_roots`: list of 1-3 ints — root positions (or empty if no subtrees masked)
- `bag_of_keys_targets`: list of multi-hot vectors of length V_atomic, one per root — entry `v` is 1.0 if atomic key `v` appears at any position inside that subtree (including the root key itself), 0.0 otherwise. Repeated occurrences of the same key are counted as 1 (true bag-of-keys, not bag-of-counts).

### v8_collate_fn additions (extends the Phase 1 precompute path)

- `subtree_mask`: `(B, N)` bool — batched from per-item masks
- `subtree_roots_flat`: `(M, 2)` long — `[batch_idx, root_pos]` pairs across the batch
- `bag_of_keys_targets_flat`: `(M, V_atomic)` float — bag-of-keys targets, flattened across batch

`M` is variable across batches (sum of per-doc subtree counts).

### Aggregator change — leak exclusion

`TreeAggregator._forward_vectorized` accepts a new optional kwarg `subtree_mask: (B, N) bool | None`.

When provided, the scatter ops skip edges where either endpoint (child OR parent) is inside a masked subtree. Concretely:
- In each depth's `edges_at_depth` scatter, filter out edges where `subtree_mask[doc_idx, child_pos]` is True OR `subtree_mask[doc_idx, parent_pos]` is True
- In the top-level doc_vec computation, masked positions are excluded from the mean (use `top_level_key_mask & ~subtree_mask[:, ...]`)

The masked subtree root positions are EXCLUDED from `subtree_vecs` updates — they keep their original encoder hidden state. The Reconstruction Head doesn't use `subtree_vecs[root]`; it uses `doc_vec` + the position embedding of the root.

The reference path (`_forward_reference`) gets the same `subtree_mask` kwarg for the numerical-equivalence test.

### Mutual exclusivity (MLM mask vs subtree mask)

Subtree picking happens AFTER MLM mask sampling. The candidate filter `if descs & mlm_masked_positions: continue` ensures no subtree overlaps with the MLM mask. This is the spec's "make masks mutually exclusive at corpus-construction time" decision.

## Training loop

New script: `scripts/train_v8_phase1_recon.py` (forked from `train_v8_phase0.py`).

### Two run modes via CLI flag

- `--reconstruction off` → MLM-only control (β=0). Equivalent to Phase 1 vectorized trainer.
- `--reconstruction on` → MLM + reconstruction, β=0.5 (default α=1.0).

Both modes run for 10 epochs on the same 5K-doc subset with the same seed (42).

### Loss

```python
total_loss = 1.0 * mlm_loss + (0.5 if recon_enabled else 0.0) * recon_loss
```

### Per-epoch monitoring (both modes)

1. **Separate loss logging**: report `mlm_loss` and `recon_loss` separately per epoch
2. **Held-out validation loss**: 90/10 train/val split of 5K docs (val = last 500 docs by index, deterministic given seed). Compute mlm_loss + recon_loss on val set after each epoch
3. **Per-epoch doc_vec dump**: after each epoch, dump per-doc `doc_vec` to a per-epoch file (`doc_vecs_epoch_<N>.pt`). Probes are NOT run on the instance — they run locally after JL training completes, against all per-epoch dumps. This keeps the trainer simple (no sklearn dependency on the instance, no label extraction during training) and lets us see the full epoch-by-epoch trajectory locally.

### Validation set construction

Deterministic: `train_docs = docs[:4500]`, `val_docs = docs[4500:5000]`. Same split for both modes so val losses are directly comparable.

## Smoke probes

New script: `scripts/eval_v8_probes.py`. Reusable for both runs; called after training completes.

### Probes

| Probe | Label extraction (from parsed YAML dict) |
|---|---|
| **kind** | `doc.get("kind")` — restrict to top-10 most frequent kinds in corpus |
| **has-containers** | `True` if `doc.spec.containers` exists (Pod) OR `doc.spec.template.spec.containers` exists (Deployment, StatefulSet, etc.) |
| **has-initContainers** | `True` if either path above has an `initContainers` field with len ≥ 1 |
| **has-volume-mounts** | `True` if any container in either path has `volumeMounts` with len ≥ 1 |

### Method (per probe)

1. Build label vector from raw YAMLs (deterministic)
2. Random 80/20 split using `random_state=42`
3. Fit `sklearn.linear_model.LogisticRegression(max_iter=2000)` on 80%
4. Report accuracy on 20%
5. Multi-class probes (kind) use multinomial; binary probes use default

### Why these probes

- **kind** = sanity check (don't regress Phase 0/1's 99.87%)
- **has-containers** = top-level structural feature, splits workload vs non-workload kinds
- **has-initContainers** = within-kind variation, ~5-15% of workload docs have it
- **has-volume-mounts** = nested structural feature, tests whether doc_vec sees past top-level

These are *smoke tests*, not a real benchmark. Picked because they're easy to label automatically and probe different "levels" of structural information. A real eval framework would do this systematically — deferred.

## Two-condition comparison

Two JarvisLabs training runs:

1. **Control (MLM-only)**: `--reconstruction off`. ~2.5 min wall time (10 forward passes + 10 doc_vec dumps), ~$0.05.
2. **Treatment (MLM+recon)**: `--reconstruction on`. ~3 min wall time (slightly more compute per step due to recon head + extra subtree-masking work in collate), ~$0.07.

Both use:
- 5K docs, 10 epochs, batch_size 32, seed 42 — identical to Phase 0/1 setup
- Fresh L4 instance, destroyed after benchmark
- Same train/val split, same probe labels

### Acceptance gate

The mini-cycle is GO for the next phase if **both** hold:

1. **Reconstruction trains stably**: `recon_loss` decreases monotonically, no NaN, no MLM loss regression (treatment MLM loss within 10% *relative* of control MLM loss at epoch 10 — i.e., if control hits 0.82, treatment must be ≤ 0.90).
2. **At least one non-kind probe improves**: ≥1 of (has-containers, has-initContainers, has-volume-mounts) shows treatment > control by ≥2 percentage points absolute. (Kind probe is expected to stay similar; the question is whether recon teaches doc_vec NEW information.)

If recon trains stably but no probes improve: AMBIGUOUS — the smoke probes may be too coarse; consider escalating to a real eval framework as the next mini-cycle.

If recon doesn't train stably: NO-GO. Investigate (leak handling bug, loss scale, masking bug).

## Files

### New

- `yaml_bert/reconstruction_head.py` (~30 lines): `ReconstructionHead` class
- `yaml_bert/subtree_masking.py` (~60 lines): `pick_subtrees` + helpers
- `scripts/train_v8_phase1_recon.py` (~250 lines): forked + extended trainer
- `scripts/eval_v8_probes.py` (~120 lines): 4-probe evaluation
- `docs/v8-phase1-reconstruction-results.md`: results doc with comparison table + decision
- `tests/test_subtree_masking.py`: unit tests for `pick_subtrees` corner cases
- `tests/test_reconstruction_head.py`: shape + backward tests
- `tests/test_v8_dataset_with_subtree_mask.py`: integration test for new collate outputs

### Modified

- `yaml_bert/v8_dataset.py`: `V8Dataset.__getitem__` adds subtree masking; `v8_collate_fn` batches new fields
- `yaml_bert/aggregator.py`: `_forward_vectorized` and `_forward_reference` accept `subtree_mask`
- `yaml_bert/v8_model.py`: holds `recon_head`; `forward` returns `recon_logits` when subtree info provided
- `yaml_bert/config.py`: `recon_enabled: bool = False`, `recon_loss_weight: float = 0.5`
- `tests/test_aggregator_vectorized.py`: extend equivalence test to cover the `subtree_mask` path

## Acceptance gate (mini-cycle done)

1. All existing tests still pass (20+ from prior cycles)
2. New tests pass (subtree_masking, reconstruction_head, v8_dataset_with_subtree_mask, extended aggregator equivalence)
3. Both JL runs complete cleanly
4. Per-epoch probes captured for both runs
5. Comparison table + decision written to results doc
6. Mini-cycle documented in `docs/v8-phase1-reconstruction-results.md` with GO / NO-GO / AMBIGUOUS decision

## Cost / time estimate

- Implementation: ~1 day of coding (8 subagent-driven tasks similar to vectorization cycle)
- Local testing: ~30 min
- JarvisLabs runs: 2 × ~$0.10 = $0.20 total
- Results writeup + decision: ~30 min
- **Total: ~1-1.5 days end-to-end**

## Decision after this mini-cycle

**GO** → start the next Phase 1 mini-cycle: proper evaluation framework design (since the smoke probes told us *something*, but we need a real benchmark to scale up the comparison).

**AMBIGUOUS** → recon trains but smoke probes can't see the difference. Skip ahead to eval framework mini-cycle (without recon for now), then come back to recon with a real benchmark.

**NO-GO** → recon objective is broken or doesn't help. Either re-think the objective (different prediction target — tree shape? token sequence?) or abandon and move on to combine-function or eval framework.
