# v8 Aggregator Vectorization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace per-document Python loops in `TreeAggregator` and `V8Model`'s `s_parent` construction with batched PyTorch scatter ops, restoring training speed within 25% of v7 (target ≥ 7 it/s) while keeping all behavior identical.

**Architecture:** Vanilla PyTorch scatter ops (no external graph libs). `v8_collate_fn` precomputes per-batch tensors that group children by depth; aggregator processes depth-by-depth with batched `index_add_`. `V8Model`'s s_parent uses `torch.gather` + `torch.where`. Backward-compat fallback (per-doc loops) stays so existing tests pass unchanged.

**Tech Stack:** Python 3, PyTorch (vanilla scatter ops, no torch_scatter/PyG/DGL), pytest, the existing yaml_bert package.

---

## File Structure

**Modified files:**
- `yaml_bert/v8_dataset.py` — extend `v8_collate_fn` to precompute batched aggregator tensors
- `yaml_bert/aggregator.py` — add vectorized forward path with backward-compat fallback
- `yaml_bert/v8_model.py` — add vectorized `s_parent` with backward-compat fallback; pass precomputed tensors through to aggregator

**New files:**
- `tests/test_aggregator_vectorized.py` — numerical equivalence between per-doc and vectorized paths
- `tests/test_aggregator_perf_smoke.py` — local microbenchmark, asserts vectorized ≥5× faster on CPU
- `docs/v8-phase1-vectorization-results.md` — written at end with benchmark numbers + go/no-go for next mini-cycle

---

## Task 1: Add precomputed tensors to `v8_collate_fn`

**Files:**
- Modify: `yaml_bert/v8_dataset.py` (`v8_collate_fn` only — leave `compute_children_info` and `V8Dataset` alone)
- Test: `tests/test_v8_dataset.py` (append one test)

**Why:** The vectorized aggregator and s_parent both need per-batch tensors that aren't easy to derive on the fly. Precomputing in collate (CPU-side, per batch, runs in DataLoader workers in parallel with training) keeps the GPU path clean.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_v8_dataset.py`:

```python
def test_v8_collate_includes_aggregator_precompute():
    """v8_collate_fn precomputes tensors needed by the vectorized aggregator."""
    from yaml_bert.linearizer import YamlLinearizer
    from yaml_bert.config import YamlBertConfig
    from yaml_bert.vocab import VocabBuilder
    from yaml_bert.v8_dataset import V8Dataset, v8_collate_fn

    docs = [
        YamlLinearizer().linearize("apiVersion: v1\nkind: Pod\nspec:\n  x: 1\n"),
        YamlLinearizer().linearize("apiVersion: v1\nkind: Service\n"),
    ]
    vocab = VocabBuilder().build([n for d in docs for n in d], min_freq=1)
    config = YamlBertConfig(v8_mode=True, mask_prob=0.0)
    ds = V8Dataset(documents=docs, vocab=vocab, config=config)
    batch = v8_collate_fn([ds[0], ds[1]])

    # parent_of_tensor: (B, N) long, -1 sentinel for no-parent or padding
    assert "parent_of_tensor" in batch
    pt = batch["parent_of_tensor"]
    assert pt.dim() == 2
    assert pt.dtype == torch.long
    assert pt.shape[0] == 2  # B
    # Doc 0: "spec" at pos 0 is root → parent_of = -1
    #        "x" at pos 1 is child of spec → parent_of points to spec's index
    # Doc 1: "apiVersion" root → -1, "kind" root → -1

    # top_level_key_mask: (B, N) bool, True at depth-0 KEY positions
    assert "top_level_key_mask" in batch
    tlkm = batch["top_level_key_mask"]
    assert tlkm.dim() == 2
    assert tlkm.dtype == torch.bool
    assert tlkm.shape == pt.shape

    # edges_by_depth: dict[int, tensor (E, 3)] of [doc_idx, child_pos, parent_pos]
    # parents_by_depth: dict[int, tensor (P, 2)] of [doc_idx, parent_pos] with at-least-one-child
    assert "edges_by_depth" in batch
    assert "parents_by_depth" in batch
    assert isinstance(batch["edges_by_depth"], dict)
    assert isinstance(batch["parents_by_depth"], dict)
    # Same set of depth keys in both
    assert set(batch["edges_by_depth"].keys()) == set(batch["parents_by_depth"].keys())

    # Per-depth shape check: edges has (E, 3), parents has (P, 2)
    for d, edges in batch["edges_by_depth"].items():
        assert edges.dim() == 2 and edges.shape[1] == 3
        assert edges.dtype == torch.long
    for d, parents in batch["parents_by_depth"].items():
        assert parents.dim() == 2 and parents.shape[1] == 2
        assert parents.dtype == torch.long
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_v8_dataset.py::test_v8_collate_includes_aggregator_precompute -v`
Expected: FAIL with `KeyError: 'parent_of_tensor'` or `AssertionError`.

- [ ] **Step 3: Implement the precompute additions in `v8_collate_fn`**

Replace the current end of `v8_collate_fn` (right before `return result`) with the following. The existing batching of 1D tensors + `padding_mask` + `batch_info` stays the same; we only ADD new fields.

In `yaml_bert/v8_dataset.py`, find the `v8_collate_fn` function. After the `result["batch_info"] = batch_info` line and before `return result`, insert:

```python
    # Vectorized-aggregator precompute. CPU work, runs in DataLoader workers.
    B = len(batch)
    N = max_len

    # parent_of_tensor: (B, N) long. -1 sentinel for no-parent, non-key, or padding.
    parent_of_tensor = torch.full((B, N), -1, dtype=torch.long)
    for b_idx, info in enumerate(batch_info):
        parent_of = info["parent_of"]  # list[int] of length n_b
        n_b = len(parent_of)
        if n_b > 0:
            parent_of_tensor[b_idx, :n_b] = torch.tensor(parent_of, dtype=torch.long)

    # top_level_key_mask: (B, N) bool. True where depth==0 AND position is a KEY.
    top_level_key_mask = torch.zeros((B, N), dtype=torch.bool)
    for b_idx, info in enumerate(batch_info):
        for kp in info["key_positions"]:
            if info["depth_of"][kp] == 0:
                top_level_key_mask[b_idx, kp] = True

    # edges_by_depth: dict[depth, (E, 3) long] of [doc_idx, child_pos, parent_pos] across batch.
    # parents_by_depth: dict[depth, (P, 2) long] of unique [doc_idx, parent_pos] with at-least-one-child.
    edges_by_depth: dict[int, list[tuple[int, int, int]]] = {}
    parents_set_by_depth: dict[int, set[tuple[int, int]]] = {}
    for b_idx, info in enumerate(batch_info):
        children_of = info["children_of"]
        depth_of = info["depth_of"]
        for parent_pos in info["key_positions"]:
            kids = children_of[parent_pos]
            if not kids:
                continue
            parent_depth = depth_of[parent_pos]
            edges_by_depth.setdefault(parent_depth, []).extend(
                (b_idx, child_pos, parent_pos) for child_pos in kids
            )
            parents_set_by_depth.setdefault(parent_depth, set()).add(
                (b_idx, parent_pos),
            )

    result["parent_of_tensor"] = parent_of_tensor
    result["top_level_key_mask"] = top_level_key_mask
    result["edges_by_depth"] = {
        d: torch.tensor(edges, dtype=torch.long)
        for d, edges in edges_by_depth.items()
    }
    result["parents_by_depth"] = {
        d: torch.tensor(sorted(parents_set), dtype=torch.long)
        for d, parents_set in parents_set_by_depth.items()
    }
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_v8_dataset.py -v`
Expected: all PASS (existing 9 + new 1 = 10)

- [ ] **Step 5: Commit**

```bash
git add yaml_bert/v8_dataset.py tests/test_v8_dataset.py
git commit -m "feat(v8): precompute batched aggregator tensors in v8_collate_fn"
```

---

## Task 2: Vectorize `TreeAggregator.forward`

**Files:**
- Modify: `yaml_bert/aggregator.py`
- Test: `tests/test_aggregator_vectorized.py` (new — numerical equivalence)

**Why:** Heart of the perf fix. The per-doc loop becomes batched scatter ops, one per depth level. Keep the per-doc reference path as fallback (when tensor kwargs aren't provided) so the existing 3 aggregator tests stay green and the equivalence test has a reference to compare against.

- [ ] **Step 1: Write the failing test (numerical equivalence)**

Create `tests/test_aggregator_vectorized.py`:

```python
"""Numerical equivalence: per-doc reference path vs vectorized path."""
import torch

from yaml_bert.aggregator import TreeAggregator
from yaml_bert.linearizer import YamlLinearizer
from yaml_bert.v8_dataset import (
    V8Dataset, compute_children_info, v8_collate_fn,
)
from yaml_bert.vocab import VocabBuilder
from yaml_bert.config import YamlBertConfig


def test_vectorized_aggregator_equals_per_doc_reference():
    """Vectorized aggregator produces numerically identical output to the
    per-doc reference path, given the same hidden states + batch_info."""
    docs = [
        YamlLinearizer().linearize(
            "apiVersion: v1\nkind: Pod\nmetadata:\n  name: a\n"
            "spec:\n  containers:\n  - name: x\n"),
        YamlLinearizer().linearize(
            "apiVersion: apps/v1\nkind: Deployment\nspec:\n  replicas: 3\n"
            "  selector:\n    matchLabels:\n      app: y\n"),
    ]
    vocab = VocabBuilder().build([n for d in docs for n in d], min_freq=1)
    config = YamlBertConfig(v8_mode=True, mask_prob=0.0, d_model=16)
    ds = V8Dataset(docs, vocab, config)
    batch = v8_collate_fn([ds[0], ds[1]])

    B, N = batch["token_ids"].shape
    d_model = 16
    torch.manual_seed(0)
    hidden = torch.randn(B, N, d_model)

    agg = TreeAggregator(d_model=d_model)

    # Reference path: legacy, no tensor kwargs
    ref_subtree, ref_doc = agg(hidden, batch["batch_info"])

    # Vectorized path: pass precomputed tensors as kwargs
    vec_subtree, vec_doc = agg(
        hidden, batch["batch_info"],
        parent_of_tensor=batch["parent_of_tensor"],
        top_level_key_mask=batch["top_level_key_mask"],
        edges_by_depth=batch["edges_by_depth"],
        parents_by_depth=batch["parents_by_depth"],
    )

    assert torch.allclose(ref_subtree, vec_subtree, atol=1e-6), (
        f"subtree_vecs mismatch: max diff = "
        f"{(ref_subtree - vec_subtree).abs().max().item()}"
    )
    assert torch.allclose(ref_doc, vec_doc, atol=1e-6), (
        f"doc_vec mismatch: max diff = "
        f"{(ref_doc - vec_doc).abs().max().item()}"
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_aggregator_vectorized.py -v`
Expected: FAIL with `TypeError: forward() got an unexpected keyword argument 'parent_of_tensor'`.

- [ ] **Step 3: Implement the vectorized path with fallback**

Edit `yaml_bert/aggregator.py`. Replace the entire `TreeAggregator.forward` method with the dispatching version. The existing per-doc loop becomes a private method `_forward_reference`; a new `_forward_vectorized` handles the tensor-kwarg path.

Full replacement for the `TreeAggregator` class (keep imports + the class docstring + `__init__` from current file):

```python
class TreeAggregator(nn.Module):
    """Bottom-up tree aggregation with mean combine.

    Two execution paths:
    - Reference path (default): per-doc Python loop. Used when the batch
      doesn't provide vectorized precompute tensors. Kept for tests and
      for backward compatibility.
    - Vectorized path (preferred during training): batched PyTorch scatter
      ops, processed depth-by-depth. Activated when caller passes the
      precomputed tensors as kwargs.

    Both paths produce numerically equivalent output on the same inputs
    (guaranteed by tests/test_aggregator_vectorized.py).
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.d_model = d_model

    def forward(
        self,
        hidden_states: torch.Tensor,
        batch_info: list[dict],
        *,
        parent_of_tensor: torch.Tensor | None = None,
        top_level_key_mask: torch.Tensor | None = None,
        edges_by_depth: dict[int, torch.Tensor] | None = None,
        parents_by_depth: dict[int, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden_states: (B, N, d_model) per-position hidden states.
            batch_info: list of B dicts (legacy path; required even when
                vectorized — kept for fallback compat).
            parent_of_tensor / top_level_key_mask / edges_by_depth /
                parents_by_depth: when ALL provided, use vectorized path.

        Returns:
            (subtree_vecs, doc_vec)
        """
        if (parent_of_tensor is not None
                and top_level_key_mask is not None
                and edges_by_depth is not None
                and parents_by_depth is not None):
            return self._forward_vectorized(
                hidden_states,
                top_level_key_mask=top_level_key_mask,
                edges_by_depth=edges_by_depth,
                parents_by_depth=parents_by_depth,
            )
        return self._forward_reference(hidden_states, batch_info)

    def _forward_reference(
        self,
        hidden_states: torch.Tensor,
        batch_info: list[dict],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-document Python loop. Original Phase 0 implementation."""
        b, n, d = hidden_states.shape
        subtree_vecs = hidden_states.clone()
        doc_vec = torch.zeros(b, d, device=hidden_states.device,
                              dtype=hidden_states.dtype)

        for doc_idx in range(b):
            info = batch_info[doc_idx]
            children_of = info["children_of"]
            depth_of = info["depth_of"]
            key_positions = info["key_positions"]

            keys_by_depth: dict[int, list[int]] = {}
            for kp in key_positions:
                keys_by_depth.setdefault(depth_of[kp], []).append(kp)

            for depth in sorted(keys_by_depth.keys(), reverse=True):
                for parent_pos in keys_by_depth[depth]:
                    children = children_of[parent_pos]
                    if children:
                        child_vecs = subtree_vecs[doc_idx, children]
                        own = hidden_states[doc_idx, parent_pos:parent_pos + 1]
                        combined = torch.cat(
                            [own, child_vecs], dim=0,
                        ).mean(dim=0)
                    else:
                        combined = hidden_states[doc_idx, parent_pos]
                    subtree_vecs[doc_idx, parent_pos] = combined

            top_level = keys_by_depth.get(0, [])
            if top_level:
                doc_vec[doc_idx] = subtree_vecs[doc_idx, top_level].mean(dim=0)

        return subtree_vecs, doc_vec

    def _forward_vectorized(
        self,
        hidden_states: torch.Tensor,
        *,
        top_level_key_mask: torch.Tensor,
        edges_by_depth: dict[int, torch.Tensor],
        parents_by_depth: dict[int, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Batched scatter-based path. One PyTorch op per depth level."""
        B, N, d = hidden_states.shape
        subtree_vecs = hidden_states.clone()

        # Process depths deepest-first.
        # parents_by_depth[d] = (P, 2) of [doc_idx, parent_pos]
        # edges_by_depth[d]   = (E, 3) of [doc_idx, child_pos, parent_pos]
        for depth in sorted(edges_by_depth.keys(), reverse=True):
            edges = edges_by_depth[depth].to(hidden_states.device)
            parents = parents_by_depth[depth].to(hidden_states.device)

            doc_idx_e = edges[:, 0]   # (E,)
            child_pos = edges[:, 1]   # (E,)
            parent_pos_e = edges[:, 2]  # (E,)

            # Gather child subtree vectors (E, d)
            child_vecs = subtree_vecs[doc_idx_e, child_pos]

            # Linear index (doc_idx, parent_pos) → flat
            parent_linear_e = doc_idx_e * N + parent_pos_e  # (E,)

            # Accumulate sum and count into per-(B*N) slots
            sum_acc = torch.zeros(
                B * N, d,
                dtype=hidden_states.dtype, device=hidden_states.device,
            )
            sum_acc.index_add_(0, parent_linear_e, child_vecs)

            count_acc = torch.zeros(
                B * N, dtype=hidden_states.dtype,
                device=hidden_states.device,
            )
            count_acc.index_add_(
                0, parent_linear_e,
                torch.ones_like(parent_linear_e, dtype=hidden_states.dtype),
            )

            # For each parent at this depth: mean = (sum + own) / (count + 1)
            parent_doc_idx = parents[:, 0]   # (P,)
            parent_pos_p = parents[:, 1]     # (P,)
            parent_linear_p = parent_doc_idx * N + parent_pos_p  # (P,)

            sum_at_parents = sum_acc[parent_linear_p]       # (P, d)
            count_at_parents = count_acc[parent_linear_p]   # (P,)
            own_at_parents = hidden_states[parent_doc_idx, parent_pos_p]  # (P, d)

            mean_at_parents = (sum_at_parents + own_at_parents) / (
                count_at_parents.unsqueeze(-1) + 1.0
            )

            # Write back
            subtree_vecs[parent_doc_idx, parent_pos_p] = mean_at_parents

        # doc_vec: mean of top-level key subtree vectors per doc.
        # top_level_key_mask: (B, N) bool; compute masked mean per doc.
        mask_f = top_level_key_mask.to(hidden_states.dtype).unsqueeze(-1)  # (B, N, 1)
        weighted = subtree_vecs * mask_f                                    # (B, N, d)
        sum_per_doc = weighted.sum(dim=1)                                   # (B, d)
        count_per_doc = top_level_key_mask.sum(dim=1, dtype=hidden_states.dtype).clamp(min=1).unsqueeze(-1)
        doc_vec = sum_per_doc / count_per_doc

        # If a doc has no top-level keys (count was 0 → clamped to 1), the
        # numerator is also 0, so doc_vec is zero — matches reference path.
        return subtree_vecs, doc_vec
```

- [ ] **Step 4: Run the new equivalence test**

Run: `python -m pytest tests/test_aggregator_vectorized.py -v`
Expected: PASS.

- [ ] **Step 5: Run existing aggregator tests (must still pass via reference fallback)**

Run: `python -m pytest tests/test_aggregator.py -v`
Expected: 3/3 PASS (unchanged behavior; they don't provide tensor kwargs so the reference path runs).

- [ ] **Step 6: Commit**

```bash
git add yaml_bert/aggregator.py tests/test_aggregator_vectorized.py
git commit -m "feat(aggregator): batched scatter-based vectorized forward path"
```

---

## Task 3: Vectorize `s_parent` in `V8Model`

**Files:**
- Modify: `yaml_bert/v8_model.py`
- Test: existing `tests/test_v8_model_e2e.py` (no new test — equivalence is exercised by the smoke test once Task 4 wires things together)

**Why:** Same kind of fix at the V8Model level. Use `torch.gather` + `torch.where` for s_parent in the fast path; keep the per-doc loop as a fallback when `parent_of_tensor` isn't provided.

- [ ] **Step 1: Edit `V8Model.forward` to accept and use precomputed tensors**

Find `V8Model.forward` in `yaml_bert/v8_model.py`. Replace its signature and body with:

```python
    def forward(
        self,
        token_ids: torch.Tensor,
        node_types: torch.Tensor,
        depths: torch.Tensor,
        sibling_indices: torch.Tensor,
        batch_info: list[dict],
        padding_mask: torch.Tensor | None = None,
        *,
        parent_of_tensor: torch.Tensor | None = None,
        top_level_key_mask: torch.Tensor | None = None,
        edges_by_depth: dict[int, torch.Tensor] | None = None,
        parents_by_depth: dict[int, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (logits, doc_vec). Vectorized path activates when the
        precomputed tensor kwargs are provided (always true at training)."""
        x = self.embedding(token_ids, node_types, depths, sibling_indices)
        x = self.encoder(x, src_key_padding_mask=padding_mask)

        # Aggregator: forwards through to its own vectorized/reference dispatch.
        subtree_vecs, doc_vec = self.aggregator(
            x, batch_info,
            parent_of_tensor=parent_of_tensor,
            top_level_key_mask=top_level_key_mask,
            edges_by_depth=edges_by_depth,
            parents_by_depth=parents_by_depth,
        )

        b, n, d = x.shape

        if parent_of_tensor is not None:
            # Vectorized s_parent
            safe_parent = parent_of_tensor.clamp(min=0)  # (B, N)
            s_parent = torch.gather(
                subtree_vecs, dim=1,
                index=safe_parent.unsqueeze(-1).expand(-1, -1, d),
            )  # (B, N, d)
            no_parent_mask = (parent_of_tensor == -1).unsqueeze(-1)  # (B, N, 1)
            s_parent = torch.where(
                no_parent_mask, doc_vec.unsqueeze(1), s_parent,
            )
        else:
            # Reference path: per-doc Python loop (kept for tests / fallback).
            s_parent = torch.zeros_like(x)
            for doc_idx in range(b):
                parent_of = batch_info[doc_idx]["parent_of"]
                for i in range(min(n, len(parent_of))):
                    p = parent_of[i]
                    if p >= 0:
                        s_parent[doc_idx, i] = subtree_vecs[doc_idx, p]
                    else:
                        s_parent[doc_idx, i] = doc_vec[doc_idx]

        doc_vec_broadcast = doc_vec.unsqueeze(1).expand(b, n, d)
        head_input = torch.cat([x, doc_vec_broadcast, s_parent], dim=-1)
        logits = self.token_head(head_input)

        return logits, doc_vec
```

- [ ] **Step 2: Run existing V8Model tests**

Run: `python -m pytest tests/test_v8_model_e2e.py -v`
Expected: 3/3 PASS. (Both `test_v8_model_forward_pass_shape` and `test_v8_model_backward_no_nan` don't pass tensor kwargs → use reference path → identical behavior. `test_v8_smoke_e2e_small_batch` calls with `batch["batch_info"]` only, also reference path.)

- [ ] **Step 3: Commit**

```bash
git add yaml_bert/v8_model.py
git commit -m "feat(v8_model): vectorized s_parent via torch.gather + where"
```

---

## Task 4: Wire collate tensors through to V8Model at training time

**Files:**
- Modify: `scripts/train_v8_phase0.py` (call V8Model.forward with the new kwargs)
- Test: add one e2e test that confirms the vectorized path is exercised end-to-end

**Why:** Until the trainer plumbs the precomputed tensors through, the vectorization is dead code. This task makes the fast path actually run during training, and adds a test that catches regressions where the kwargs stop being passed.

- [ ] **Step 1: Write the failing e2e test**

Append to `tests/test_v8_model_e2e.py`:

```python
def test_v8_smoke_e2e_vectorized_path():
    """End-to-end with v8_collate_fn precompute kwargs passed to V8Model.

    Asserts the vectorized path produces logits and a loss, AND that the
    logits are numerically equivalent to the reference path on the same
    inputs (catches regressions where the two paths diverge)."""
    from yaml_bert.linearizer import YamlLinearizer
    from yaml_bert.vocab import VocabBuilder
    from yaml_bert.config import YamlBertConfig
    from yaml_bert.embedding import YamlBertEmbedding
    from yaml_bert.v8_dataset import V8Dataset, v8_collate_fn

    yamls = [
        "apiVersion: v1\nkind: Pod\nmetadata:\n  name: a\n",
        "apiVersion: v1\nkind: Service\nspec:\n  x: 1\n",
        "apiVersion: apps/v1\nkind: Deployment\nspec:\n  replicas: 3\n",
    ]
    documents = [YamlLinearizer().linearize(y) for y in yamls]
    flat = [n for doc in documents for n in doc]
    vocab = VocabBuilder().build(flat, min_freq=1)

    config = YamlBertConfig(d_model=32, num_layers=2, num_heads=4,
                            v8_mode=True, mask_prob=0.5)
    ds = V8Dataset(documents, vocab, config)
    batch = v8_collate_fn([ds[i] for i in range(len(ds))])

    emb = YamlBertEmbedding(config=config,
                            key_vocab_size=vocab.key_vocab_size,
                            value_vocab_size=vocab.value_vocab_size)
    model = V8Model(config=config, embedding=emb,
                    atomic_vocab_size=vocab.atomic_target_vocab_size)
    model.eval()  # disable dropout for deterministic comparison

    # Reference path: no tensor kwargs
    with torch.no_grad():
        ref_logits, ref_doc = model(
            token_ids=batch["token_ids"],
            node_types=batch["node_types"],
            depths=batch["depths"],
            sibling_indices=batch["sibling_indices"],
            batch_info=batch["batch_info"],
            padding_mask=batch["padding_mask"],
        )

    # Vectorized path: pass tensor kwargs
    with torch.no_grad():
        vec_logits, vec_doc = model(
            token_ids=batch["token_ids"],
            node_types=batch["node_types"],
            depths=batch["depths"],
            sibling_indices=batch["sibling_indices"],
            batch_info=batch["batch_info"],
            padding_mask=batch["padding_mask"],
            parent_of_tensor=batch["parent_of_tensor"],
            top_level_key_mask=batch["top_level_key_mask"],
            edges_by_depth=batch["edges_by_depth"],
            parents_by_depth=batch["parents_by_depth"],
        )

    # Both paths should produce identical logits and doc vectors
    assert torch.allclose(ref_logits, vec_logits, atol=1e-5), (
        f"logits diverge: max diff = "
        f"{(ref_logits - vec_logits).abs().max().item()}"
    )
    assert torch.allclose(ref_doc, vec_doc, atol=1e-5)

    # Loss + backward must still work on vectorized path
    model.train()
    loss = torch.nn.functional.cross_entropy(
        vec_logits.view(-1, vec_logits.size(-1)),
        batch["atomic_labels"].view(-1),
        ignore_index=-100,
    )
    assert torch.isfinite(loss)
```

- [ ] **Step 2: Run test, expect it to pass already**

Run: `python -m pytest tests/test_v8_model_e2e.py::test_v8_smoke_e2e_vectorized_path -v`

If the test passes immediately (V8Model is already wired correctly per Task 3), good — proceed. If it fails with a dimension mismatch or path-selection bug, fix the bug in `V8Model.forward` and re-run.

Expected: PASS.

- [ ] **Step 3: Update `scripts/train_v8_phase0.py` to pass tensor kwargs**

In `scripts/train_v8_phase0.py`, find the training loop. The current forward call is:

```python
            logits, doc_vec = model(
                token_ids=tensors["token_ids"],
                node_types=tensors["node_types"],
                depths=tensors["depths"],
                sibling_indices=tensors["sibling_indices"],
                batch_info=tensors["batch_info"],
                padding_mask=tensors["padding_mask"],
            )
```

Note: `tensors["batch_info"]` is the legacy list. The precompute tensors come from `batch` directly (they are torch.Tensor and got moved to device by the dict comprehension just before, but only if the comprehension catches them — verify by reading the surrounding code; `batch_info` is excluded by the `isinstance(v, torch.Tensor)` filter, but `parent_of_tensor`/`top_level_key_mask` ARE tensors so they DO get moved to device). The dicts `edges_by_depth` and `parents_by_depth` contain tensors but the dict itself isn't a tensor — they need explicit device moves.

Replace the forward call AND the surrounding `tensors = ...` dict comprehension with:

```python
            # Move tensors to device, leave dict-of-tensors and list-of-dicts on CPU
            tensors = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            # Move the per-depth tensor dicts to device
            tensors["edges_by_depth"] = {
                d: t.to(device) for d, t in batch["edges_by_depth"].items()
            }
            tensors["parents_by_depth"] = {
                d: t.to(device) for d, t in batch["parents_by_depth"].items()
            }

            optimizer.zero_grad()
            logits, doc_vec = model(
                token_ids=tensors["token_ids"],
                node_types=tensors["node_types"],
                depths=tensors["depths"],
                sibling_indices=tensors["sibling_indices"],
                batch_info=tensors["batch_info"],
                padding_mask=tensors["padding_mask"],
                parent_of_tensor=tensors["parent_of_tensor"],
                top_level_key_mask=tensors["top_level_key_mask"],
                edges_by_depth=tensors["edges_by_depth"],
                parents_by_depth=tensors["parents_by_depth"],
            )
```

Apply the same change to the eval-time forward call further down in the same script (Step 6 "Dump per-doc doc vectors for probe"). Replace that forward call's args list the same way (drop the `tensors["..."]` for batch_info but pass `batch["batch_info"]` plus the new kwargs).

Actually — to keep it simple and DRY, factor out into a helper inside the script. After the `import` block but before `def main`, add:

```python
def _forward_v8(model, batch, device):
    """Forward V8Model with vectorized path active. Returns (logits, doc_vec)."""
    return model(
        token_ids=batch["token_ids"].to(device),
        node_types=batch["node_types"].to(device),
        depths=batch["depths"].to(device),
        sibling_indices=batch["sibling_indices"].to(device),
        batch_info=batch["batch_info"],
        padding_mask=batch["padding_mask"].to(device),
        parent_of_tensor=batch["parent_of_tensor"].to(device),
        top_level_key_mask=batch["top_level_key_mask"].to(device),
        edges_by_depth={
            d: t.to(device) for d, t in batch["edges_by_depth"].items()
        },
        parents_by_depth={
            d: t.to(device) for d, t in batch["parents_by_depth"].items()
        },
    )
```

Then in the training loop:

```python
            optimizer.zero_grad()
            logits, doc_vec = _forward_v8(model, batch, device)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                batch["atomic_labels"].to(device).view(-1),
                ignore_index=-100,
            )
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
            if not torch.isfinite(loss):
                print(f"  !! NaN/Inf loss at batch {n_batches}; stopping early")
                return
```

And in the eval (doc-vector dump) loop:

```python
    with torch.no_grad():
        for batch_idx, batch in enumerate(eval_loader):
            _, dvec = _forward_v8(model, batch, device)
            doc_vecs.append(dvec.cpu())
            for doc_idx_in_batch in range(dvec.size(0)):
                global_idx = batch_idx * args.batch_size + doc_idx_in_batch
                if global_idx < len(cached):
                    doc_kinds.append(_extract_kind(cached[global_idx]))
```

- [ ] **Step 4: Smoke-run the training script locally (50 docs)**

Run: `python scripts/train_v8_phase0.py --max-docs 50 --epochs 1 --batch-size 4 --output-dir /tmp/v8_vec_smoke`

Expected: completes without crashes. Output shows epoch loss + saves checkpoint. The vectorized path now runs (no error from missing kwargs).

- [ ] **Step 5: Run all v8 tests for regression**

Run: `python -m pytest tests/test_aggregator.py tests/test_aggregator_vectorized.py tests/test_v8_dataset.py tests/test_v8_model_e2e.py -v`

Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add yaml_bert/v8_model.py scripts/train_v8_phase0.py tests/test_v8_model_e2e.py
git commit -m "feat(v8): wire vectorized path through trainer + e2e equivalence test"
```

---

## Task 5: Local perf smoke test

**Files:**
- Test: `tests/test_aggregator_perf_smoke.py` (new)

**Why:** Catch regressions where the vectorized path silently falls back, or where the speedup is too small to matter. CPU-only, fast (no GPU needed). Will be the first signal if Task 6's GPU benchmark would have problems.

- [ ] **Step 1: Write the perf smoke test**

Create `tests/test_aggregator_perf_smoke.py`:

```python
"""Local microbenchmark: vectorized aggregator must be substantially
faster than the per-doc reference path on a representative batch.

CPU-only. Soft acceptance: ≥5× speedup. If this fails, the GPU
benchmark in Phase 1 Task 6 is going to look bad too.
"""
import time

import torch

from yaml_bert.aggregator import TreeAggregator
from yaml_bert.linearizer import YamlLinearizer
from yaml_bert.v8_dataset import V8Dataset, v8_collate_fn
from yaml_bert.vocab import VocabBuilder
from yaml_bert.config import YamlBertConfig


def _make_batch(batch_size: int = 32, d_model: int = 256):
    """Build a synthetic batch of ~32 docs with realistic K8s manifests."""
    yamls = [
        "apiVersion: apps/v1\nkind: Deployment\nmetadata:\n"
        "  name: web\n  labels:\n    app: nginx\n"
        "spec:\n  replicas: 3\n  selector:\n    matchLabels:\n      app: nginx\n"
        "  template:\n    metadata:\n      labels:\n        app: nginx\n"
        "    spec:\n      containers:\n      - name: nginx\n        image: nginx:1.25\n"
        "        ports:\n        - containerPort: 80\n",

        "apiVersion: v1\nkind: Service\nmetadata:\n"
        "  name: web\nspec:\n  selector:\n    app: nginx\n"
        "  ports:\n  - port: 80\n    targetPort: 8080\n",

        "apiVersion: v1\nkind: Pod\nmetadata:\n  name: x\n"
        "spec:\n  containers:\n  - name: c\n    image: nginx\n",

        "apiVersion: v1\nkind: ConfigMap\nmetadata:\n  name: cfg\n"
        "data:\n  config.yaml: |\n    key: value\n",
    ]
    # Repeat to reach batch_size
    yamls = (yamls * ((batch_size // len(yamls)) + 1))[:batch_size]
    docs = [YamlLinearizer().linearize(y) for y in yamls]
    flat = [n for d in docs for n in d]
    vocab = VocabBuilder().build(flat, min_freq=1)
    config = YamlBertConfig(v8_mode=True, mask_prob=0.0, d_model=d_model)
    ds = V8Dataset(docs, vocab, config)
    batch = v8_collate_fn([ds[i] for i in range(len(ds))])
    B, N = batch["token_ids"].shape
    torch.manual_seed(0)
    hidden = torch.randn(B, N, d_model)
    return hidden, batch


def test_vectorized_aggregator_is_at_least_5x_faster():
    """Vectorized path on a synthetic 32-doc batch should be ≥5× faster
    than the per-doc reference path."""
    d_model = 256
    hidden, batch = _make_batch(batch_size=32, d_model=d_model)
    agg = TreeAggregator(d_model=d_model)

    # Warmup once for each path (catches first-call compilation overhead)
    _ = agg(hidden, batch["batch_info"])
    _ = agg(
        hidden, batch["batch_info"],
        parent_of_tensor=batch["parent_of_tensor"],
        top_level_key_mask=batch["top_level_key_mask"],
        edges_by_depth=batch["edges_by_depth"],
        parents_by_depth=batch["parents_by_depth"],
    )

    n_iters = 30

    # Reference path
    t0 = time.perf_counter()
    for _ in range(n_iters):
        agg(hidden, batch["batch_info"])
    ref_time = time.perf_counter() - t0

    # Vectorized path
    t0 = time.perf_counter()
    for _ in range(n_iters):
        agg(
            hidden, batch["batch_info"],
            parent_of_tensor=batch["parent_of_tensor"],
            top_level_key_mask=batch["top_level_key_mask"],
            edges_by_depth=batch["edges_by_depth"],
            parents_by_depth=batch["parents_by_depth"],
        )
    vec_time = time.perf_counter() - t0

    speedup = ref_time / vec_time
    print(f"\nreference: {ref_time:.3f}s for {n_iters} iters "
          f"({ref_time / n_iters * 1000:.1f}ms/iter)")
    print(f"vectorized: {vec_time:.3f}s for {n_iters} iters "
          f"({vec_time / n_iters * 1000:.1f}ms/iter)")
    print(f"speedup: {speedup:.1f}x")
    assert speedup >= 5.0, (
        f"Vectorized path only {speedup:.1f}× faster — expected ≥5×. "
        f"Investigate before running GPU benchmark."
    )
```

- [ ] **Step 2: Run the perf test**

Run: `python -m pytest tests/test_aggregator_perf_smoke.py -v -s`

Expected: PASS. The `-s` flag lets the print statements show; you'll see the actual timings (useful diagnostic). If speedup < 5×, do NOT proceed to Task 6 (GPU benchmark) — debug what's slow on CPU first.

- [ ] **Step 3: Commit**

```bash
git add tests/test_aggregator_perf_smoke.py
git commit -m "test(aggregator): local microbenchmark — vectorized must be ≥5× faster"
```

---

## Task 6: Re-run the JarvisLabs Phase 0 benchmark

**Files:**
- No code; deployment + execution only

**Why:** Acceptance gate per the spec. Same setup as Phase 0 (5K docs, 10 epochs, random init, mean combine, no reconstruction head), now with the vectorized aggregator + s_parent. Target: ≥ 7 it/s, kind probe ≥ 95%.

- [ ] **Step 1: Verify v7 full-corpus training is complete (or accept running a separate instance)**

Run: `jl list --json | python3 -c "import json,sys; [print(i['machine_id'], i['status']) for i in json.load(sys.stdin)]"`

If v7's instance (`415123`) is `Running` and v7 training is done, we can reuse it. If still training, either wait or spin up a fresh L4 (~$0.55 for ~1h).

For a fresh instance:

```bash
jl create --gpu L4 --storage 100 --yes --json | tail -3
# Capture the new machine_id
jl list --json | python3 -c "import json,sys; d=json.load(sys.stdin); print(d[0]['machine_id'])"
```

- [ ] **Step 2: Bundle current main and push to the chosen instance**

```bash
git bundle create /tmp/v8-vec.bundle main
jl upload <machine_id> /tmp/v8-vec.bundle /home/v8-vec.bundle
```

If the instance has `/home/yaml-bert` already (reused v7 instance):
```bash
jl exec <machine_id> -- sh -lc 'cd /home/yaml-bert && git fetch /home/v8-vec.bundle main:vec && git merge --ff-only vec'
```

If it's a fresh instance with no yaml-bert checkout:
```bash
jl exec <machine_id> -- sh -lc 'cd /home && git clone -b main /home/v8-vec.bundle yaml-bert && cd yaml-bert && pip install -q -r requirements.txt'
```

- [ ] **Step 3: Launch the benchmark training run**

```bash
jl run --on <machine_id> --json --yes -- sh -lc \
  'cd /home/yaml-bert && PYTHONPATH=. python scripts/train_v8_phase0.py \
    --max-docs 5000 --epochs 10 --batch-size 32 \
    --output-dir output_v8_phase1_vec_seed42 --seed 42'
```

Save the returned `run_id`.

- [ ] **Step 4: Monitor and wait for completion**

Run: `jl run logs <run_id> --tail 30`

Expected: linearization (fast — uses any cached doc_cache or builds fresh) → vocab build → training loop. Total should complete in ≤ 4 min (~3.7 min projected).

If it crashes, read the error and fix the smallest issue, repeat. Common gotchas: a tensor not moved to device (would manifest as device mismatch), a typo in the kwargs.

- [ ] **Step 5: Download outputs**

```bash
mkdir -p output_v8_phase1_vec_seed42
jl download <machine_id> /home/yaml-bert/output_v8_phase1_vec_seed42 \
  ./output_v8_phase1_vec_seed42 -r
```

- [ ] **Step 6: Run kind probe locally**

```bash
PYTHONPATH=. python scripts/eval_v8_phase0.py \
  --doc-vecs output_v8_phase1_vec_seed42/doc_vecs.pt \
  --top-k-kinds 10
```

Note the probe accuracy. Acceptance gate: ≥ 95% (loose floor; Phase 0 hit 99.87%).

- [ ] **Step 7: Pause or destroy the benchmark instance**

If used a fresh instance:
```bash
jl destroy <machine_id> --yes --json
```

If reused v7's instance: leave it as-is (v7's other usage continues).

- [ ] **Step 8: No commit (execution only). Record numbers for Task 7.**

---

## Task 7: Compile results doc + go/no-go decision

**Files:**
- Create: `docs/v8-phase1-vectorization-results.md`

**Why:** Lock in the numbers and the decision. Same format as Phase 0 results so they're easy to compare side-by-side.

- [ ] **Step 1: Create the results doc**

Create `docs/v8-phase1-vectorization-results.md`:

```markdown
# v8 Phase 1 — Aggregator Vectorization Results

## Setup

- **Date:** <YYYY-MM-DD>
- **Hardware:** JarvisLabs L4 GPU (instance `<machine_id>`)
- **Training subset:** 5,000 docs from substratusai/the-stack-yaml-k8s (same as Phase 0)
- **Epochs:** 10
- **Batch size:** 32
- **Model params:** <N>
- **Changes since Phase 0:** TreeAggregator + V8Model s_parent vectorized (batched scatter ops, no Python loops). Behavior identical (locked by numerical-equivalence tests).

## Measured Metrics

| Metric | Phase 0 | Phase 1 (vectorized) | Target | Verdict |
|---|---|---|---|---|
| Training step time (it/s) | 3.6 | <X> | ≥ 7 | <PASS/FAIL> |
| Total training time (10 epochs) | 7.3 min | <T> | ≤ 4 min | <PASS/FAIL> |
| Kind probe accuracy (top 10 kinds) | 99.87% | <PCT>% | ≥ 95% | <PASS/FAIL> |
| Loss at epoch 10 | 0.82 | <L> | trending down | <PASS/FAIL> |
| Training stability | clean | <observed> | NaN-free | <PASS/FAIL> |

## Loss trajectory

(Paste per-epoch losses from training log.)

## Observations

- (Speed comparison vs v7 baseline: how close did we get to 9 it/s?)
- (Any surprises: GPU util, memory, anything unexpected during training?)
- (Equivalence-test status: did existing aggregator tests + the new equivalence test stay green throughout?)

## Decision

<GO / NO-GO for next mini-cycle>

If GO: proceed to next Phase 1 mini-cycle — reconstruction objective design.
  → Start new brainstorming session for reconstruction.

If NO-GO: speed still below threshold OR kind probe regressed.
  → Diagnose: is the bottleneck still in aggregator/s_parent, or somewhere else
    (collate overhead, encoder, head)? Consider Approach 2 (PyG/DGL) from the
    spec, or fundamentally rethink data path.
```

- [ ] **Step 2: Fill in measured values from Task 6**

Manually edit with real numbers from the JarvisLabs run + the kind probe.

- [ ] **Step 3: Add concrete decision rationale**

In the Decision section, write a paragraph tying each measurement to verdict. Don't just say "all PASS → GO" — say *why*. E.g.: "Vectorization restored speed to within X% of v7 (9 it/s baseline). Kind discrimination held at Y% (still well above 95% floor). The architectural validation from Phase 0 transfers cleanly; we can now layer reconstruction objective on a fast foundation."

- [ ] **Step 4: Commit**

```bash
git add docs/v8-phase1-vectorization-results.md
git commit -m "docs(v8): Phase 1 vectorization results + decision"
```

**Mini-cycle complete. Decide go/no-go for next Phase 1 mini-cycle (reconstruction objective).**

---

## Self-Review Notes

**Spec coverage:**
- ✓ Vectorize TreeAggregator (Task 2)
- ✓ Vectorize V8Model s_parent (Task 3)
- ✓ Precompute tensors in v8_collate_fn (Task 1)
- ✓ Numerical equivalence test (Task 2 — test_aggregator_vectorized.py; Task 4 — test_v8_smoke_e2e_vectorized_path)
- ✓ Local perf smoke (Task 5)
- ✓ Re-run Phase 0 benchmark on JarvisLabs (Task 6)
- ✓ Results report + go/no-go decision (Task 7)
- ✓ Backward-compat fallback (per spec): both aggregator and V8Model have reference path active when tensor kwargs absent

**Placeholder scan:** Only intentional placeholders are in the results-doc template (Task 7); those get filled in at execution time with measured numbers.

**Type consistency:** `parent_of_tensor`, `top_level_key_mask`, `edges_by_depth`, `parents_by_depth` — all four use the same names everywhere they appear (collate, aggregator forward signature, V8Model forward signature, training script forward call helper). `_forward_vectorized` and `_forward_reference` are the internal method names used consistently in the aggregator.
