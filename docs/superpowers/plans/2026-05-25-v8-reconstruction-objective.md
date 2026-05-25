# v8 Reconstruction Objective Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add subtree-reconstruction objective alongside MLM on the v8 architecture; train two conditions (MLM-only control vs MLM+reconstruction treatment) on the 5K-doc Phase 0/1 setup, decide go/no-go via 4 smoke-test probes.

**Architecture:** Subtree masking in `V8Dataset.__getitem__` (1-3 mutually disjoint subtrees per doc), leak-aware aggregator path (excludes masked-subtree positions from doc_vec), small MLP `ReconstructionHead` reading `[doc_vec ; pos_emb(masked_root)]` and predicting bag-of-keys via BCE. Joint loss `α·L_mlm + β·L_recon` with α=1.0, β=0.5. Backward-compat: every new pathway (aggregator's `subtree_mask`, V8Model's `recon_head`, trainer's `--reconstruction` flag) dispatches on optional kwargs so the existing 19 Phase 1 tests keep passing.

**Tech Stack:** Python 3, PyTorch (vanilla scatter ops, no new external deps), pytest, sklearn (probes only, local), the existing yaml_bert package.

---

## File Structure

**New files:**
- `yaml_bert/subtree_masking.py` — `pick_subtrees` + `descendants_of` + `bag_of_keys_target` (pure functions)
- `yaml_bert/reconstruction_head.py` — `ReconstructionHead` nn.Module
- `scripts/train_v8_phase1_recon.py` — forked trainer with `--reconstruction on|off` flag and per-epoch monitoring
- `scripts/eval_v8_probes.py` — local script: 4 sklearn probes against per-epoch doc_vec dumps
- `tests/test_subtree_masking.py` — unit tests for pick_subtrees corner cases
- `tests/test_reconstruction_head.py` — shape + backward tests
- `tests/test_v8_dataset_subtree.py` — integration test for new collate fields
- `docs/v8-phase1-reconstruction-results.md` — results doc, written at end

**Modified files:**
- `yaml_bert/v8_dataset.py` — `V8Dataset.__init__` precomputes descendants; `__getitem__` adds subtree masking + new outputs; `v8_collate_fn` batches new fields
- `yaml_bert/aggregator.py` — both forward paths accept `subtree_mask` kwarg
- `yaml_bert/v8_model.py` — owns `recon_head`; `forward` returns `recon_logits` when subtree info provided
- `yaml_bert/config.py` — `recon_enabled: bool = False`, `recon_loss_weight: float = 0.5`
- `tests/test_aggregator_vectorized.py` — extend equivalence test with `subtree_mask` path

---

## Task 1: subtree_masking module (pure functions, fully tested)

**Files:**
- Create: `yaml_bert/subtree_masking.py`
- Test: `tests/test_subtree_masking.py`

**Why:** All masking logic lives in one place as pure functions: easy to unit-test corner cases (small docs, no candidates, overlap) without involving the dataset or torch.

- [ ] **Step 1: Write the failing test file**

Create `tests/test_subtree_masking.py`:

```python
"""Unit tests for subtree_masking primitives (pure functions, no torch)."""
import random

from yaml_bert.subtree_masking import (
    descendants_of,
    pick_subtrees,
    bag_of_keys_target,
    MIN_DOC_NODES,
    MAX_SUBTREE_FRACTION,
    MAX_TOTAL_SUBTREE_FRACTION,
)


def test_descendants_of_single_leaf():
    """A KEY with no children has only itself as descendant."""
    children_of = {0: [], 1: [], 2: []}
    assert descendants_of(0, children_of) == {0}


def test_descendants_of_tree():
    """Tree shape:
       0
       ├─ 1
       │   └─ 3
       └─ 2
    """
    children_of = {0: [1, 2], 1: [3], 2: [], 3: []}
    assert descendants_of(0, children_of) == {0, 1, 2, 3}
    assert descendants_of(1, children_of) == {1, 3}
    assert descendants_of(2, children_of) == {2}


def test_pick_subtrees_returns_empty_for_tiny_doc():
    """Doc smaller than MIN_DOC_NODES → no picks."""
    rng = random.Random(0)
    picks = pick_subtrees(
        N=5,  # below threshold
        key_positions=[0, 1, 2],
        depth_of=[0, 1, 1],
        children_of={0: [1, 2], 1: [], 2: []},
        mlm_masked_positions=set(),
        rng=rng,
    )
    assert picks == []


def test_pick_subtrees_returns_empty_when_no_candidates():
    """Doc large enough but no depth>=1 KEY with children → empty."""
    rng = random.Random(0)
    # Only the root has children; depth-0 is excluded
    picks = pick_subtrees(
        N=20,
        key_positions=[0, 1, 2],
        depth_of=[0, 1, 1],
        children_of={0: [1, 2], 1: [], 2: []},  # 1 and 2 are leaves
        mlm_masked_positions=set(),
        rng=rng,
    )
    assert picks == []


def test_pick_subtrees_basic_pick():
    """Doc with a valid depth>=1 KEY with children → at least one pick."""
    # Tree:  0 (root, depth 0)
    #        ├── 1 (depth 1, has children 3, 4)
    #        │      ├── 3 (leaf)
    #        │      └── 4 (leaf)
    #        └── 2 (depth 1, leaf)
    # plus filler nodes 5..14 to satisfy MIN_DOC_NODES=10
    N = 15
    rng = random.Random(0)
    picks = pick_subtrees(
        N=N,
        key_positions=[0, 1, 2, 3, 4],
        depth_of={0: 0, 1: 1, 2: 1, 3: 2, 4: 2},
        children_of={0: [1, 2], 1: [3, 4], 2: [], 3: [], 4: []},
        mlm_masked_positions=set(),
        rng=rng,
    )
    # Only candidate is position 1 (depth>=1, has children). Subtree size 3
    # = 20% of 15, below MAX_SUBTREE_FRACTION=0.30. Should pick it.
    assert picks == [1]


def test_pick_subtrees_excludes_too_large_subtrees():
    """A subtree larger than MAX_SUBTREE_FRACTION * N is excluded."""
    # Tree: 0 root, 1 depth-1 with 8 children (positions 2-9).
    # Subtree at 1 = {1,2,3,4,5,6,7,8,9} = 9 nodes. N=10. 9/10 = 90% > 30%.
    rng = random.Random(0)
    picks = pick_subtrees(
        N=10,
        key_positions=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        depth_of={i: (0 if i == 0 else (1 if i == 1 else 2)) for i in range(10)},
        children_of={0: [1], 1: [2, 3, 4, 5, 6, 7, 8, 9],
                     2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: []},
        mlm_masked_positions=set(),
        rng=rng,
    )
    assert picks == []  # only candidate (1) exceeds size cap


def test_pick_subtrees_excludes_mlm_overlap():
    """Candidate whose subtree overlaps MLM mask is excluded."""
    rng = random.Random(0)
    picks = pick_subtrees(
        N=15,
        key_positions=[0, 1, 2, 3, 4],
        depth_of={0: 0, 1: 1, 2: 1, 3: 2, 4: 2},
        children_of={0: [1, 2], 1: [3, 4], 2: [], 3: [], 4: []},
        mlm_masked_positions={3},  # position 3 inside subtree-at-1
        rng=rng,
    )
    assert picks == []  # subtree-at-1 contains 3, which is MLM-masked


def test_pick_subtrees_disjoint_picks():
    """Multiple picks are mutually disjoint."""
    # Tree:   0 root
    #         ├── 1 (subtree {1,3,4})
    #         └── 2 (subtree {2,5,6})
    # plus filler 7..16 to meet MIN_DOC_NODES and not blow size caps
    rng = random.Random(0)
    picks = pick_subtrees(
        N=50,  # large so size cap 5%=2.5 nodes won't limit much; total cap is 5%
        key_positions=[0, 1, 2, 3, 4, 5, 6],
        depth_of={0: 0, 1: 1, 2: 1, 3: 2, 4: 2, 5: 2, 6: 2},
        children_of={0: [1, 2], 1: [3, 4], 2: [5, 6],
                     3: [], 4: [], 5: [], 6: []},
        mlm_masked_positions=set(),
        rng=rng,
    )
    # Subtree at 1 = 3 nodes; at 2 = 3 nodes. Total 6 / 50 = 12% > 5%, so only ONE picks.
    # But: total cap 5% × 50 = 2.5, so even 3-node subtree exceeds → 0 picks
    # Let's pick N=100 so 5% = 5, each subtree of 3 fits, both together = 6 > 5 still.
    # The implementation should pick the FIRST candidate then skip the second.
    # Re-run with N=200 to give more headroom:
    picks = pick_subtrees(
        N=200,
        key_positions=[0, 1, 2, 3, 4, 5, 6],
        depth_of={0: 0, 1: 1, 2: 1, 3: 2, 4: 2, 5: 2, 6: 2},
        children_of={0: [1, 2], 1: [3, 4], 2: [5, 6],
                     3: [], 4: [], 5: [], 6: []},
        mlm_masked_positions=set(),
        rng=rng,
    )
    # Total cap 5% × 200 = 10. Both subtrees (3+3=6) fit. Picks should be disjoint.
    assert set(picks).issubset({1, 2})
    # If both picked, they must be {1, 2} — not contain each other.
    if len(picks) == 2:
        assert set(picks) == {1, 2}


def test_pick_subtrees_respects_total_cap():
    """Sum of subtree sizes across picks ≤ MAX_TOTAL_SUBTREE_FRACTION * N."""
    rng = random.Random(0)
    picks = pick_subtrees(
        N=20,  # 5% cap = 1 node → no single subtree of ≥2 nodes fits in total cap
        key_positions=[0, 1, 2, 3],
        depth_of={0: 0, 1: 1, 2: 1, 3: 2},
        children_of={0: [1, 2], 1: [3], 2: [], 3: []},
        mlm_masked_positions=set(),
        rng=rng,
    )
    # Only candidate: pos 1, subtree {1, 3} = 2 nodes. Total cap = 1.0. Skip.
    assert picks == []


def test_bag_of_keys_target_basic():
    """Multi-hot vector with 1s at atomic-vocab indices of keys in subtree."""
    # Subtree includes positions 1, 2, 3 with key strings 'name', 'image', 'name'
    # vocab maps name=10, image=12
    atomic_vocab = {"name": 10, "image": 12, "ports": 14}
    position_to_key_str = {1: "name", 2: "image", 3: "name"}
    target = bag_of_keys_target(
        subtree_positions={1, 2, 3},
        position_to_key_str=position_to_key_str,
        atomic_vocab=atomic_vocab,
        vocab_size=20,
    )
    assert target.shape == (20,)
    assert target[10].item() == 1.0  # name present (deduped)
    assert target[12].item() == 1.0  # image present
    assert target[14].item() == 0.0  # ports not present
    assert target.sum().item() == 2.0  # exactly two distinct keys


def test_bag_of_keys_target_skips_unknown_keys():
    """Keys not in atomic_vocab are silently skipped."""
    atomic_vocab = {"name": 10}
    position_to_key_str = {1: "name", 2: "obscure_key_not_in_vocab"}
    target = bag_of_keys_target(
        subtree_positions={1, 2},
        position_to_key_str=position_to_key_str,
        atomic_vocab=atomic_vocab,
        vocab_size=20,
    )
    assert target[10].item() == 1.0
    assert target.sum().item() == 1.0
```

- [ ] **Step 2: Run tests to confirm they all fail**

Run: `python -m pytest tests/test_subtree_masking.py -v`
Expected: ImportError or all 9 tests FAIL (module doesn't exist yet).

- [ ] **Step 3: Implement the module**

Create `yaml_bert/subtree_masking.py`:

```python
"""Subtree-masking primitives for the v8 reconstruction objective.

Pure functions (no torch in the masking logic itself; targets returned as
torch tensors). All randomness flows through a caller-provided random.Random
instance for reproducibility.
"""
from __future__ import annotations

import random
from typing import Mapping, Sequence

import torch


MIN_DOC_NODES = 10
MAX_SUBTREE_FRACTION = 0.30  # no single subtree may exceed this fraction of doc
MAX_TOTAL_SUBTREE_FRACTION = 0.05  # sum of all picked subtrees ≤ this fraction


def descendants_of(pos: int, children_of: Mapping[int, Sequence[int]]) -> set[int]:
    """DFS descendant set of `pos`, including pos itself."""
    out = {pos}
    stack = list(children_of.get(pos, []))
    while stack:
        p = stack.pop()
        out.add(p)
        stack.extend(children_of.get(p, []))
    return out


def pick_subtrees(
    N: int,
    key_positions: Sequence[int],
    depth_of: Mapping[int, int],
    children_of: Mapping[int, Sequence[int]],
    mlm_masked_positions: set[int],
    rng: random.Random,
    descendants_cache: Mapping[int, set[int]] | None = None,
) -> list[int]:
    """Pick 1-3 mutually-disjoint subtree root positions (or [] if none qualify).

    Args:
        N: total node count in the doc.
        key_positions: positions that are KEY/LIST_KEY.
        depth_of: per-position depth lookup.
        children_of: per-position children list.
        mlm_masked_positions: positions already masked by token-level MLM.
        rng: random source.
        descendants_cache: optional precomputed descendants per position
            (avoids redoing DFS on every call). If None, computed on the fly.

    Returns:
        List of 1-3 root positions, mutually-disjoint, total size within cap,
        or [] if doc too small / no valid candidates / cap too tight.
    """
    if N < MIN_DOC_NODES:
        return []

    def _descs(p: int) -> set[int]:
        if descendants_cache is not None and p in descendants_cache:
            return descendants_cache[p]
        return descendants_of(p, children_of)

    candidates: list[tuple[int, set[int]]] = []
    for kp in key_positions:
        if depth_of[kp] < 1:
            continue
        if not children_of.get(kp):
            continue
        descs = _descs(kp)
        if len(descs) > MAX_SUBTREE_FRACTION * N:
            continue
        if descs & mlm_masked_positions:
            continue
        candidates.append((kp, descs))

    if not candidates:
        return []

    rng.shuffle(candidates)
    num_to_pick = rng.randint(1, min(3, len(candidates)))

    picked: list[int] = []
    picked_positions: set[int] = set()
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


def bag_of_keys_target(
    subtree_positions: set[int],
    position_to_key_str: Mapping[int, str],
    atomic_vocab: Mapping[str, int],
    vocab_size: int,
) -> torch.Tensor:
    """Build a multi-hot bag-of-keys target for one masked subtree.

    Args:
        subtree_positions: all positions inside the subtree (root + descendants).
        position_to_key_str: per-position key string lookup (positions not in this
            map are skipped — e.g., VALUE positions don't contribute key tokens).
        atomic_vocab: atomic key string → vocab index.
        vocab_size: V_atomic, size of the output multi-hot vector.

    Returns:
        Float tensor of shape (vocab_size,). target[v]=1.0 iff vocab index v
        appears as a key inside the subtree. De-duplicated (true bag-of-keys,
        not bag-of-counts).
    """
    target = torch.zeros(vocab_size, dtype=torch.float)
    seen: set[int] = set()
    for pos in subtree_positions:
        key_str = position_to_key_str.get(pos)
        if key_str is None:
            continue
        vocab_idx = atomic_vocab.get(key_str)
        if vocab_idx is None:
            continue
        if vocab_idx in seen:
            continue
        target[vocab_idx] = 1.0
        seen.add(vocab_idx)
    return target
```

- [ ] **Step 4: Run tests to confirm all pass**

Run: `python -m pytest tests/test_subtree_masking.py -v`
Expected: 9/9 PASS.

- [ ] **Step 5: Commit**

```bash
git add yaml_bert/subtree_masking.py tests/test_subtree_masking.py
git commit -m "feat(v8): subtree_masking module (pick_subtrees, descendants_of, bag_of_keys_target)"
```

---

## Task 2: V8Dataset + v8_collate_fn subtree integration

**Files:**
- Modify: `yaml_bert/v8_dataset.py`
- Test: `tests/test_v8_dataset_subtree.py` (new)

**Why:** Wire `pick_subtrees` into `V8Dataset.__getitem__` to produce subtree-masking outputs, and extend `v8_collate_fn` to batch them. `V8Dataset.__init__` precomputes descendant sets per doc (cached) so per-call cost is just the picker.

- [ ] **Step 1: Write the failing integration test**

Create `tests/test_v8_dataset_subtree.py`:

```python
"""Integration test: V8Dataset emits subtree-masking outputs; v8_collate_fn batches them."""
import torch

from yaml_bert.linearizer import YamlLinearizer
from yaml_bert.config import YamlBertConfig
from yaml_bert.vocab import VocabBuilder
from yaml_bert.v8_dataset import V8Dataset, v8_collate_fn


def _build_dataset_and_vocab(yamls: list[str], recon_enabled: bool):
    docs = [YamlLinearizer().linearize(y) for y in yamls]
    flat = [n for d in docs for n in d]
    vocab = VocabBuilder().build(flat, min_freq=1)
    config = YamlBertConfig(v8_mode=True, mask_prob=0.0,
                            recon_enabled=recon_enabled)
    return V8Dataset(docs, vocab, config), vocab


def test_v8_dataset_item_includes_subtree_fields_when_recon_enabled():
    """When recon_enabled=True, items carry subtree_mask + subtree_roots
    + bag_of_keys_targets."""
    yamls = [
        "apiVersion: apps/v1\nkind: Deployment\nmetadata:\n  name: web\n"
        "spec:\n  replicas: 3\n  template:\n    spec:\n      containers:\n"
        "      - name: nginx\n        image: nginx:1.25\n        ports:\n"
        "        - containerPort: 80\n",
    ]
    ds, _ = _build_dataset_and_vocab(yamls, recon_enabled=True)
    item = ds[0]

    assert "subtree_mask" in item
    assert item["subtree_mask"].dtype == torch.bool
    assert item["subtree_mask"].shape[0] == item["token_ids"].shape[0]

    assert "subtree_roots" in item
    assert isinstance(item["subtree_roots"], list)
    assert all(isinstance(r, int) for r in item["subtree_roots"])

    assert "bag_of_keys_targets" in item
    # If subtree_roots is non-empty, each root has a multi-hot target of length V_atomic
    if item["subtree_roots"]:
        assert len(item["bag_of_keys_targets"]) == len(item["subtree_roots"])
        for target in item["bag_of_keys_targets"]:
            assert target.dtype == torch.float
            assert target.dim() == 1


def test_v8_dataset_item_omits_subtree_fields_when_recon_disabled():
    """When recon_enabled=False, no subtree-related fields appear in item."""
    yamls = ["apiVersion: v1\nkind: Pod\nspec:\n  x: 1\n"]
    ds, _ = _build_dataset_and_vocab(yamls, recon_enabled=False)
    item = ds[0]
    assert "subtree_mask" not in item
    assert "subtree_roots" not in item
    assert "bag_of_keys_targets" not in item


def test_v8_collate_batches_subtree_fields_when_present():
    """v8_collate_fn batches per-item subtree fields into (B,N) + flat (M,*)."""
    yamls = [
        "apiVersion: apps/v1\nkind: Deployment\nmetadata:\n  name: a\n"
        "spec:\n  template:\n    spec:\n      containers:\n      - name: x\n",
        "apiVersion: v1\nkind: Pod\nmetadata:\n  name: b\n"
        "spec:\n  containers:\n  - name: y\n        image: nginx\n",
    ]
    ds, vocab = _build_dataset_and_vocab(yamls, recon_enabled=True)
    batch = v8_collate_fn([ds[0], ds[1]])

    assert "subtree_mask" in batch
    assert batch["subtree_mask"].dim() == 2  # (B, N)
    assert batch["subtree_mask"].shape[0] == 2
    assert batch["subtree_mask"].dtype == torch.bool

    assert "subtree_roots_flat" in batch
    sr = batch["subtree_roots_flat"]
    assert sr.dtype == torch.long
    assert sr.dim() == 2 and sr.shape[1] == 2  # (M, 2) of [batch_idx, root_pos]

    assert "bag_of_keys_targets_flat" in batch
    bot = batch["bag_of_keys_targets_flat"]
    assert bot.dtype == torch.float
    assert bot.dim() == 2  # (M, V_atomic)
    assert bot.shape[0] == sr.shape[0]
    assert bot.shape[1] == vocab.atomic_target_vocab_size


def test_v8_collate_omits_subtree_fields_when_absent():
    """If items don't carry subtree fields, neither does the batched dict."""
    yamls = ["apiVersion: v1\nkind: Pod\nspec:\n  x: 1\n",
             "apiVersion: v1\nkind: Service\n"]
    ds, _ = _build_dataset_and_vocab(yamls, recon_enabled=False)
    batch = v8_collate_fn([ds[0], ds[1]])
    assert "subtree_mask" not in batch
    assert "subtree_roots_flat" not in batch
    assert "bag_of_keys_targets_flat" not in batch
```

- [ ] **Step 2: Run tests to confirm they fail**

Run: `python -m pytest tests/test_v8_dataset_subtree.py -v`
Expected: 4 FAIL (config doesn't have recon_enabled; dataset doesn't add subtree fields).

- [ ] **Step 3: Add config flags**

In `yaml_bert/config.py`, find the `YamlBertConfig` dataclass and add two new fields. After the existing `v8_mode: bool = False` line, add:

```python
    recon_enabled: bool = False
    recon_loss_weight: float = 0.5
```

- [ ] **Step 4: Extend `V8Dataset.__init__` to precompute descendants per doc**

Edit `yaml_bert/v8_dataset.py`. Find the `V8Dataset.__init__` method. Replace it with:

```python
    def __init__(
        self,
        documents: list[list[YamlNode]],
        vocab: Vocabulary,
        config: YamlBertConfig,
    ) -> None:
        self.documents = documents
        self.vocab = vocab
        self.mask_prob = config.mask_prob
        self.max_seq_len = config.max_seq_len
        self.recon_enabled = config.recon_enabled

        # Precompute per-doc children_info AND descendant cache (subtree picker
        # uses descendants on every __getitem__ call; cache once at init).
        self._cached_children_info: list[dict] = []
        self._cached_descendants: list[dict[int, set[int]] | None] = []
        for doc in documents:
            ci = compute_children_info(doc[: self.max_seq_len])
            self._cached_children_info.append(ci)
            if self.recon_enabled:
                # Precompute descendants for every KEY-with-children
                from yaml_bert.subtree_masking import descendants_of
                desc_cache: dict[int, set[int]] = {}
                for kp in ci["key_positions"]:
                    if ci["children_of"][kp]:
                        desc_cache[kp] = descendants_of(kp, ci["children_of"])
                self._cached_descendants.append(desc_cache)
            else:
                self._cached_descendants.append(None)
```

(Note: `compute_children_info` returns dicts/lists keyed by position; `children_of[kp]` works whether children_of is a dict or list-of-lists. Check the file — it's a list-of-lists.)

- [ ] **Step 5: Extend `V8Dataset.__getitem__` to emit subtree fields when recon_enabled**

Edit `yaml_bert/v8_dataset.py` `V8Dataset.__getitem__`. Find the existing `return { ... }` block (around line 163). Replace the entire method body with:

```python
    def __getitem__(self, idx: int) -> dict:
        nodes = self.documents[idx]
        if len(nodes) > self.max_seq_len:
            nodes = nodes[: self.max_seq_len]

        token_ids: list[int] = []
        node_types: list[int] = []
        depths: list[int] = []
        sibling_indices: list[int] = []

        for node in nodes:
            if node.node_type in (NodeType.KEY, NodeType.LIST_KEY):
                token_ids.append(self.vocab.encode_key(node.token))
            else:
                token_ids.append(self.vocab.encode_value(node.token))
            node_types.append(_NODE_TYPE_INDEX[node.node_type])
            depths.append(min(node.depth, 15))
            sibling_indices.append(min(node.sibling_index, 31))

        atomic_labels: list[int] = [-100] * len(nodes)
        mask_id: int = self.vocab.special_tokens["[MASK]"]
        unk_id: int = self.vocab.special_tokens["[UNK]"]

        mlm_masked_positions: set[int] = set()
        for i, node in enumerate(nodes):
            if node.node_type not in _MASKABLE_TYPES:
                continue
            if random.random() >= self.mask_prob:
                continue
            atomic_id = self.vocab.encode_atomic_target(node.token)
            if atomic_id == unk_id:
                continue
            atomic_labels[i] = atomic_id
            mlm_masked_positions.add(i)
            r = random.random()
            if r < 0.8:
                token_ids[i] = mask_id
            elif r < 0.9:
                token_ids[i] = random.randint(
                    len(self.vocab.special_tokens),
                    len(self.vocab.key_vocab) + len(self.vocab.special_tokens) - 1,
                )

        result = {
            "token_ids": torch.tensor(token_ids, dtype=torch.long),
            "node_types": torch.tensor(node_types, dtype=torch.long),
            "depths": torch.tensor(depths, dtype=torch.long),
            "sibling_indices": torch.tensor(sibling_indices, dtype=torch.long),
            "atomic_labels": torch.tensor(atomic_labels, dtype=torch.long),
            "children_info": self._cached_children_info[idx],
        }

        if self.recon_enabled:
            from yaml_bert.subtree_masking import pick_subtrees, bag_of_keys_target
            ci = self._cached_children_info[idx]
            picked_roots = pick_subtrees(
                N=len(nodes),
                key_positions=ci["key_positions"],
                depth_of=ci["depth_of"],
                children_of=ci["children_of"],
                mlm_masked_positions=mlm_masked_positions,
                rng=random,  # use module-level random for parity with MLM masking
                descendants_cache=self._cached_descendants[idx],
            )
            # Build subtree_mask + apply [MASK] to subtree positions
            subtree_mask = torch.zeros(len(nodes), dtype=torch.bool)
            picked_positions_all: set[int] = set()
            bag_targets: list[torch.Tensor] = []
            # position → key string lookup for bag-of-keys building
            position_to_key_str = {
                i: nodes[i].token
                for i in range(len(nodes))
                if nodes[i].node_type in (NodeType.KEY, NodeType.LIST_KEY)
            }
            for root_pos in picked_roots:
                descs = self._cached_descendants[idx][root_pos]
                picked_positions_all |= descs
                bag_targets.append(bag_of_keys_target(
                    subtree_positions=descs,
                    position_to_key_str=position_to_key_str,
                    atomic_vocab=self.vocab.atomic_target_vocab,
                    vocab_size=self.vocab.atomic_target_vocab_size,
                ))
            for pos in picked_positions_all:
                subtree_mask[pos] = True
                token_ids[pos] = mask_id
            # Re-encode token_ids tensor since we mutated the list
            result["token_ids"] = torch.tensor(token_ids, dtype=torch.long)
            result["subtree_mask"] = subtree_mask
            result["subtree_roots"] = picked_roots  # list[int]
            result["bag_of_keys_targets"] = bag_targets  # list[Tensor (V,)]

        return result
```

- [ ] **Step 6: Extend `v8_collate_fn` to batch the new fields**

Edit `yaml_bert/v8_dataset.py` `v8_collate_fn`. Find the end of the function, just before `return result`. Add this block:

```python
    # Subtree-mask batching (only present when recon is enabled per-item).
    if "subtree_mask" in batch[0]:
        # subtree_mask: (B, N) bool, pad with False
        subtree_masks: list[torch.Tensor] = []
        for item in batch:
            sm = item["subtree_mask"]
            pad_len = max_len - sm.size(0)
            if pad_len > 0:
                subtree_masks.append(torch.cat([
                    sm, torch.zeros(pad_len, dtype=torch.bool),
                ]))
            else:
                subtree_masks.append(sm)
        result["subtree_mask"] = torch.stack(subtree_masks)

        # subtree_roots_flat: (M, 2) of [batch_idx, root_pos]; M = total roots
        # bag_of_keys_targets_flat: (M, V_atomic)
        flat_roots: list[tuple[int, int]] = []
        flat_targets: list[torch.Tensor] = []
        for b_idx, item in enumerate(batch):
            for root_pos, target in zip(
                item["subtree_roots"], item["bag_of_keys_targets"]
            ):
                flat_roots.append((b_idx, root_pos))
                flat_targets.append(target)
        if flat_roots:
            result["subtree_roots_flat"] = torch.tensor(
                flat_roots, dtype=torch.long,
            )
            result["bag_of_keys_targets_flat"] = torch.stack(flat_targets)
        else:
            # Even with recon enabled, all docs in this batch may have empty
            # picks (small docs, no candidates). Emit empty tensors so the
            # consumer can detect "no subtrees this batch."
            result["subtree_roots_flat"] = torch.zeros(
                (0, 2), dtype=torch.long,
            )
            v = batch[0]["bag_of_keys_targets"][0].size(0) if (
                batch[0]["bag_of_keys_targets"]
            ) else 0
            result["bag_of_keys_targets_flat"] = torch.zeros(
                (0, v), dtype=torch.float,
            )
```

- [ ] **Step 7: Run the new test plus existing v8_dataset tests for regression**

Run: `python -m pytest tests/test_v8_dataset.py tests/test_v8_dataset_subtree.py -v`
Expected: all PASS (9 existing + 4 new = 13).

- [ ] **Step 8: Commit**

```bash
git add yaml_bert/config.py yaml_bert/v8_dataset.py tests/test_v8_dataset_subtree.py
git commit -m "feat(v8): V8Dataset + collate emit subtree-mask outputs when recon_enabled"
```

---

## Task 3: Leak-aware aggregator path

**Files:**
- Modify: `yaml_bert/aggregator.py`
- Test: `tests/test_aggregator_vectorized.py` (extend with a subtree-mask equivalence test)

**Why:** When `subtree_mask` is provided, the aggregator must exclude masked-subtree positions from sums into `doc_vec` and ancestor `subtree_vecs`. Both the reference path (per-doc loop) and the vectorized path (scatter ops) need to honor this. A new equivalence test locks the two paths together under the new kwarg.

- [ ] **Step 1: Write the failing equivalence test**

Append to `tests/test_aggregator_vectorized.py`:

```python
def test_vectorized_aggregator_with_subtree_mask_equals_reference():
    """Vectorized path with subtree_mask matches reference path with same mask."""
    from yaml_bert.aggregator import TreeAggregator
    from yaml_bert.linearizer import YamlLinearizer
    from yaml_bert.v8_dataset import V8Dataset, v8_collate_fn
    from yaml_bert.vocab import VocabBuilder
    from yaml_bert.config import YamlBertConfig

    docs = [
        YamlLinearizer().linearize(
            "apiVersion: apps/v1\nkind: Deployment\nspec:\n"
            "  replicas: 3\n  template:\n    spec:\n      containers:\n"
            "      - name: x\n        image: nginx\n"),
        YamlLinearizer().linearize(
            "apiVersion: v1\nkind: Pod\nmetadata:\n  name: y\n"
            "spec:\n  containers:\n  - name: z\n"),
    ]
    vocab = VocabBuilder().build([n for d in docs for n in d], min_freq=1)
    config = YamlBertConfig(v8_mode=True, mask_prob=0.0, d_model=16,
                            recon_enabled=True)
    ds = V8Dataset(docs, vocab, config)
    batch = v8_collate_fn([ds[0], ds[1]])
    # Synthesize a subtree_mask manually so the test doesn't depend on
    # random picker state. Mask all depth>=2 positions in doc 0.
    sm = batch["subtree_mask"].clone()
    # Force a known mask: cover position 3 in doc 0 (some inner position)
    if sm.shape[1] > 3:
        sm[0, 3] = True

    B, N = batch["token_ids"].shape
    d_model = 16
    torch.manual_seed(0)
    hidden = torch.randn(B, N, d_model)

    agg = TreeAggregator(d_model=d_model)

    # Reference path with subtree_mask
    ref_subtree, ref_doc = agg(hidden, batch["batch_info"], subtree_mask=sm)

    # Vectorized path with subtree_mask
    vec_subtree, vec_doc = agg(
        hidden, batch["batch_info"],
        parent_of_tensor=batch["parent_of_tensor"],
        top_level_key_mask=batch["top_level_key_mask"],
        edges_by_depth=batch["edges_by_depth"],
        parents_by_depth=batch["parents_by_depth"],
        subtree_mask=sm,
    )

    assert torch.allclose(ref_subtree, vec_subtree, atol=1e-6), (
        f"subtree mismatch: max diff = "
        f"{(ref_subtree - vec_subtree).abs().max().item()}"
    )
    assert torch.allclose(ref_doc, vec_doc, atol=1e-6)


def test_aggregator_subtree_mask_excludes_positions_from_doc_vec():
    """A subtree_mask covering a top-level key removes it from doc_vec mean."""
    from yaml_bert.aggregator import TreeAggregator
    from yaml_bert.linearizer import YamlLinearizer
    from yaml_bert.v8_dataset import V8Dataset, v8_collate_fn
    from yaml_bert.vocab import VocabBuilder
    from yaml_bert.config import YamlBertConfig

    docs = [
        YamlLinearizer().linearize(
            "apiVersion: v1\nkind: Pod\nspec:\n  x: 1\n  y: 2\n"),
    ]
    vocab = VocabBuilder().build([n for d in docs for n in d], min_freq=1)
    config = YamlBertConfig(v8_mode=True, mask_prob=0.0, d_model=8,
                            recon_enabled=True)
    ds = V8Dataset(docs, vocab, config)
    batch = v8_collate_fn([ds[0]])

    # Find the position of "spec" (depth-0 KEY)
    info = batch["batch_info"][0]
    spec_pos = next(
        kp for kp in info["key_positions"]
        if info["depth_of"][kp] == 0 and info["full_path_of"][kp] == "spec"
    )

    B, N = batch["token_ids"].shape
    torch.manual_seed(0)
    hidden = torch.randn(B, N, 8)

    # No-mask baseline
    agg = TreeAggregator(d_model=8)
    _, doc_no_mask = agg(hidden, batch["batch_info"])

    # Mask the entire spec subtree (spec + x + y)
    sm = torch.zeros((B, N), dtype=torch.bool)
    descendants = {spec_pos}
    for child in info["children_of"][spec_pos]:
        descendants.add(child)
    for pos in descendants:
        sm[0, pos] = True

    _, doc_with_mask = agg(
        hidden, batch["batch_info"],
        parent_of_tensor=batch["parent_of_tensor"],
        top_level_key_mask=batch["top_level_key_mask"],
        edges_by_depth=batch["edges_by_depth"],
        parents_by_depth=batch["parents_by_depth"],
        subtree_mask=sm,
    )
    # The two doc_vecs should differ — masking out spec changes the mean
    assert not torch.allclose(doc_no_mask, doc_with_mask, atol=1e-5)
```

- [ ] **Step 2: Run tests to confirm they fail**

Run: `python -m pytest tests/test_aggregator_vectorized.py -v -k "subtree_mask"`
Expected: 2 FAIL (`subtree_mask` not a recognized kwarg).

- [ ] **Step 3: Extend the aggregator forward to accept subtree_mask**

Edit `yaml_bert/aggregator.py`. Find the `forward` method's signature. Replace it with:

```python
    def forward(
        self,
        hidden_states: torch.Tensor,
        batch_info: list[dict],
        *,
        parent_of_tensor: torch.Tensor | None = None,
        top_level_key_mask: torch.Tensor | None = None,
        edges_by_depth: dict[int, torch.Tensor] | None = None,
        parents_by_depth: dict[int, torch.Tensor] | None = None,
        subtree_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            ... (existing)
            subtree_mask: (B, N) bool tensor. When provided, positions where
                subtree_mask[b, i]=True are excluded from sums into doc_vec
                and from contributing to ancestor subtree_vecs. Used for the
                v8 reconstruction objective's leak-prevention.

        Returns:
            (subtree_vecs, doc_vec)
        """
        provided = (
            parent_of_tensor is not None,
            top_level_key_mask is not None,
            edges_by_depth is not None,
            parents_by_depth is not None,
        )
        if any(provided):
            if not all(provided):
                raise ValueError(
                    "TreeAggregator.forward: vectorized kwargs must be passed "
                    "all-or-none. Got: "
                    f"parent_of_tensor={'set' if provided[0] else 'None'}, "
                    f"top_level_key_mask={'set' if provided[1] else 'None'}, "
                    f"edges_by_depth={'set' if provided[2] else 'None'}, "
                    f"parents_by_depth={'set' if provided[3] else 'None'}"
                )
            return self._forward_vectorized(
                hidden_states,
                top_level_key_mask=top_level_key_mask,
                edges_by_depth=edges_by_depth,
                parents_by_depth=parents_by_depth,
                subtree_mask=subtree_mask,
            )
        return self._forward_reference(
            hidden_states, batch_info, subtree_mask=subtree_mask,
        )
```

- [ ] **Step 4: Update `_forward_reference` to honor subtree_mask**

Edit `_forward_reference` to accept and apply `subtree_mask`. Replace it with:

```python
    def _forward_reference(
        self,
        hidden_states: torch.Tensor,
        batch_info: list[dict],
        *,
        subtree_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-document Python loop. Original Phase 0 implementation,
        extended with leak-aware subtree_mask exclusion."""
        b, n, d = hidden_states.shape
        subtree_vecs = hidden_states.clone()
        doc_vec = torch.zeros(b, d, device=hidden_states.device,
                              dtype=hidden_states.dtype)

        def is_masked(doc_idx: int, pos: int) -> bool:
            if subtree_mask is None:
                return False
            return bool(subtree_mask[doc_idx, pos].item())

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
                    if is_masked(doc_idx, parent_pos):
                        # Masked root: keep its hidden state as-is, don't aggregate
                        continue
                    children = [
                        c for c in children_of[parent_pos]
                        if not is_masked(doc_idx, c)
                    ]
                    if children:
                        child_vecs = subtree_vecs[doc_idx, children]
                        own = hidden_states[doc_idx, parent_pos:parent_pos + 1]
                        combined = torch.cat(
                            [own, child_vecs], dim=0,
                        ).mean(dim=0)
                    else:
                        combined = hidden_states[doc_idx, parent_pos]
                    subtree_vecs[doc_idx, parent_pos] = combined

            top_level = [
                kp for kp in keys_by_depth.get(0, [])
                if not is_masked(doc_idx, kp)
            ]
            if top_level:
                doc_vec[doc_idx] = subtree_vecs[doc_idx, top_level].mean(dim=0)

        return subtree_vecs, doc_vec
```

- [ ] **Step 5: Update `_forward_vectorized` to honor subtree_mask**

Edit `_forward_vectorized` to filter edges + top-level positions by the mask. Replace it with:

```python
    def _forward_vectorized(
        self,
        hidden_states: torch.Tensor,
        *,
        top_level_key_mask: torch.Tensor,
        edges_by_depth: dict[int, torch.Tensor],
        parents_by_depth: dict[int, torch.Tensor],
        subtree_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Batched scatter-based path with leak-aware subtree_mask filtering."""
        B, N, d = hidden_states.shape
        subtree_vecs = hidden_states.clone()

        for depth in sorted(edges_by_depth.keys(), reverse=True):
            edges = edges_by_depth[depth].to(hidden_states.device)
            parents = parents_by_depth[depth].to(hidden_states.device)

            doc_idx_e = edges[:, 0]
            child_pos = edges[:, 1]
            parent_pos_e = edges[:, 2]

            # Filter edges where either endpoint is in a masked subtree
            if subtree_mask is not None:
                sm = subtree_mask.to(hidden_states.device)
                keep_edge = ~(sm[doc_idx_e, child_pos] | sm[doc_idx_e, parent_pos_e])
                doc_idx_e = doc_idx_e[keep_edge]
                child_pos = child_pos[keep_edge]
                parent_pos_e = parent_pos_e[keep_edge]

            child_vecs = subtree_vecs[doc_idx_e, child_pos]
            parent_linear_e = doc_idx_e * N + parent_pos_e

            sum_acc = torch.zeros(
                B * N, d,
                dtype=hidden_states.dtype, device=hidden_states.device,
            )
            sum_acc.index_add_(0, parent_linear_e, child_vecs)

            count_acc = torch.zeros(
                B * N, dtype=torch.float32,
                device=hidden_states.device,
            )
            count_acc.index_add_(
                0, parent_linear_e,
                torch.ones_like(parent_linear_e, dtype=torch.float32),
            )

            # Filter parents: skip those that are themselves masked
            parent_doc_idx = parents[:, 0]
            parent_pos_p = parents[:, 1]
            if subtree_mask is not None:
                keep_parent = ~subtree_mask.to(hidden_states.device)[
                    parent_doc_idx, parent_pos_p
                ]
                parent_doc_idx = parent_doc_idx[keep_parent]
                parent_pos_p = parent_pos_p[keep_parent]

            parent_linear_p = parent_doc_idx * N + parent_pos_p

            sum_at_parents = sum_acc[parent_linear_p]
            count_at_parents = count_acc[parent_linear_p].to(hidden_states.dtype)
            own_at_parents = hidden_states[parent_doc_idx, parent_pos_p]

            mean_at_parents = (sum_at_parents + own_at_parents) / (
                count_at_parents.unsqueeze(-1) + 1.0
            )

            subtree_vecs[parent_doc_idx, parent_pos_p] = mean_at_parents

        # doc_vec: masked positions excluded from top-level mean
        effective_top_level = top_level_key_mask
        if subtree_mask is not None:
            effective_top_level = top_level_key_mask & (~subtree_mask.to(
                hidden_states.device
            ))

        mask_f = effective_top_level.to(hidden_states.dtype).unsqueeze(-1)
        weighted = subtree_vecs * mask_f
        sum_per_doc = weighted.sum(dim=1)
        count_per_doc = effective_top_level.sum(
            dim=1, dtype=torch.float32,
        ).clamp(min=1).to(hidden_states.dtype).unsqueeze(-1)
        doc_vec = sum_per_doc / count_per_doc

        return subtree_vecs, doc_vec
```

- [ ] **Step 6: Run the full aggregator test suite + the new tests**

Run: `python -m pytest tests/test_aggregator.py tests/test_aggregator_vectorized.py tests/test_aggregator_perf_smoke.py -v`
Expected: 8 PASS (3 + 4 + 1, including 2 new subtree_mask tests).

- [ ] **Step 7: Commit**

```bash
git add yaml_bert/aggregator.py tests/test_aggregator_vectorized.py
git commit -m "feat(aggregator): leak-aware subtree_mask path in both reference + vectorized"
```

---

## Task 4: ReconstructionHead module

**Files:**
- Create: `yaml_bert/reconstruction_head.py`
- Test: `tests/test_reconstruction_head.py`

**Why:** Standalone small MLP module. Tested in isolation (shapes + backward) before integration into V8Model. Pure module, no surrounding context needed.

- [ ] **Step 1: Write the failing test file**

Create `tests/test_reconstruction_head.py`:

```python
"""ReconstructionHead unit tests: shapes + backward."""
import torch

from yaml_bert.reconstruction_head import ReconstructionHead


def test_reconstruction_head_output_shape():
    """Output has shape (M, V_atomic)."""
    head = ReconstructionHead(d_model=64, d_pos=16, atomic_vocab_size=100)
    M = 7
    doc_vec = torch.randn(M, 64)
    pos_emb = torch.randn(M, 16)
    logits = head(doc_vec, pos_emb)
    assert logits.shape == (M, 100)


def test_reconstruction_head_backward_produces_gradients():
    """Backward through BCE loss produces finite gradients on all params."""
    head = ReconstructionHead(d_model=32, d_pos=8, atomic_vocab_size=50)
    M = 4
    doc_vec = torch.randn(M, 32, requires_grad=True)
    pos_emb = torch.randn(M, 8, requires_grad=True)
    target = (torch.rand(M, 50) > 0.5).float()

    logits = head(doc_vec, pos_emb)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
    loss.backward()

    assert torch.isfinite(loss)
    for name, p in head.named_parameters():
        assert p.grad is not None, f"no grad on {name}"
        assert torch.isfinite(p.grad).all(), f"non-finite grad on {name}"
    assert doc_vec.grad is not None
    assert torch.isfinite(doc_vec.grad).all()


def test_reconstruction_head_param_count_is_small():
    """Sanity: head is ~205K params at default sizes (d_model=256, d_pos=48, V=427)."""
    head = ReconstructionHead(d_model=256, d_pos=48, atomic_vocab_size=427)
    n = sum(p.numel() for p in head.parameters())
    # Linear(304→256) = 304*256 + 256 = 78,080
    # Linear(256→427) = 256*427 + 427 = 109,739
    # Total = 187,819 (close to ~205K target)
    assert 180_000 < n < 220_000, f"unexpected param count: {n}"
```

- [ ] **Step 2: Run tests to confirm they fail**

Run: `python -m pytest tests/test_reconstruction_head.py -v`
Expected: ImportError (module doesn't exist).

- [ ] **Step 3: Implement the module**

Create `yaml_bert/reconstruction_head.py`:

```python
"""Reconstruction Head: predict bag of atomic keys in a masked subtree.

Reads [doc_vec ; pos_emb(masked_root)] and outputs BCE logits over the atomic
key vocabulary. The position embedding is intentionally only depth + sibling
(not the root's key identity) so the head must use doc_vec to disambiguate
which keys are in the subtree — this is the pretraining pressure on doc_vec.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ReconstructionHead(nn.Module):
    def __init__(self, d_model: int, d_pos: int, atomic_vocab_size: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model + d_pos, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, atomic_vocab_size),
        )

    def forward(
        self,
        doc_vec_per_subtree: torch.Tensor,
        pos_emb_per_subtree: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            doc_vec_per_subtree: (M, d_model) — doc_vec repeated per masked
                subtree in the batch (M = total subtrees across batch).
            pos_emb_per_subtree: (M, d_pos) — [depth_emb ; sibling_emb] of
                each masked subtree's root.
        Returns:
            (M, atomic_vocab_size) logits — pass to BCE-with-logits loss.
        """
        return self.mlp(torch.cat(
            [doc_vec_per_subtree, pos_emb_per_subtree], dim=-1,
        ))
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_reconstruction_head.py -v`
Expected: 3/3 PASS.

- [ ] **Step 5: Commit**

```bash
git add yaml_bert/reconstruction_head.py tests/test_reconstruction_head.py
git commit -m "feat(v8): ReconstructionHead MLP module + tests"
```

---

## Task 5: V8Model integration of reconstruction head

**Files:**
- Modify: `yaml_bert/v8_model.py`
- Test: `tests/test_v8_model_e2e.py` (append a test)

**Why:** Wire the head into V8Model; `forward` returns `(logits, doc_vec, recon_logits)` when subtree info is provided, `(logits, doc_vec)` otherwise. Build the per-root position embedding inside `forward` using the existing depth+sibling embeddings.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_v8_model_e2e.py`:

```python
def test_v8_model_returns_recon_logits_when_subtree_info_provided():
    """V8Model.forward returns recon_logits with shape (M, V_atomic) when
    the batch contains subtree_roots_flat (i.e., recon is active)."""
    from yaml_bert.linearizer import YamlLinearizer
    from yaml_bert.vocab import VocabBuilder
    from yaml_bert.config import YamlBertConfig
    from yaml_bert.embedding import YamlBertEmbedding
    from yaml_bert.v8_dataset import V8Dataset, v8_collate_fn

    yamls = [
        "apiVersion: apps/v1\nkind: Deployment\nmetadata:\n  name: a\n"
        "spec:\n  replicas: 3\n  template:\n    spec:\n      containers:\n"
        "      - name: x\n        image: nginx\n        ports:\n"
        "        - containerPort: 80\n",
        "apiVersion: v1\nkind: Pod\nmetadata:\n  name: b\n"
        "spec:\n  containers:\n  - name: y\n        image: nginx\n",
    ]
    docs = [YamlLinearizer().linearize(y) for y in yamls]
    flat = [n for doc in docs for n in doc]
    vocab = VocabBuilder().build(flat, min_freq=1)
    config = YamlBertConfig(d_model=32, num_layers=2, num_heads=4,
                            v8_mode=True, mask_prob=0.5, recon_enabled=True)
    ds = V8Dataset(docs, vocab, config)
    batch = v8_collate_fn([ds[i] for i in range(len(ds))])
    if batch["subtree_roots_flat"].size(0) == 0:
        # Unlikely on these docs but possible — re-seed and retry once
        import random
        random.seed(7)
        batch = v8_collate_fn([ds[i] for i in range(len(ds))])

    emb = YamlBertEmbedding(config=config,
                            key_vocab_size=vocab.key_vocab_size,
                            value_vocab_size=vocab.value_vocab_size)
    model = V8Model(config=config, embedding=emb,
                    atomic_vocab_size=vocab.atomic_target_vocab_size)
    model.train()

    out = model(
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
        subtree_mask=batch["subtree_mask"],
        subtree_roots_flat=batch["subtree_roots_flat"],
    )
    assert len(out) == 3, "expected (logits, doc_vec, recon_logits) tuple"
    logits, doc_vec, recon_logits = out

    M = batch["subtree_roots_flat"].size(0)
    if M > 0:
        assert recon_logits.shape == (M, vocab.atomic_target_vocab_size)
        # Verify recon loss flows back through doc_vec
        target = batch["bag_of_keys_targets_flat"]
        recon_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            recon_logits, target,
        )
        recon_loss.backward()
        # Doc_vec is internal; check at least the head's params got gradients
        for name, p in model.recon_head.named_parameters():
            assert p.grad is not None
            assert torch.isfinite(p.grad).all()


def test_v8_model_omits_recon_logits_when_no_subtree_info():
    """V8Model.forward returns (logits, doc_vec) — old shape — when subtree
    kwargs are absent. Backward-compat with Phase 0/1 callers."""
    from yaml_bert.linearizer import YamlLinearizer
    from yaml_bert.vocab import VocabBuilder
    from yaml_bert.config import YamlBertConfig
    from yaml_bert.embedding import YamlBertEmbedding
    from yaml_bert.v8_dataset import V8Dataset, v8_collate_fn

    yamls = ["apiVersion: v1\nkind: Pod\nspec:\n  x: 1\n"]
    docs = [YamlLinearizer().linearize(y) for y in yamls]
    flat = [n for doc in docs for n in doc]
    vocab = VocabBuilder().build(flat, min_freq=1)
    # recon_enabled=False → dataset omits subtree fields → forward omits recon
    config = YamlBertConfig(d_model=16, num_layers=1, num_heads=2,
                            v8_mode=True, mask_prob=0.0, recon_enabled=False)
    ds = V8Dataset(docs, vocab, config)
    batch = v8_collate_fn([ds[0]])

    emb = YamlBertEmbedding(config=config,
                            key_vocab_size=vocab.key_vocab_size,
                            value_vocab_size=vocab.value_vocab_size)
    model = V8Model(config=config, embedding=emb,
                    atomic_vocab_size=vocab.atomic_target_vocab_size)
    model.eval()
    out = model(
        token_ids=batch["token_ids"],
        node_types=batch["node_types"],
        depths=batch["depths"],
        sibling_indices=batch["sibling_indices"],
        batch_info=batch["batch_info"],
        padding_mask=batch["padding_mask"],
    )
    assert len(out) == 2  # (logits, doc_vec) — no recon
```

- [ ] **Step 2: Run the test, expect FAIL**

Run: `python -m pytest tests/test_v8_model_e2e.py -v -k "recon_logits or omits_recon"`
Expected: 2 FAIL — V8Model doesn't have `recon_head` or accept `subtree_roots_flat`.

- [ ] **Step 3: Add `recon_head` to V8Model and extend forward**

Edit `yaml_bert/v8_model.py`. Find the `V8Model.__init__` method. After the existing initialization of `token_head`, add:

```python
        # Reconstruction Head: built unconditionally; only USED when caller
        # passes subtree_roots_flat. Cost when unused: ~0 (no forward call).
        # Position embedding dimension = d_depth + d_sibling
        d_pos = config.depth_emb_dim + config.sibling_emb_dim
        from yaml_bert.reconstruction_head import ReconstructionHead
        self.recon_head = ReconstructionHead(
            d_model=config.d_model,
            d_pos=d_pos,
            atomic_vocab_size=atomic_vocab_size,
        )
```

(If `config.depth_emb_dim` and `config.sibling_emb_dim` aren't already on `YamlBertConfig`, check `yaml_bert/config.py` and `yaml_bert/embedding.py` to find the names actually used. The embedding module's `__init__` will tell you. If they're called something else like `depth_dim`, use that name.)

Then find `V8Model.forward` and replace its signature + body with:

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
        subtree_mask: torch.Tensor | None = None,
        subtree_roots_flat: torch.Tensor | None = None,
    ) -> tuple:
        """Returns (logits, doc_vec) or (logits, doc_vec, recon_logits).

        recon_logits only returned when subtree_roots_flat is provided AND
        has at least one row."""
        x = self.embedding(token_ids, node_types, depths, sibling_indices)
        x = self.encoder(x, src_key_padding_mask=padding_mask)

        subtree_vecs, doc_vec = self.aggregator(
            x, batch_info,
            parent_of_tensor=parent_of_tensor,
            top_level_key_mask=top_level_key_mask,
            edges_by_depth=edges_by_depth,
            parents_by_depth=parents_by_depth,
            subtree_mask=subtree_mask,
        )

        b, n, d = x.shape

        if parent_of_tensor is not None:
            # Vectorized s_parent. parent_of_tensor being set implies all four
            # precompute kwargs were provided (aggregator enforces all-or-none).
            safe_parent = parent_of_tensor.clamp(min=0)
            s_parent = torch.gather(
                subtree_vecs, dim=1,
                index=safe_parent.unsqueeze(-1).expand(-1, -1, d),
            )
            no_parent_mask = (parent_of_tensor == -1).unsqueeze(-1)
            s_parent = torch.where(
                no_parent_mask, doc_vec.unsqueeze(1), s_parent,
            )
        else:
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

        # Reconstruction path: only if caller provided subtree roots
        if subtree_roots_flat is not None and subtree_roots_flat.size(0) > 0:
            # Build per-root inputs
            # subtree_roots_flat: (M, 2) of [batch_idx, root_pos]
            M = subtree_roots_flat.size(0)
            batch_idx_per_root = subtree_roots_flat[:, 0]   # (M,)
            root_pos_per_root = subtree_roots_flat[:, 1]    # (M,)

            doc_vec_per_root = doc_vec[batch_idx_per_root]  # (M, d_model)

            # Build pos_emb_per_root from existing depth + sibling embeddings.
            # Use the embedding module's depth_embedding / sibling_embedding
            # parameters to keep training pressure on the same embedding params.
            root_depths = depths[batch_idx_per_root, root_pos_per_root]  # (M,)
            root_siblings = sibling_indices[
                batch_idx_per_root, root_pos_per_root
            ]  # (M,)
            depth_e = self.embedding.depth_embedding(root_depths)       # (M, d_depth)
            sibling_e = self.embedding.sibling_embedding(root_siblings)  # (M, d_sibling)
            pos_emb_per_root = torch.cat([depth_e, sibling_e], dim=-1)  # (M, d_pos)

            recon_logits = self.recon_head(doc_vec_per_root, pos_emb_per_root)
            return logits, doc_vec, recon_logits

        return logits, doc_vec
```

NOTE: the `self.embedding.depth_embedding` / `self.embedding.sibling_embedding` attribute names must match what `YamlBertEmbedding` actually uses internally. Read `yaml_bert/embedding.py` to confirm. If they're named differently (e.g., `self.depth_emb`), adjust the two lookups accordingly. If the embedding module doesn't expose them as named submodules, expose them — modify `embedding.py` to store them as named attributes.

- [ ] **Step 4: Run the new tests + all v8 tests for regression**

Run: `python -m pytest tests/test_v8_model_e2e.py tests/test_aggregator.py tests/test_aggregator_vectorized.py tests/test_v8_dataset.py tests/test_v8_dataset_subtree.py tests/test_reconstruction_head.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add yaml_bert/v8_model.py tests/test_v8_model_e2e.py
git commit -m "feat(v8): V8Model.forward returns recon_logits when subtree info provided"
```

---

## Task 6: Forked trainer with --reconstruction flag + per-epoch monitoring

**Files:**
- Create: `scripts/train_v8_phase1_recon.py`
- (No new tests — trainer scripts are validated by smoke-run at the end of this task)

**Why:** Trainer is forked rather than extending Phase 0/1's trainer so the prior reproducibility stays clean. The new trainer adds: `--reconstruction on|off` flag, per-epoch separate loss logging, held-out val split (4500/500), per-epoch `doc_vecs_epoch_<N>.pt` dumps.

- [ ] **Step 1: Create the new trainer script**

Create `scripts/train_v8_phase1_recon.py`:

```python
"""v8 Phase 1 reconstruction benchmark: train on 5K-doc subset, MLM-only OR
MLM+reconstruction. Per-epoch loss + val + doc_vec dumps for probe trajectory.
"""
from __future__ import annotations
import _setup_path  # noqa: F401

import argparse
import os
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from yaml_bert.cache import build_or_load_cache
from yaml_bert.config import YamlBertConfig
from yaml_bert.embedding import YamlBertEmbedding
from yaml_bert.v8_dataset import V8Dataset, v8_collate_fn
from yaml_bert.v8_model import V8Model
from yaml_bert.vocab import VocabBuilder

DATASET_NAME = "substratusai/the-stack-yaml-k8s"


def _forward_v8(model, batch, device, recon_enabled: bool):
    """Forward V8Model. Returns (logits, doc_vec, recon_logits|None)."""
    kwargs = dict(
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
    if recon_enabled and "subtree_mask" in batch:
        kwargs["subtree_mask"] = batch["subtree_mask"].to(device)
        kwargs["subtree_roots_flat"] = batch["subtree_roots_flat"].to(device)
        out = model(**kwargs)
        if len(out) == 3:
            return out  # (logits, doc_vec, recon_logits)
        return (*out, None)  # (logits, doc_vec, None) — no subtrees this batch
    out = model(**kwargs)
    return (*out, None)


def _compute_losses(out, batch, device, recon_enabled: bool, recon_weight: float):
    logits, _, recon_logits = out
    mlm_loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        batch["atomic_labels"].to(device).view(-1),
        ignore_index=-100,
    )
    if not recon_enabled or recon_logits is None:
        return mlm_loss, mlm_loss, torch.tensor(0.0, device=device)
    recon_target = batch["bag_of_keys_targets_flat"].to(device)
    recon_loss = F.binary_cross_entropy_with_logits(recon_logits, recon_target)
    total = mlm_loss + recon_weight * recon_loss
    return total, mlm_loss, recon_loss


def _dump_doc_vecs(model, dataset, batch_size, device, recon_enabled,
                   output_path, cached, num_workers):
    """One pass over the FULL 5K corpus dumping doc_vecs to disk."""
    from yaml_bert.dataset import _extract_kind
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                       collate_fn=v8_collate_fn, num_workers=num_workers)
    doc_vecs: list[torch.Tensor] = []
    doc_kinds: list[str] = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            _, dvec, _ = _forward_v8(model, batch, device, recon_enabled)
            doc_vecs.append(dvec.cpu())
            for j in range(dvec.size(0)):
                gi = batch_idx * batch_size + j
                if gi < len(cached):
                    doc_kinds.append(_extract_kind(cached[gi]))
    torch.save({
        "doc_vecs": torch.cat(doc_vecs, dim=0),
        "kinds": doc_kinds,
    }, output_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-docs", type=int, default=5000)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reconstruction", choices=["on", "off"], default="off")
    parser.add_argument("--recon-weight", type=float, default=0.5)
    args = parser.parse_args()

    recon_enabled = args.reconstruction == "on"

    import random
    import numpy as np
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    cache_path = os.path.join(args.output_dir, "doc_cache.pkl")
    print(f"Step 0: Linearize → {cache_path}")
    cached = build_or_load_cache(DATASET_NAME, cache_path=cache_path,
                                 max_docs=args.max_docs)

    print("Step 1: Build vocab (v8 mode — atomic targets)")
    all_nodes = [n for doc in cached for n in doc]
    vocab = VocabBuilder().build(
        all_nodes,
        key_min_freq=10,
        value_min_freq=10,
        simple_target_min_freq=5,
        kind_target_min_freq=2,
    )
    vocab.save(os.path.join(args.output_dir, "vocab.json"))
    print(f"  key vocab: {vocab.key_vocab_size}")
    print(f"  atomic vocab: {vocab.atomic_target_vocab_size}")
    print(f"  reconstruction: {args.reconstruction} (weight={args.recon_weight})")

    print("Step 2: Build dataset (train: 4500, val: 500)")
    config = YamlBertConfig(num_epochs=args.epochs, batch_size=args.batch_size,
                            v8_mode=True, recon_enabled=recon_enabled,
                            recon_loss_weight=args.recon_weight)
    full_dataset = V8Dataset(cached, vocab, config)
    train_indices = list(range(len(cached) - 500))
    val_indices = list(range(len(cached) - 500, len(cached)))
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)

    print("Step 3: Build model")
    emb = YamlBertEmbedding(config=config,
                            key_vocab_size=vocab.key_vocab_size,
                            value_vocab_size=vocab.value_vocab_size)
    model = V8Model(config=config, embedding=emb,
                    atomic_vocab_size=vocab.atomic_target_vocab_size)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  device: {device}")
    print(f"  params: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)

    num_workers = min(8, max(2, (os.cpu_count() or 4) // 2))
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, collate_fn=v8_collate_fn,
                              num_workers=num_workers, persistent_workers=True,
                              pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, collate_fn=v8_collate_fn,
                            num_workers=num_workers)

    print("Step 4: Training")
    train_start = time.time()
    epoch_log: list[dict] = []
    for epoch in range(args.epochs):
        epoch_start = time.time()
        model.train()
        sums = {"total": 0.0, "mlm": 0.0, "recon": 0.0}
        n_batches = 0
        for batch in train_loader:
            optimizer.zero_grad()
            out = _forward_v8(model, batch, device, recon_enabled)
            total_loss, mlm_loss, recon_loss = _compute_losses(
                out, batch, device, recon_enabled, args.recon_weight,
            )
            total_loss.backward()
            optimizer.step()
            sums["total"] += total_loss.item()
            sums["mlm"] += mlm_loss.item()
            sums["recon"] += recon_loss.item()
            n_batches += 1
            if not torch.isfinite(total_loss):
                print(f"  !! NaN/Inf loss at batch {n_batches}; stopping early")
                return

        avg = {k: v / max(1, n_batches) for k, v in sums.items()}

        # Validation pass (no_grad)
        model.eval()
        val_sums = {"total": 0.0, "mlm": 0.0, "recon": 0.0}
        n_val = 0
        with torch.no_grad():
            for vb in val_loader:
                out = _forward_v8(model, vb, device, recon_enabled)
                tl, ml, rl = _compute_losses(
                    out, vb, device, recon_enabled, args.recon_weight,
                )
                val_sums["total"] += tl.item()
                val_sums["mlm"] += ml.item()
                val_sums["recon"] += rl.item()
                n_val += 1
        val_avg = {k: v / max(1, n_val) for k, v in val_sums.items()}

        epoch_dur = time.time() - epoch_start
        print(
            f"  Epoch {epoch+1}/{args.epochs} — "
            f"train total {avg['total']:.4f} mlm {avg['mlm']:.4f} "
            f"recon {avg['recon']:.4f}  |  "
            f"val total {val_avg['total']:.4f} mlm {val_avg['mlm']:.4f} "
            f"recon {val_avg['recon']:.4f}  "
            f"({n_batches} batches, {epoch_dur:.1f}s, "
            f"{n_batches/epoch_dur:.2f} it/s)"
        )
        epoch_log.append({"epoch": epoch + 1, "train": avg, "val": val_avg,
                          "dur_sec": epoch_dur, "n_batches": n_batches})

        # Per-epoch doc_vec dump (full 5K corpus, for probe trajectory)
        dump_path = os.path.join(
            args.output_dir, f"doc_vecs_epoch_{epoch+1}.pt",
        )
        _dump_doc_vecs(model, full_dataset, args.batch_size, device,
                       recon_enabled, dump_path, cached, num_workers)

    total_dur = time.time() - train_start
    print(f"Step 5: Save final checkpoint")
    ckpt_path = os.path.join(args.output_dir, "v8_phase1_recon.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "epoch_log": epoch_log,
        "n_params": n_params,
        "total_train_sec": total_dur,
        "reconstruction": args.reconstruction,
        "recon_weight": args.recon_weight,
    }, ckpt_path)

    print(f"Done. Total: {total_dur:.1f}s")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-run the trainer locally with --reconstruction off (50 docs, 1 epoch)**

Run: `PYTHONPATH=. python scripts/train_v8_phase1_recon.py --max-docs 50 --epochs 1 --batch-size 4 --output-dir /tmp/v8_recon_off_smoke --reconstruction off`

Expected: completes; loss prints with mlm/recon split (recon=0.0 since disabled); doc_vecs_epoch_1.pt saved; final checkpoint saved.

- [ ] **Step 3: Smoke-run the trainer locally with --reconstruction on (50 docs, 1 epoch)**

Run: `PYTHONPATH=. python scripts/train_v8_phase1_recon.py --max-docs 50 --epochs 1 --batch-size 4 --output-dir /tmp/v8_recon_on_smoke --reconstruction on`

Expected: completes; loss prints with both mlm and recon non-zero; doc_vecs_epoch_1.pt saved.

If either smoke crashes, fix the smallest issue (likely an attribute name mismatch like `depth_embedding` vs `depth_emb` in V8Model's pos_emb construction) and retry.

- [ ] **Step 4: Run full v8 test suite for regression**

Run: `python -m pytest tests/test_aggregator.py tests/test_aggregator_vectorized.py tests/test_v8_dataset.py tests/test_v8_dataset_subtree.py tests/test_v8_model_e2e.py tests/test_subtree_masking.py tests/test_reconstruction_head.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/train_v8_phase1_recon.py
git commit -m "feat(v8): train_v8_phase1_recon.py — forked trainer with --reconstruction flag and per-epoch val + doc_vec dumps"
```

---

## Task 7: eval_v8_probes script — 4 smoke probes

**Files:**
- Create: `scripts/eval_v8_probes.py`
- (No new unit test — validated by running against Phase 1's existing doc_vecs as a fixture)

**Why:** Local probe evaluation: read per-epoch doc_vec dumps, build labels from the raw YAML corpus, fit a `LogisticRegression` per probe, print a per-epoch trajectory table.

- [ ] **Step 1: Create the script**

Create `scripts/eval_v8_probes.py`:

```python
"""Run 4 smoke-test probes on doc_vec dumps from train_v8_phase1_recon.py.

Reads doc_vecs_epoch_<N>.pt files + the raw doc_cache.pkl, builds labels
from parsed YAML, fits sklearn LogisticRegression per probe, prints a
trajectory table.
"""
from __future__ import annotations
import _setup_path  # noqa: F401

import argparse
import os
import pickle
from collections import Counter

import numpy as np
import torch
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from yaml_bert.dataset import _extract_kind


def _parse_yaml_safe(yaml_text: str) -> dict:
    try:
        out = yaml.safe_load(yaml_text)
        return out if isinstance(out, dict) else {}
    except Exception:
        return {}


def _get_workload_containers(parsed: dict) -> list[dict]:
    """Return containers list from either Pod (spec.containers) or
    Deployment-like (spec.template.spec.containers)."""
    spec = parsed.get("spec") or {}
    direct = spec.get("containers")
    if isinstance(direct, list):
        return direct
    tmpl = (spec.get("template") or {}).get("spec") or {}
    nested = tmpl.get("containers")
    if isinstance(nested, list):
        return nested
    return []


def _get_init_containers(parsed: dict) -> list[dict]:
    spec = parsed.get("spec") or {}
    direct = spec.get("initContainers")
    if isinstance(direct, list):
        return direct
    tmpl = (spec.get("template") or {}).get("spec") or {}
    nested = tmpl.get("initContainers")
    if isinstance(nested, list):
        return nested
    return []


def _label_has_containers(parsed: dict) -> bool:
    return bool(_get_workload_containers(parsed))


def _label_has_init_containers(parsed: dict) -> bool:
    return bool(_get_init_containers(parsed))


def _label_has_volume_mounts(parsed: dict) -> bool:
    for c in _get_workload_containers(parsed):
        vm = c.get("volumeMounts") if isinstance(c, dict) else None
        if isinstance(vm, list) and vm:
            return True
    return False


def _build_labels(yaml_texts: list[str], top_k_kinds: int = 10):
    """Build label arrays for the 4 probes from raw YAML texts.

    Returns dict:
        - kind: int labels (or -1 if outside top-K) and the kind name list
        - has_containers, has_init_containers, has_volume_mounts: bool labels
    """
    parsed_docs = [_parse_yaml_safe(t) for t in yaml_texts]
    kinds = [_extract_kind_from_dict(d) for d in parsed_docs]
    counter = Counter(k for k in kinds if k is not None)
    top_kinds = [k for k, _ in counter.most_common(top_k_kinds)]
    kind_to_idx = {k: i for i, k in enumerate(top_kinds)}
    kind_labels = np.array(
        [kind_to_idx.get(k, -1) for k in kinds], dtype=int,
    )
    return {
        "kind_labels": kind_labels,
        "kind_names": top_kinds,
        "has_containers": np.array(
            [_label_has_containers(d) for d in parsed_docs], dtype=int,
        ),
        "has_init_containers": np.array(
            [_label_has_init_containers(d) for d in parsed_docs], dtype=int,
        ),
        "has_volume_mounts": np.array(
            [_label_has_volume_mounts(d) for d in parsed_docs], dtype=int,
        ),
    }


def _extract_kind_from_dict(d: dict) -> str | None:
    k = d.get("kind") if isinstance(d, dict) else None
    return k if isinstance(k, str) else None


def _probe_accuracy(X: np.ndarray, y: np.ndarray, multi_class: bool = False,
                    label_filter: np.ndarray | None = None) -> float:
    """Fit LogisticRegression on 80% of (X, y), report accuracy on 20%."""
    if label_filter is not None:
        keep = label_filter
        X = X[keep]
        y = y[keep]
    if len(np.unique(y)) < 2:
        return float("nan")
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y,
    )
    clf = LogisticRegression(max_iter=2000)
    clf.fit(X_tr, y_tr)
    return float(clf.score(X_te, y_te))


def _eval_one_dump(doc_vecs_path: str, labels: dict) -> dict:
    """Run all 4 probes on one doc_vec dump file."""
    data = torch.load(doc_vecs_path, map_location="cpu", weights_only=False)
    X = data["doc_vecs"].numpy()  # (D, d_model)
    # Trim labels to match X (in case dataset was smaller)
    n = X.shape[0]
    kind_mask = labels["kind_labels"][:n] >= 0
    return {
        "kind": _probe_accuracy(
            X, labels["kind_labels"][:n],
            multi_class=True, label_filter=kind_mask,
        ),
        "has_containers": _probe_accuracy(X, labels["has_containers"][:n]),
        "has_init_containers": _probe_accuracy(
            X, labels["has_init_containers"][:n]),
        "has_volume_mounts": _probe_accuracy(
            X, labels["has_volume_mounts"][:n]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True,
                       help="dir containing doc_vecs_epoch_*.pt and doc_cache.pkl")
    parser.add_argument("--top-k-kinds", type=int, default=10)
    args = parser.parse_args()

    cache_path = os.path.join(args.output_dir, "doc_cache.pkl")
    with open(cache_path, "rb") as f:
        cached = pickle.load(f)
    yaml_texts = [doc[0].raw_text if hasattr(doc[0], "raw_text") else "" for doc in cached]
    # Alternate: rebuild yaml texts from the linearized doc — but if raw_text
    # isn't carried, we need to source them differently. Cleanest: re-download
    # the dataset slice. For 5K docs this is fast (~10s).
    if not any(yaml_texts):
        from datasets import load_dataset
        print("doc_cache doesn't carry raw_text; re-fetching from HF dataset")
        ds = load_dataset("substratusai/the-stack-yaml-k8s",
                          split="train", streaming=False)
        yaml_texts = [ds[i]["content"] for i in range(len(cached))]

    print(f"Building labels for {len(yaml_texts)} docs...")
    labels = _build_labels(yaml_texts, top_k_kinds=args.top_k_kinds)
    print(f"Top kinds: {labels['kind_names']}")
    print(f"Counts: containers={labels['has_containers'].sum()}, "
          f"init={labels['has_init_containers'].sum()}, "
          f"vol_mounts={labels['has_volume_mounts'].sum()}")

    # Find all per-epoch dumps
    dumps = sorted([
        f for f in os.listdir(args.output_dir)
        if f.startswith("doc_vecs_epoch_") and f.endswith(".pt")
    ], key=lambda f: int(f.split("_")[3].split(".")[0]))
    if not dumps:
        print(f"No per-epoch dumps in {args.output_dir}; trying doc_vecs.pt")
        dumps = ["doc_vecs.pt"] if os.path.exists(
            os.path.join(args.output_dir, "doc_vecs.pt")
        ) else []

    print(f"\n{'epoch':>6} | {'kind':>8} | {'containers':>10} "
          f"| {'init':>8} | {'vol_mounts':>10}")
    print("-" * 60)
    for fn in dumps:
        epoch_n = (
            int(fn.split("_")[3].split(".")[0])
            if "epoch_" in fn else "final"
        )
        results = _eval_one_dump(os.path.join(args.output_dir, fn), labels)
        print(
            f"{str(epoch_n):>6} | "
            f"{results['kind']*100:>7.2f}% | "
            f"{results['has_containers']*100:>9.2f}% | "
            f"{results['has_init_containers']*100:>7.2f}% | "
            f"{results['has_volume_mounts']*100:>9.2f}%"
        )


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Validate the script against the Phase 1 vectorization output (already local)**

Run: `PYTHONPATH=. python scripts/eval_v8_probes.py --output-dir output_v8_phase1_vec_seed42`

Expected: prints a single-row table (Phase 1 saved only a final doc_vecs.pt). Kind probe should hit ~99-100% (consistent with Phase 0/1 result). has-containers, has-init, has-volume-mounts: report whatever they are — this is the smoke test.

If the script fails, fix the smallest issue (probably an attribute name on cached docs — read `yaml_bert/cache.py` and `yaml_bert/types.py` to see what's actually stored) and retry.

- [ ] **Step 3: Commit**

```bash
git add scripts/eval_v8_probes.py
git commit -m "feat(v8): eval_v8_probes.py — local 4-probe smoke evaluation"
```

---

## Task 8: JarvisLabs benchmark + results doc

**Files:**
- Create: `docs/v8-phase1-reconstruction-results.md`

**Why:** Two-condition comparison on fresh L4. Acceptance gate per the spec. Results doc captures comparison table + go/no-go decision.

- [ ] **Step 1: Create a fresh L4 instance**

Run: `jl create --gpu L4 --storage 100 --template pytorch --yes --json | tail -30`

Capture `machine_id` from the JSON.

- [ ] **Step 2: Bundle current main + push to instance + install deps**

```bash
git bundle create /tmp/v8-recon.bundle main
jl upload <machine_id> /tmp/v8-recon.bundle /home/v8-recon.bundle
jl exec <machine_id> -- sh -lc 'cd /home && git clone -b main /home/v8-recon.bundle yaml-bert && cd yaml-bert && pip install -q -r requirements.txt'
```

Verify HEAD commit landed:

```bash
jl exec <machine_id> -- sh -lc 'cd /home/yaml-bert && git log --oneline -3'
```

- [ ] **Step 3: Launch the control run (MLM-only)**

```bash
jl run --on <machine_id> --json --yes -- sh -lc \
  'cd /home/yaml-bert && PYTHONPATH=. python scripts/train_v8_phase1_recon.py \
    --max-docs 5000 --epochs 10 --batch-size 32 --seed 42 \
    --reconstruction off \
    --output-dir output_v8_phase1_control'
```

Capture `run_id`. Poll until done:

```bash
until jl run status <run_id> --json 2>/dev/null | python3 -c "import sys,json; d=json.load(sys.stdin); s=d.get('state','unknown'); print(s,file=sys.stderr); sys.exit(0 if s in ('succeeded','failed','exited','cancelled') else 1)"; do sleep 30; done; jl run logs <run_id> --tail 60
```

Expected: ~2.5 min wall time. Per-epoch lines should show mlm=non-zero, recon=0.0.

- [ ] **Step 4: Launch the treatment run (MLM+reconstruction)**

```bash
jl run --on <machine_id> --json --yes -- sh -lc \
  'cd /home/yaml-bert && PYTHONPATH=. python scripts/train_v8_phase1_recon.py \
    --max-docs 5000 --epochs 10 --batch-size 32 --seed 42 \
    --reconstruction on \
    --output-dir output_v8_phase1_treatment'
```

Poll until done (same pattern).

Expected: ~3 min wall time. Per-epoch lines show mlm + recon both non-zero, recon trending down.

- [ ] **Step 5: Download both output directories**

```bash
mkdir -p output_v8_phase1_control output_v8_phase1_treatment
jl download <machine_id> /home/yaml-bert/output_v8_phase1_control \
  ./output_v8_phase1_control -r
jl download <machine_id> /home/yaml-bert/output_v8_phase1_treatment \
  ./output_v8_phase1_treatment -r
```

If downloads land nested (Phase 1 vec mini-cycle had this), flatten:

```bash
mv output_v8_phase1_control/output_v8_phase1_control/* output_v8_phase1_control/ 2>/dev/null
rmdir output_v8_phase1_control/output_v8_phase1_control 2>/dev/null
mv output_v8_phase1_treatment/output_v8_phase1_treatment/* output_v8_phase1_treatment/ 2>/dev/null
rmdir output_v8_phase1_treatment/output_v8_phase1_treatment 2>/dev/null
```

- [ ] **Step 6: Destroy the JL instance**

```bash
jl destroy <machine_id> --yes --json
```

- [ ] **Step 7: Run probes locally on both outputs**

```bash
PYTHONPATH=. python scripts/eval_v8_probes.py --output-dir output_v8_phase1_control \
  > /tmp/probes_control.txt
PYTHONPATH=. python scripts/eval_v8_probes.py --output-dir output_v8_phase1_treatment \
  > /tmp/probes_treatment.txt
cat /tmp/probes_control.txt
echo "---"
cat /tmp/probes_treatment.txt
```

- [ ] **Step 8: Write the results doc**

Create `docs/v8-phase1-reconstruction-results.md`:

```markdown
# v8 Phase 1 — Reconstruction Objective Results

## Setup

- **Date:** <YYYY-MM-DD>
- **Hardware:** JarvisLabs L4 GPU (instance `<machine_id>`, destroyed after benchmark)
- **Training subset:** 5,000 docs from `substratusai/the-stack-yaml-k8s`
- **Epochs:** 10
- **Batch size:** 32
- **Train/val split:** 4500/500 deterministic by index
- **Model params (both conditions):** <N> total; recon head adds ~205K
- **Conditions:**
  - Control: MLM-only (`--reconstruction off`)
  - Treatment: MLM + reconstruction with α=1.0, β=0.5 (`--reconstruction on`)
- **Both conditions:** same seed=42, same train/val split, same probe labels

## Per-epoch loss curves

Control (MLM-only):
```
<paste epoch-by-epoch train+val table from run log>
```

Treatment (MLM+recon):
```
<paste epoch-by-epoch train+val table from run log>
```

## Probe accuracies — final epoch (epoch 10)

| Probe | Control (MLM-only) | Treatment (MLM+recon) | Δ |
|---|---|---|---|
| kind (top 10) | <X>% | <Y>% | <Δ> |
| has-containers | <X>% | <Y>% | <Δ> |
| has-init-containers | <X>% | <Y>% | <Δ> |
| has-volume-mounts | <X>% | <Y>% | <Δ> |

## Probe trajectories (per epoch)

(Paste full per-epoch tables from `eval_v8_probes.py` for both conditions, side-by-side.)

## Acceptance gate check

1. **Reconstruction trains stably?** `recon_loss` trajectory: <values>. Monotonic ↓: <YES/NO>. NaN: <NO>. Treatment MLM loss at epoch 10 = <T>; control = <C>; ratio T/C = <r>. Within 10% relative? <YES/NO>.
2. **At least one non-kind probe improves by ≥2pp absolute?** has-containers Δ=<Δ>, has-init Δ=<Δ>, has-volume-mounts Δ=<Δ>. ≥1 above 2pp? <YES/NO>.

## Observations

(Surprises, unexpected patterns in trajectories, anything worth recording for the next mini-cycle.)

## Decision

<GO / AMBIGUOUS / NO-GO>

Rationale: <paragraph tying each measurement to verdict>.

If GO: next mini-cycle is the proper evaluation framework (the smoke probes hinted reconstruction added value; we need a real benchmark to measure it at scale).

If AMBIGUOUS: recon trained stably but the 3 smoke probes can't see a difference. Likely meaning: either reconstruction didn't actually add useful signal, or the probes are too coarse to detect what it added. Skip ahead to eval-framework mini-cycle and revisit reconstruction with that benchmark.

If NO-GO: reconstruction is broken or counterproductive. Investigate (leak bug? loss imbalance? bag-of-keys signal not informative?) before retrying.

## Files

- `output_v8_phase1_control/`: control run outputs
- `output_v8_phase1_treatment/`: treatment run outputs (incl. per-epoch doc_vec dumps)
- Both directories include `doc_cache.pkl`, `vocab.json`, `v8_phase1_recon.pt`, `doc_vecs_epoch_*.pt`
```

Then fill in the placeholders manually with the measured numbers from Steps 3-7.

- [ ] **Step 9: Commit the results doc**

```bash
git add docs/v8-phase1-reconstruction-results.md
git commit -m "docs(v8): Phase 1 reconstruction-objective results + decision"
```

**Mini-cycle complete. Decide go/no-go for the next Phase 1 mini-cycle (eval framework).**

---

## Self-Review Notes

**Spec coverage:**
- ✓ Subtree masking module (Task 1)
- ✓ V8Dataset + collate subtree integration (Task 2)
- ✓ Leak-aware aggregator (Task 3)
- ✓ ReconstructionHead module (Task 4)
- ✓ V8Model integration (Task 5)
- ✓ Forked trainer with --reconstruction + per-epoch monitoring + val split (Task 6)
- ✓ eval_v8_probes script + 4 probes (Task 7)
- ✓ Two-condition JL benchmark + results doc + acceptance-gate decision (Task 8)
- ✓ Config flags `recon_enabled` + `recon_loss_weight` (Task 2 Step 3)
- ✓ Reuse `[MASK]` sentinel (Task 2 — `token_ids[pos] = mask_id`)
- ✓ Mutual exclusivity MLM↔subtree (Task 2 — `mlm_masked_positions` passed to `pick_subtrees` candidate filter)
- ✓ Backward-compat (Tasks 2-5 — all new pathways gated on optional kwargs; existing tests must pass)

**Placeholder scan:** Two intentional placeholders in Task 8's results doc template (per-epoch numbers + acceptance-gate values) — filled in at execution time. No `TBD`/`TODO` in code-step content.

**Type consistency:** All four precompute tensor names match across Phase 1 cycles. New names introduced this plan — `subtree_mask` (B, N) bool, `subtree_roots_flat` (M, 2) long, `bag_of_keys_targets_flat` (M, V_atomic) float — are used identically in V8Dataset, v8_collate_fn, aggregator, V8Model, and the trainer's `_forward_v8` helper.

**One risk worth flagging upfront:** Task 5's pos_emb construction assumes `self.embedding.depth_embedding` and `self.embedding.sibling_embedding` exist as named submodules on `YamlBertEmbedding`. The implementer must read `yaml_bert/embedding.py` to confirm the actual attribute names (they may be `depth_emb` or otherwise). If different, adjust the two lookups. If embedding doesn't expose them as named submodules, expose them in a small `embedding.py` edit. The plan calls this out at Task 5 Step 3.
