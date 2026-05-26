# v9 Sub-tokenization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace v8's separate atomic key + atomic value embedding tables with one unified byte-level BPE subword vocabulary, while keeping the v8 encoder, tree aggregator, structural `doc_vec`, MLM-on-keys, and reconstruction objective unchanged.

**Architecture:** The dataset BPE-expands each linearizer node into 1..K subword positions, all replicating the node's `node_type`/`depth`/`sibling`/`parent_path` plus a new `logical_id` tensor. The encoder runs over subword positions. A new `_pool_subwords` step inside `TreeAggregator.forward` mean-pools subwords per logical node, after which the existing v8 vectorized aggregator runs unchanged on per-logical-node vectors. MLM masks all subwords of a chosen logical KEY and predicts the atomic key id from `atomic_target_vocab` (same head, same target vocab as v8). The embedding module collapses `key_embedding + value_embedding` into one `subword_embedding`.

**Tech Stack:** Python 3.13, PyTorch 2.12 (existing), `tokenizers` 0.22 (already installed for v9 tokenizer training), pytest, the existing `yaml_bert` package. The trained BPE artifact at `output_v8_276K_recon_seed42/unified_bpe_8k.json` is the source of truth for the new vocab.

**Backward-compat strategy:** **In-place replacement of v8 code.** v8 is already mainline and shipped. We follow the same archival pattern used for v7: snapshot v8 to `archive/v8/` (mirroring `archive/v7/`) for relic value, then modify the canonical `yaml_bert/` modules. Old v8 checkpoints stay on disk where they are; the v9 trainer doesn't try to load them. Existing tests that reference removed v8 APIs (separate `encode_key`/`encode_value`, key+value embedding tables) are updated to the v9 shape — they're the same kind of test, just with a different vocab interface.

---

## File Structure

**New files:**
- `yaml_bert/tokenizer.py` — `SubwordTokenizer` wrapper around HF `Tokenizer`, providing the ID lookups and special-token constants the rest of the package needs (`pad_id`, `mask_id`, `long_value_id`, plus `encode_token` and `encode_value_with_long`).
- `tests/test_tokenizer.py` — unit tests for the wrapper (round-trip, special tokens, long-value rule).
- `tests/test_aggregator_subword_pooling.py` — unit tests for the new `_pool_subwords` step in isolation.
- `scripts/audit_v9_batch.py` — runs one batch through dataset → collate → model.forward, prints all shapes, asserts NaN-free; runs locally without GPU.
- `docs/v9-subword-results.md` — results doc, written at end of Task 9.

**Modified files:**
- `yaml_bert/vocab.py` — `Vocabulary` now stores a `SubwordTokenizer` instead of separate key/value vocabs; keeps `atomic_target_vocab` (unchanged semantics) and `special_tokens` for compat with the head's existing API.
- `yaml_bert/embedding.py` — `YamlBertEmbedding.__init__` takes a single `subword_vocab_size` instead of `key_vocab_size + value_vocab_size`; `forward` drops the `is_key` gate and the `torch.where` selection.
- `yaml_bert/dataset.py` — `YamlBertDataset.__getitem__` BPE-expands each node, emits `logical_ids`. Whole-word masking replaces per-position masking. `collate_fn` adds `logical_ids` to the padded tensors and computes per-batch `logical_id_offsets` for the aggregator pool step.
- `yaml_bert/aggregator.py` — `TreeAggregator.forward` accepts a new required kwarg `logical_ids` (B, N_sub) and `n_logical_per_doc` (B,); calls `_pool_subwords(hidden_states, logical_ids, n_logical_per_doc)` to produce per-logical-node hidden states, then runs the existing vectorized path on those.
- `yaml_bert/model.py` — `YamlBertModel.forward` threads `logical_ids` and `n_logical_per_doc` from the batch into the aggregator; the Token Head still consumes per-logical-node hidden states (pooled output from the aggregator step is now the input).
- `yaml_bert/config.py` — `max_seq_len: int = 768` (was 512), add `vocab_size: int = 8192` for the new subword vocab.
- `scripts/train.py` — load the BPE tokenizer instead of building a vocab from scratch; pass `vocab` (now subword-backed) to dataset and model; new default output dir `output_v9_*`.
- `tests/test_vocab.py` — rewritten for the subword vocabulary.
- `tests/test_embedding.py` — updated to construct embedding with `subword_vocab_size`.
- `tests/test_dataset.py` — updated for subword expansion + `logical_ids`.
- `tests/test_aggregator.py`, `tests/test_aggregator_vectorized.py` — updated to pass `logical_ids` + `n_logical_per_doc`.
- `tests/test_model_e2e.py` — updated to use the new vocab + dataset.
- `tests/test_atomic_vocab.py` — atomic target vocab logic itself didn't change, but its construction did (now reads from the tokenizer's atomic-key set).
- `tests/test_dataset_subtree.py`, `tests/test_reconstruction_head.py`, `tests/test_subtree_masking.py` — reconstruction logic unchanged; tests need vocab-construction touchup.
- `archive/v8/` (NEW directory) — snapshot of `yaml_bert/`, `scripts/`, key `tests/` at v8 head.

---

## Task 1: Snapshot v8 to archive, baseline-lock current tests

**Files:**
- Create directory: `archive/v8/yaml_bert/`, `archive/v8/scripts/`, `archive/v8/tests/`, `archive/v8/README.md`
- Read-only copies of v8 source

**Why:** v8 is the mainline that's currently deployed on the HF Space. The v7-archive pattern from prior cycles preserves it as a relic, off the PYTHONPATH, so it can never be accidentally imported but stays available for reference. Doing this *first* protects us if v9 needs to roll back.

- [ ] **Step 1: Create archive directory structure**

```bash
mkdir -p archive/v8/yaml_bert archive/v8/scripts archive/v8/tests
```

- [ ] **Step 2: Copy v8 source files into archive (no .pyc, no __pycache__)**

```bash
cp -v yaml_bert/*.py archive/v8/yaml_bert/
cp -v scripts/train.py scripts/eval_probes.py scripts/export_model.py archive/v8/scripts/
cp -v tests/test_*.py archive/v8/tests/
```

- [ ] **Step 3: Write archive/v8/README.md to mark it as a relic**

Create `archive/v8/README.md`:

```markdown
# v8 Archive (read-only, off-path)

Snapshot of YAML-BERT v8 (MLM + reconstruction, 276K atomic-vocab model) taken on 2026-05-27 before the v9 sub-tokenization rewrite. This directory is intentionally not on `PYTHONPATH` — these files are for reference only, not for runtime import.

Mirrors the same pattern used for `archive/v7/`.

To rehydrate v8 locally:

    git checkout <pre-v9-commit-sha> -- yaml_bert/ scripts/train.py

For deployed v8 checkpoints + tokenizer artifacts, see `output_v8_276K_recon_seed42/`.
```

- [ ] **Step 4: Establish the test baseline before any v9 code changes**

Run: `pytest tests/ -q --tb=line 2>&1 | tail -5`

Record the pass count and any pre-existing failures (e.g. `test_linearizer.py` + `test_integration.py` have pre-existing `FileNotFoundError` per session memory). Write the baseline to a temporary note (you'll diff against it after each subsequent task).

Expected: ~85+ passing; only the known data-file-missing failures.

- [ ] **Step 5: Commit**

```bash
git add archive/v8/
git commit -m "archive: snapshot v8 source to archive/v8/ before v9 rewrite"
```

---

## Task 2: SubwordTokenizer wrapper + unit tests

**Files:**
- Create: `yaml_bert/tokenizer.py`
- Test: `tests/test_tokenizer.py`

**Why:** Keep all knowledge of the HF `tokenizers` library in one wrapper file. The rest of the package depends on the wrapper's stable interface (`encode_token`, `mask_id`, etc.), not on HF internals. Makes future tokenizer changes (different lib, different vocab size) a one-file edit.

- [ ] **Step 1: Write the failing test file**

Create `tests/test_tokenizer.py`:

```python
"""Unit tests for SubwordTokenizer wrapper."""
import os
import pytest

from yaml_bert.tokenizer import (
    SubwordTokenizer,
    LONG_VALUE_TOKEN,
    PAD_TOKEN,
    MASK_TOKEN,
    UNK_TOKEN,
    MAX_CHARS_VALUE,
    LONG_VALUE_THRESHOLD,
)

TOKENIZER_PATH = "output_v8_276K_recon_seed42/unified_bpe_8k.json"


@pytest.fixture(scope="module")
def tok():
    if not os.path.exists(TOKENIZER_PATH):
        pytest.skip(f"Tokenizer artifact missing at {TOKENIZER_PATH}")
    return SubwordTokenizer.load(TOKENIZER_PATH)


def test_special_token_ids_are_distinct(tok):
    ids = {tok.pad_id, tok.unk_id, tok.mask_id, tok.long_value_id}
    assert len(ids) == 4


def test_vocab_size_is_8192(tok):
    assert tok.vocab_size == 8192


def test_encode_token_short_value_returns_subwords(tok):
    ids = tok.encode_token("web-1", is_value=True)
    assert isinstance(ids, list)
    assert all(isinstance(i, int) for i in ids)
    assert len(ids) >= 2  # 'web', '-', '1' or similar


def test_encode_token_key_does_not_truncate_or_long_replace(tok):
    long_key = "x" * 500
    ids = tok.encode_token(long_key, is_value=False)
    # Keys are encoded as-is (no LONG_VALUE substitution for keys)
    assert tok.long_value_id not in ids


def test_encode_token_long_value_returns_single_long_value_token(tok):
    long = "x" * (LONG_VALUE_THRESHOLD + 1)
    ids = tok.encode_token(long, is_value=True)
    assert ids == [tok.long_value_id]


def test_encode_token_mid_length_value_is_truncated_then_bped(tok):
    """Value between MAX_CHARS_VALUE and LONG_VALUE_THRESHOLD:
    truncate to MAX_CHARS_VALUE chars, then BPE-encode."""
    mid = "a" * (MAX_CHARS_VALUE + 10)
    assert MAX_CHARS_VALUE < len(mid) < LONG_VALUE_THRESHOLD
    ids = tok.encode_token(mid, is_value=True)
    # Should not be the single long-value sentinel
    assert ids != [tok.long_value_id]
    # And the encoded length should be ~equal to encoding the first MAX_CHARS_VALUE chars
    ids_trunc = tok.encode_token(mid[:MAX_CHARS_VALUE], is_value=True)
    assert ids == ids_trunc


def test_encode_token_known_atomic_schema_key_stays_single_subword(tok):
    # apiVersion was confirmed atomic in the trained vocab
    ids = tok.encode_token("apiVersion", is_value=False)
    assert len(ids) == 1


def test_roundtrip_via_decode(tok):
    """Decoding the encoded ids reproduces the original short string."""
    cases = ["nginx", "ClusterIP", "Pod", "web-1", "apps/v1"]
    for s in cases:
        ids = tok.encode_token(s, is_value=True)
        decoded = tok.decode(ids)
        assert decoded == s, f"{s!r} → {ids} → {decoded!r}"
```

- [ ] **Step 2: Run tests, confirm they all fail with import error**

Run: `pytest tests/test_tokenizer.py -v 2>&1 | tail -10`
Expected: All fail with `ModuleNotFoundError: No module named 'yaml_bert.tokenizer'` or `ImportError`.

- [ ] **Step 3: Write minimal `yaml_bert/tokenizer.py`**

Create `yaml_bert/tokenizer.py`:

```python
"""Subword tokenizer wrapper for YAML-BERT v9.

Wraps the HF `tokenizers` library so the rest of yaml_bert depends on
this stable interface, not on HF internals.

Vocabulary semantics:
  - Special tokens reserved at training time: [PAD], [UNK], [MASK], [LONG_VALUE]
  - Otherwise byte-level BPE; any string can be encoded.

Long-value rule (values only; keys never get this treatment):
  - value length >= LONG_VALUE_THRESHOLD chars → single [LONG_VALUE] token
  - MAX_CHARS_VALUE < value length < LONG_VALUE_THRESHOLD → truncate to MAX_CHARS_VALUE chars, then BPE
  - shorter → BPE in full
"""
from __future__ import annotations

from tokenizers import Tokenizer

PAD_TOKEN = "[PAD]"
UNK_TOKEN = "[UNK]"
MASK_TOKEN = "[MASK]"
LONG_VALUE_TOKEN = "[LONG_VALUE]"

MAX_CHARS_VALUE = 64
LONG_VALUE_THRESHOLD = 256


class SubwordTokenizer:
    """Wraps an HF Tokenizer with YAML-BERT-specific value-length rules."""

    def __init__(self, hf_tokenizer: Tokenizer) -> None:
        self._tok = hf_tokenizer
        self.pad_id = hf_tokenizer.token_to_id(PAD_TOKEN)
        self.unk_id = hf_tokenizer.token_to_id(UNK_TOKEN)
        self.mask_id = hf_tokenizer.token_to_id(MASK_TOKEN)
        self.long_value_id = hf_tokenizer.token_to_id(LONG_VALUE_TOKEN)
        for name, val in (
            (PAD_TOKEN, self.pad_id), (UNK_TOKEN, self.unk_id),
            (MASK_TOKEN, self.mask_id), (LONG_VALUE_TOKEN, self.long_value_id),
        ):
            if val is None:
                raise ValueError(
                    f"SubwordTokenizer: required special token {name!r} not "
                    f"in tokenizer vocab — was the tokenizer trained with "
                    f"this special token reserved?"
                )

    @classmethod
    def load(cls, path: str) -> "SubwordTokenizer":
        return cls(Tokenizer.from_file(path))

    @property
    def vocab_size(self) -> int:
        return self._tok.get_vocab_size()

    def encode_token(self, token: str, *, is_value: bool) -> list[int]:
        """Encode one linearizer-node token to a list of subword ids.

        See module docstring for the value-length rule. Keys are always
        BPE-encoded in full regardless of length.
        """
        if is_value:
            if len(token) >= LONG_VALUE_THRESHOLD:
                return [self.long_value_id]
            if len(token) > MAX_CHARS_VALUE:
                token = token[:MAX_CHARS_VALUE]
        return self._tok.encode(token).ids

    def decode(self, ids: list[int]) -> str:
        return self._tok.decode(ids)
```

- [ ] **Step 4: Run tests, confirm they pass**

Run: `pytest tests/test_tokenizer.py -v 2>&1 | tail -15`
Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
git add yaml_bert/tokenizer.py tests/test_tokenizer.py
git commit -m "feat(v9): SubwordTokenizer wrapper + tests"
```

---

## Task 3: Vocabulary rewrite — subword-backed, keep atomic_target_vocab

**Files:**
- Modify: `yaml_bert/vocab.py` (full rewrite of `Vocabulary` class; keep `VocabBuilder.build_from_huggingface` for atomic_target_vocab construction)
- Modify: `tests/test_vocab.py` (rewrite for new shape)

**Why:** v9 still needs an `atomic_target_vocab` (the Token Head's output classes are atomic keys, not subwords — per spec). But the input-side vocab is now just "ask the tokenizer". `Vocabulary` becomes a thin holder: a `SubwordTokenizer` plus the `atomic_target_vocab` dict, dropping the `key_vocab`/`value_vocab` dicts and their related methods.

- [ ] **Step 1: Rewrite `tests/test_vocab.py`**

Replace contents of `tests/test_vocab.py` with:

```python
"""Tests for v9 Vocabulary (subword-backed + atomic_target_vocab)."""
import json
import os
import pytest

from yaml_bert.vocab import Vocabulary

TOKENIZER_PATH = "output_v8_276K_recon_seed42/unified_bpe_8k.json"


@pytest.fixture(scope="module")
def vocab():
    if not os.path.exists(TOKENIZER_PATH):
        pytest.skip(f"Tokenizer artifact missing at {TOKENIZER_PATH}")
    atomic_target_vocab = {"apiVersion": 4, "kind": 5, "metadata": 6}
    return Vocabulary.from_tokenizer_path(
        tokenizer_path=TOKENIZER_PATH,
        atomic_target_vocab=atomic_target_vocab,
    )


def test_subword_vocab_size(vocab):
    assert vocab.subword_vocab_size == 8192


def test_special_token_ids(vocab):
    assert vocab.pad_id >= 0
    assert vocab.unk_id >= 0
    assert vocab.mask_id >= 0
    assert vocab.long_value_id >= 0


def test_atomic_target_vocab_size(vocab):
    # 3 user entries + 4 special tokens (pad/unk/mask/long_value)
    assert vocab.atomic_target_vocab_size == 3 + 4


def test_encode_atomic_target_known_key(vocab):
    assert vocab.encode_atomic_target("apiVersion") == 4


def test_encode_atomic_target_unknown_returns_unk(vocab):
    assert vocab.encode_atomic_target("totally-unknown-key") == vocab.unk_id


def test_encode_token_value_short(vocab):
    ids = vocab.encode_token("web-1", is_value=True)
    assert len(ids) >= 2


def test_save_and_load_round_trip(vocab, tmp_path):
    path = tmp_path / "vocab.json"
    vocab.save(str(path))
    loaded = Vocabulary.load(str(path))
    assert loaded.subword_vocab_size == vocab.subword_vocab_size
    assert loaded.atomic_target_vocab == vocab.atomic_target_vocab
    assert loaded.encode_atomic_target("kind") == 5


def test_saved_vocab_json_references_tokenizer_path(vocab, tmp_path):
    path = tmp_path / "vocab.json"
    vocab.save(str(path))
    payload = json.loads(path.read_text())
    assert "tokenizer_path" in payload
    assert "atomic_target_vocab" in payload
```

- [ ] **Step 2: Run tests, confirm they fail**

Run: `pytest tests/test_vocab.py -v 2>&1 | tail -10`
Expected: all fail (Vocabulary has the old shape).

- [ ] **Step 3: Replace `yaml_bert/vocab.py`**

Overwrite `yaml_bert/vocab.py` with:

```python
"""v9 Vocabulary: subword tokenizer + atomic_target_vocab.

The v8 key_vocab + value_vocab were merged into a single subword vocabulary
exposed via SubwordTokenizer. The Token Head still predicts over an atomic
target vocab built from frequent KEY tokens in the training corpus.
"""
from __future__ import annotations

import json
from yaml_bert.tokenizer import SubwordTokenizer

# Canonical casing for known Kubernetes kinds.
# Maps lowercase → canonical form. Populated by VocabBuilder.build_atomic_target_vocab.
_KIND_CANONICAL: dict[str, str] = {}


def normalize_kind(kind: str) -> str:
    return _KIND_CANONICAL.get(kind.lower(), kind)


class Vocabulary:
    """Holds a subword tokenizer + an atomic target vocab.

    `atomic_target_vocab` maps whole KEY strings (e.g. "containers",
    "restartPolicy") to integer class ids for the Token Head's output.
    The 4 special tokens (pad/unk/mask/long_value) get the first 4 ids.
    """

    def __init__(
        self,
        tokenizer: SubwordTokenizer,
        atomic_target_vocab: dict[str, int],
        tokenizer_path: str | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.atomic_target_vocab = atomic_target_vocab
        self.tokenizer_path = tokenizer_path
        # Convenience: expose the 4 special-token ids at the vocab level
        self.pad_id = tokenizer.pad_id
        self.unk_id = tokenizer.unk_id
        self.mask_id = tokenizer.mask_id
        self.long_value_id = tokenizer.long_value_id

    @classmethod
    def from_tokenizer_path(
        cls,
        tokenizer_path: str,
        atomic_target_vocab: dict[str, int],
    ) -> "Vocabulary":
        return cls(
            tokenizer=SubwordTokenizer.load(tokenizer_path),
            atomic_target_vocab=atomic_target_vocab,
            tokenizer_path=tokenizer_path,
        )

    def encode_token(self, token: str, *, is_value: bool) -> list[int]:
        return self.tokenizer.encode_token(token, is_value=is_value)

    def encode_atomic_target(self, key: str) -> int:
        return self.atomic_target_vocab.get(key, self.unk_id)

    @property
    def subword_vocab_size(self) -> int:
        return self.tokenizer.vocab_size

    @property
    def atomic_target_vocab_size(self) -> int:
        # +4 to reserve ids 0-3 for special tokens (consistent with v8 layout)
        return len(self.atomic_target_vocab) + 4

    def save(self, path: str) -> None:
        if self.tokenizer_path is None:
            raise ValueError(
                "Vocabulary.save requires tokenizer_path to be set "
                "(use Vocabulary.from_tokenizer_path)."
            )
        with open(path, "w") as f:
            json.dump({
                "tokenizer_path": self.tokenizer_path,
                "atomic_target_vocab": self.atomic_target_vocab,
            }, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "Vocabulary":
        with open(path) as f:
            data = json.load(f)
        return cls.from_tokenizer_path(
            tokenizer_path=data["tokenizer_path"],
            atomic_target_vocab=data["atomic_target_vocab"],
        )


class VocabBuilder:
    """Builds the atomic_target_vocab from a corpus.

    The subword tokenizer is built separately by scripts/train_unified_tokenizer.py.
    """

    @staticmethod
    def build_atomic_target_vocab(
        nodes_per_doc: list[list],
        min_freq: int,
    ) -> dict[str, int]:
        """Scan KEY tokens across docs, return {token: id} for keys appearing >= min_freq times.

        IDs start at 4 (0-3 are reserved for special tokens).
        """
        from yaml_bert.types import NodeType
        counts: dict[str, int] = {}
        for nodes in nodes_per_doc:
            for n in nodes:
                if n.node_type in (NodeType.KEY, NodeType.LIST_KEY):
                    counts[n.token] = counts.get(n.token, 0) + 1

        # Also populate kind-canonical map as a side effect (used by suggest.py)
        prev_kind_key = False
        for nodes in nodes_per_doc:
            for n in nodes:
                if n.node_type in (NodeType.KEY, NodeType.LIST_KEY):
                    prev_kind_key = (n.token == "kind" and n.depth == 0)
                elif n.node_type in (NodeType.VALUE, NodeType.LIST_VALUE):
                    if prev_kind_key:
                        lower = n.token.lower()
                        existing = _KIND_CANONICAL.get(lower)
                        if existing is None:
                            _KIND_CANONICAL[lower] = n.token
                    prev_kind_key = False

        kept = sorted(t for t, c in counts.items() if c >= min_freq)
        return {t: 4 + i for i, t in enumerate(kept)}
```

- [ ] **Step 4: Run tests, confirm pass**

Run: `pytest tests/test_vocab.py -v 2>&1 | tail -15`
Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
git add yaml_bert/vocab.py tests/test_vocab.py
git commit -m "feat(v9): subword-backed Vocabulary + atomic_target_vocab builder"
```

---

## Task 4: YamlBertEmbedding — merge key + value tables

**Files:**
- Modify: `yaml_bert/embedding.py`
- Modify: `tests/test_embedding.py`

**Why:** With one subword vocabulary, the dual-table-with-`torch.where`-gate in v8 collapses to a single `nn.Embedding` indexed directly by `token_ids`. The `node_type_embedding`, `depth_embedding`, `sibling_embedding` are unchanged. This is the smallest, most isolated v9 code change — does not require any dataset changes to test in isolation.

- [ ] **Step 1: Rewrite `tests/test_embedding.py`**

Replace its contents with:

```python
"""Tests for v9 YamlBertEmbedding (single subword embedding table)."""
import pytest
import torch

from yaml_bert.config import YamlBertConfig, TreePosVariant
from yaml_bert.embedding import YamlBertEmbedding


def _make_emb(vocab_size=200, d=16):
    cfg = YamlBertConfig(
        d_model=d, num_layers=1, num_heads=1, d_ff=32,
        max_depth=8, max_sibling=8, max_seq_len=64,
    )
    return YamlBertEmbedding(config=cfg, subword_vocab_size=vocab_size)


def test_init_accepts_subword_vocab_size():
    emb = _make_emb()
    assert emb.subword_embedding.num_embeddings == 200


def test_forward_output_shape():
    emb = _make_emb(vocab_size=128, d=16)
    B, N = 2, 5
    out = emb(
        token_ids=torch.zeros(B, N, dtype=torch.long),
        node_types=torch.zeros(B, N, dtype=torch.long),
        depths=torch.zeros(B, N, dtype=torch.long),
        sibling_indices=torch.zeros(B, N, dtype=torch.long),
    )
    assert out.shape == (B, N, 16)


def test_node_type_embedding_still_present_and_used():
    emb = _make_emb()
    # Two positions with same token id but different node_types should differ
    ids = torch.tensor([[5, 5]])
    nt = torch.tensor([[0, 1]])
    z = torch.zeros_like(ids)
    out = emb(ids, nt, z, z)
    assert not torch.allclose(out[0, 0], out[0, 1])


def test_old_key_value_tables_no_longer_exist():
    emb = _make_emb()
    assert not hasattr(emb, "key_embedding")
    assert not hasattr(emb, "value_embedding")


def test_tree_pos_variant_no_depth_still_works():
    cfg = YamlBertConfig(
        d_model=16, num_layers=1, num_heads=1, d_ff=32,
        max_depth=8, max_sibling=8, max_seq_len=64,
        tree_pos_variant=TreePosVariant.NO_DEPTH,
    )
    emb = YamlBertEmbedding(config=cfg, subword_vocab_size=64)
    assert emb.depth_embedding is None
```

- [ ] **Step 2: Run tests, confirm fail**

Run: `pytest tests/test_embedding.py -v 2>&1 | tail -10`
Expected: All fail with `TypeError: __init__() got an unexpected keyword argument 'subword_vocab_size'` or `key_vocab_size`.

- [ ] **Step 3: Rewrite `yaml_bert/embedding.py`**

Overwrite with:

```python
from __future__ import annotations

import torch
import torch.nn as nn

from yaml_bert.config import TreePosVariant, YamlBertConfig


class YamlBertEmbedding(nn.Module):
    """v9 embedding layer: single subword table + tree positional encoding.

    Produces input vectors by summing:
    - Subword embedding (looked up by token_id; same table for KEY and VALUE
      positions — what they ARE is signalled separately via node_type_emb)
    - Tree positional encoding (composition depends on config.tree_pos_variant)
    """

    def __init__(
        self,
        config: YamlBertConfig,
        subword_vocab_size: int,
    ) -> None:
        super().__init__()
        d: int = config.d_model
        variant: TreePosVariant = config.tree_pos_variant
        self.variant: TreePosVariant = variant

        self.subword_embedding: nn.Embedding = nn.Embedding(subword_vocab_size, d)
        self.node_type_embedding: nn.Embedding = nn.Embedding(4, d)

        use_depth: bool = variant in (TreePosVariant.FULL, TreePosVariant.NO_SIBLING)
        use_sibling: bool = variant in (TreePosVariant.FULL, TreePosVariant.NO_DEPTH)
        use_seq_pos: bool = variant == TreePosVariant.SEQUENTIAL

        self.depth_embedding: nn.Embedding | None = (
            nn.Embedding(config.max_depth, d) if use_depth else None
        )
        self.sibling_embedding: nn.Embedding | None = (
            nn.Embedding(config.max_sibling, d) if use_sibling else None
        )
        self.pos_embedding: nn.Embedding | None = (
            nn.Embedding(config.max_seq_len, d) if use_seq_pos else None
        )

        self.layer_norm: nn.LayerNorm = nn.LayerNorm(d)

    def forward(
        self,
        token_ids: torch.Tensor,
        node_types: torch.Tensor,
        depths: torch.Tensor,
        sibling_indices: torch.Tensor,
    ) -> torch.Tensor:
        token_emb = self.subword_embedding(token_ids)

        tree_pos = self.node_type_embedding(node_types)
        if self.depth_embedding is not None:
            tree_pos = tree_pos + self.depth_embedding(depths)
        if self.sibling_embedding is not None:
            tree_pos = tree_pos + self.sibling_embedding(sibling_indices)
        if self.pos_embedding is not None:
            seq_len: int = token_ids.size(1)
            max_pos: int = self.pos_embedding.num_embeddings
            positions = (
                torch.arange(seq_len, device=token_ids.device)
                .clamp(max=max_pos - 1)
                .unsqueeze(0)
                .expand(token_ids.size(0), seq_len)
            )
            tree_pos = tree_pos + self.pos_embedding(positions)

        return self.layer_norm(token_emb + tree_pos)
```

- [ ] **Step 4: Run tests, confirm pass**

Run: `pytest tests/test_embedding.py -v 2>&1 | tail -10`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add yaml_bert/embedding.py tests/test_embedding.py
git commit -m "feat(v9): merge key+value embedding tables into unified subword_embedding"
```

---

## Task 5: Dataset — BPE expansion + whole-word MLM masking

**Files:**
- Modify: `yaml_bert/dataset.py`
- Modify: `yaml_bert/config.py` (bump `max_seq_len` to 768)
- Modify: `tests/test_dataset.py`

**Why:** This is the largest change in v9. Each linearizer node becomes 1..K subword positions. All BPE knowledge is concentrated in `__getitem__`. Whole-word MLM masking replaces v8's per-position masking. `collate_fn` learns to pad/stack the new `logical_ids` tensor and to compute per-batch `n_logical_per_doc` for the aggregator.

- [ ] **Step 1: Bump max_seq_len in config**

Edit `yaml_bert/config.py`:

```python
    max_seq_len: int = 768  # was 512 in v8; v9 BPE-expands sequences ~2.3x
```

- [ ] **Step 2: Rewrite `tests/test_dataset.py`**

Replace contents with:

```python
"""Tests for v9 YamlBertDataset (subword expansion + whole-key masking)."""
import os
import pytest
import torch

from yaml_bert.annotator import DomainAnnotator
from yaml_bert.config import YamlBertConfig
from yaml_bert.dataset import YamlBertDataset, collate_fn
from yaml_bert.linearizer import YamlLinearizer
from yaml_bert.vocab import Vocabulary, VocabBuilder

TOKENIZER_PATH = "output_v8_276K_recon_seed42/unified_bpe_8k.json"

SIMPLE_YAML = """apiVersion: v1
kind: Pod
metadata:
  name: web
"""


@pytest.fixture(scope="module")
def vocab():
    if not os.path.exists(TOKENIZER_PATH):
        pytest.skip("tokenizer missing")
    return Vocabulary.from_tokenizer_path(
        tokenizer_path=TOKENIZER_PATH,
        atomic_target_vocab=VocabBuilder.build_atomic_target_vocab(
            [YamlLinearizer().linearize(SIMPLE_YAML)], min_freq=1,
        ),
    )


@pytest.fixture(scope="module")
def docs():
    lin = YamlLinearizer()
    ann = DomainAnnotator()
    nodes = lin.linearize(SIMPLE_YAML)
    ann.annotate(nodes)
    return [nodes]


def _cfg(**kw):
    base = dict(
        d_model=16, num_layers=1, num_heads=1, d_ff=32,
        max_depth=8, max_sibling=8, max_seq_len=64,
        mask_prob=0.0,  # determinism for shape tests
    )
    base.update(kw)
    return YamlBertConfig(**base)


def test_getitem_subword_expansion(vocab, docs):
    ds = YamlBertDataset(docs, vocab, _cfg())
    item = ds[0]
    # All per-position tensors are the same length (the subword length)
    n_sub = item["token_ids"].size(0)
    assert item["node_types"].size(0) == n_sub
    assert item["depths"].size(0) == n_sub
    assert item["sibling_indices"].size(0) == n_sub
    assert item["logical_ids"].size(0) == n_sub
    # Some logical nodes (e.g. 'apiVersion') BPE to 1 subword;
    # at least one must BPE to >1 (e.g. 'v1' → 'v' '1') or this test is wrong
    n_logical = item["logical_ids"].max().item() + 1
    assert n_sub >= n_logical


def test_logical_ids_are_contiguous_and_increasing(vocab, docs):
    ds = YamlBertDataset(docs, vocab, _cfg())
    item = ds[0]
    lids = item["logical_ids"].tolist()
    # Each block of identical logical_ids must be contiguous
    seen_max = -1
    for lid in lids:
        assert lid >= seen_max, f"logical_ids must be non-decreasing: {lids}"
        seen_max = max(seen_max, lid)


def test_whole_key_masking_masks_all_subwords_of_chosen_key(vocab, docs):
    """With mask_prob=1.0 and a fixed seed, every KEY's subwords get [MASK]."""
    import random
    random.seed(0)
    ds = YamlBertDataset(docs, vocab, _cfg(mask_prob=1.0))
    item = ds[0]
    mask_id = vocab.mask_id
    # For each masked logical KEY: every position with that logical_id
    # should have token_id == mask_id (whole-key masking)
    # Find masked logicals via atomic_labels != -100
    labels = item["atomic_labels"]  # (n_logical,) — one label per LOGICAL node
    masked_lids = (labels != -100).nonzero(as_tuple=True)[0].tolist()
    assert len(masked_lids) > 0
    for lid in masked_lids:
        sub_positions = (item["logical_ids"] == lid).nonzero(as_tuple=True)[0]
        for p in sub_positions:
            assert item["token_ids"][p].item() == mask_id, \
                f"logical {lid} subword at {p} not masked"


def test_atomic_labels_are_per_logical_not_per_subword(vocab, docs):
    ds = YamlBertDataset(docs, vocab, _cfg())
    item = ds[0]
    n_logical = item["logical_ids"].max().item() + 1
    assert item["atomic_labels"].size(0) == n_logical


def test_collate_pads_logical_ids_and_emits_n_logical_per_doc(vocab, docs):
    ds = YamlBertDataset(docs * 3, vocab, _cfg())
    batch = collate_fn([ds[0], ds[1], ds[2]])
    assert "logical_ids" in batch
    assert "n_logical_per_doc" in batch
    assert batch["n_logical_per_doc"].shape == (3,)
    # Subword pad value is 0 (== pad_id slot); logical-id pad is -1 (out-of-range marker)
    # Atomic labels pad to -100
    assert batch["atomic_labels"].shape[1] == int(batch["n_logical_per_doc"].max())
```

- [ ] **Step 3: Run tests, confirm fail**

Run: `pytest tests/test_dataset.py -v 2>&1 | tail -10`
Expected: All fail (no subword expansion in v8 dataset).

- [ ] **Step 4: Rewrite `yaml_bert/dataset.py`**

Replace its contents with:

```python
"""v9 YAML-BERT dataset: BPE-expand each linearizer node into subword
positions, mask whole logical KEYs, emit per-logical-node atomic labels."""
from __future__ import annotations

import random
import re

import torch
from torch.utils.data import Dataset

from yaml_bert.config import YamlBertConfig
from yaml_bert.subtree_masking import descendants_of
from yaml_bert.types import NodeType, YamlNode
from yaml_bert.vocab import Vocabulary


_LIST_INDEX_RE = re.compile(r"\.\d+$")


def _strip_trailing_list_index(path: str) -> str:
    return _LIST_INDEX_RE.sub("", path)


def compute_children_info(nodes: list[YamlNode]) -> dict:
    """Same as v8 — operates on LOGICAL nodes (not subwords)."""
    n = len(nodes)
    full_path_of: list[str] = []
    for node in nodes:
        if node.parent_path:
            full_path_of.append(f"{node.parent_path}.{node.token}")
        else:
            full_path_of.append(node.token)

    key_positions: list[int] = [
        i for i, node in enumerate(nodes)
        if node.node_type in (NodeType.KEY, NodeType.LIST_KEY)
    ]
    path_to_key_pos: dict[str, int] = {
        full_path_of[p]: p for p in key_positions
    }

    children_of: list[list[int]] = [[] for _ in range(n)]
    parent_of: list[int] = [-1] * n
    depth_of: list[int] = [node.depth for node in nodes]

    for p in key_positions:
        parent_path = nodes[p].parent_path
        if not parent_path:
            continue
        parent_pos = path_to_key_pos.get(parent_path)
        if parent_pos is None:
            stripped = _strip_trailing_list_index(parent_path)
            if stripped != parent_path:
                parent_pos = path_to_key_pos.get(stripped)
        if parent_pos is not None:
            parent_of[p] = parent_pos
            children_of[parent_pos].append(p)

    return {
        "children_of": children_of,
        "parent_of": parent_of,
        "key_positions": key_positions,
        "depth_of": depth_of,
        "full_path_of": full_path_of,
    }


_NODE_TYPE_INDEX = {
    NodeType.KEY: 0,
    NodeType.VALUE: 1,
    NodeType.LIST_KEY: 2,
    NodeType.LIST_VALUE: 3,
}
_MASKABLE_TYPES = (NodeType.KEY, NodeType.LIST_KEY)


class YamlBertDataset(Dataset):
    """v9 dataset: subword expansion + whole-key MLM masking + recon."""

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

        self._cached_children_info: list[dict] = []
        self._cached_descendants: list[dict[int, set[int]] | None] = []
        for doc in documents:
            # Cap LOGICAL nodes here; BPE expansion may still exceed max_seq_len
            # at the subword level (handled below by truncation).
            ci = compute_children_info(doc)
            self._cached_children_info.append(ci)
            if self.recon_enabled:
                desc_cache: dict[int, set[int]] = {}
                for kp in ci["key_positions"]:
                    if ci["children_of"][kp]:
                        desc_cache[kp] = descendants_of(kp, ci["children_of"])
                self._cached_descendants.append(desc_cache)
            else:
                self._cached_descendants.append(None)

    def __len__(self) -> int:
        return len(self.documents)

    def __getitem__(self, idx: int) -> dict:
        nodes = self.documents[idx]

        # Pass 1: BPE-expand each logical node, building per-subword tensors.
        sub_token_ids: list[int] = []
        sub_node_types: list[int] = []
        sub_depths: list[int] = []
        sub_sibling: list[int] = []
        sub_logical_ids: list[int] = []
        per_logical_subword_spans: list[tuple[int, int]] = []
        # We may need to drop trailing logical nodes if subword expansion
        # blows the max_seq_len cap.
        kept_logical: int = 0
        for logical_idx, node in enumerate(nodes):
            is_value = node.node_type in (NodeType.VALUE, NodeType.LIST_VALUE)
            ids = self.vocab.encode_token(node.token, is_value=is_value)
            if len(sub_token_ids) + len(ids) > self.max_seq_len:
                break
            start = len(sub_token_ids)
            sub_token_ids.extend(ids)
            sub_node_types.extend([_NODE_TYPE_INDEX[node.node_type]] * len(ids))
            sub_depths.extend([min(node.depth, 15)] * len(ids))
            sub_sibling.extend([min(node.sibling_index, 31)] * len(ids))
            sub_logical_ids.extend([kept_logical] * len(ids))
            per_logical_subword_spans.append((start, start + len(ids)))
            kept_logical += 1

        n_logical = kept_logical
        # Truncate cached children_info to kept logicals
        ci = self._cached_children_info[idx]
        kept_set = set(range(n_logical))
        children_of_t = [
            [c for c in ci["children_of"][p] if c in kept_set]
            for p in range(n_logical)
        ]
        parent_of_t = [
            ci["parent_of"][p] if (ci["parent_of"][p] in kept_set or ci["parent_of"][p] == -1) else -1
            for p in range(n_logical)
        ]
        key_positions_t = [p for p in ci["key_positions"] if p < n_logical]
        depth_of_t = ci["depth_of"][:n_logical]
        full_path_of_t = ci["full_path_of"][:n_logical]
        ci_t = {
            "children_of": children_of_t,
            "parent_of": parent_of_t,
            "key_positions": key_positions_t,
            "depth_of": depth_of_t,
            "full_path_of": full_path_of_t,
        }

        # Pass 2: whole-key MLM masking, one decision per LOGICAL KEY.
        atomic_labels: list[int] = [-100] * n_logical
        mask_id = self.vocab.mask_id
        unk_id = self.vocab.unk_id
        subword_vocab_size = self.vocab.subword_vocab_size

        mlm_masked_positions: set[int] = set()
        for logical_idx in key_positions_t:
            if random.random() >= self.mask_prob:
                continue
            tok = nodes[logical_idx].token
            atomic_id = self.vocab.encode_atomic_target(tok)
            if atomic_id == unk_id:
                continue  # skip [UNK] targets (Lever 1, carried from v8)
            atomic_labels[logical_idx] = atomic_id
            mlm_masked_positions.add(logical_idx)
            r = random.random()
            start, end = per_logical_subword_spans[logical_idx]
            if r < 0.8:
                for p in range(start, end):
                    sub_token_ids[p] = mask_id
            elif r < 0.9:
                for p in range(start, end):
                    sub_token_ids[p] = random.randint(4, subword_vocab_size - 1)

        result = {
            "token_ids": torch.tensor(sub_token_ids, dtype=torch.long),
            "node_types": torch.tensor(sub_node_types, dtype=torch.long),
            "depths": torch.tensor(sub_depths, dtype=torch.long),
            "sibling_indices": torch.tensor(sub_sibling, dtype=torch.long),
            "logical_ids": torch.tensor(sub_logical_ids, dtype=torch.long),
            "atomic_labels": torch.tensor(atomic_labels, dtype=torch.long),
            "children_info": ci_t,
            "n_logical": n_logical,
        }

        if self.recon_enabled:
            from yaml_bert.subtree_masking import pick_subtrees, bag_of_keys_target
            picked_roots = pick_subtrees(
                N=n_logical,
                key_positions=key_positions_t,
                depth_of=depth_of_t,
                children_of=children_of_t,
                mlm_masked_positions=mlm_masked_positions,
                rng=random,
                descendants_cache={
                    kp: descendants_of(kp, children_of_t)
                    for kp in key_positions_t
                    if children_of_t[kp]
                },
            )
            subtree_mask = torch.zeros(n_logical, dtype=torch.bool)
            picked_positions_all: set[int] = set()
            bag_targets: list[torch.Tensor] = []
            position_to_key_str = {
                i: nodes[i].token for i in key_positions_t
            }
            for root_pos in picked_roots:
                descs = {
                    d for d in descendants_of(root_pos, children_of_t)
                    if d < n_logical
                }
                picked_positions_all |= descs
                bag_targets.append(bag_of_keys_target(
                    subtree_positions=descs,
                    position_to_key_str=position_to_key_str,
                    atomic_vocab=self.vocab.atomic_target_vocab,
                    vocab_size=self.vocab.atomic_target_vocab_size,
                ))
            # Apply [MASK] to ALL subwords of each logical position in the picked subtree
            for lpos in picked_positions_all:
                subtree_mask[lpos] = True
                start, end = per_logical_subword_spans[lpos]
                for p in range(start, end):
                    sub_token_ids[p] = mask_id
            result["token_ids"] = torch.tensor(sub_token_ids, dtype=torch.long)
            result["subtree_mask"] = subtree_mask
            result["subtree_roots"] = picked_roots
            result["bag_of_keys_targets"] = bag_targets
            result["_atomic_vocab_size"] = self.vocab.atomic_target_vocab_size

        return result


_COLLATE_NON_TENSOR_KEYS = frozenset({
    "children_info",
    "subtree_roots",
    "bag_of_keys_targets",
    "subtree_mask",
    "_atomic_vocab_size",
    "n_logical",
})


def collate_fn(batch: list[dict]) -> dict:
    """Pad subword-level tensors AND logical-level tensors.

    Subword-level (per-position): token_ids, node_types, depths, sibling_indices,
                                  logical_ids  — padded to max subword length
    Logical-level (per-logical-node): atomic_labels  — padded to max logical count
    """
    max_sub_len = max(item["token_ids"].size(0) for item in batch)
    max_logical = max(item["n_logical"] for item in batch)

    subword_keys = ("token_ids", "node_types", "depths", "sibling_indices",
                    "logical_ids")
    padded_sub: dict[str, list[torch.Tensor]] = {k: [] for k in subword_keys}
    padded_labels: list[torch.Tensor] = []
    padding_masks: list[torch.Tensor] = []
    batch_info: list[dict] = []
    n_logical_per_doc: list[int] = []

    for item in batch:
        sub_len = item["token_ids"].size(0)
        pad_sub = max_sub_len - sub_len
        for k in subword_keys:
            pad_value = -1 if k == "logical_ids" else 0
            if pad_sub > 0:
                padding = torch.full((pad_sub,), pad_value, dtype=torch.long)
                padded_sub[k].append(torch.cat([item[k], padding]))
            else:
                padded_sub[k].append(item[k])

        labels = item["atomic_labels"]
        pad_lab = max_logical - labels.size(0)
        if pad_lab > 0:
            padded_labels.append(torch.cat([
                labels, torch.full((pad_lab,), -100, dtype=torch.long),
            ]))
        else:
            padded_labels.append(labels)

        mask = torch.cat([
            torch.zeros(sub_len, dtype=torch.bool),
            torch.ones(pad_sub, dtype=torch.bool),
        ]) if pad_sub > 0 else torch.zeros(sub_len, dtype=torch.bool)
        padding_masks.append(mask)
        batch_info.append(item["children_info"])
        n_logical_per_doc.append(item["n_logical"])

    result = {k: torch.stack(v) for k, v in padded_sub.items()}
    result["atomic_labels"] = torch.stack(padded_labels)
    result["padding_mask"] = torch.stack(padding_masks)
    result["batch_info"] = batch_info
    result["n_logical_per_doc"] = torch.tensor(n_logical_per_doc, dtype=torch.long)

    # parent_of_tensor and top_level_key_mask now operate at LOGICAL level
    B = len(batch)
    L = max_logical
    parent_of_tensor = torch.full((B, L), -1, dtype=torch.long)
    top_level_key_mask = torch.zeros((B, L), dtype=torch.bool)
    for b_idx, info in enumerate(batch_info):
        parent_of = info["parent_of"]
        n_b = len(parent_of)
        if n_b > 0:
            parent_of_tensor[b_idx, :n_b] = torch.tensor(parent_of, dtype=torch.long)
        depth_of = info["depth_of"]
        depth_zero_kps = [kp for kp in info["key_positions"] if depth_of[kp] == 0]
        if depth_zero_kps:
            top_level_key_mask[b_idx, depth_zero_kps] = True

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

    if "subtree_mask" in batch[0]:
        subtree_masks: list[torch.Tensor] = []
        for item in batch:
            sm = item["subtree_mask"]
            pad_len = max_logical - sm.size(0)
            if pad_len > 0:
                subtree_masks.append(torch.cat([
                    sm, torch.zeros(pad_len, dtype=torch.bool),
                ]))
            else:
                subtree_masks.append(sm)
        result["subtree_mask"] = torch.stack(subtree_masks)

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
            result["subtree_roots_flat"] = torch.zeros((0, 2), dtype=torch.long)
            v = batch[0].get("_atomic_vocab_size", 0)
            result["bag_of_keys_targets_flat"] = torch.zeros(
                (0, v), dtype=torch.float,
            )

    return result
```

- [ ] **Step 5: Run dataset tests**

Run: `pytest tests/test_dataset.py -v 2>&1 | tail -15`
Expected: 5 passed.

- [ ] **Step 6: Commit**

```bash
git add yaml_bert/dataset.py yaml_bert/config.py tests/test_dataset.py
git commit -m "feat(v9): subword expansion + whole-key MLM masking in dataset"
```

---

## Task 6: Aggregator — add subword pooling step

**Files:**
- Modify: `yaml_bert/aggregator.py`
- Create: `tests/test_aggregator_subword_pooling.py`
- Modify: `tests/test_aggregator.py` and `tests/test_aggregator_vectorized.py` (update existing tests to pass new kwargs)

**Why:** The aggregator becomes the single place where subword-level hidden states get pooled into logical-level hidden states. Once pooled, the existing v8 vectorized path runs unchanged on per-logical-node vectors. Isolating this step means the rest of the aggregator code is touched minimally.

- [ ] **Step 1: Write `tests/test_aggregator_subword_pooling.py`**

```python
"""Unit tests for the subword-pooling step inside TreeAggregator."""
import torch

from yaml_bert.aggregator import _pool_subwords


def test_pool_subwords_basic():
    """3 subwords of logical 0, 2 subwords of logical 1, 1 of logical 2."""
    B, N_sub, d = 1, 6, 4
    hidden = torch.tensor([[
        [1.0, 0, 0, 0],
        [3.0, 0, 0, 0],
        [5.0, 0, 0, 0],   # logical 0: mean = 3.0
        [10.0, 0, 0, 0],
        [20.0, 0, 0, 0],  # logical 1: mean = 15.0
        [7.0, 0, 0, 0],   # logical 2: mean = 7.0
    ]])
    logical_ids = torch.tensor([[0, 0, 0, 1, 1, 2]])
    n_logical = torch.tensor([3])
    out = _pool_subwords(hidden, logical_ids, n_logical)
    assert out.shape == (B, 3, d)
    assert torch.allclose(out[0, 0, 0], torch.tensor(3.0))
    assert torch.allclose(out[0, 1, 0], torch.tensor(15.0))
    assert torch.allclose(out[0, 2, 0], torch.tensor(7.0))


def test_pool_subwords_ignores_negative_logical_ids():
    """logical_ids == -1 means padding; those subwords shouldn't affect pools."""
    hidden = torch.tensor([[
        [1.0, 0],
        [3.0, 0],
        [999.0, 0],  # pad, ignored
    ]])
    logical_ids = torch.tensor([[0, 0, -1]])
    n_logical = torch.tensor([1])
    out = _pool_subwords(hidden, logical_ids, n_logical)
    assert out.shape == (1, 1, 2)
    assert torch.allclose(out[0, 0, 0], torch.tensor(2.0))


def test_pool_subwords_batched_with_different_n_logical():
    """Two docs, different number of logical nodes."""
    hidden = torch.tensor([
        [[1.0, 0], [2.0, 0], [3.0, 0], [0.0, 0]],  # doc 0: logical [0,0,1]
        [[10.0, 0], [20.0, 0], [30.0, 0], [40.0, 0]],  # doc 1: logical [0,1,2,3]
    ])
    logical_ids = torch.tensor([[0, 0, 1, -1], [0, 1, 2, 3]])
    n_logical = torch.tensor([2, 4])
    out = _pool_subwords(hidden, logical_ids, n_logical)
    assert out.shape == (2, 4, 2)
    # Doc 0: logical 0 = mean(1, 2) = 1.5; logical 1 = 3
    assert torch.allclose(out[0, 0, 0], torch.tensor(1.5))
    assert torch.allclose(out[0, 1, 0], torch.tensor(3.0))
    # Doc 1: logical 0-3 = 10, 20, 30, 40
    assert torch.allclose(out[1, :, 0], torch.tensor([10., 20., 30., 40.]))
```

- [ ] **Step 2: Run tests, confirm fail with ImportError**

Run: `pytest tests/test_aggregator_subword_pooling.py -v 2>&1 | tail -10`
Expected: ImportError on `_pool_subwords`.

- [ ] **Step 3: Add `_pool_subwords` to `yaml_bert/aggregator.py`**

Add at the top of the file (after imports), and modify `forward` to require + use the new kwargs. Here is the full new `aggregator.py`:

```python
"""Tree aggregator v9: pool subwords per logical node, then bottom-up
combine of logical KEY nodes into subtree vectors + a document vector.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def _pool_subwords(
    hidden_states: torch.Tensor,
    logical_ids: torch.Tensor,
    n_logical_per_doc: torch.Tensor,
) -> torch.Tensor:
    """Mean-pool subword hidden states into per-logical-node vectors.

    Args:
        hidden_states: (B, N_sub, d) per-subword hidden states from the encoder.
        logical_ids:  (B, N_sub) int tensor; -1 marks padding (ignored).
        n_logical_per_doc: (B,) number of logical nodes per doc; pooled output
            shape is (B, max(n_logical_per_doc), d).

    Returns:
        (B, L_max, d) where L_max = int(n_logical_per_doc.max()).
    """
    B, N_sub, d = hidden_states.shape
    L_max = int(n_logical_per_doc.max().item())
    out = torch.zeros(B, L_max, d, device=hidden_states.device, dtype=hidden_states.dtype)
    count = torch.zeros(B, L_max, device=hidden_states.device, dtype=torch.float32)

    valid = logical_ids >= 0  # (B, N_sub)
    safe_lids = logical_ids.clamp(min=0)  # (B, N_sub)

    # Doc index broadcast over N_sub
    doc_idx = torch.arange(B, device=hidden_states.device).unsqueeze(1).expand(B, N_sub)

    # Linear (doc, logical) → flat slot
    flat = doc_idx * L_max + safe_lids  # (B, N_sub)
    flat_valid = flat[valid]
    h_valid = hidden_states[valid]

    out_flat = out.view(B * L_max, d)
    count_flat = count.view(B * L_max)
    out_flat.index_add_(0, flat_valid, h_valid)
    count_flat.index_add_(
        0, flat_valid, torch.ones_like(flat_valid, dtype=torch.float32),
    )

    pooled = out_flat / count_flat.clamp(min=1.0).unsqueeze(-1).to(out_flat.dtype)
    return pooled.view(B, L_max, d)


class TreeAggregator(nn.Module):
    """v9: pool subwords first, then run v8 logical-level aggregator."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.d_model = d_model

    def forward(
        self,
        hidden_states: torch.Tensor,
        batch_info: list[dict],
        *,
        logical_ids: torch.Tensor,
        n_logical_per_doc: torch.Tensor,
        parent_of_tensor: torch.Tensor | None = None,
        top_level_key_mask: torch.Tensor | None = None,
        edges_by_depth: dict[int, torch.Tensor] | None = None,
        parents_by_depth: dict[int, torch.Tensor] | None = None,
        subtree_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden_states: (B, N_sub, d) per-subword hidden states.
            logical_ids:   (B, N_sub) per-subword logical-node id (-1 for pad).
            n_logical_per_doc: (B,) number of logical nodes per doc.
            (Others: same as v8.)

        Returns:
            (subtree_vecs, doc_vec) where subtree_vecs is (B, L_max, d) —
            indexed by LOGICAL position, not subword.
        """
        pooled = _pool_subwords(hidden_states, logical_ids, n_logical_per_doc)

        provided = (
            parent_of_tensor is not None,
            top_level_key_mask is not None,
            edges_by_depth is not None,
            parents_by_depth is not None,
        )
        if any(provided):
            if not all(provided):
                raise ValueError(
                    "TreeAggregator.forward: vectorized kwargs must be passed all-or-none."
                )
            return self._forward_vectorized(
                pooled,
                top_level_key_mask=top_level_key_mask,
                edges_by_depth=edges_by_depth,
                parents_by_depth=parents_by_depth,
                subtree_mask=subtree_mask,
            )
        return self._forward_reference(pooled, batch_info, subtree_mask=subtree_mask)

    # _forward_reference and _forward_vectorized are unchanged from v8.
    # (Copy from v8 verbatim — they operate on per-logical-node hidden states,
    # which is what `pooled` provides.)
```

For the body of `_forward_reference` and `_forward_vectorized`, copy them VERBATIM from the v8 `aggregator.py` (lines 88-243). They take per-position hidden states; in v9 those positions are logical positions, which is exactly what `pooled` provides.

- [ ] **Step 4: Update `tests/test_aggregator.py` and `tests/test_aggregator_vectorized.py`**

In each file, the `TreeAggregator.forward(...)` calls need the two new required kwargs. For a single-doc test with N hidden states, pass:

```python
logical_ids = torch.arange(N).unsqueeze(0)  # each subword is its own logical (1:1)
n_logical_per_doc = torch.tensor([N])
```

This makes the test data behave like "no BPE expansion happened" — equivalent to v8 behavior — which is exactly what the existing tests assert against.

For batched tests, build `logical_ids` and `n_logical_per_doc` per-doc, padding `logical_ids` with -1.

- [ ] **Step 5: Run all aggregator tests**

Run: `pytest tests/test_aggregator_subword_pooling.py tests/test_aggregator.py tests/test_aggregator_vectorized.py -v 2>&1 | tail -20`
Expected: All pass.

- [ ] **Step 6: Commit**

```bash
git add yaml_bert/aggregator.py tests/test_aggregator_subword_pooling.py tests/test_aggregator.py tests/test_aggregator_vectorized.py
git commit -m "feat(v9): subword pooling step in aggregator"
```

---

## Task 7: Model — thread logical_ids + n_logical_per_doc through forward

**Files:**
- Modify: `yaml_bert/model.py`
- Modify: `tests/test_model_e2e.py`
- Modify: `tests/test_reconstruction_head.py` (vocab construction only)
- Modify: `tests/test_dataset_subtree.py` (vocab construction only)
- Modify: `tests/test_atomic_vocab.py` (vocab construction only)
- Modify: `tests/test_subtree_masking.py` (no-op if it doesn't touch vocab)
- Modify: `yaml_bert/__init__.py` (re-exports unchanged, but verify)

**Why:** The model is the glue. It accepts batches from the new dataset, asks the embedding for subword-level reps, forwards them through the encoder, then hands them (with `logical_ids` + `n_logical_per_doc`) to the aggregator. From the aggregator's pooled output onward, the rest of the model (Token Head, recon head) sees a logical-level world that matches v8.

- [ ] **Step 1: Update `yaml_bert/model.py`'s `forward` signature**

In `forward`, accept and forward the new kwargs. Specifically:

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
        logical_ids: torch.Tensor,           # NEW required
        n_logical_per_doc: torch.Tensor,     # NEW required
        parent_of_tensor: torch.Tensor | None = None,
        top_level_key_mask: torch.Tensor | None = None,
        edges_by_depth: dict[int, torch.Tensor] | None = None,
        parents_by_depth: dict[int, torch.Tensor] | None = None,
        subtree_mask: torch.Tensor | None = None,
        subtree_roots_flat: torch.Tensor | None = None,
    ) -> tuple:
        x = self.embedding(token_ids, node_types, depths, sibling_indices)
        x = self.encoder(x, src_key_padding_mask=padding_mask)

        subtree_vecs, doc_vec = self.aggregator(
            x, batch_info,
            logical_ids=logical_ids,
            n_logical_per_doc=n_logical_per_doc,
            parent_of_tensor=parent_of_tensor,
            top_level_key_mask=top_level_key_mask,
            edges_by_depth=edges_by_depth,
            parents_by_depth=parents_by_depth,
            subtree_mask=subtree_mask,
        )

        # From here, subtree_vecs is (B, L_max, d) — logical-level, NOT subword.
        # The Token Head input must come from the LOGICAL hidden state.
        # The aggregator's internal pooling produced `pooled`; we recompute it
        # here (cheap, one index_add) to feed the head.
        from yaml_bert.aggregator import _pool_subwords
        h_logical = _pool_subwords(x, logical_ids, n_logical_per_doc)  # (B, L_max, d)

        b, L_max, d = h_logical.shape

        if parent_of_tensor is not None:
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
            s_parent = torch.zeros_like(h_logical)
            for doc_idx in range(b):
                parent_of = batch_info[doc_idx]["parent_of"]
                for i in range(min(L_max, len(parent_of))):
                    p = parent_of[i]
                    if p >= 0:
                        s_parent[doc_idx, i] = subtree_vecs[doc_idx, p]
                    else:
                        s_parent[doc_idx, i] = doc_vec[doc_idx]

        doc_vec_broadcast = doc_vec.unsqueeze(1).expand(b, L_max, d)
        head_input = torch.cat([h_logical, doc_vec_broadcast, s_parent], dim=-1)
        logits = self.token_head(head_input)  # (B, L_max, atomic_vocab_size)

        if subtree_roots_flat is not None and subtree_roots_flat.size(0) > 0:
            batch_idx_per_root = subtree_roots_flat[:, 0]
            root_pos_per_root = subtree_roots_flat[:, 1]
            doc_vec_per_root = doc_vec[batch_idx_per_root]
            # Root positions live in the LOGICAL coordinate system, so look up
            # depth/sibling from the dataset's children_info (depth_of), not from
            # the per-subword depths tensor.
            root_depths = torch.tensor(
                [batch_info[b]["depth_of"][r]
                 for b, r in zip(batch_idx_per_root.tolist(), root_pos_per_root.tolist())],
                device=depths.device, dtype=torch.long,
            )
            # sibling: in v9 there's no per-logical sibling tensor in batch_info;
            # subwords share their logical's sibling. Use the first subword's
            # sibling for each (doc, root) by inverting logical_ids.
            root_siblings = []
            for b_i, r_i in zip(batch_idx_per_root.tolist(), root_pos_per_root.tolist()):
                # Find any subword in doc b_i with logical_id == r_i
                pos = (logical_ids[b_i] == r_i).nonzero(as_tuple=True)[0]
                root_siblings.append(
                    int(sibling_indices[b_i, pos[0]].item()) if len(pos) else 0,
                )
            root_siblings = torch.tensor(root_siblings, device=depths.device, dtype=torch.long)

            depth_e = self.embedding.depth_embedding(root_depths)
            sibling_e = self.embedding.sibling_embedding(root_siblings)
            pos_emb_per_root = torch.cat([depth_e, sibling_e], dim=-1)

            recon_logits = self.recon_head(doc_vec_per_root, pos_emb_per_root)
            return logits, doc_vec, recon_logits

        return logits, doc_vec
```

Also update the constructor signature: `YamlBertModel.__init__` no longer needs separate vocab sizes — it accepts `atomic_vocab_size` only (same as v8) and the embedding object itself.

- [ ] **Step 2: Update `tests/test_model_e2e.py` to construct the new vocab/embedding/model**

The pattern to use throughout:

```python
from yaml_bert.vocab import Vocabulary, VocabBuilder
vocab = Vocabulary.from_tokenizer_path(
    tokenizer_path="output_v8_276K_recon_seed42/unified_bpe_8k.json",
    atomic_target_vocab=VocabBuilder.build_atomic_target_vocab(docs, min_freq=1),
)
emb = YamlBertEmbedding(config=cfg, subword_vocab_size=vocab.subword_vocab_size)
model = YamlBertModel(
    config=cfg, embedding=emb,
    atomic_vocab_size=vocab.atomic_target_vocab_size,
)
```

Forward calls must include `logical_ids` and `n_logical_per_doc` from the batch.

- [ ] **Step 3: Sweep other tests for vocab construction**

Edit `tests/test_atomic_vocab.py`, `tests/test_dataset_subtree.py`, `tests/test_reconstruction_head.py` to use the new vocab-construction pattern from Step 2. `tests/test_subtree_masking.py` operates on pure functions and shouldn't need vocab changes.

- [ ] **Step 4: Run the full test suite**

Run: `pytest tests/ -q --tb=line 2>&1 | tail -10`
Expected: same pass count as the Task 1 baseline, with the known pre-existing FileNotFoundError test failures still present and nothing new failing.

- [ ] **Step 5: Commit**

```bash
git add yaml_bert/model.py tests/test_model_e2e.py tests/test_atomic_vocab.py tests/test_dataset_subtree.py tests/test_reconstruction_head.py
git commit -m "feat(v9): thread logical_ids through model forward; update integration tests"
```

---

## Task 8: Per-batch shape audit script

**Files:**
- Create: `scripts/audit_v9_batch.py`

**Why:** A single-batch dry-run end-to-end is the cheapest insurance against shape mismatches between the dataset, collate, model, and loss before we burn GPU time. No GPU required.

- [ ] **Step 1: Write `scripts/audit_v9_batch.py`**

```python
"""v9 audit: one batch through dataset → collate → model.forward, print all shapes.

Run: `PYTHONPATH=. python scripts/audit_v9_batch.py`
"""
from __future__ import annotations

import torch

from yaml_bert.annotator import DomainAnnotator
from yaml_bert.config import YamlBertConfig
from yaml_bert.dataset import YamlBertDataset, collate_fn
from yaml_bert.embedding import YamlBertEmbedding
from yaml_bert.linearizer import YamlLinearizer
from yaml_bert.model import YamlBertModel
from yaml_bert.vocab import Vocabulary, VocabBuilder

TOKENIZER_PATH = "output_v8_276K_recon_seed42/unified_bpe_8k.json"

YAMLS = [
    """apiVersion: apps/v1
kind: Deployment
metadata:
  name: web
spec:
  replicas: 3
""",
    """apiVersion: v1
kind: Service
metadata:
  name: web
spec:
  type: ClusterIP
  ports:
  - port: 80
""",
]


def main():
    lin = YamlLinearizer()
    ann = DomainAnnotator()
    docs = []
    for y in YAMLS:
        nodes = lin.linearize(y)
        ann.annotate(nodes)
        docs.append(nodes)

    atv = VocabBuilder.build_atomic_target_vocab(docs, min_freq=1)
    vocab = Vocabulary.from_tokenizer_path(TOKENIZER_PATH, atomic_target_vocab=atv)

    cfg = YamlBertConfig(
        d_model=64, num_layers=2, num_heads=4, d_ff=128,
        max_depth=8, max_sibling=8, max_seq_len=128,
        mask_prob=0.15, recon_enabled=True,
    )
    ds = YamlBertDataset(docs, vocab, cfg)
    batch = collate_fn([ds[0], ds[1]])

    print("=" * 60)
    print("BATCH SHAPES")
    print("=" * 60)
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k:25s}: tuple{tuple(v.shape)} {v.dtype}")
        elif isinstance(v, dict):
            print(f"  {k:25s}: dict of {len(v)} entries")
        else:
            print(f"  {k:25s}: {type(v).__name__} of {len(v) if hasattr(v, '__len__') else '?'}")

    emb = YamlBertEmbedding(config=cfg, subword_vocab_size=vocab.subword_vocab_size)
    model = YamlBertModel(
        config=cfg, embedding=emb,
        atomic_vocab_size=vocab.atomic_target_vocab_size,
    )

    out = model(
        token_ids=batch["token_ids"],
        node_types=batch["node_types"],
        depths=batch["depths"],
        sibling_indices=batch["sibling_indices"],
        batch_info=batch["batch_info"],
        padding_mask=batch["padding_mask"],
        logical_ids=batch["logical_ids"],
        n_logical_per_doc=batch["n_logical_per_doc"],
        parent_of_tensor=batch["parent_of_tensor"],
        top_level_key_mask=batch["top_level_key_mask"],
        edges_by_depth=batch["edges_by_depth"],
        parents_by_depth=batch["parents_by_depth"],
        subtree_mask=batch.get("subtree_mask"),
        subtree_roots_flat=batch.get("subtree_roots_flat"),
    )

    print()
    print("=" * 60)
    print("MODEL OUTPUTS")
    print("=" * 60)
    if len(out) == 2:
        logits, doc_vec = out
        recon = None
    else:
        logits, doc_vec, recon = out
    print(f"  logits:   {tuple(logits.shape)}  finite={torch.isfinite(logits).all().item()}")
    print(f"  doc_vec:  {tuple(doc_vec.shape)}  finite={torch.isfinite(doc_vec).all().item()}")
    if recon is not None:
        print(f"  recon:    {tuple(recon.shape)}  finite={torch.isfinite(recon).all().item()}")

    assert torch.isfinite(logits).all(), "non-finite logits"
    assert torch.isfinite(doc_vec).all(), "non-finite doc_vec"
    if recon is not None:
        assert torch.isfinite(recon).all(), "non-finite recon"

    print()
    print("AUDIT PASSED.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the audit**

Run: `PYTHONPATH=. python scripts/audit_v9_batch.py 2>&1 | tail -25`
Expected: AUDIT PASSED.

- [ ] **Step 3: Commit**

```bash
git add scripts/audit_v9_batch.py
git commit -m "feat(v9): single-batch audit script"
```

---

## Task 9: Quick-mode 5K training run + acceptance gate

**Files:**
- Modify: `scripts/train.py`

**Why:** Before committing to a multi-hour 276K run, confirm the v9 pipeline trains end-to-end on a small subset — MLM loss decreases, recon loss decreases, no NaN, no OOM, output is sensible.

- [ ] **Step 1: Modify `scripts/train.py` to use the v9 vocab**

Replace the `VocabBuilder().build(...)` block in `train.py` with:

```python
from yaml_bert.vocab import Vocabulary, VocabBuilder

# v9: tokenizer is trained offline; we just load it.
TOKENIZER_PATH = os.environ.get(
    "YAML_BERT_TOKENIZER",
    "output_v8_276K_recon_seed42/unified_bpe_8k.json",
)
atomic_target_vocab = VocabBuilder.build_atomic_target_vocab(
    cached.docs, min_freq=args.min_freq,
)
vocab = Vocabulary.from_tokenizer_path(
    tokenizer_path=TOKENIZER_PATH,
    atomic_target_vocab=atomic_target_vocab,
)
print(f"  subword vocab size: {vocab.subword_vocab_size}")
print(f"  atomic target vocab: {vocab.atomic_target_vocab_size}")
```

Also update the model construction:

```python
emb = YamlBertEmbedding(config=cfg, subword_vocab_size=vocab.subword_vocab_size)
model = YamlBertModel(
    config=cfg, embedding=emb,
    atomic_vocab_size=vocab.atomic_target_vocab_size,
)
```

And update the `_forward_v8` helper to include `logical_ids` and `n_logical_per_doc` from the batch. Rename it to `_forward_v9` for clarity.

Update default output dir naming: `output_v9_<N>K_recon_seed<S>`.

- [ ] **Step 2: Run a 5K-doc training**

Run:
```bash
PYTHONPATH=. python scripts/train.py --max-docs 5000 --epochs 5 --reconstruction on --output-dir output_v9_5K_recon_seed42 --seed 42 2>&1 | tee /tmp/v9_5k.log | tail -30
```

Expected after ~15-30 minutes on CPU (or 5 min on GPU):
- MLM loss decreases epoch over epoch
- Recon loss decreases epoch over epoch
- No NaN/Inf reported
- Final checkpoint written to `output_v9_5K_recon_seed42/`

- [ ] **Step 3: Sanity-check the trained model's doc_vec on the C/E collision case**

Use the suggest.py or a small custom script to encode the two YAMLs:

```yaml
apiVersion: v1
kind: Pod
metadata: { name: web-1, namespace: staging }
spec: { containers: [{ name: app, image: nginx, ports: [{ containerPort: 80 }] }] }
```
and
```yaml
apiVersion: v1
kind: Pod
metadata: { name: web-3, namespace: staging }
spec: { containers: [{ name: app, image: nginx, ports: [{ containerPort: 80 }] }] }
```

Compute `cos(doc_vec_C, doc_vec_E)`. With v9 sub-tokenization, the inputs are no longer literally identical (the names BPE differently), so the cosine should be **< 1.0** (was exactly 1.0 in v8). A drop into 0.97-0.99 range is the expected and good outcome.

If this still returns 1.0, the dataset wiring is wrong — investigate before proceeding to Task 10.

- [ ] **Step 4: Commit**

```bash
git add scripts/train.py
git commit -m "feat(v9): wire v9 vocab+model into train.py; 5K quick-mode validated"
```

---

## Task 10: Full 276K JarvisLabs training run (acceptance gate)

**Files:**
- `scripts/train_service.sh` (existing JL launcher; verify it works with v9)
- Create: `docs/v9-subword-results.md`

**Why:** This is the production-scale training run. Acceptance criteria match the spec's validation plan.

- [ ] **Step 1: Verify the JL launcher**

```bash
cat scripts/train_service.sh | head -40
```

Confirm it accepts the same flags as `scripts/train.py`. If it uses a hardcoded reference to v8 paths, update those to v9 equivalents.

- [ ] **Step 2: Read the JarvisLabs skill before launching**

Run: `cat ~/.claude/skills/jarvislabs/SKILL.md`

Follow the JL launch procedure documented there. Use the same L4 GPU configuration as the v8 run.

- [ ] **Step 3: Launch full v9 training**

Per JL skill:
```bash
jl start --gpu L4 --script scripts/train_service.sh \
    --env MAX_DOCS=276520 \
    --env EPOCHS=30 \
    --env RECONSTRUCTION=on \
    --env OUTPUT_DIR=output_v9_276K_recon_seed42 \
    --env SEED=42
```

(Adjust to actual JL CLI syntax from the skill doc.)

Monitor via `jl logs <job_id>`.

- [ ] **Step 4: Acceptance gate — write `docs/v9-subword-results.md`**

After training completes, the results doc must record:

1. **Loss trajectory** — MLM and recon loss per epoch, vs v8's published trajectory.

2. **Capability test pass rate** — run `pytest model_tests/test_capabilities.py model_tests/test_structural.py model_tests/test_bigger_boat.py` against the v9 checkpoint. v9 must be within ±10% of v8's pass rates.

3. **The 4 HF Space structural probes**, with explicit pass/fail:
   - Pod ± initContainers
   - Service type (ClusterIP / NodePort / LoadBalancer)
   - Pods same/different namespace (v9-specific finding: did this start passing now that namespace values are no longer [UNK]?)
   - Pod vs Deployment wrapping the same Pod

4. **The C/E collision case** — adding `web-3` should no longer collide with `web-1`. Record the cosine.

5. **k-NN purity by kind** on a held-out 5K — must not be worse than v8's purity.

6. **Final param count** — confirm the ~13.25M estimate from the spec.

7. **Go/No-Go decision** — if all gates pass, v9 replaces v8 in the HF Space and in the active checkpoint pointer. If any fail, the doc records the failure mode, and a follow-up plan addresses it.

- [ ] **Step 5: Commit results**

```bash
git add docs/v9-subword-results.md
git commit -m "docs(v9): full-training results + go/no-go"
```

---

## Self-Review

**1. Spec coverage check:**

| Spec requirement | Task |
|---|---|
| Unified byte-level BPE, vocab=8192 | Task 2 (tokenizer wrapper) + already-trained artifact |
| Long-value rule (`[LONG_VALUE]` ≥ 256, truncate 64-255) | Task 2 (`SubwordTokenizer.encode_token`) |
| Dataset emits `logical_id` per subword | Task 5 |
| Replicated `node_type`/`depth`/`sibling`/`parent_path` across subwords | Task 5 |
| Subword embedding merges key+value tables | Task 4 |
| Aggregator subword pooling step | Task 6 |
| Doc_vec structural-only (KEYs) | Task 6 (uses v8 vectorized path verbatim — KEY-only by construction) |
| Whole-word MLM masking | Task 5 |
| Token Head predicts atomic_target_vocab | Task 7 (head input is pooled logical hidden) |
| Recon objective kept, structural target | Task 7 (recon path preserved) |
| max_seq_len=768 | Task 5 |
| d_model=256 | unchanged from v8 default |
| From-scratch initialization | Task 9 (training script) |
| Per-batch shape audit | Task 8 |
| 5K validation run | Task 9 |
| 276K training + acceptance gate | Task 10 |
| Archive v8 | Task 1 |

All spec items covered.

**2. Placeholder scan:** No TBDs. Every step has either complete code or an exact command. Task 6 Step 3 says "copy `_forward_reference` and `_forward_vectorized` verbatim from v8 (lines 88-243)" — the actual verbatim code is what the engineer will copy, and the line range gives them the exact source.

**3. Type consistency:** `SubwordTokenizer.encode_token(token, is_value=...)`, `Vocabulary.encode_token(token, is_value=...)`, `Vocabulary.from_tokenizer_path(tokenizer_path, atomic_target_vocab)`, `YamlBertEmbedding(config, subword_vocab_size)`, `TreeAggregator.forward(... logical_ids, n_logical_per_doc, ...)`, `YamlBertModel.forward(... logical_ids, n_logical_per_doc, ...)` — names used consistently across tasks.

One nuance worth flagging: in Task 7 the model's `forward` recomputes `_pool_subwords(x, logical_ids, n_logical_per_doc)` to get `h_logical` for the Token Head. The aggregator also pools internally. That's two pooling calls per forward — inefficient. An optimization for later: have the aggregator return both `pooled` AND `(subtree_vecs, doc_vec)`. Out of scope for the initial plan to keep the diff small; flagged in `docs/v9-subword-results.md` as a follow-up.
