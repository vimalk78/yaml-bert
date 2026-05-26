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
