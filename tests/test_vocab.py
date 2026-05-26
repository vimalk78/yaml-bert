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
