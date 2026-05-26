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
