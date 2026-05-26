import os
import pytest

from yaml_bert.linearizer import YamlLinearizer
from yaml_bert.vocab import VocabBuilder

_TOKENIZER_PATH = "output_v8_276K_recon_seed42/unified_bpe_8k.json"


def _skip_if_no_tokenizer():
    if not os.path.exists(_TOKENIZER_PATH):
        pytest.skip(f"tokenizer not found: {_TOKENIZER_PATH}")


def _build_vocab_v9(docs):
    """Build a v9 Vocabulary (requires tokenizer on disk)."""
    from yaml_bert.vocab import Vocabulary
    return Vocabulary.from_tokenizer_path(
        tokenizer_path=_TOKENIZER_PATH,
        atomic_target_vocab=VocabBuilder.build_atomic_target_vocab(docs, min_freq=1),
    )


def test_atomic_target_vocab_contains_keys():
    """atomic_target_vocab should contain key tokens appearing in training data."""
    nodes = YamlLinearizer().linearize("""\
apiVersion: v1
kind: Pod
metadata:
  name: x
spec:
  containers:
  - name: c
    image: nginx
""")
    atomic = VocabBuilder.build_atomic_target_vocab([nodes], min_freq=1)

    # Atomic targets are key tokens
    assert "name" in atomic
    assert "image" in atomic
    assert "containers" in atomic
    assert "spec" in atomic


def test_atomic_target_vocab_size_property():
    """atomic_target_vocab_size includes special tokens.

    v9: 4 special token slots (ids 0-3: pad/unk/mask/long_value).
    """
    _skip_if_no_tokenizer()
    nodes = YamlLinearizer().linearize("apiVersion: v1\nkind: Pod\n")
    vocab = _build_vocab_v9([nodes])
    # 2 keys (apiVersion, kind) + 4 special token ids (0-3 reserved)
    assert vocab.atomic_target_vocab_size == 2 + 4
    assert vocab.atomic_target_vocab_size == len(vocab.atomic_target_vocab) + 4


def test_atomic_target_vocab_save_load_roundtrip(tmp_path):
    """Vocabulary.save → Vocabulary.load preserves atomic_target_vocab.

    v9: save/load requires a valid tokenizer_path on disk.
    """
    _skip_if_no_tokenizer()
    from yaml_bert.vocab import Vocabulary

    nodes = YamlLinearizer().linearize("apiVersion: v1\nkind: Pod\nspec:\n  x: 1\n")
    vocab = _build_vocab_v9([nodes])
    path = str(tmp_path / "vocab.json")
    vocab.save(path)

    loaded = Vocabulary.load(path)
    assert loaded.atomic_target_vocab == vocab.atomic_target_vocab
    assert loaded.atomic_target_vocab_size == vocab.atomic_target_vocab_size


def test_load_v7_era_vocab_without_atomic_field(tmp_path):
    """v7-era vocab.json (no tokenizer_path field) is not loadable in v9.

    v9 Vocabulary.load() requires a tokenizer_path to reconstruct the subword
    tokenizer. v7-era files only had key_vocab/value_vocab. Skipped pending a
    v9 migration path (e.g., a compatibility shim in Vocabulary.load).
    """
    pytest.skip(
        "v9 migration: Vocabulary.load() requires 'tokenizer_path' in JSON "
        "(v7-era files have key_vocab/value_vocab only). "
        "Compatibility shim not yet implemented."
    )


def test_encode_atomic_target_known_and_unknown():
    """encode_atomic_target returns the right id for known tokens, unk for unknown."""
    _skip_if_no_tokenizer()
    nodes = YamlLinearizer().linearize("spec:\n  x: 1\n")
    vocab = _build_vocab_v9([nodes])
    # "spec" is in the vocab — should not be unk
    assert vocab.encode_atomic_target("spec") != vocab.unk_id
    # "nonexistent" is not — should be unk
    assert vocab.encode_atomic_target("nonexistent") == vocab.unk_id
