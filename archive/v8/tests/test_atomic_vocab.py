from yaml_bert.linearizer import YamlLinearizer
from yaml_bert.vocab import VocabBuilder


def test_atomic_target_vocab_contains_keys():
    """atomic_target_vocab should contain key tokens appearing in training data."""
    linearizer = YamlLinearizer()
    nodes = linearizer.linearize("""\
apiVersion: v1
kind: Pod
metadata:
  name: x
spec:
  containers:
  - name: c
    image: nginx
""")
    builder = VocabBuilder()
    vocab = builder.build(nodes, min_freq=1)

    # Atomic targets are key tokens
    assert "name" in vocab.atomic_target_vocab
    assert "image" in vocab.atomic_target_vocab
    assert "containers" in vocab.atomic_target_vocab
    assert "spec" in vocab.atomic_target_vocab


def test_atomic_target_vocab_size_property():
    """atomic_target_vocab_size includes special tokens."""
    linearizer = YamlLinearizer()
    nodes = linearizer.linearize("apiVersion: v1\nkind: Pod\n")
    vocab = VocabBuilder().build(nodes, min_freq=1)
    # 2 keys (apiVersion, kind) + 3 special tokens ([PAD], [UNK], [MASK])
    assert vocab.atomic_target_vocab_size == 2 + 3
    assert vocab.atomic_target_vocab_size == len(vocab.atomic_target_vocab) + 3


def test_atomic_target_vocab_save_load_roundtrip(tmp_path):
    """Vocabulary.save → Vocabulary.load preserves atomic_target_vocab."""
    from yaml_bert.vocab import Vocabulary

    nodes = YamlLinearizer().linearize("apiVersion: v1\nkind: Pod\nspec:\n  x: 1\n")
    vocab = VocabBuilder().build(nodes, min_freq=1)
    path = str(tmp_path / "vocab.json")
    vocab.save(path)

    loaded = Vocabulary.load(path)
    assert loaded.atomic_target_vocab == vocab.atomic_target_vocab
    assert loaded.atomic_target_vocab_size == vocab.atomic_target_vocab_size


def test_load_v7_era_vocab_without_atomic_field(tmp_path):
    """A vocab.json without atomic_target_vocab (v7-era) loads with empty atomic vocab."""
    import json

    from yaml_bert.vocab import Vocabulary

    path = tmp_path / "v7_vocab.json"
    v7_data = {
        "key_vocab": {"foo": 3, "bar": 4},
        "value_vocab": {"v1": 3},
        "special_tokens": {"[PAD]": 0, "[UNK]": 1, "[MASK]": 2},
        "kind_vocab": {"[NO_KIND]": 0, "Pod": 1},
        "simple_target_vocab": {},
        "kind_target_vocab": {},
        # NOTE: no atomic_target_vocab field
    }
    path.write_text(json.dumps(v7_data))

    loaded = Vocabulary.load(str(path))
    assert loaded.atomic_target_vocab == {}
    assert loaded.atomic_target_vocab_size == 3  # just special tokens
    # Should not crash on encode lookup
    assert loaded.encode_atomic_target("unknown") == loaded.special_tokens["[UNK]"]


def test_encode_atomic_target_known_and_unknown():
    """encode_atomic_target returns the right id for known tokens, [UNK] for unknown."""
    nodes = YamlLinearizer().linearize("spec:\n  x: 1\n")
    vocab = VocabBuilder().build(nodes, min_freq=1)
    # "spec" is in the vocab — should not be [UNK]
    assert vocab.encode_atomic_target("spec") != vocab.special_tokens["[UNK]"]
    # "nonexistent" is not — should be [UNK]
    assert vocab.encode_atomic_target("nonexistent") == vocab.special_tokens["[UNK]"]
