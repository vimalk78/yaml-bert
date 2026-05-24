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
    # 4 keys (apiVersion, kind) + 3 special tokens ([PAD], [UNK], [MASK])
    assert vocab.atomic_target_vocab_size == len(vocab.atomic_target_vocab) + 3
