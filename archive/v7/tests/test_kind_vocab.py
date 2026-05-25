from yaml_bert.linearizer import YamlLinearizer
from yaml_bert.vocab import Vocabulary, VocabBuilder


def test_build_kind_vocab():
    linearizer = YamlLinearizer()
    nodes1 = linearizer.linearize("""\
apiVersion: apps/v1
kind: Deployment
metadata:
  name: test
""")
    nodes2 = linearizer.linearize("""\
apiVersion: v1
kind: Service
metadata:
  name: test
""")
    nodes3 = linearizer.linearize("""\
apiVersion: v1
kind: Pod
metadata:
  name: test
""")

    builder = VocabBuilder()
    vocab = builder.build(nodes1 + nodes2 + nodes3)

    assert vocab.kind_vocab is not None
    assert "Deployment" in vocab.kind_vocab
    assert "Service" in vocab.kind_vocab
    assert "Pod" in vocab.kind_vocab


def test_encode_kind():
    linearizer = YamlLinearizer()
    nodes = linearizer.linearize("""\
apiVersion: v1
kind: Pod
metadata:
  name: test
""")
    vocab = VocabBuilder().build(nodes)

    kind_id = vocab.encode_kind("Pod")
    assert kind_id == vocab.kind_vocab["Pod"]

    unknown_id = vocab.encode_kind("UnknownResource")
    assert unknown_id == vocab.kind_vocab["[NO_KIND]"]


def test_no_kind_document():
    linearizer = YamlLinearizer()
    nodes = linearizer.linearize("""\
apiVersion: v1
data:
  key: value
""")
    vocab = VocabBuilder().build(nodes)

    no_kind_id = vocab.encode_kind("")
    assert no_kind_id == vocab.kind_vocab["[NO_KIND]"]


def test_kind_vocab_saved_and_loaded(tmp_path):
    linearizer = YamlLinearizer()
    nodes = linearizer.linearize("""\
apiVersion: v1
kind: Pod
metadata:
  name: test
""")
    vocab = VocabBuilder().build(nodes)

    path = str(tmp_path / "vocab.json")
    vocab.save(path)
    loaded = Vocabulary.load(path)

    assert loaded.kind_vocab == vocab.kind_vocab
    assert loaded.encode_kind("Pod") == vocab.encode_kind("Pod")


def test_kind_vocab_size():
    linearizer = YamlLinearizer()
    nodes = linearizer.linearize("""\
apiVersion: v1
kind: Pod
metadata:
  name: test
""")
    vocab = VocabBuilder().build(nodes)

    assert vocab.kind_vocab_size >= 2  # at least [NO_KIND] + Pod


def test_backward_compatible_load(tmp_path):
    import json
    v1_data = {
        "key_vocab": {"apiVersion": 3, "kind": 4},
        "value_vocab": {"v1": 3, "Pod": 4},
        "special_tokens": {"[PAD]": 0, "[UNK]": 1, "[MASK]": 2},
    }
    path = str(tmp_path / "v1_vocab.json")
    with open(path, "w") as f:
        json.dump(v1_data, f)

    loaded = Vocabulary.load(path)
    assert loaded.kind_vocab == {"[NO_KIND]": 0}
    assert loaded.encode_kind("Pod") == 0
