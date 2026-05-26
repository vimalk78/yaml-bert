from yaml_bert.linearizer import YamlLinearizer
from yaml_bert.vocab import Vocabulary, VocabBuilder


def test_build_vocab_from_simple_yaml():
    yaml_str = "apiVersion: v1\nkind: Pod\nmetadata:\n  name: nginx\n"
    linearizer = YamlLinearizer()
    nodes = linearizer.linearize(yaml_str)

    builder = VocabBuilder()
    vocab = builder.build(nodes)

    assert "apiVersion" in vocab.key_vocab
    assert "kind" in vocab.key_vocab
    assert "metadata" in vocab.key_vocab
    assert "name" in vocab.key_vocab

    assert "v1" in vocab.value_vocab
    assert "Pod" in vocab.value_vocab
    assert "nginx" in vocab.value_vocab

    assert "apiVersion" not in vocab.value_vocab
    assert "v1" not in vocab.key_vocab


def test_special_tokens_present():
    builder = VocabBuilder()
    vocab = builder.build([])

    assert "[UNK]" in vocab.special_tokens
    assert "[PAD]" in vocab.special_tokens
    assert "[MASK]" in vocab.special_tokens

    assert vocab.special_tokens["[PAD]"] == 0
    assert vocab.special_tokens["[UNK]"] == 1
    assert vocab.special_tokens["[MASK]"] == 2


def test_encode_decode_roundtrip():
    yaml_str = "apiVersion: v1\nkind: Pod\n"
    from yaml_bert.linearizer import YamlLinearizer
    linearizer = YamlLinearizer()
    nodes = linearizer.linearize(yaml_str)

    builder = VocabBuilder()
    vocab = builder.build(nodes)

    key_id = vocab.encode_key("apiVersion")
    assert vocab.decode_key(key_id) == "apiVersion"

    value_id = vocab.encode_value("v1")
    assert vocab.decode_value(value_id) == "v1"

    unk_id = vocab.special_tokens["[UNK]"]
    assert vocab.encode_key("nonexistent_key") == unk_id
    assert vocab.encode_value("nonexistent_value") == unk_id


def test_save_and_load(tmp_path):
    yaml_str = "apiVersion: v1\nkind: Pod\n"
    from yaml_bert.linearizer import YamlLinearizer
    linearizer = YamlLinearizer()
    nodes = linearizer.linearize(yaml_str)

    builder = VocabBuilder()
    vocab = builder.build(nodes)

    vocab_path = str(tmp_path / "vocab.json")
    vocab.save(vocab_path)

    loaded = Vocabulary.load(vocab_path)

    assert loaded.key_vocab == vocab.key_vocab
    assert loaded.value_vocab == vocab.value_vocab
    assert loaded.special_tokens == vocab.special_tokens
    assert loaded.encode_key("apiVersion") == vocab.encode_key("apiVersion")
    assert loaded.encode_value("v1") == vocab.encode_value("v1")


def test_min_freq_filtering():
    yaml_str = "a: x\nb: y\na: x\n"
    from yaml_bert.linearizer import YamlLinearizer
    linearizer = YamlLinearizer()
    nodes1 = linearizer.linearize("a: x\nb: y\n")
    nodes2 = linearizer.linearize("a: x\nc: z\n")
    all_nodes = nodes1 + nodes2

    builder = VocabBuilder()
    vocab = builder.build(all_nodes, min_freq=2)

    assert "a" in vocab.key_vocab
    assert "b" not in vocab.key_vocab
    assert "c" not in vocab.key_vocab
    assert "x" in vocab.value_vocab
    assert "y" not in vocab.value_vocab
