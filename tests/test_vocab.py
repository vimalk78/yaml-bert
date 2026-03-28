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
