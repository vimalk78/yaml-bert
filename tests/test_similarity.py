import torch
from yaml_bert.config import YamlBertConfig
from yaml_bert.embedding import YamlBertEmbedding
from yaml_bert.model import YamlBertModel
from yaml_bert.similarity import extract_hidden_states, get_document_embedding
from yaml_bert.pooling import DocumentPooling
from yaml_bert.linearizer import YamlLinearizer
from yaml_bert.annotator import DomainAnnotator
from yaml_bert.vocab import VocabBuilder


def _build_model():
    linearizer = YamlLinearizer()
    annotator = DomainAnnotator()
    nodes = linearizer.linearize("""\
apiVersion: v1
kind: Pod
metadata:
  name: test
spec:
  containers:
  - name: app
    image: nginx
""")
    annotator.annotate(nodes)
    vocab = VocabBuilder().build(nodes)

    config = YamlBertConfig(d_model=64, num_layers=2, num_heads=2)
    emb = YamlBertEmbedding(
        config=config,
        key_vocab_size=vocab.key_vocab_size,
        value_vocab_size=vocab.value_vocab_size,
        kind_vocab_size=vocab.kind_vocab_size,
    )
    model = YamlBertModel(
        config=config, embedding=emb,
        key_vocab_size=vocab.key_vocab_size,
        kind_vocab_size=vocab.kind_vocab_size,
    )
    return model, vocab


def test_extract_hidden_states():
    model, vocab = _build_model()
    yaml_text = """\
apiVersion: v1
kind: Pod
metadata:
  name: test
"""
    hidden, kind_pos = extract_hidden_states(model, vocab, yaml_text)
    assert hidden.dim() == 2
    assert hidden.shape[1] == 64
    assert kind_pos >= 0


def test_get_document_embedding():
    model, vocab = _build_model()
    pooling = DocumentPooling(d_model=64, num_heads=2)

    yaml_text = """\
apiVersion: v1
kind: Pod
metadata:
  name: test
"""
    emb = get_document_embedding(model, pooling, vocab, yaml_text)
    assert emb.shape == (64,)


def test_different_yamls_different_embeddings():
    model, vocab = _build_model()
    pooling = DocumentPooling(d_model=64, num_heads=2)

    yaml_a = """\
apiVersion: v1
kind: Pod
metadata:
  name: a
spec:
  containers:
  - name: app
    image: nginx
"""
    yaml_b = """\
apiVersion: v1
kind: Pod
metadata:
  name: b
"""
    emb_a = get_document_embedding(model, pooling, vocab, yaml_a)
    emb_b = get_document_embedding(model, pooling, vocab, yaml_b)
    assert not torch.allclose(emb_a, emb_b)
