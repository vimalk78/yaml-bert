import torch
from yaml_bert.config import YamlBertConfig
from yaml_bert.embedding import YamlBertEmbedding
from yaml_bert.model import YamlBertModel


TEST_CONFIG = YamlBertConfig(d_model=64, num_layers=2, num_heads=2)
SIMPLE_VOCAB = 100
KIND_VOCAB = 30


def _make_model():
    emb = YamlBertEmbedding(
        config=TEST_CONFIG, key_vocab_size=50, value_vocab_size=80,
    )
    return YamlBertModel(
        config=TEST_CONFIG, embedding=emb,
        simple_vocab_size=SIMPLE_VOCAB, kind_vocab_size=KIND_VOCAB,
    )


def test_v4_output_shapes():
    model = _make_model()
    batch, seq = 2, 8
    token_ids = torch.randint(0, 30, (batch, seq))
    node_types = torch.zeros(batch, seq, dtype=torch.long)
    depths = torch.randint(0, 5, (batch, seq))
    siblings = torch.randint(0, 3, (batch, seq))

    simple_logits, kind_logits = model(token_ids, node_types, depths, siblings)

    assert simple_logits.shape == (batch, seq, SIMPLE_VOCAB)
    assert kind_logits.shape == (batch, seq, KIND_VOCAB)


def test_v4_with_padding_mask():
    model = _make_model()
    batch, seq = 2, 8
    token_ids = torch.randint(0, 30, (batch, seq))
    node_types = torch.zeros(batch, seq, dtype=torch.long)
    depths = torch.randint(0, 5, (batch, seq))
    siblings = torch.randint(0, 3, (batch, seq))
    mask = torch.tensor([[False]*8, [False]*5 + [True]*3])

    simple_logits, kind_logits = model(token_ids, node_types, depths, siblings, padding_mask=mask)
    assert simple_logits.shape == (batch, seq, SIMPLE_VOCAB)


def test_v4_loss():
    model = _make_model()
    batch, seq = 2, 8
    token_ids = torch.randint(0, 30, (batch, seq))
    node_types = torch.zeros(batch, seq, dtype=torch.long)
    depths = torch.randint(0, 5, (batch, seq))
    siblings = torch.randint(0, 3, (batch, seq))

    simple_logits, kind_logits = model(token_ids, node_types, depths, siblings)

    # simple_labels: some masked, rest -100
    simple_labels = torch.full((batch, seq), -100, dtype=torch.long)
    simple_labels[0, 1] = 10
    simple_labels[1, 3] = 20

    # kind_labels: some masked, rest -100
    kind_labels = torch.full((batch, seq), -100, dtype=torch.long)
    kind_labels[0, 4] = 5

    loss, breakdown = model.compute_loss(simple_logits, simple_labels, kind_logits, kind_labels)

    assert loss.dim() == 0
    assert loss.item() > 0
    assert loss.requires_grad
    assert "simple" in breakdown
    assert "kind" in breakdown


def test_v4_no_kind_ids_needed():
    """v4 forward doesn't accept kind_ids — not in the architecture."""
    model = _make_model()
    import inspect
    sig = inspect.signature(model.forward)
    assert "kind_ids" not in sig.parameters
    assert "parent_key_ids" not in sig.parameters
