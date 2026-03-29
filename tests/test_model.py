import torch
from yaml_bert.config import YamlBertConfig
from yaml_bert.embedding import YamlBertEmbedding
from yaml_bert.model import YamlBertModel


TEST_CONFIG: YamlBertConfig = YamlBertConfig(d_model=64, num_layers=2, num_heads=2)
KEY_VOCAB_SIZE: int = 100
VALUE_VOCAB_SIZE: int = 200


def _make_model() -> YamlBertModel:
    emb = YamlBertEmbedding(
        config=TEST_CONFIG,
        key_vocab_size=KEY_VOCAB_SIZE,
        value_vocab_size=VALUE_VOCAB_SIZE,
    )
    return YamlBertModel(
        config=TEST_CONFIG,
        embedding=emb,
        key_vocab_size=KEY_VOCAB_SIZE,
    )


def test_model_output_shape():
    model = _make_model()

    batch_size = 2
    seq_len = 5
    token_ids = torch.randint(0, 50, (batch_size, seq_len))
    node_types = torch.zeros(batch_size, seq_len, dtype=torch.long)
    depths = torch.randint(0, 5, (batch_size, seq_len))
    siblings = torch.randint(0, 3, (batch_size, seq_len))
    parent_keys = torch.randint(0, 50, (batch_size, seq_len))

    key_logits = model(token_ids, node_types, depths, siblings, parent_keys)

    assert key_logits.shape == (batch_size, seq_len, KEY_VOCAB_SIZE)


def test_model_with_padding_mask():
    model = _make_model()

    batch_size = 2
    seq_len = 5
    token_ids = torch.randint(0, 50, (batch_size, seq_len))
    node_types = torch.zeros(batch_size, seq_len, dtype=torch.long)
    depths = torch.randint(0, 5, (batch_size, seq_len))
    siblings = torch.randint(0, 3, (batch_size, seq_len))
    parent_keys = torch.randint(0, 50, (batch_size, seq_len))

    padding_mask = torch.tensor([
        [False, False, False, False, False],
        [False, False, False, True, True],
    ])

    key_logits = model(
        token_ids, node_types, depths, siblings, parent_keys,
        padding_mask=padding_mask,
    )

    assert key_logits.shape == (batch_size, seq_len, KEY_VOCAB_SIZE)


def test_model_loss_computation():
    model = _make_model()

    batch_size = 2
    seq_len = 5
    token_ids = torch.randint(0, 50, (batch_size, seq_len))
    node_types = torch.zeros(batch_size, seq_len, dtype=torch.long)
    depths = torch.randint(0, 5, (batch_size, seq_len))
    siblings = torch.randint(0, 3, (batch_size, seq_len))
    parent_keys = torch.randint(0, 50, (batch_size, seq_len))

    labels = torch.full((batch_size, seq_len), -100, dtype=torch.long)
    labels[0, 1] = 10
    labels[1, 0] = 20

    key_logits = model(token_ids, node_types, depths, siblings, parent_keys)
    loss = model.compute_loss(key_logits, labels)

    assert loss.dim() == 0
    assert loss.item() > 0
    assert loss.requires_grad


def test_model_with_kind_ids():
    config = YamlBertConfig(d_model=64, num_layers=2, num_heads=2)
    emb = YamlBertEmbedding(
        config=config, key_vocab_size=100, value_vocab_size=200, kind_vocab_size=10,
    )
    model = YamlBertModel(config=config, embedding=emb, key_vocab_size=100)

    batch_size = 2
    seq_len = 5
    token_ids = torch.randint(0, 50, (batch_size, seq_len))
    node_types = torch.zeros(batch_size, seq_len, dtype=torch.long)
    depths = torch.randint(0, 5, (batch_size, seq_len))
    siblings = torch.randint(0, 3, (batch_size, seq_len))
    parent_keys = torch.randint(0, 50, (batch_size, seq_len))
    kind_ids = torch.tensor([[1, 1, 1, 1, 1], [2, 2, 2, 2, 2]])

    key_logits = model(token_ids, node_types, depths, siblings, parent_keys, kind_ids=kind_ids)
    assert key_logits.shape == (batch_size, seq_len, 100)


def test_model_without_kind_ids_backward_compatible():
    model = _make_model()
    batch_size = 2
    seq_len = 5
    token_ids = torch.randint(0, 50, (batch_size, seq_len))
    node_types = torch.zeros(batch_size, seq_len, dtype=torch.long)
    depths = torch.randint(0, 5, (batch_size, seq_len))
    siblings = torch.randint(0, 3, (batch_size, seq_len))
    parent_keys = torch.randint(0, 50, (batch_size, seq_len))

    key_logits = model(token_ids, node_types, depths, siblings, parent_keys)
    assert key_logits.shape == (batch_size, seq_len, KEY_VOCAB_SIZE)
