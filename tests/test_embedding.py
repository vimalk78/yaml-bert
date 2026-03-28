import torch
from yaml_bert.config import YamlBertConfig
from yaml_bert.embedding import YamlBertEmbedding
from yaml_bert.types import NodeType


def _make_embedding(d_model: int = 64, key_vocab: int = 100, val_vocab: int = 200) -> YamlBertEmbedding:
    config = YamlBertConfig(d_model=d_model)
    return YamlBertEmbedding(
        config=config,
        key_vocab_size=key_vocab,
        value_vocab_size=val_vocab,
    )


def test_embedding_output_shape():
    d_model = 64
    emb = _make_embedding(d_model=d_model)

    batch_size = 2
    seq_len = 5
    token_ids = torch.randint(0, 50, (batch_size, seq_len))
    node_types = torch.zeros(batch_size, seq_len, dtype=torch.long)
    depths = torch.randint(0, 5, (batch_size, seq_len))
    sibling_indices = torch.randint(0, 3, (batch_size, seq_len))
    parent_key_ids = torch.randint(0, 50, (batch_size, seq_len))

    output = emb(token_ids, node_types, depths, sibling_indices, parent_key_ids)

    assert output.shape == (batch_size, seq_len, d_model)


def test_embedding_routes_by_node_type():
    emb = _make_embedding(d_model=32, key_vocab=10, val_vocab=10)

    token_ids = torch.tensor([[5, 5]])
    depths = torch.tensor([[0, 0]])
    siblings = torch.tensor([[0, 0]])
    parent_keys = torch.tensor([[0, 0]])

    node_types_key = torch.tensor([[0, 0]])
    node_types_value = torch.tensor([[1, 1]])

    out_key = emb(token_ids, node_types_key, depths, siblings, parent_keys)
    out_value = emb(token_ids, node_types_value, depths, siblings, parent_keys)

    assert not torch.allclose(out_key, out_value)


def test_different_parent_keys_produce_different_embeddings():
    emb = _make_embedding(d_model=32, key_vocab=10, val_vocab=10)

    token_ids = torch.tensor([[3]])
    node_types = torch.tensor([[0]])
    depths = torch.tensor([[1]])
    siblings = torch.tensor([[0]])

    parent_a = torch.tensor([[4]])
    parent_b = torch.tensor([[5]])

    out_a = emb(token_ids, node_types, depths, siblings, parent_a)
    out_b = emb(token_ids, node_types, depths, siblings, parent_b)

    assert not torch.allclose(out_a, out_b)
