import torch
from yaml_bert.config import TreePosVariant, YamlBertConfig
from yaml_bert.embedding import YamlBertEmbedding


def test_embedding_output_shape():
    config = YamlBertConfig(d_model=64)
    emb = YamlBertEmbedding(config=config, key_vocab_size=100, value_vocab_size=200)

    token_ids = torch.tensor([[3, 4, 5]])
    node_types = torch.tensor([[0, 1, 0]])
    depths = torch.tensor([[0, 0, 1]])
    siblings = torch.tensor([[0, 0, 0]])

    output = emb(token_ids, node_types, depths, siblings)
    assert output.shape == (1, 3, 64)


def test_embedding_no_kind_or_parent_params():
    config = YamlBertConfig(d_model=32)
    emb = YamlBertEmbedding(config=config, key_vocab_size=10, value_vocab_size=10)

    # Should NOT have kind_embedding or parent_key_embedding
    assert not hasattr(emb, 'kind_embedding') or emb.kind_embedding is None
    assert not hasattr(emb, 'parent_key_embedding')

    # Default (FULL) variant: 5 embedding tables
    embedding_count = sum(1 for name, _ in emb.named_modules() if isinstance(_, torch.nn.Embedding))
    assert embedding_count == 5  # key, value, depth, sibling, node_type


def test_embedding_routes_by_node_type():
    config = YamlBertConfig(d_model=32)
    emb = YamlBertEmbedding(config=config, key_vocab_size=10, value_vocab_size=10)

    token_ids = torch.tensor([[5, 5]])
    depths = torch.tensor([[0, 0]])
    siblings = torch.tensor([[0, 0]])

    node_types_key = torch.tensor([[0, 0]])
    node_types_value = torch.tensor([[1, 1]])

    out_key = emb(token_ids, node_types_key, depths, siblings)
    out_value = emb(token_ids, node_types_value, depths, siblings)

    assert not torch.allclose(out_key, out_value)


def test_different_depths_produce_different_embeddings():
    config = YamlBertConfig(d_model=32)
    emb = YamlBertEmbedding(config=config, key_vocab_size=10, value_vocab_size=10)

    token_ids = torch.tensor([[3, 3]])
    node_types = torch.tensor([[0, 0]])
    siblings = torch.tensor([[0, 0]])

    depths_a = torch.tensor([[0, 0]])
    depths_b = torch.tensor([[2, 2]])

    out_a = emb(token_ids, node_types, depths_a, siblings)
    out_b = emb(token_ids, node_types, depths_b, siblings)

    assert not torch.allclose(out_a, out_b)


# --- Ablation variants ---

def _make_inputs():
    token_ids = torch.tensor([[3, 4, 5]])
    node_types = torch.tensor([[0, 1, 0]])
    depths = torch.tensor([[0, 1, 2]])
    siblings = torch.tensor([[0, 1, 2]])
    return token_ids, node_types, depths, siblings


def test_no_depth_variant_ignores_depth():
    config = YamlBertConfig(d_model=32, tree_pos_variant=TreePosVariant.NO_DEPTH)
    emb = YamlBertEmbedding(config=config, key_vocab_size=10, value_vocab_size=10)
    assert emb.depth_embedding is None
    assert emb.sibling_embedding is not None
    assert emb.pos_embedding is None

    token_ids, node_types, _, siblings = _make_inputs()
    out_a = emb(token_ids, node_types, torch.tensor([[0, 0, 0]]), siblings)
    out_b = emb(token_ids, node_types, torch.tensor([[5, 7, 9]]), siblings)
    assert torch.allclose(out_a, out_b)


def test_no_sibling_variant_ignores_sibling():
    config = YamlBertConfig(d_model=32, tree_pos_variant=TreePosVariant.NO_SIBLING)
    emb = YamlBertEmbedding(config=config, key_vocab_size=10, value_vocab_size=10)
    assert emb.depth_embedding is not None
    assert emb.sibling_embedding is None
    assert emb.pos_embedding is None

    token_ids, node_types, depths, _ = _make_inputs()
    out_a = emb(token_ids, node_types, depths, torch.tensor([[0, 0, 0]]))
    out_b = emb(token_ids, node_types, depths, torch.tensor([[5, 7, 9]]))
    assert torch.allclose(out_a, out_b)


def test_sequential_variant_uses_position_only():
    config = YamlBertConfig(d_model=32, tree_pos_variant=TreePosVariant.SEQUENTIAL)
    emb = YamlBertEmbedding(config=config, key_vocab_size=10, value_vocab_size=10)
    assert emb.depth_embedding is None
    assert emb.sibling_embedding is None
    assert emb.pos_embedding is not None

    token_ids, node_types, _, _ = _make_inputs()
    # depth/sibling are ignored → must produce identical outputs
    out_a = emb(token_ids, node_types, torch.tensor([[0, 1, 2]]), torch.tensor([[0, 1, 2]]))
    out_b = emb(token_ids, node_types, torch.tensor([[7, 8, 9]]), torch.tensor([[3, 4, 5]]))
    assert torch.allclose(out_a, out_b)


def test_sequential_variant_positions_differ():
    config = YamlBertConfig(d_model=32, tree_pos_variant=TreePosVariant.SEQUENTIAL)
    emb = YamlBertEmbedding(config=config, key_vocab_size=10, value_vocab_size=10)

    # Same single token at two different sequence positions → different embedding
    token_ids = torch.tensor([[3, 3]])
    node_types = torch.tensor([[0, 0]])
    depths = torch.tensor([[0, 0]])
    siblings = torch.tensor([[0, 0]])
    out = emb(token_ids, node_types, depths, siblings)
    assert not torch.allclose(out[:, 0, :], out[:, 1, :])
