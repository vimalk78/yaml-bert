"""Tests for v9 YamlBertEmbedding (single subword embedding table)."""
import pytest
import torch

from yaml_bert.config import YamlBertConfig, TreePosVariant
from yaml_bert.embedding import YamlBertEmbedding


def _make_emb(vocab_size=200, d=16):
    cfg = YamlBertConfig(
        d_model=d, num_layers=1, num_heads=1, d_ff=32,
        max_depth=8, max_sibling=8, max_seq_len=64,
    )
    return YamlBertEmbedding(config=cfg, subword_vocab_size=vocab_size)


def test_init_accepts_subword_vocab_size():
    emb = _make_emb()
    assert emb.subword_embedding.num_embeddings == 200


def test_forward_output_shape():
    emb = _make_emb(vocab_size=128, d=16)
    B, N = 2, 5
    out = emb(
        token_ids=torch.zeros(B, N, dtype=torch.long),
        node_types=torch.zeros(B, N, dtype=torch.long),
        depths=torch.zeros(B, N, dtype=torch.long),
        sibling_indices=torch.zeros(B, N, dtype=torch.long),
    )
    assert out.shape == (B, N, 16)


def test_node_type_embedding_still_present_and_used():
    emb = _make_emb()
    # Two positions with same token id but different node_types should differ
    ids = torch.tensor([[5, 5]])
    nt = torch.tensor([[0, 1]])
    z = torch.zeros_like(ids)
    out = emb(ids, nt, z, z)
    assert not torch.allclose(out[0, 0], out[0, 1])


def test_old_key_value_tables_no_longer_exist():
    emb = _make_emb()
    assert not hasattr(emb, "key_embedding")
    assert not hasattr(emb, "value_embedding")


def test_tree_pos_variant_no_depth_still_works():
    cfg = YamlBertConfig(
        d_model=16, num_layers=1, num_heads=1, d_ff=32,
        max_depth=8, max_sibling=8, max_seq_len=64,
        tree_pos_variant=TreePosVariant.NO_DEPTH,
    )
    emb = YamlBertEmbedding(config=cfg, subword_vocab_size=64)
    assert emb.depth_embedding is None


def test_tree_pos_variant_no_sibling_still_works():
    cfg = YamlBertConfig(
        d_model=16, num_layers=1, num_heads=1, d_ff=32,
        max_depth=8, max_sibling=8, max_seq_len=64,
        tree_pos_variant=TreePosVariant.NO_SIBLING,
    )
    emb = YamlBertEmbedding(config=cfg, subword_vocab_size=64)
    assert emb.sibling_embedding is None
    assert emb.depth_embedding is not None  # NO_SIBLING keeps depth


def test_tree_pos_variant_sequential_uses_pos_embedding():
    cfg = YamlBertConfig(
        d_model=16, num_layers=1, num_heads=1, d_ff=32,
        max_depth=8, max_sibling=8, max_seq_len=64,
        tree_pos_variant=TreePosVariant.SEQUENTIAL,
    )
    emb = YamlBertEmbedding(config=cfg, subword_vocab_size=64)
    assert emb.depth_embedding is None
    assert emb.sibling_embedding is None
    assert emb.pos_embedding is not None
    # Smoke: forward with the SEQUENTIAL variant produces non-NaN output
    import torch
    out = emb(
        token_ids=torch.zeros(1, 5, dtype=torch.long),
        node_types=torch.zeros(1, 5, dtype=torch.long),
        depths=torch.zeros(1, 5, dtype=torch.long),
        sibling_indices=torch.zeros(1, 5, dtype=torch.long),
    )
    assert torch.isfinite(out).all()
