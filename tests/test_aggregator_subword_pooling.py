"""Unit tests for the subword-pooling step inside TreeAggregator."""
import torch

from yaml_bert.aggregator import _pool_subwords


def test_pool_subwords_basic():
    """3 subwords of logical 0, 2 subwords of logical 1, 1 of logical 2."""
    B, N_sub, d = 1, 6, 4
    hidden = torch.tensor([[
        [1.0, 0, 0, 0],
        [3.0, 0, 0, 0],
        [5.0, 0, 0, 0],   # logical 0: mean = 3.0
        [10.0, 0, 0, 0],
        [20.0, 0, 0, 0],  # logical 1: mean = 15.0
        [7.0, 0, 0, 0],   # logical 2: mean = 7.0
    ]])
    logical_ids = torch.tensor([[0, 0, 0, 1, 1, 2]])
    n_logical = torch.tensor([3])
    out = _pool_subwords(hidden, logical_ids, n_logical)
    assert out.shape == (B, 3, d)
    assert torch.allclose(out[0, 0, 0], torch.tensor(3.0))
    assert torch.allclose(out[0, 1, 0], torch.tensor(15.0))
    assert torch.allclose(out[0, 2, 0], torch.tensor(7.0))


def test_pool_subwords_ignores_negative_logical_ids():
    """logical_ids == -1 means padding; those subwords shouldn't affect pools."""
    hidden = torch.tensor([[
        [1.0, 0],
        [3.0, 0],
        [999.0, 0],  # pad, ignored
    ]])
    logical_ids = torch.tensor([[0, 0, -1]])
    n_logical = torch.tensor([1])
    out = _pool_subwords(hidden, logical_ids, n_logical)
    assert out.shape == (1, 1, 2)
    assert torch.allclose(out[0, 0, 0], torch.tensor(2.0))


def test_pool_subwords_batched_with_different_n_logical():
    """Two docs, different number of logical nodes."""
    hidden = torch.tensor([
        [[1.0, 0], [2.0, 0], [3.0, 0], [0.0, 0]],  # doc 0: logical [0,0,1]
        [[10.0, 0], [20.0, 0], [30.0, 0], [40.0, 0]],  # doc 1: logical [0,1,2,3]
    ])
    logical_ids = torch.tensor([[0, 0, 1, -1], [0, 1, 2, 3]])
    n_logical = torch.tensor([2, 4])
    out = _pool_subwords(hidden, logical_ids, n_logical)
    assert out.shape == (2, 4, 2)
    # Doc 0: logical 0 = mean(1, 2) = 1.5; logical 1 = 3
    assert torch.allclose(out[0, 0, 0], torch.tensor(1.5))
    assert torch.allclose(out[0, 1, 0], torch.tensor(3.0))
    # Doc 1: logical 0-3 = 10, 20, 30, 40
    assert torch.allclose(out[1, :, 0], torch.tensor([10., 20., 30., 40.]))
