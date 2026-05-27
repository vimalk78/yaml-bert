import torch

from yaml_bert.attention_pooling import AttentionPooling


def test_output_shape():
    ap = AttentionPooling(d_model=8)
    input: torch.Tensor
    output: torch.Tensor

    input = torch.randn([2, 5, 8])
    output, _ = ap(input)

    assert output.shape == (2, 8)

    input = torch.randn([2, 10, 8])
    output, _ = ap(input)
    assert output.shape == (2, 8)

    input = torch.randn([2, 100, 8])
    output, _ = ap(input)
    assert output.shape == (2, 8)


def test_weights():
    ap: AttentionPooling = AttentionPooling(d_model=8)
    input: torch.Tensor
    weights: torch.Tensor

    input = torch.randn([2, 5, 8])
    _, weights = ap(input)

    weights_sum: torch.Tensor = weights.sum(dim=-1)
    assert weights_sum.shape == (2, 1)
    assert torch.allclose(weights_sum, torch.ones(2, 1))


def test_single_token():
    ap = AttentionPooling(d_model=8)
    input: torch.Tensor
    output: torch.Tensor
    weights: torch.Tensor

    input = torch.randn([1, 1, 8])
    output, weights = ap(input)
    print(f"{input.shape=}")
    print(f"{output.shape=}")
    assert torch.equal(input, output.unsqueeze(1))
    sum: torch.Tensor = weights.sum(dim=-1)  #
    assert sum[-1] == 1.0
