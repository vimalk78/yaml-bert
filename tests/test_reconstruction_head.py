"""ReconstructionHead unit tests: shapes + backward."""
import torch

from yaml_bert.reconstruction_head import ReconstructionHead


def test_reconstruction_head_output_shape():
    """Output has shape (M, V_atomic)."""
    head = ReconstructionHead(d_model=64, d_pos=16, atomic_vocab_size=100)
    M = 7
    doc_vec = torch.randn(M, 64)
    pos_emb = torch.randn(M, 16)
    logits = head(doc_vec, pos_emb)
    assert logits.shape == (M, 100)


def test_reconstruction_head_backward_produces_gradients():
    """Backward through BCE loss produces finite gradients on all params."""
    head = ReconstructionHead(d_model=32, d_pos=8, atomic_vocab_size=50)
    M = 4
    doc_vec = torch.randn(M, 32, requires_grad=True)
    pos_emb = torch.randn(M, 8, requires_grad=True)
    target = (torch.rand(M, 50) > 0.5).float()

    logits = head(doc_vec, pos_emb)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
    loss.backward()

    assert torch.isfinite(loss)
    for name, p in head.named_parameters():
        assert p.grad is not None, f"no grad on {name}"
        assert torch.isfinite(p.grad).all(), f"non-finite grad on {name}"
    assert doc_vec.grad is not None
    assert torch.isfinite(doc_vec.grad).all()


def test_reconstruction_head_param_count_is_small():
    """Sanity: head is ~205K params at default sizes (d_model=256, d_pos=48, V=427)."""
    head = ReconstructionHead(d_model=256, d_pos=48, atomic_vocab_size=427)
    n = sum(p.numel() for p in head.parameters())
    # Linear(304→256) = 304*256 + 256 = 78,080
    # Linear(256→427) = 256*427 + 427 = 109,739
    # Total = 187,819 (close to ~205K target)
    assert 180_000 < n < 220_000, f"unexpected param count: {n}"
