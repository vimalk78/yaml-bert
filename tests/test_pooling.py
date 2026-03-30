import torch
from yaml_bert.pooling import DocumentPooling, supervised_contrastive_loss


def test_pooling_output_shape():
    d_model = 64
    pooling = DocumentPooling(d_model=d_model, num_heads=4)

    batch_size = 2
    seq_len = 10
    kind_hidden = torch.randn(batch_size, 1, d_model)
    all_hidden = torch.randn(batch_size, seq_len, d_model)

    doc_emb = pooling(kind_hidden, all_hidden)
    assert doc_emb.shape == (batch_size, d_model)


def test_pooling_different_inputs_different_outputs():
    d_model = 64
    pooling = DocumentPooling(d_model=d_model, num_heads=4)

    kind = torch.randn(1, 1, d_model)
    hidden_a = torch.randn(1, 10, d_model)
    hidden_b = torch.randn(1, 10, d_model)

    emb_a = pooling(kind, hidden_a)
    emb_b = pooling(kind, hidden_b)
    assert not torch.allclose(emb_a, emb_b)


def test_pooling_deterministic():
    d_model = 64
    pooling = DocumentPooling(d_model=d_model, num_heads=4)
    pooling.eval()

    kind = torch.randn(1, 1, d_model)
    hidden = torch.randn(1, 10, d_model)

    with torch.no_grad():
        emb1 = pooling(kind, hidden)
        emb2 = pooling(kind, hidden)
    assert torch.equal(emb1, emb2)


def test_contrastive_loss_same_labels_lower():
    torch.manual_seed(42)
    emb_good = torch.tensor([
        [1.0, 0.0], [0.9, 0.1],
        [0.0, 1.0], [0.1, 0.9],
    ])
    labels = torch.tensor([0, 0, 1, 1])

    emb_bad = torch.tensor([
        [1.0, 0.0], [0.0, 1.0],
        [0.9, 0.1], [0.1, 0.9],
    ])

    loss_good = supervised_contrastive_loss(emb_good, labels)
    loss_bad = supervised_contrastive_loss(emb_bad, labels)
    assert loss_good < loss_bad


def test_contrastive_loss_positive():
    emb = torch.randn(8, 64, requires_grad=True)
    labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    loss = supervised_contrastive_loss(emb, labels)
    assert loss.item() > 0
    assert loss.requires_grad
