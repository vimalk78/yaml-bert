import os
import pytest
import torch
from yaml_bert.config import YamlBertConfig
from yaml_bert.embedding import YamlBertEmbedding
from yaml_bert.linearizer import YamlLinearizer
from yaml_bert.dataset import compute_children_info
from yaml_bert.model import YamlBertModel
from yaml_bert.vocab import Vocabulary, VocabBuilder

_TOKENIZER_PATH = "output_v8_276K_recon_seed42/unified_bpe_8k.json"


def _skip_if_no_tokenizer():
    if not os.path.exists(_TOKENIZER_PATH):
        pytest.skip(f"tokenizer not found: {_TOKENIZER_PATH}")


def _build_vocab(docs):
    """Build a v9 Vocabulary from a list-of-node-lists."""
    return Vocabulary.from_tokenizer_path(
        tokenizer_path=_TOKENIZER_PATH,
        atomic_target_vocab=VocabBuilder.build_atomic_target_vocab(docs, min_freq=1),
    )


def test_model_forward_pass_shape():
    """YamlBertModel forward returns logits of shape (B, L_max, atomic_vocab_size).

    v9: logits are LOGICAL-level, not subword-level.
    """
    _skip_if_no_tokenizer()
    from yaml_bert.dataset import YamlBertDataset, collate_fn

    yaml_str = "apiVersion: v1\nkind: Pod\nspec:\n  containers:\n  - name: x\n"
    docs = [YamlLinearizer().linearize(yaml_str)]
    vocab = _build_vocab(docs)

    config = YamlBertConfig(d_model=32, num_layers=2, num_heads=4)
    ds = YamlBertDataset(docs, vocab, config)
    batch = collate_fn([ds[0]])

    emb = YamlBertEmbedding(config=config, subword_vocab_size=vocab.subword_vocab_size)
    model = YamlBertModel(config=config, embedding=emb,
                          atomic_vocab_size=vocab.atomic_target_vocab_size)

    logits, doc_vec = model(
        token_ids=batch["token_ids"],
        node_types=batch["node_types"],
        depths=batch["depths"],
        sibling_indices=batch["sibling_indices"],
        batch_info=batch["batch_info"],
        padding_mask=batch["padding_mask"],
        logical_ids=batch["logical_ids"],
        n_logical_per_doc=batch["n_logical_per_doc"],
    )

    L_max = int(batch["n_logical_per_doc"].max().item())
    assert logits.shape == (1, L_max, vocab.atomic_target_vocab_size)
    assert doc_vec.shape == (1, 32)


def test_model_backward_no_nan():
    """Loss is finite and gradients exist after one backward pass."""
    _skip_if_no_tokenizer()
    from yaml_bert.dataset import YamlBertDataset, collate_fn

    yaml_str = "apiVersion: v1\nkind: Pod\nmetadata:\n  name: x\n"
    docs = [YamlLinearizer().linearize(yaml_str)]
    vocab = _build_vocab(docs)

    # mask_prob=1.0 ensures at least one key is supervised (avoids all-ignored NaN).
    config = YamlBertConfig(d_model=32, num_layers=2, num_heads=4, mask_prob=1.0)
    ds = YamlBertDataset(docs, vocab, config)
    batch = collate_fn([ds[0]])

    emb = YamlBertEmbedding(config=config, subword_vocab_size=vocab.subword_vocab_size)
    model = YamlBertModel(config=config, embedding=emb,
                          atomic_vocab_size=vocab.atomic_target_vocab_size)

    logits, _ = model(
        token_ids=batch["token_ids"],
        node_types=batch["node_types"],
        depths=batch["depths"],
        sibling_indices=batch["sibling_indices"],
        batch_info=batch["batch_info"],
        padding_mask=batch["padding_mask"],
        logical_ids=batch["logical_ids"],
        n_logical_per_doc=batch["n_logical_per_doc"],
    )

    loss = torch.nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)),
        batch["atomic_labels"].view(-1),
        ignore_index=-100,
    )
    assert torch.isfinite(loss)
    loss.backward()
    grads_nonzero = any(p.grad is not None and p.grad.abs().sum() > 0
                        for p in model.parameters() if p.requires_grad)
    assert grads_nonzero


def test_smoke_e2e_small_batch():
    """End-to-end: dataset → collate → model → loss → backward."""
    _skip_if_no_tokenizer()
    from yaml_bert.dataset import YamlBertDataset, collate_fn

    yamls = [
        "apiVersion: v1\nkind: Pod\nmetadata:\n  name: a\n",
        "apiVersion: v1\nkind: Service\nspec:\n  x: 1\n",
        "apiVersion: apps/v1\nkind: Deployment\nspec:\n  replicas: 3\n",
    ]
    docs = [YamlLinearizer().linearize(y) for y in yamls]
    vocab = _build_vocab(docs)

    config = YamlBertConfig(d_model=32, num_layers=2, num_heads=4, mask_prob=0.5)
    ds = YamlBertDataset(docs, vocab, config)
    batch = collate_fn([ds[i] for i in range(len(ds))])

    emb = YamlBertEmbedding(config=config, subword_vocab_size=vocab.subword_vocab_size)
    model = YamlBertModel(config=config, embedding=emb,
                          atomic_vocab_size=vocab.atomic_target_vocab_size)

    logits, doc_vec = model(
        token_ids=batch["token_ids"],
        node_types=batch["node_types"],
        depths=batch["depths"],
        sibling_indices=batch["sibling_indices"],
        batch_info=batch["batch_info"],
        padding_mask=batch["padding_mask"],
        logical_ids=batch["logical_ids"],
        n_logical_per_doc=batch["n_logical_per_doc"],
    )

    loss = torch.nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)),
        batch["atomic_labels"].view(-1),
        ignore_index=-100,
    )
    assert torch.isfinite(loss)
    loss.backward()


def test_smoke_e2e_vectorized_path():
    """End-to-end with collate_fn precompute kwargs passed to YamlBertModel.

    Asserts the vectorized path produces logits and a loss, AND that the
    logits are numerically equivalent to the reference path on the same
    inputs (catches regressions where the two paths diverge).

    v9: both paths produce LOGICAL-level logits (B, L_max, V_atomic).
    The reference path (no edges_by_depth etc.) still requires logical_ids
    and n_logical_per_doc.
    """
    _skip_if_no_tokenizer()
    from yaml_bert.dataset import YamlBertDataset, collate_fn

    yamls = [
        "apiVersion: v1\nkind: Pod\nmetadata:\n  name: a\n",
        "apiVersion: v1\nkind: Service\nspec:\n  x: 1\n",
        "apiVersion: apps/v1\nkind: Deployment\nspec:\n  replicas: 3\n",
    ]
    docs = [YamlLinearizer().linearize(y) for y in yamls]
    vocab = _build_vocab(docs)

    config = YamlBertConfig(d_model=32, num_layers=2, num_heads=4, mask_prob=0.5)
    ds = YamlBertDataset(docs, vocab, config)
    batch = collate_fn([ds[i] for i in range(len(ds))])

    emb = YamlBertEmbedding(config=config, subword_vocab_size=vocab.subword_vocab_size)
    model = YamlBertModel(config=config, embedding=emb,
                          atomic_vocab_size=vocab.atomic_target_vocab_size)
    model.eval()

    # Reference path: no vectorized tree kwargs, but logical_ids required
    with torch.no_grad():
        ref_logits, ref_doc = model(
            token_ids=batch["token_ids"],
            node_types=batch["node_types"],
            depths=batch["depths"],
            sibling_indices=batch["sibling_indices"],
            batch_info=batch["batch_info"],
            padding_mask=batch["padding_mask"],
            logical_ids=batch["logical_ids"],
            n_logical_per_doc=batch["n_logical_per_doc"],
        )

    # Vectorized path: pass tree tensor kwargs
    with torch.no_grad():
        vec_logits, vec_doc = model(
            token_ids=batch["token_ids"],
            node_types=batch["node_types"],
            depths=batch["depths"],
            sibling_indices=batch["sibling_indices"],
            batch_info=batch["batch_info"],
            padding_mask=batch["padding_mask"],
            logical_ids=batch["logical_ids"],
            n_logical_per_doc=batch["n_logical_per_doc"],
            parent_of_tensor=batch["parent_of_tensor"],
            top_level_key_mask=batch["top_level_key_mask"],
            edges_by_depth=batch["edges_by_depth"],
            parents_by_depth=batch["parents_by_depth"],
        )

    assert torch.allclose(ref_logits, vec_logits, atol=1e-5), (
        f"logits diverge: max diff = "
        f"{(ref_logits - vec_logits).abs().max().item()}"
    )
    assert torch.allclose(ref_doc, vec_doc, atol=1e-5)

    # Backward must actually work on the vectorized path.
    model.train()
    train_logits, _ = model(
        token_ids=batch["token_ids"],
        node_types=batch["node_types"],
        depths=batch["depths"],
        sibling_indices=batch["sibling_indices"],
        batch_info=batch["batch_info"],
        padding_mask=batch["padding_mask"],
        logical_ids=batch["logical_ids"],
        n_logical_per_doc=batch["n_logical_per_doc"],
        parent_of_tensor=batch["parent_of_tensor"],
        top_level_key_mask=batch["top_level_key_mask"],
        edges_by_depth=batch["edges_by_depth"],
        parents_by_depth=batch["parents_by_depth"],
    )
    loss = torch.nn.functional.cross_entropy(
        train_logits.view(-1, train_logits.size(-1)),
        batch["atomic_labels"].view(-1),
        ignore_index=-100,
    )
    assert torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads, "no parameters received gradients via vectorized path"
    assert all(torch.isfinite(g).all() for g in grads), "non-finite gradient"


def test_model_returns_recon_logits_when_subtree_info_provided():
    """YamlBertModel.forward returns recon_logits with shape (M, V_atomic) when
    the batch contains subtree_roots_flat (i.e., recon is active).

    Uses a large YAML (55+ nodes) to guarantee pick_subtrees returns at least
    one root (MIN_DOC_NODES=10; MAX_TOTAL_SUBTREE_FRACTION=0.05 means N>20
    is required for even the smallest subtree).
    """
    _skip_if_no_tokenizer()
    from yaml_bert.dataset import YamlBertDataset, collate_fn

    # Large YAML docs (55+ nodes each) to satisfy pick_subtrees' size threshold.
    yamls = [
        "apiVersion: apps/v1\nkind: Deployment\nmetadata:\n  name: myapp\n"
        "  namespace: prod\n  labels:\n    app: myapp\n    version: v1\n"
        "spec:\n  replicas: 3\n  selector:\n    matchLabels:\n      app: myapp\n"
        "  template:\n    metadata:\n      labels:\n        app: myapp\n"
        "    spec:\n      containers:\n      - name: app\n        image: myapp:latest\n"
        "        ports:\n        - containerPort: 8080\n"
        "        env:\n        - name: DB_HOST\n          value: localhost\n"
        "        - name: DB_PORT\n          value: \"5432\"\n"
        "        resources:\n          requests:\n            memory: 128Mi\n"
        "            cpu: 100m\n          limits:\n            memory: 256Mi\n"
        "            cpu: 500m\n",
        "apiVersion: v1\nkind: Pod\nmetadata:\n  name: myapp\n"
        "  namespace: prod\n  labels:\n    app: myapp\n    version: v1\n"
        "spec:\n  containers:\n  - name: app\n    image: myapp:latest\n"
        "    ports:\n    - containerPort: 8080\n"
        "    env:\n    - name: DB_HOST\n      value: localhost\n"
        "    - name: DB_PORT\n      value: \"5432\"\n"
        "    resources:\n      requests:\n        memory: 128Mi\n"
        "        cpu: 100m\n      limits:\n        memory: 256Mi\n"
        "        cpu: 500m\n",
    ]
    docs = [YamlLinearizer().linearize(y) for y in yamls]
    vocab = _build_vocab(docs)
    config = YamlBertConfig(d_model=32, num_layers=2, num_heads=4,
                            mask_prob=0.0, recon_enabled=True)
    ds = YamlBertDataset(docs, vocab, config)
    batch = collate_fn([ds[i] for i in range(len(ds))])
    assert batch["subtree_roots_flat"].size(0) > 0, (
        "test precondition failed: no subtree roots picked. "
        "Increase doc size or check pick_subtrees thresholds."
    )

    emb = YamlBertEmbedding(config=config, subword_vocab_size=vocab.subword_vocab_size)
    model = YamlBertModel(config=config, embedding=emb,
                          atomic_vocab_size=vocab.atomic_target_vocab_size)
    model.train()

    out = model(
        token_ids=batch["token_ids"],
        node_types=batch["node_types"],
        depths=batch["depths"],
        sibling_indices=batch["sibling_indices"],
        batch_info=batch["batch_info"],
        padding_mask=batch["padding_mask"],
        logical_ids=batch["logical_ids"],
        n_logical_per_doc=batch["n_logical_per_doc"],
        parent_of_tensor=batch["parent_of_tensor"],
        top_level_key_mask=batch["top_level_key_mask"],
        edges_by_depth=batch["edges_by_depth"],
        parents_by_depth=batch["parents_by_depth"],
        subtree_mask=batch["subtree_mask"],
        subtree_roots_flat=batch["subtree_roots_flat"],
    )
    assert len(out) == 3, "expected (logits, doc_vec, recon_logits) tuple"
    logits, doc_vec, recon_logits = out

    M = batch["subtree_roots_flat"].size(0)
    assert recon_logits.shape == (M, vocab.atomic_target_vocab_size)
    # Verify recon loss flows back through doc_vec
    target = batch["bag_of_keys_targets_flat"]
    recon_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        recon_logits, target,
    )
    recon_loss.backward()
    for name, p in model.recon_head.named_parameters():
        assert p.grad is not None, f"recon_head.{name} has no gradient"
        assert torch.isfinite(p.grad).all(), f"recon_head.{name} gradient is non-finite"


def test_model_omits_recon_logits_when_no_subtree_info():
    """YamlBertModel.forward returns (logits, doc_vec) — old shape — when subtree
    kwargs are absent. Backward-compat with Phase 0/1 callers."""
    _skip_if_no_tokenizer()
    from yaml_bert.dataset import YamlBertDataset, collate_fn

    yamls = ["apiVersion: v1\nkind: Pod\nspec:\n  x: 1\n"]
    docs = [YamlLinearizer().linearize(y) for y in yamls]
    vocab = _build_vocab(docs)
    config = YamlBertConfig(d_model=16, num_layers=1, num_heads=2,
                            mask_prob=0.0, recon_enabled=False)
    ds = YamlBertDataset(docs, vocab, config)
    batch = collate_fn([ds[0]])

    emb = YamlBertEmbedding(config=config, subword_vocab_size=vocab.subword_vocab_size)
    model = YamlBertModel(config=config, embedding=emb,
                          atomic_vocab_size=vocab.atomic_target_vocab_size)
    model.eval()
    out = model(
        token_ids=batch["token_ids"],
        node_types=batch["node_types"],
        depths=batch["depths"],
        sibling_indices=batch["sibling_indices"],
        batch_info=batch["batch_info"],
        padding_mask=batch["padding_mask"],
        logical_ids=batch["logical_ids"],
        n_logical_per_doc=batch["n_logical_per_doc"],
    )
    assert len(out) == 2  # (logits, doc_vec) — no recon
