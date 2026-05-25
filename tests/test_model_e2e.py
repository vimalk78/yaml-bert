import torch
from yaml_bert.config import YamlBertConfig
from yaml_bert.embedding import YamlBertEmbedding
from yaml_bert.linearizer import YamlLinearizer
from yaml_bert.v8_dataset import compute_children_info
from yaml_bert.v8_model import V8Model
from yaml_bert.vocab import VocabBuilder


def test_v8_model_forward_pass_shape():
    """V8Model forward returns logits of shape (B, N, atomic_vocab_size)."""
    yaml_str = "apiVersion: v1\nkind: Pod\nspec:\n  containers:\n  - name: x\n"
    nodes = YamlLinearizer().linearize(yaml_str)
    vocab = VocabBuilder().build(nodes, min_freq=1)
    info = compute_children_info(nodes)

    config = YamlBertConfig(d_model=32, num_layers=2, num_heads=4,
                            tree_bias_enabled=False, v8_mode=True)
    emb = YamlBertEmbedding(config=config,
                            key_vocab_size=vocab.key_vocab_size,
                            value_vocab_size=vocab.value_vocab_size)
    model = V8Model(config=config, embedding=emb,
                    atomic_vocab_size=vocab.atomic_target_vocab_size)

    n = len(nodes)
    token_ids = torch.zeros(1, n, dtype=torch.long)
    node_types = torch.zeros(1, n, dtype=torch.long)
    depths = torch.zeros(1, n, dtype=torch.long)
    siblings = torch.zeros(1, n, dtype=torch.long)

    logits, doc_vec = model(
        token_ids=token_ids,
        node_types=node_types,
        depths=depths,
        sibling_indices=siblings,
        batch_info=[info],
    )

    assert logits.shape == (1, n, vocab.atomic_target_vocab_size)
    assert doc_vec.shape == (1, 32)


def test_v8_model_backward_no_nan():
    """Loss is finite and gradients exist after one backward pass."""
    yaml_str = "apiVersion: v1\nkind: Pod\nmetadata:\n  name: x\n"
    nodes = YamlLinearizer().linearize(yaml_str)
    vocab = VocabBuilder().build(nodes, min_freq=1)
    info = compute_children_info(nodes)

    config = YamlBertConfig(d_model=32, num_layers=2, num_heads=4,
                            tree_bias_enabled=False, v8_mode=True)
    emb = YamlBertEmbedding(config=config,
                            key_vocab_size=vocab.key_vocab_size,
                            value_vocab_size=vocab.value_vocab_size)
    model = V8Model(config=config, embedding=emb,
                    atomic_vocab_size=vocab.atomic_target_vocab_size)

    n = len(nodes)
    token_ids = torch.zeros(1, n, dtype=torch.long)
    node_types = torch.zeros(1, n, dtype=torch.long)
    depths = torch.zeros(1, n, dtype=torch.long)
    siblings = torch.zeros(1, n, dtype=torch.long)
    labels = torch.full((1, n), -100, dtype=torch.long)
    labels[0, 0] = 1  # supervise one position

    logits, _ = model(token_ids=token_ids, node_types=node_types,
                       depths=depths, sibling_indices=siblings,
                       batch_info=[info])

    loss = torch.nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)),
        labels.view(-1),
        ignore_index=-100,
    )
    assert torch.isfinite(loss)
    loss.backward()
    # At least one parameter should have a non-zero gradient
    grads_nonzero = any(p.grad is not None and p.grad.abs().sum() > 0
                       for p in model.parameters() if p.requires_grad)
    assert grads_nonzero


def test_v8_smoke_e2e_small_batch():
    """End-to-end: dataset → collate → model → loss → backward."""
    from yaml_bert.linearizer import YamlLinearizer
    from yaml_bert.vocab import VocabBuilder
    from yaml_bert.config import YamlBertConfig
    from yaml_bert.embedding import YamlBertEmbedding
    from yaml_bert.v8_dataset import V8Dataset, v8_collate_fn

    yamls = [
        "apiVersion: v1\nkind: Pod\nmetadata:\n  name: a\n",
        "apiVersion: v1\nkind: Service\nspec:\n  x: 1\n",
        "apiVersion: apps/v1\nkind: Deployment\nspec:\n  replicas: 3\n",
    ]
    documents = [YamlLinearizer().linearize(y) for y in yamls]
    flat = [n for doc in documents for n in doc]
    vocab = VocabBuilder().build(flat, min_freq=1)

    config = YamlBertConfig(d_model=32, num_layers=2, num_heads=4,
                            v8_mode=True, mask_prob=0.5)
    ds = V8Dataset(documents, vocab, config)
    batch = v8_collate_fn([ds[i] for i in range(len(ds))])

    emb = YamlBertEmbedding(config=config,
                            key_vocab_size=vocab.key_vocab_size,
                            value_vocab_size=vocab.value_vocab_size)
    model = V8Model(config=config, embedding=emb,
                    atomic_vocab_size=vocab.atomic_target_vocab_size)

    logits, doc_vec = model(
        token_ids=batch["token_ids"],
        node_types=batch["node_types"],
        depths=batch["depths"],
        sibling_indices=batch["sibling_indices"],
        batch_info=batch["batch_info"],
        padding_mask=batch["padding_mask"],
    )

    loss = torch.nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)),
        batch["atomic_labels"].view(-1),
        ignore_index=-100,
    )
    assert torch.isfinite(loss)
    loss.backward()


def test_v8_smoke_e2e_vectorized_path():
    """End-to-end with v8_collate_fn precompute kwargs passed to V8Model.

    Asserts the vectorized path produces logits and a loss, AND that the
    logits are numerically equivalent to the reference path on the same
    inputs (catches regressions where the two paths diverge)."""
    from yaml_bert.linearizer import YamlLinearizer
    from yaml_bert.vocab import VocabBuilder
    from yaml_bert.config import YamlBertConfig
    from yaml_bert.embedding import YamlBertEmbedding
    from yaml_bert.v8_dataset import V8Dataset, v8_collate_fn

    yamls = [
        "apiVersion: v1\nkind: Pod\nmetadata:\n  name: a\n",
        "apiVersion: v1\nkind: Service\nspec:\n  x: 1\n",
        "apiVersion: apps/v1\nkind: Deployment\nspec:\n  replicas: 3\n",
    ]
    documents = [YamlLinearizer().linearize(y) for y in yamls]
    flat = [n for doc in documents for n in doc]
    vocab = VocabBuilder().build(flat, min_freq=1)

    config = YamlBertConfig(d_model=32, num_layers=2, num_heads=4,
                            v8_mode=True, mask_prob=0.5)
    ds = V8Dataset(documents, vocab, config)
    batch = v8_collate_fn([ds[i] for i in range(len(ds))])

    emb = YamlBertEmbedding(config=config,
                            key_vocab_size=vocab.key_vocab_size,
                            value_vocab_size=vocab.value_vocab_size)
    model = V8Model(config=config, embedding=emb,
                    atomic_vocab_size=vocab.atomic_target_vocab_size)
    model.eval()  # disable dropout for deterministic comparison

    # Reference path: no tensor kwargs
    with torch.no_grad():
        ref_logits, ref_doc = model(
            token_ids=batch["token_ids"],
            node_types=batch["node_types"],
            depths=batch["depths"],
            sibling_indices=batch["sibling_indices"],
            batch_info=batch["batch_info"],
            padding_mask=batch["padding_mask"],
        )

    # Vectorized path: pass tensor kwargs
    with torch.no_grad():
        vec_logits, vec_doc = model(
            token_ids=batch["token_ids"],
            node_types=batch["node_types"],
            depths=batch["depths"],
            sibling_indices=batch["sibling_indices"],
            batch_info=batch["batch_info"],
            padding_mask=batch["padding_mask"],
            parent_of_tensor=batch["parent_of_tensor"],
            top_level_key_mask=batch["top_level_key_mask"],
            edges_by_depth=batch["edges_by_depth"],
            parents_by_depth=batch["parents_by_depth"],
        )

    # Both paths should produce identical logits and doc vectors
    assert torch.allclose(ref_logits, vec_logits, atol=1e-5), (
        f"logits diverge: max diff = "
        f"{(ref_logits - vec_logits).abs().max().item()}"
    )
    assert torch.allclose(ref_doc, vec_doc, atol=1e-5)

    # Backward must actually work on the vectorized path. Run a fresh
    # forward in train mode (not under no_grad) so autograd has a graph.
    model.train()
    train_logits, _ = model(
        token_ids=batch["token_ids"],
        node_types=batch["node_types"],
        depths=batch["depths"],
        sibling_indices=batch["sibling_indices"],
        batch_info=batch["batch_info"],
        padding_mask=batch["padding_mask"],
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


def test_v8_model_returns_recon_logits_when_subtree_info_provided():
    """V8Model.forward returns recon_logits with shape (M, V_atomic) when
    the batch contains subtree_roots_flat (i.e., recon is active).

    Uses a large YAML (55+ nodes) to guarantee pick_subtrees returns at least
    one root (MIN_DOC_NODES=10; MAX_TOTAL_SUBTREE_FRACTION=0.05 means N>20
    is required for even the smallest subtree).
    """
    from yaml_bert.linearizer import YamlLinearizer
    from yaml_bert.vocab import VocabBuilder
    from yaml_bert.config import YamlBertConfig
    from yaml_bert.embedding import YamlBertEmbedding
    from yaml_bert.v8_dataset import V8Dataset, v8_collate_fn

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
    flat = [n for doc in docs for n in doc]
    vocab = VocabBuilder().build(flat, min_freq=1)
    config = YamlBertConfig(d_model=32, num_layers=2, num_heads=4,
                            v8_mode=True, mask_prob=0.0, recon_enabled=True)
    ds = V8Dataset(docs, vocab, config)
    batch = v8_collate_fn([ds[i] for i in range(len(ds))])
    # With large docs and mask_prob=0.0 (no MLM competing for positions),
    # pick_subtrees reliably returns >= 1 root.
    assert batch["subtree_roots_flat"].size(0) > 0, (
        "test precondition failed: no subtree roots picked. "
        "Increase doc size or check pick_subtrees thresholds."
    )

    emb = YamlBertEmbedding(config=config,
                            key_vocab_size=vocab.key_vocab_size,
                            value_vocab_size=vocab.value_vocab_size)
    model = V8Model(config=config, embedding=emb,
                    atomic_vocab_size=vocab.atomic_target_vocab_size)
    model.train()

    out = model(
        token_ids=batch["token_ids"],
        node_types=batch["node_types"],
        depths=batch["depths"],
        sibling_indices=batch["sibling_indices"],
        batch_info=batch["batch_info"],
        padding_mask=batch["padding_mask"],
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
    # Check at least the head's params got gradients
    for name, p in model.recon_head.named_parameters():
        assert p.grad is not None, f"recon_head.{name} has no gradient"
        assert torch.isfinite(p.grad).all(), f"recon_head.{name} gradient is non-finite"


def test_v8_model_omits_recon_logits_when_no_subtree_info():
    """V8Model.forward returns (logits, doc_vec) — old shape — when subtree
    kwargs are absent. Backward-compat with Phase 0/1 callers."""
    from yaml_bert.linearizer import YamlLinearizer
    from yaml_bert.vocab import VocabBuilder
    from yaml_bert.config import YamlBertConfig
    from yaml_bert.embedding import YamlBertEmbedding
    from yaml_bert.v8_dataset import V8Dataset, v8_collate_fn

    yamls = ["apiVersion: v1\nkind: Pod\nspec:\n  x: 1\n"]
    docs = [YamlLinearizer().linearize(y) for y in yamls]
    flat = [n for doc in docs for n in doc]
    vocab = VocabBuilder().build(flat, min_freq=1)
    # recon_enabled=False → dataset omits subtree fields → forward omits recon
    config = YamlBertConfig(d_model=16, num_layers=1, num_heads=2,
                            v8_mode=True, mask_prob=0.0, recon_enabled=False)
    ds = V8Dataset(docs, vocab, config)
    batch = v8_collate_fn([ds[0]])

    emb = YamlBertEmbedding(config=config,
                            key_vocab_size=vocab.key_vocab_size,
                            value_vocab_size=vocab.value_vocab_size)
    model = V8Model(config=config, embedding=emb,
                    atomic_vocab_size=vocab.atomic_target_vocab_size)
    model.eval()
    out = model(
        token_ids=batch["token_ids"],
        node_types=batch["node_types"],
        depths=batch["depths"],
        sibling_indices=batch["sibling_indices"],
        batch_info=batch["batch_info"],
        padding_mask=batch["padding_mask"],
    )
    assert len(out) == 2  # (logits, doc_vec) — no recon
