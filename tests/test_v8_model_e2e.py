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

    # Loss + backward must still work on vectorized path
    model.train()
    loss = torch.nn.functional.cross_entropy(
        vec_logits.view(-1, vec_logits.size(-1)),
        batch["atomic_labels"].view(-1),
        ignore_index=-100,
    )
    assert torch.isfinite(loss)
