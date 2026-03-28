import glob
import os

import torch
from yaml_bert.config import YamlBertConfig
from yaml_bert.annotator import DomainAnnotator
from yaml_bert.dataset import YamlDataset
from yaml_bert.embedding import YamlBertEmbedding
from yaml_bert.linearizer import YamlLinearizer
from yaml_bert.model import YamlBertModel
from yaml_bert.trainer import YamlBertTrainer
from yaml_bert.vocab import VocabBuilder

TEMPLATES_DIR = os.path.join(
    os.path.dirname(__file__), "..", "data", "k8s-yamls"
)

TEST_CONFIG: YamlBertConfig = YamlBertConfig(
    d_model=64, num_layers=2, num_heads=2, num_epochs=10, batch_size=8,
)


def test_end_to_end_pipeline():
    """Full pipeline: corpus -> vocab -> dataset -> model -> train -> converges."""
    linearizer = YamlLinearizer()
    annotator = DomainAnnotator()

    all_nodes = []
    for path in glob.glob(os.path.join(TEMPLATES_DIR, "**", "*.yaml"), recursive=True):
        nodes = linearizer.linearize_file(path)
        annotator.annotate(nodes)
        all_nodes.extend(nodes)

    vocab = VocabBuilder().build(all_nodes)
    print(f"Key vocab: {len(vocab.key_vocab)} tokens")
    print(f"Value vocab: {len(vocab.value_vocab)} tokens")

    dataset = YamlDataset(
        yaml_dir=TEMPLATES_DIR,
        vocab=vocab,
        linearizer=YamlLinearizer(),
        annotator=DomainAnnotator(),
        config=TEST_CONFIG,
    )
    print(f"Dataset: {len(dataset)} documents")

    emb = YamlBertEmbedding(
        config=TEST_CONFIG,
        key_vocab_size=vocab.key_vocab_size,
        value_vocab_size=vocab.value_vocab_size,
    )
    model = YamlBertModel(
        config=TEST_CONFIG,
        embedding=emb,
        key_vocab_size=vocab.key_vocab_size,
    )

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {num_params:,} parameters")

    trainer = YamlBertTrainer(
        config=TEST_CONFIG,
        model=model,
        dataset=dataset,
    )
    losses = trainer.train()

    assert losses[-1] < losses[0], (
        f"Loss did not decrease: first={losses[0]:.4f}, last={losses[-1]:.4f}"
    )
    print(f"Loss: {losses[0]:.4f} -> {losses[-1]:.4f}")


def test_tree_position_differentiation():
    """Verify that 'spec' at different tree positions gets different embeddings."""
    linearizer = YamlLinearizer()
    annotator = DomainAnnotator()

    all_nodes = []
    for path in glob.glob(os.path.join(TEMPLATES_DIR, "**", "*.yaml"), recursive=True):
        nodes = linearizer.linearize_file(path)
        annotator.annotate(nodes)
        all_nodes.extend(nodes)

    vocab = VocabBuilder().build(all_nodes)

    emb = YamlBertEmbedding(
        config=TEST_CONFIG,
        key_vocab_size=vocab.key_vocab_size,
        value_vocab_size=vocab.value_vocab_size,
    )

    spec_id = vocab.encode_key("spec")
    token_ids = torch.tensor([[spec_id, spec_id]])
    node_types = torch.tensor([[0, 0]])
    depths = torch.tensor([[0, 2]])
    siblings = torch.tensor([[0, 0]])

    root_parent = vocab.encode_key("")
    template_parent = vocab.encode_key("template")
    parent_keys = torch.tensor([[root_parent, template_parent]])

    output = emb(token_ids, node_types, depths, siblings, parent_keys)

    spec_at_depth0 = output[0, 0]
    spec_at_depth2 = output[0, 1]

    cosine_sim = torch.nn.functional.cosine_similarity(
        spec_at_depth0.unsqueeze(0), spec_at_depth2.unsqueeze(0)
    ).item()

    assert cosine_sim < 0.99, (
        f"spec at different depths too similar: cosine_sim={cosine_sim:.4f}"
    )
    print(f"spec at depth 0 vs depth 2: cosine_similarity={cosine_sim:.4f}")
