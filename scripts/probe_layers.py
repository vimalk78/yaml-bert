"""Probe what information each transformer layer encodes.

Trains linear classifiers on frozen hidden states at each layer to predict:
- depth (0-15): Does the model know how deep a node is?
- parent_key: Does the model know the parent key?
- kind: Does the model know the document kind?
- node_type: Does the model know if it's a key or value?

For v4: kind and parent are NOT in the input embedding.
Any probe accuracy for these was genuinely learned, not leaked.

Usage:
    PYTHONPATH=. python scripts/probe_layers.py output_v4/checkpoints/yaml_bert_v4_epoch_15.pt --vocab output_v4/vocab.json
"""
from __future__ import annotations
import _setup_path  # noqa: F401

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from yaml_bert.config import YamlBertConfig
from yaml_bert.dataset import YamlDataset, _extract_kind
from yaml_bert.embedding import YamlBertEmbedding
from yaml_bert.model import YamlBertModel
from yaml_bert.vocab import Vocabulary
from yaml_bert.types import NodeType


def extract_layer_states(
    model: YamlBertModel,
    dataset: YamlDataset,
    vocab: Vocabulary,
    max_docs: int = 300,
) -> dict:
    """Run forward pass, extract hidden states at embedding + each layer."""
    model.eval()
    type_map = {NodeType.KEY: 0, NodeType.VALUE: 1, NodeType.LIST_KEY: 2, NodeType.LIST_VALUE: 3}
    num_layers = len(model.encoder.layers)

    all_embedding: list[torch.Tensor] = []
    all_layers: dict[int, list[torch.Tensor]] = {l: [] for l in range(num_layers)}
    all_depths: list[int] = []
    all_node_types: list[int] = []
    all_parent_keys: list[int] = []
    all_kinds: list[int] = []

    for idx in range(min(max_docs, len(dataset))):
        item = dataset[idx]
        nodes = dataset.documents[idx]
        kind = _extract_kind(nodes)
        kind_id = vocab.encode_kind(kind)

        seq_len = item["token_ids"].shape[0]
        nodes = nodes[:seq_len]

        token_ids = item["token_ids"].unsqueeze(0)
        node_types_t = item["node_types"].unsqueeze(0)
        depths_t = item["depths"].unsqueeze(0)
        siblings_t = item["sibling_indices"].unsqueeze(0)

        with torch.no_grad():
            x = model.embedding(token_ids, node_types_t, depths_t, siblings_t)
            all_embedding.append(x.squeeze(0))

            for l, layer in enumerate(model.encoder.layers):
                x = layer(x)
                all_layers[l].append(x.squeeze(0))

        for node in nodes:
            all_depths.append(min(node.depth, 15))
            all_node_types.append(type_map[node.node_type])
            parent_key = Vocabulary.extract_parent_key(node.parent_path)
            all_parent_keys.append(vocab.encode_key(parent_key))
            all_kinds.append(kind_id)

        if (idx + 1) % 100 == 0:
            print(f"  {idx + 1}/{min(max_docs, len(dataset))} docs")

    labels = {
        "depth": torch.tensor(all_depths),
        "node_type": torch.tensor(all_node_types),
        "parent_key": torch.tensor(all_parent_keys),
        "kind": torch.tensor(all_kinds),
    }

    layer_tensors = {l: torch.cat(v) for l, v in all_layers.items()}
    embedding_tensor = torch.cat(all_embedding)

    print(f"Extracted {len(all_depths)} nodes from {min(max_docs, len(dataset))} docs")
    return {
        "embedding": embedding_tensor,
        "layers": layer_tensors,
        "labels": labels,
        "num_layers": num_layers,
    }


def train_probe(states: torch.Tensor, labels: torch.Tensor, num_classes: int, epochs: int = 20) -> float:
    """Train a linear probe, return validation accuracy."""
    n = states.shape[0]
    perm = torch.randperm(n)
    split = int(n * 0.8)
    train_idx, val_idx = perm[:split], perm[split:]

    X_train, y_train = states[train_idx], labels[train_idx]
    X_val, y_val = states[val_idx], labels[val_idx]

    probe = nn.Linear(states.shape[1], num_classes)
    optimizer = torch.optim.Adam(probe.parameters(), lr=1e-3)

    for _ in range(epochs):
        probe.train()
        logits = probe(X_train)
        loss = F.cross_entropy(logits, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    probe.eval()
    with torch.no_grad():
        preds = probe(X_val).argmax(dim=-1)
        return (preds == y_val).float().mean().item()


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe layer representations")
    parser.add_argument("checkpoint", type=str)
    parser.add_argument("--vocab", type=str, default="output_v4/vocab.json")
    parser.add_argument("--max-docs", type=int, default=300)
    parser.add_argument("--cached-docs", type=str, default=None,
                        help="Path to cached docs pickle (faster than parsing)")
    args = parser.parse_args()

    torch.manual_seed(42)
    vocab = Vocabulary.load(args.vocab)
    config = YamlBertConfig()
    emb = YamlBertEmbedding(config=config, key_vocab_size=vocab.key_vocab_size,
                            value_vocab_size=vocab.value_vocab_size)
    model = YamlBertModel(config=config, embedding=emb,
                          simple_vocab_size=vocab.simple_target_vocab_size,
                          kind_vocab_size=vocab.kind_target_vocab_size)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Loaded epoch {checkpoint['epoch']}")

    # Load dataset
    if args.cached_docs:
        dataset = YamlDataset.from_cached_docs_v4(args.cached_docs, vocab)
    else:
        from yaml_bert.linearizer import YamlLinearizer
        from yaml_bert.annotator import DomainAnnotator
        import os
        data_dir = os.path.join(os.path.dirname(__file__), "..", "data", "k8s-yamls")
        dataset = YamlDataset(
            yaml_dir=data_dir, vocab=vocab,
            linearizer=YamlLinearizer(), annotator=DomainAnnotator(),
        )
    print(f"Dataset: {len(dataset)} docs\n")

    print("Extracting hidden states...")
    data = extract_layer_states(model, dataset, vocab, max_docs=args.max_docs)

    num_layers = data["num_layers"]
    labels = data["labels"]

    num_classes = {
        "depth": 16,
        "node_type": 4,
        "parent_key": vocab.key_vocab_size,
        "kind": vocab.kind_vocab_size,
    }

    # Probe at each layer
    print(f"\n{'Layer':<12} {'Depth':>8} {'Type':>8} {'Parent':>8} {'Kind':>8}")
    print("-" * 48)

    properties = ["depth", "node_type", "parent_key", "kind"]

    # Embedding layer
    accs = []
    for prop in properties:
        acc = train_probe(data["embedding"], labels[prop], num_classes[prop])
        accs.append(acc)
    print(f"{'Embedding':<12} {accs[0]:>7.1%} {accs[1]:>7.1%} {accs[2]:>7.1%} {accs[3]:>7.1%}")

    # Transformer layers
    for l in range(num_layers):
        accs = []
        for prop in properties:
            acc = train_probe(data["layers"][l], labels[prop], num_classes[prop])
            accs.append(acc)
        print(f"{'Layer ' + str(l):<12} {accs[0]:>7.1%} {accs[1]:>7.1%} {accs[2]:>7.1%} {accs[3]:>7.1%}")

    print(f"\nNote: kind and parent_key are NOT in the v4 input embedding.")
    print(f"Any accuracy above random for these was learned from the prediction target.")
    print(f"Random baselines: depth={1/16:.1%}, type={1/4:.1%}, parent=~{1/100:.1%}, kind=~{1/vocab.kind_vocab_size:.1%}")


if __name__ == "__main__":
    main()
