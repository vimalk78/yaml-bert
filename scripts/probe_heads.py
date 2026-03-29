"""Probe what each attention head has learned.

Extracts hidden states from each head, trains tiny linear classifiers
to predict tree properties (depth, kind, parent_key, node_type).
Shows which heads specialize in which structural information.

Usage:
    CUDA_VISIBLE_DEVICES="" python scripts/probe_heads.py output_v2/checkpoints/yaml_bert_epoch_5.pt --vocab output_v2/vocab.json
"""
from __future__ import annotations
import _setup_path  # noqa: F401

import argparse
import os
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from yaml_bert.config import YamlBertConfig
from yaml_bert.annotator import DomainAnnotator
from yaml_bert.dataset import YamlDataset, _extract_kind
from yaml_bert.embedding import YamlBertEmbedding
from yaml_bert.linearizer import YamlLinearizer
from yaml_bert.model import YamlBertModel
from yaml_bert.vocab import Vocabulary
from yaml_bert.types import NodeType


def extract_head_states(
    model: YamlBertModel,
    dataset: YamlDataset,
    vocab: Vocabulary,
    max_docs: int = 200,
) -> dict[str, Any]:
    """Run forward pass and extract per-head hidden states + labels."""
    model.eval()

    type_map = {NodeType.KEY: 0, NodeType.VALUE: 1, NodeType.LIST_KEY: 2, NodeType.LIST_VALUE: 3}
    num_layers: int = len(model.encoder.layers)
    num_heads: int = model.encoder.layers[0].self_attn.num_heads
    d_head: int = model.embedding.key_embedding.embedding_dim // num_heads

    print(f"Model: {num_layers} layers, {num_heads} heads, {d_head} dims/head")

    # Storage for all head outputs and labels
    all_head_outputs: dict[tuple[int, int], list[torch.Tensor]] = {
        (l, h): [] for l in range(num_layers) for h in range(num_heads)
    }
    # Storage for full residual stream (256-dim) after each layer
    all_layer_outputs: dict[int, list[torch.Tensor]] = {
        l: [] for l in range(num_layers)
    }
    # Also capture the embedding output (before any transformer layer)
    all_embedding_outputs: list[torch.Tensor] = []
    all_depths: list[int] = []
    all_node_types: list[int] = []
    all_parent_keys: list[int] = []
    all_kinds: list[int] = []

    docs_processed: int = 0
    for idx in range(min(max_docs, len(dataset))):
        item: dict[str, torch.Tensor] = dataset[idx]
        nodes = dataset.documents[idx]

        # Get kind_id
        kind: str = _extract_kind(nodes)
        kind_id: int = vocab.encode_kind(kind)

        # Truncate to match dataset
        seq_len: int = item["token_ids"].shape[0]
        nodes = nodes[:seq_len]

        # Add batch dimension
        token_ids = item["token_ids"].unsqueeze(0)
        node_types_t = item["node_types"].unsqueeze(0)
        depths_t = item["depths"].unsqueeze(0)
        siblings_t = item["sibling_indices"].unsqueeze(0)
        parent_keys_t = item["parent_key_ids"].unsqueeze(0)
        kind_ids_t = item["kind_ids"].unsqueeze(0) if "kind_ids" in item else None

        # Forward through embedding
        with torch.no_grad():
            x: torch.Tensor = model.embedding(
                token_ids, node_types_t, depths_t, siblings_t, parent_keys_t,
                kind_ids=kind_ids_t,
            )

            # Capture embedding output
            all_embedding_outputs.append(x.squeeze(0).clone())

            # Forward through each layer, extracting per-head outputs
            for layer_idx, layer in enumerate(model.encoder.layers):
                # Multi-head attention: get per-head outputs
                # self_attn projects Q, K, V internally
                # We need to access the internal head split
                attn = layer.self_attn

                # Compute Q, K, V
                q = attn.in_proj_weight[:attn.embed_dim] @ x.squeeze(0).T + attn.in_proj_bias[:attn.embed_dim].unsqueeze(1)
                k = attn.in_proj_weight[attn.embed_dim:2*attn.embed_dim] @ x.squeeze(0).T + attn.in_proj_bias[attn.embed_dim:2*attn.embed_dim].unsqueeze(1)
                v = attn.in_proj_weight[2*attn.embed_dim:] @ x.squeeze(0).T + attn.in_proj_bias[2*attn.embed_dim:].unsqueeze(1)

                # Reshape to (seq, heads, d_head) then transpose
                q = q.T.view(seq_len, num_heads, d_head)
                k = k.T.view(seq_len, num_heads, d_head)
                v = v.T.view(seq_len, num_heads, d_head)

                # Compute attention per head
                for h in range(num_heads):
                    scores = (q[:, h] @ k[:, h].T) / (d_head ** 0.5)
                    weights = F.softmax(scores, dim=-1)
                    head_out = weights @ v[:, h]  # (seq_len, d_head)
                    all_head_outputs[(layer_idx, h)].append(head_out)

                # Pass through full layer for next layer's input
                x = layer(x)

                # Capture full residual stream output (256-dim)
                all_layer_outputs[layer_idx].append(x.squeeze(0).clone())

        # Collect labels for each node
        for i, node in enumerate(nodes):
            all_depths.append(min(node.depth, 15))
            all_node_types.append(type_map[node.node_type])
            parent_key: str = Vocabulary.extract_parent_key(node.parent_path)
            all_parent_keys.append(vocab.encode_key(parent_key))
            all_kinds.append(kind_id)

        docs_processed += 1
        if (docs_processed) % 50 == 0:
            print(f"  {docs_processed}/{min(max_docs, len(dataset))} docs processed")

    # Stack all head outputs
    head_tensors: dict[tuple[int, int], torch.Tensor] = {}
    for key, outputs in all_head_outputs.items():
        head_tensors[key] = torch.cat(outputs, dim=0)  # (total_nodes, d_head)

    # Stack layer outputs
    layer_tensors: dict[int, torch.Tensor] = {}
    for key, outputs in all_layer_outputs.items():
        layer_tensors[key] = torch.cat(outputs, dim=0)  # (total_nodes, d_model)

    # Stack embedding output
    embedding_tensor: torch.Tensor = torch.cat(all_embedding_outputs, dim=0)

    labels = {
        "depth": torch.tensor(all_depths),
        "node_type": torch.tensor(all_node_types),
        "parent_key": torch.tensor(all_parent_keys),
        "kind": torch.tensor(all_kinds),
    }

    print(f"Extracted {len(all_depths)} node states from {docs_processed} docs")
    return {
        "head_states": head_tensors,
        "layer_states": layer_tensors,
        "embedding_states": embedding_tensor,
        "labels": labels,
        "num_layers": num_layers,
        "num_heads": num_heads,
    }


def train_probe(
    states: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    epochs: int = 20,
) -> float:
    """Train a linear probe and return accuracy."""
    # Split 80/20
    n: int = states.shape[0]
    split: int = int(n * 0.8)
    perm = torch.randperm(n)
    train_idx = perm[:split]
    val_idx = perm[split:]

    X_train, y_train = states[train_idx], labels[train_idx]
    X_val, y_val = states[val_idx], labels[val_idx]

    # Filter out classes that don't appear enough
    d_in: int = states.shape[1]
    probe = nn.Linear(d_in, num_classes)
    optimizer = torch.optim.Adam(probe.parameters(), lr=1e-3)

    for epoch in range(epochs):
        probe.train()
        logits = probe(X_train)
        loss = F.cross_entropy(logits, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    probe.eval()
    with torch.no_grad():
        val_logits = probe(X_val)
        preds = val_logits.argmax(dim=-1)
        accuracy: float = (preds == y_val).float().mean().item()

    return accuracy


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe attention heads")
    parser.add_argument("checkpoint", type=str)
    parser.add_argument("--vocab", type=str, default="output_v1/vocab.json")
    parser.add_argument("--max-docs", type=int, default=200)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    # Load model
    print("Loading model...")
    vocab: Vocabulary = Vocabulary.load(args.vocab)
    config: YamlBertConfig = YamlBertConfig()

    emb = YamlBertEmbedding(
        config=config,
        key_vocab_size=vocab.key_vocab_size,
        value_vocab_size=vocab.value_vocab_size,
        kind_vocab_size=vocab.kind_vocab_size,
    )
    model = YamlBertModel(config=config, embedding=emb, key_vocab_size=vocab.key_vocab_size, kind_vocab_size=vocab.kind_vocab_size)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    print(f"Loaded epoch {checkpoint['epoch']}")

    # Load dataset
    print("Loading dataset...")
    linearizer = YamlLinearizer()
    annotator = DomainAnnotator()

    data_dir = os.path.join(os.path.dirname(__file__), "..", "data", "k8s-yamls")
    if os.path.exists(data_dir):
        dataset = YamlDataset(
            yaml_dir=data_dir,
            vocab=vocab,
            linearizer=linearizer,
            annotator=annotator,
        )
    else:
        dataset = YamlDataset.from_huggingface(
            "substratusai/the-stack-yaml-k8s",
            vocab=vocab,
            linearizer=linearizer,
            annotator=annotator,
            max_docs=args.max_docs,
        )

    print(f"Dataset: {len(dataset)} docs")

    # Extract head states
    print("\nExtracting head states...")
    data = extract_head_states(model, dataset, vocab, max_docs=args.max_docs)

    head_states = data["head_states"]
    labels = data["labels"]
    num_layers: int = data["num_layers"]
    num_heads: int = data["num_heads"]

    # Determine number of classes for each property
    num_classes = {
        "depth": 16,
        "node_type": 4,
        "parent_key": vocab.key_vocab_size,
        "kind": vocab.kind_vocab_size,
    }

    # Train probes for each head and property
    print("\nTraining probes...")
    print(f"{'Head':<10} {'Depth':>8} {'Type':>8} {'Parent':>8} {'Kind':>8}")
    print("-" * 46)

    results: dict[str, dict[tuple[int, int], float]] = {
        prop: {} for prop in ["depth", "node_type", "parent_key", "kind"]
    }

    for layer_idx in range(num_layers):
        for head_idx in range(num_heads):
            states = head_states[(layer_idx, head_idx)]
            accs: list[str] = []

            for prop in ["depth", "node_type", "parent_key", "kind"]:
                acc = train_probe(states, labels[prop], num_classes[prop])
                results[prop][(layer_idx, head_idx)] = acc
                accs.append(f"{acc:.1%}")

            print(f"L{layer_idx}H{head_idx:<5} {accs[0]:>8} {accs[1]:>8} {accs[2]:>8} {accs[3]:>8}")

    # Summary: best head for each property
    print(f"\n{'=' * 46}")
    print("Best head for each property:")
    for prop in ["depth", "node_type", "parent_key", "kind"]:
        best_head = max(results[prop], key=results[prop].get)
        best_acc = results[prop][best_head]
        print(f"  {prop:<12}: L{best_head[0]}H{best_head[1]} ({best_acc:.1%})")

    # ============================================================
    # Probe the RESIDUAL STREAM (full 256-dim layer outputs)
    # ============================================================
    layer_states = data["layer_states"]
    embedding_states = data["embedding_states"]

    print(f"\n{'=' * 60}")
    print("RESIDUAL STREAM PROBING (full d_model={} dims)".format(embedding_states.shape[1]))
    print(f"{'=' * 60}")
    print(f"{'Layer':<12} {'Depth':>8} {'Type':>8} {'Parent':>8} {'Kind':>8}")
    print("-" * 48)

    residual_results: dict[str, dict[str, float]] = {
        prop: {} for prop in ["depth", "node_type", "parent_key", "kind"]
    }

    # Probe embedding output (before any transformer layer)
    accs_emb: list[str] = []
    for prop in ["depth", "node_type", "parent_key", "kind"]:
        acc = train_probe(embedding_states, labels[prop], num_classes[prop])
        residual_results[prop]["embedding"] = acc
        accs_emb.append(f"{acc:.1%}")
    print(f"{'Embedding':<12} {accs_emb[0]:>8} {accs_emb[1]:>8} {accs_emb[2]:>8} {accs_emb[3]:>8}")

    # Probe each layer's output
    for layer_idx in range(num_layers):
        states = layer_states[layer_idx]
        accs_layer: list[str] = []
        for prop in ["depth", "node_type", "parent_key", "kind"]:
            acc = train_probe(states, labels[prop], num_classes[prop])
            residual_results[prop][f"layer_{layer_idx}"] = acc
            accs_layer.append(f"{acc:.1%}")
        print(f"Layer {layer_idx:<6} {accs_layer[0]:>8} {accs_layer[1]:>8} {accs_layer[2]:>8} {accs_layer[3]:>8}")

    print(f"\n{'=' * 60}")
    print("Summary — Residual stream vs per-head:")
    for prop in ["depth", "node_type", "parent_key", "kind"]:
        best_head = max(results[prop], key=results[prop].get)
        best_head_acc = results[prop][best_head]
        best_layer = max(residual_results[prop], key=residual_results[prop].get)
        best_layer_acc = residual_results[prop][best_layer]
        print(f"  {prop:<12}: best head = {best_head_acc:.1%} (L{best_head[0]}H{best_head[1]})  |  best residual = {best_layer_acc:.1%} ({best_layer})")

    # Save results
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        for prop in ["depth", "node_type", "parent_key", "kind"]:
            grid = np.zeros((num_layers, num_heads))
            for (l, h), acc in results[prop].items():
                grid[l, h] = acc

            fig, ax = plt.subplots(figsize=(10, 6))
            im = ax.imshow(grid, cmap="YlOrRd", vmin=0, vmax=1)
            ax.set_xlabel("Head")
            ax.set_ylabel("Layer")
            ax.set_xticks(range(num_heads))
            ax.set_yticks(range(num_layers))
            ax.set_title(f"Probing Accuracy: {prop}")

            for l in range(num_layers):
                for h in range(num_heads):
                    ax.text(h, l, f"{grid[l, h]:.0%}", ha="center", va="center", fontsize=8)

            fig.colorbar(im, ax=ax, shrink=0.8)
            fig.tight_layout()
            fig.savefig(os.path.join(args.output_dir, f"probe_{prop}.png"), dpi=150)
            plt.close(fig)
            print(f"Saved: probe_{prop}.png")


if __name__ == "__main__":
    main()
