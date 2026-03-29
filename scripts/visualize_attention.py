"""Visualize attention patterns from a trained YAML-BERT checkpoint.

Usage:
    python visualize_attention.py output_v1/checkpoints/yaml_bert_epoch_10.pt
    python visualize_attention.py output_v1/checkpoints/yaml_bert_epoch_10.pt --doc-idx 5
"""
from __future__ import annotations
import _setup_path  # noqa: F401

import argparse
import os

import torch

from yaml_bert.config import YamlBertConfig
from yaml_bert.annotator import DomainAnnotator
from yaml_bert.dataset import YamlDataset
from yaml_bert.embedding import YamlBertEmbedding
from yaml_bert.linearizer import YamlLinearizer
from yaml_bert.model import YamlBertModel
from yaml_bert.visualize import plot_attention_patterns
from yaml_bert.vocab import Vocabulary
from yaml_bert.types import NodeType


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize YAML-BERT attention patterns")
    parser.add_argument("checkpoint", type=str, help="Path to checkpoint .pt file")
    parser.add_argument("--vocab", type=str, default="output_v1/vocab.json")
    parser.add_argument("--doc-idx", type=int, default=0, help="Document index to visualize")
    parser.add_argument("--max-nodes", type=int, default=30,
                        help="Max nodes to show (truncate long docs for readability)")
    parser.add_argument("--output-dir", type=str, default="output_v1/attention")
    parser.add_argument("--yaml-text", type=str, default=None,
                        help="Custom YAML text to visualize (instead of HF dataset)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading vocabulary...")
    vocab: Vocabulary = Vocabulary.load(args.vocab)
    config: YamlBertConfig = YamlBertConfig()

    print("Building model and loading checkpoint...")
    emb: YamlBertEmbedding = YamlBertEmbedding(
        config=config,
        key_vocab_size=vocab.key_vocab_size,
        value_vocab_size=vocab.value_vocab_size,
    )
    model: YamlBertModel = YamlBertModel(
        config=config,
        embedding=emb,
        key_vocab_size=vocab.key_vocab_size,
    )
    checkpoint: dict = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded checkpoint: epoch {checkpoint['epoch']}")

    linearizer: YamlLinearizer = YamlLinearizer()
    annotator: DomainAnnotator = DomainAnnotator()

    if args.yaml_text:
        nodes = linearizer.linearize(args.yaml_text)
        annotator.annotate(nodes)
    else:
        print("Loading a document from HuggingFace dataset...")
        from datasets import load_dataset
        ds = load_dataset("substratusai/the-stack-yaml-k8s", split="train")
        yaml_content: str = ds[args.doc_idx]["content"]
        nodes = linearizer.linearize(yaml_content)
        annotator.annotate(nodes)
        print(f"Document {args.doc_idx}: {len(nodes)} nodes")

    # Truncate for readability
    if len(nodes) > args.max_nodes:
        print(f"Truncating from {len(nodes)} to {args.max_nodes} nodes")
        nodes = nodes[:args.max_nodes]

    # Build token labels for the plot
    token_labels: list[str] = []
    for node in nodes:
        type_prefix: str = ""
        if node.node_type in (NodeType.VALUE, NodeType.LIST_VALUE):
            type_prefix = "="
        token_labels.append(f"{type_prefix}{node.token[:20]}")

    # Encode nodes to tensors
    token_ids: list[int] = []
    node_types: list[int] = []
    depths: list[int] = []
    sibling_indices: list[int] = []
    parent_key_ids: list[int] = []

    type_map = {NodeType.KEY: 0, NodeType.VALUE: 1, NodeType.LIST_KEY: 2, NodeType.LIST_VALUE: 3}

    for node in nodes:
        if node.node_type in (NodeType.KEY, NodeType.LIST_KEY):
            token_ids.append(vocab.encode_key(node.token))
        else:
            token_ids.append(vocab.encode_value(node.token))
        node_types.append(type_map[node.node_type])
        depths.append(min(node.depth, 15))
        sibling_indices.append(min(node.sibling_index, 31))
        parent_key: str = Vocabulary.extract_parent_key(node.parent_path)
        parent_key_ids.append(vocab.encode_key(parent_key))

    # Convert to tensors with batch dimension
    t_token_ids: torch.Tensor = torch.tensor([token_ids])
    t_node_types: torch.Tensor = torch.tensor([node_types])
    t_depths: torch.Tensor = torch.tensor([depths])
    t_siblings: torch.Tensor = torch.tensor([sibling_indices])
    t_parent_keys: torch.Tensor = torch.tensor([parent_key_ids])

    print("Extracting attention weights...")
    attention_weights: list[torch.Tensor] = model.get_attention_weights(
        t_token_ids, t_node_types, t_depths, t_siblings, t_parent_keys
    )

    print(f"Got {len(attention_weights)} layers, {attention_weights[0].shape[1]} heads each")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    num_layers: int = len(attention_weights)
    num_heads: int = attention_weights[0].shape[1]

    # Save individual head plots — one per head per layer, large and readable
    for layer_idx, layer_weights in enumerate(attention_weights):
        weights: torch.Tensor = layer_weights[0]  # remove batch dim: (heads, seq, seq)

        for head_idx in range(num_heads):
            head_weights: torch.Tensor = weights[head_idx].cpu()

            fig, ax = plt.subplots(figsize=(12, 10))
            im = ax.imshow(head_weights.numpy(), cmap="Blues", vmin=0)
            ax.set_title(f"Layer {layer_idx}, Head {head_idx}", fontsize=14)
            ax.set_xticks(range(len(token_labels)))
            ax.set_yticks(range(len(token_labels)))
            ax.set_xticklabels(token_labels, rotation=45, ha="right", fontsize=8)
            ax.set_yticklabels(token_labels, fontsize=8)
            ax.set_xlabel("Attending to")
            ax.set_ylabel("Attending from")
            fig.colorbar(im, ax=ax, shrink=0.8)
            fig.tight_layout()

            output_path: str = os.path.join(
                args.output_dir, f"layer{layer_idx}_head{head_idx}.png"
            )
            fig.savefig(output_path, dpi=150)
            plt.close(fig)

        # Average across heads
        avg_weights: torch.Tensor = weights.mean(dim=0).cpu()
        fig, ax = plt.subplots(figsize=(12, 10))
        im = ax.imshow(avg_weights.numpy(), cmap="Blues", vmin=0)
        ax.set_title(f"Layer {layer_idx} — Average Across All Heads", fontsize=14)
        ax.set_xticks(range(len(token_labels)))
        ax.set_yticks(range(len(token_labels)))
        ax.set_xticklabels(token_labels, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(token_labels, fontsize=8)
        ax.set_xlabel("Attending to")
        ax.set_ylabel("Attending from")
        fig.colorbar(im, ax=ax, shrink=0.8)
        fig.tight_layout()

        output_path = os.path.join(args.output_dir, f"layer{layer_idx}_avg.png")
        fig.savefig(output_path, dpi=150)
        plt.close(fig)

    print(f"\nAll plots saved to: {args.output_dir}")
    print("Files:")
    for f in sorted(os.listdir(args.output_dir)):
        if f.endswith(".png"):
            print(f"  {f}")


if __name__ == "__main__":
    main()
