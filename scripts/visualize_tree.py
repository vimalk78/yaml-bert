"""Visualize learned tree positional encodings overlaid on a YAML tree.

Each node is colored by its tree positional encoding (reduced to RGB via PCA).
Nodes with similar colors have similar positional encodings — the model sees them
as structurally similar positions in the tree.

Usage:
    python visualize_tree.py output_hf/checkpoints/yaml_bert_epoch_10.pt
    python visualize_tree.py output_hf/checkpoints/yaml_bert_epoch_10.pt --yaml-file data/k8s-yamls/deployment/deployment-nginx.yaml
    python visualize_tree.py output_hf/checkpoints/yaml_bert_epoch_10.pt --doc-idx 73
"""
from __future__ import annotations

import argparse
import os

import torch
import numpy as np
from sklearn.decomposition import PCA

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from yaml_bert.config import YamlBertConfig
from yaml_bert.annotator import DomainAnnotator
from yaml_bert.embedding import YamlBertEmbedding
from yaml_bert.linearizer import YamlLinearizer
from yaml_bert.model import YamlBertModel
from yaml_bert.vocab import Vocabulary
from yaml_bert.types import NodeType


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize tree positional encodings")
    parser.add_argument("checkpoint", type=str)
    parser.add_argument("--vocab", type=str, default="output_hf/vocab.json")
    parser.add_argument("--yaml-file", type=str, default=None, help="Local YAML file")
    parser.add_argument("--doc-idx", type=int, default=None, help="HuggingFace dataset index")
    parser.add_argument("--max-nodes", type=int, default=40)
    parser.add_argument("--output", type=str, default="output_hf/tree_viz/tree_embeddings.png")
    parser.add_argument("--mode", type=str, default="tree_pos",
                        choices=["tree_pos", "full", "token_only"],
                        help="What to visualize: tree_pos (positional encoding only), "
                             "full (token + position), token_only (token embedding only)")
    return parser.parse_args()


def compute_embeddings(
    model: YamlBertModel,
    vocab: Vocabulary,
    nodes: list,
) -> dict[str, torch.Tensor]:
    """Compute different embedding components for each node."""
    type_map = {NodeType.KEY: 0, NodeType.VALUE: 1, NodeType.LIST_KEY: 2, NodeType.LIST_VALUE: 3}

    token_ids: list[int] = []
    node_types: list[int] = []
    depths: list[int] = []
    siblings: list[int] = []
    parent_keys: list[int] = []

    for node in nodes:
        if node.node_type in (NodeType.KEY, NodeType.LIST_KEY):
            token_ids.append(vocab.encode_key(node.token))
        else:
            token_ids.append(vocab.encode_value(node.token))
        node_types.append(type_map[node.node_type])
        depths.append(min(node.depth, 15))
        siblings.append(min(node.sibling_index, 31))
        parent_keys.append(vocab.encode_key(Vocabulary.extract_parent_key(node.parent_path)))

    t = lambda x: torch.tensor([x])
    t_ids = t(token_ids)
    t_types = t(node_types)
    t_depths = t(depths)
    t_sibs = t(siblings)
    t_parents = t(parent_keys)

    emb = model.embedding

    with torch.no_grad():
        is_key = (t_types == 0) | (t_types == 2)
        key_vocab_size = emb.key_embedding.num_embeddings
        val_vocab_size = emb.value_embedding.num_embeddings
        key_e = emb.key_embedding(t_ids.clamp(0, key_vocab_size - 1))
        val_e = emb.value_embedding(t_ids.clamp(0, val_vocab_size - 1))
        token_emb = torch.where(is_key.unsqueeze(-1), key_e, val_e)

        depth_emb = emb.depth_embedding(t_depths)
        sib_emb = emb.sibling_embedding(t_sibs)
        type_emb = emb.node_type_embedding(t_types)
        parent_emb = emb.parent_key_embedding(t_parents)

        tree_pos = depth_emb + sib_emb + type_emb + parent_emb
        full_emb = emb.layer_norm(token_emb + tree_pos)

    return {
        "token_only": token_emb[0],
        "tree_pos": tree_pos[0],
        "full": full_emb[0],
        "depth": depth_emb[0],
        "sibling": sib_emb[0],
        "node_type": type_emb[0],
        "parent_key": parent_emb[0],
    }


def embeddings_to_colors(vectors: torch.Tensor) -> np.ndarray:
    """Reduce embedding vectors to RGB colors via PCA."""
    data: np.ndarray = vectors.numpy()

    if data.shape[0] < 3:
        return np.array([[0.5, 0.5, 0.8]] * data.shape[0])

    pca = PCA(n_components=3)
    reduced: np.ndarray = pca.fit_transform(data)

    # Normalize each component to [0, 1]
    for i in range(3):
        col = reduced[:, i]
        min_val, max_val = col.min(), col.max()
        if max_val > min_val:
            reduced[:, i] = (col - min_val) / (max_val - min_val)
        else:
            reduced[:, i] = 0.5

    return reduced


def draw_tree(
    nodes: list,
    colors: np.ndarray,
    title: str,
    output_path: str,
    component_labels: dict[str, np.ndarray] | None = None,
) -> None:
    """Draw the YAML tree with nodes colored by their embeddings."""
    n: int = len(nodes)

    fig, ax = plt.subplots(figsize=(16, max(10, n * 0.4)))
    ax.set_xlim(-0.5, 10)
    ax.set_ylim(-n, 1)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.axis("off")

    y_positions: list[float] = []
    x_positions: list[float] = []

    for i, node in enumerate(nodes):
        x: float = node.depth * 1.2
        y: float = -i
        x_positions.append(x)
        y_positions.append(y)

        # Draw node box
        color = colors[i]
        rect = mpatches.FancyBboxPatch(
            (x, y - 0.3), 3.5, 0.6,
            boxstyle="round,pad=0.1",
            facecolor=color,
            edgecolor="gray",
            linewidth=0.5,
        )
        ax.add_patch(rect)

        # Node label
        prefix: str = "=" if node.node_type in (NodeType.VALUE, NodeType.LIST_VALUE) else ""
        label: str = f"{prefix}{node.token[:30]}"

        # Choose text color based on background brightness
        brightness: float = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
        text_color: str = "white" if brightness < 0.5 else "black"

        ax.text(
            x + 1.75, y,
            label,
            ha="center", va="center",
            fontsize=7, fontweight="bold",
            color=text_color,
        )

        # Depth/parent annotation
        parent_key: str = Vocabulary.extract_parent_key(node.parent_path)
        info: str = f"d={node.depth} p={parent_key}" if parent_key else f"d={node.depth}"
        ax.text(
            x + 3.7, y,
            info,
            ha="left", va="center",
            fontsize=6, color="gray",
        )

    # Draw edges (parent-child lines)
    for i, node in enumerate(nodes):
        if node.depth == 0:
            continue
        # Find parent: nearest previous node with depth = current - 1
        for j in range(i - 1, -1, -1):
            if nodes[j].depth == node.depth - 1:
                ax.plot(
                    [x_positions[j] + 0.1, x_positions[i] - 0.1],
                    [y_positions[j], y_positions[i]],
                    color="lightgray", linewidth=0.8,
                )
                break

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Tree plot saved: {output_path}")


def main() -> None:
    args = parse_args()
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    print("Loading model...")
    vocab: Vocabulary = Vocabulary.load(args.vocab)
    config: YamlBertConfig = YamlBertConfig()
    emb = YamlBertEmbedding(config=config, key_vocab_size=vocab.key_vocab_size, value_vocab_size=vocab.value_vocab_size)
    model = YamlBertModel(config=config, embedding=emb, key_vocab_size=vocab.key_vocab_size)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded epoch {checkpoint['epoch']}")

    linearizer: YamlLinearizer = YamlLinearizer()
    annotator: DomainAnnotator = DomainAnnotator()

    # Load YAML
    if args.yaml_file:
        nodes = linearizer.linearize_file(args.yaml_file)
    elif args.doc_idx is not None:
        from datasets import load_dataset
        ds = load_dataset("substratusai/the-stack-yaml-k8s", split="train")
        nodes = linearizer.linearize(ds[args.doc_idx]["content"])
    else:
        # Default: use a deployment
        yaml_text: str = """\
apiVersion: apps/v1
kind: Deployment
metadata:
  name: nginx
  labels:
    app: web
spec:
  replicas: 3
  selector:
    matchLabels:
      app: web
  template:
    metadata:
      labels:
        app: web
    spec:
      containers:
      - name: nginx
        image: nginx:1.21
        ports:
        - containerPort: 80
"""
        nodes = linearizer.linearize(yaml_text)

    annotator.annotate(nodes)
    if len(nodes) > args.max_nodes:
        print(f"Truncating from {len(nodes)} to {args.max_nodes} nodes")
        nodes = nodes[:args.max_nodes]

    print(f"Document: {len(nodes)} nodes")

    # Compute embeddings
    all_embs: dict[str, torch.Tensor] = compute_embeddings(model, vocab, nodes)

    # Draw tree colored by selected mode
    vectors: torch.Tensor = all_embs[args.mode]
    colors: np.ndarray = embeddings_to_colors(vectors)

    mode_titles: dict[str, str] = {
        "tree_pos": "Tree Positional Encoding (depth + sibling + type + parent_key)",
        "full": "Full Embedding (token + tree position)",
        "token_only": "Token Embedding Only (no positional info)",
    }

    draw_tree(
        nodes, colors,
        title=mode_titles[args.mode],
        output_path=args.output,
    )

    # Also generate all three modes for comparison
    base: str = os.path.splitext(args.output)[0]
    for mode in ["tree_pos", "full", "token_only"]:
        v = all_embs[mode]
        c = embeddings_to_colors(v)
        out: str = f"{base}_{mode}.png"
        draw_tree(nodes, c, title=mode_titles[mode], output_path=out)

    # Individual components
    for comp in ["depth", "sibling", "node_type", "parent_key"]:
        v = all_embs[comp]
        c = embeddings_to_colors(v)
        out = f"{base}_{comp}.png"
        draw_tree(nodes, c, title=f"Component: {comp}", output_path=out)

    print(f"\nAll plots saved with prefix: {base}_*.png")


if __name__ == "__main__":
    main()
