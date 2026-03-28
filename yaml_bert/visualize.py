from __future__ import annotations

from typing import Any

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import torch


def plot_training_loss(
    losses: list[float],
    output_path: str = "training_loss.png",
    title: str = "YAML-BERT Training Loss",
) -> None:
    """Plot training loss curve over epochs."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(1, len(losses) + 1), losses, marker="o", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Training loss plot saved: {output_path}")


def plot_embedding_similarity(
    results: list[dict[str, Any]],
    output_path: str = "embedding_similarity.png",
    title: str = "Tree Position Embedding Similarity",
) -> None:
    """Plot cosine similarity between same-key embeddings at different tree positions."""
    labels: list[str] = []
    similarities: list[float] = []

    for r in results:
        pa: dict[str, Any] = r["position_a"]
        pb: dict[str, Any] = r["position_b"]
        label: str = (
            f"{r['key']}\n"
            f"d={pa['depth']},p={pa['parent_key']}\n"
            f"vs d={pb['depth']},p={pb['parent_key']}"
        )
        labels.append(label)
        similarities.append(r["cosine_similarity"])

    fig, ax = plt.subplots(figsize=(max(8, len(results) * 3), 6))
    bars = ax.bar(range(len(results)), similarities, color="steelblue")
    ax.set_xticks(range(len(results)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Cosine Similarity")
    ax.set_title(title)
    ax.set_ylim(-1, 1)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)

    for bar, sim in zip(bars, similarities):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{sim:.3f}",
            ha="center",
            fontsize=10,
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Embedding similarity plot saved: {output_path}")


def plot_attention_patterns(
    attention_weights: torch.Tensor,
    token_labels: list[str],
    output_path: str = "attention_patterns.png",
    title: str = "Attention Patterns",
) -> None:
    """Plot attention heatmaps for each head.

    Args:
        attention_weights: (num_heads, seq_len, seq_len)
        token_labels: labels for each position in the sequence
    """
    num_heads: int = attention_weights.shape[0]
    fig, axes = plt.subplots(1, num_heads, figsize=(6 * num_heads, 5))

    if num_heads == 1:
        axes = [axes]

    for head_idx, ax in enumerate(axes):
        weights: torch.Tensor = attention_weights[head_idx].cpu()
        im = ax.imshow(weights.numpy(), cmap="Blues", vmin=0, vmax=1)
        ax.set_title(f"Head {head_idx}")
        ax.set_xticks(range(len(token_labels)))
        ax.set_yticks(range(len(token_labels)))
        ax.set_xticklabels(token_labels, rotation=45, ha="right", fontsize=7)
        ax.set_yticklabels(token_labels, fontsize=7)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Attention pattern plot saved: {output_path}")
