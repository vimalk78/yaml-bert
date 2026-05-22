"""Plot static key vs value embeddings to show they share one R^d space.

Loads the input embedding tables from a checkpoint, projects them to 2D
via PCA and t-SNE, and saves a side-by-side scatter colored by source
table. Used in the presentation to demonstrate that the two embedding
tables do NOT self-organize into separate clusters — they intermix
completely, with node_type_embedding carrying all explicit position-type
signal.

Usage:
    python scripts/plot_static_embeddings.py <checkpoint> --vocab <vocab.json>
"""
from __future__ import annotations
import _setup_path  # noqa: F401

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot static key vs value embeddings")
    parser.add_argument("checkpoint", type=str)
    parser.add_argument("--vocab", type=str, required=True)
    parser.add_argument("--out", type=str, default="/tmp/key_value_embeddings.png")
    parser.add_argument("--perplexity", type=int, default=30)
    args = parser.parse_args()

    cp = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    sd = cp["model_state_dict"]
    key_w = sd["embedding.key_embedding.weight"].numpy()
    val_w = sd["embedding.value_embedding.weight"].numpy()
    print(f"key_emb: {key_w.shape}, value_emb: {val_w.shape}")

    X = np.vstack([key_w, val_w])
    labels = np.array([0] * len(key_w) + [1] * len(val_w))

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    print(f"PCA top-2 explained: {pca.explained_variance_ratio_}")

    # Quantitative separation read
    key_c = key_w.mean(0)
    val_c = val_w.mean(0)
    cent_dist = float(np.linalg.norm(key_c - val_c))
    key_spread = float(np.linalg.norm(key_w - key_c, axis=1).mean())
    val_spread = float(np.linalg.norm(val_w - val_c, axis=1).mean())
    sep = cent_dist / ((key_spread + val_spread) / 2)
    print(
        f"centroid dist={cent_dist:.3f}  key_spread={key_spread:.3f}  "
        f"val_spread={val_spread:.3f}  separation ratio={sep:.3f}"
    )

    print("Running t-SNE (~30s)...")
    tsne = TSNE(
        n_components=2,
        init="pca",
        random_state=42,
        perplexity=args.perplexity,
        max_iter=1000,
    )
    X_tsne = tsne.fit_transform(X)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    panels = [
        (axes[0], X_pca, f"PCA (top2 explains {pca.explained_variance_ratio_.sum() * 100:.1f}%)"),
        (axes[1], X_tsne, f"t-SNE (perplexity={args.perplexity})"),
    ]
    for ax, X_red, title in panels:
        ax.scatter(
            X_red[labels == 0, 0], X_red[labels == 0, 1],
            c="C0", s=6, alpha=0.5, label=f"keys (n={len(key_w)})",
        )
        ax.scatter(
            X_red[labels == 1, 0], X_red[labels == 1, 1],
            c="C1", s=6, alpha=0.5, label=f"values (n={len(val_w)})",
        )
        ax.set_title(title)
        ax.legend()
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle("YAML-BERT: key vs value embeddings in R^256 (static, before attention)")
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
