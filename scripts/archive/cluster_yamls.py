"""Cluster, search, and find outliers in K8s YAML collections.

Usage:
    python scripts/cluster_yamls.py --encoder ckpt.pt --pooling pooling.pt --corpus ./manifests/ --cluster
    python scripts/cluster_yamls.py --encoder ckpt.pt --pooling pooling.pt --query my.yaml --corpus ./manifests/
    python scripts/cluster_yamls.py --encoder ckpt.pt --pooling pooling.pt --corpus ./manifests/ --outliers
    python scripts/cluster_yamls.py --encoder ckpt.pt --pooling pooling.pt --corpus ./manifests/ --filter-kind Deployment --cluster
"""
from __future__ import annotations
import _setup_path  # noqa: F401

import argparse
import glob
import os

import torch
import numpy as np

from yaml_bert.config import YamlBertConfig
from yaml_bert.embedding import YamlBertEmbedding
from yaml_bert.model import YamlBertModel
from yaml_bert.pooling import DocumentPooling
from yaml_bert.similarity import get_document_embedding, cosine_similarity_matrix
from yaml_bert.vocab import Vocabulary
from yaml_bert.linearizer import YamlLinearizer
from yaml_bert.annotator import DomainAnnotator
from yaml_bert.dataset import _extract_kind


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cluster and search K8s YAMLs")
    parser.add_argument("--encoder", type=str, required=True, help="Encoder checkpoint")
    parser.add_argument("--pooling", type=str, required=True, help="Pooling layer checkpoint")
    parser.add_argument("--vocab", type=str, required=True)
    parser.add_argument("--corpus", type=str, required=True, help="Directory of YAML files")
    parser.add_argument("--query", type=str, default=None, help="Find similar to this file")
    parser.add_argument("--cluster", action="store_true", help="Cluster the corpus")
    parser.add_argument("--n-clusters", type=int, default=5, help="Number of clusters")
    parser.add_argument("--outliers", action="store_true", help="Find outliers")
    parser.add_argument("--filter-kind", type=str, default=None, help="Only include this kind")
    parser.add_argument("--top-k", type=int, default=5, help="Top K similar results")
    return parser.parse_args()


def load_models(args) -> tuple[YamlBertModel, DocumentPooling, Vocabulary]:
    torch.manual_seed(42)
    vocab: Vocabulary = Vocabulary.load(args.vocab)
    config: YamlBertConfig = YamlBertConfig()
    emb = YamlBertEmbedding(
        config=config,
        key_vocab_size=vocab.key_vocab_size,
        value_vocab_size=vocab.value_vocab_size,
        kind_vocab_size=vocab.kind_vocab_size,
    )
    model = YamlBertModel(
        config=config, embedding=emb,
        key_vocab_size=vocab.key_vocab_size,
        kind_vocab_size=vocab.kind_vocab_size,
    )
    cp = torch.load(args.encoder, map_location="cpu", weights_only=False)
    model.load_state_dict(cp["model_state_dict"], strict=False)
    model.eval()

    pooling_cp = torch.load(args.pooling, map_location="cpu", weights_only=False)
    pooling = DocumentPooling(
        d_model=pooling_cp["d_model"],
        num_heads=pooling_cp["num_heads"],
    )
    pooling.load_state_dict(pooling_cp["pooling_state_dict"])
    pooling.eval()

    return model, pooling, vocab


def main() -> None:
    args = parse_args()
    model, pooling, vocab = load_models(args)

    linearizer: YamlLinearizer = YamlLinearizer()
    annotator: DomainAnnotator = DomainAnnotator()

    # Load corpus
    yaml_files: list[str] = sorted(
        glob.glob(os.path.join(args.corpus, "**", "*.yaml"), recursive=True)
    )

    print(f"Embedding {len(yaml_files)} files...")
    embeddings: list[torch.Tensor] = []
    file_names: list[str] = []
    file_kinds: list[str] = []

    for path in yaml_files:
        with open(path) as f:
            yaml_text: str = f.read()

        try:
            docs = linearizer.linearize_multi_doc(yaml_text)
            nodes = docs[0] if docs else []
        except Exception:
            continue
        if not nodes:
            continue
        annotator.annotate(nodes)
        kind: str = _extract_kind(nodes)

        if args.filter_kind and kind != args.filter_kind:
            continue

        try:
            emb: torch.Tensor = get_document_embedding(model, pooling, vocab, yaml_text)
        except Exception:
            continue
        embeddings.append(emb)
        file_names.append(os.path.relpath(path, args.corpus))
        file_kinds.append(kind)

    if not embeddings:
        print("No documents found.")
        return

    all_embs: torch.Tensor = torch.stack(embeddings)
    print(f"Embedded {len(embeddings)} documents")

    # Similarity search
    if args.query:
        with open(args.query) as f:
            query_text: str = f.read()
        query_emb: torch.Tensor = get_document_embedding(model, pooling, vocab, query_text)
        query_docs = linearizer.linearize_multi_doc(query_text)
        query_nodes = query_docs[0] if query_docs else []
        query_kind: str = _extract_kind(query_nodes) if query_nodes else "?"

        sims: torch.Tensor = torch.nn.functional.cosine_similarity(
            query_emb.unsqueeze(0), all_embs,
        )
        top_idx: torch.Tensor = sims.argsort(descending=True)[:args.top_k]

        print(f"\nQuery: {args.query} ({query_kind})")
        print(f"\nMost similar:")
        for rank, idx in enumerate(top_idx):
            i: int = idx.item()
            print(f"  {rank + 1}. [{sims[i]:.3f}] {file_names[i]} ({file_kinds[i]})")

    # Clustering
    if args.cluster:
        from sklearn.cluster import KMeans

        emb_np: np.ndarray = all_embs.numpy()
        n_clusters: int = min(args.n_clusters, len(embeddings))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels: np.ndarray = kmeans.fit_predict(emb_np)

        print(f"\nClusters (k={n_clusters}):")
        for c in range(n_clusters):
            members: list[int] = [i for i, l in enumerate(labels) if l == c]
            kinds_in_cluster: list[str] = [file_kinds[i] for i in members]
            kind_summary: str = ", ".join(
                f"{k}({kinds_in_cluster.count(k)})"
                for k in sorted(set(kinds_in_cluster))
            )
            print(f"\n  Cluster {c} ({len(members)} docs): {kind_summary}")
            for i in members[:5]:
                print(f"    {file_names[i]} ({file_kinds[i]})")
            if len(members) > 5:
                print(f"    ... and {len(members) - 5} more")

    # Outlier detection
    if args.outliers:
        emb_np = all_embs.numpy()
        centroid: np.ndarray = emb_np.mean(axis=0)
        distances: np.ndarray = np.linalg.norm(emb_np - centroid, axis=1)
        mean_dist: float = distances.mean()
        std_dist: float = distances.std()

        print(f"\nOutliers (>{2:.0f}σ from centroid):")
        outlier_idx: np.ndarray = np.where(distances > mean_dist + 2 * std_dist)[0]
        if len(outlier_idx) == 0:
            print("  No outliers detected.")
        else:
            for i in sorted(outlier_idx, key=lambda x: -distances[x]):
                sigma: float = (distances[i] - mean_dist) / std_dist
                print(f"  [{sigma:.1f}σ] {file_names[i]} ({file_kinds[i]})")


if __name__ == "__main__":
    main()
