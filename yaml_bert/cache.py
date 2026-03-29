"""Cache linearized YAML documents to disk.

Linearizes all documents once (with multiprocessing), saves to pickle.
Both VocabBuilder and YamlDataset load from cache — no re-parsing.

Usage:
    from yaml_bert.cache import build_or_load_cache

    docs = build_or_load_cache(
        dataset_name="substratusai/the-stack-yaml-k8s",
        cache_path="output_v3/doc_cache.pkl",
        max_docs=276520,
    )
"""
from __future__ import annotations

import os
import pickle
import time
from multiprocessing import Pool, cpu_count
from typing import Any

from yaml_bert.annotator import DomainAnnotator
from yaml_bert.linearizer import YamlLinearizer
from yaml_bert.types import YamlNode


def _linearize_one(yaml_content: str) -> list[YamlNode] | None:
    """Linearize a single YAML string. Used by multiprocessing pool."""
    try:
        linearizer = YamlLinearizer()
        annotator = DomainAnnotator()
        nodes: list[YamlNode] = linearizer.linearize(yaml_content)
        if nodes:
            annotator.annotate(nodes)
            return nodes
    except Exception:
        pass
    return None


def build_or_load_cache(
    dataset_name: str,
    cache_path: str,
    max_docs: int | None = None,
    num_workers: int | None = None,
) -> list[list[YamlNode]]:
    """Build or load cached linearized documents.

    Args:
        dataset_name: HuggingFace dataset ID
        cache_path: Path to save/load the cache pickle
        max_docs: Max documents to process (None = all)
        num_workers: Number of parallel workers (None = cpu_count)

    Returns:
        List of linearized documents (each is a list of YamlNode)
    """
    if os.path.exists(cache_path):
        print(f"Loading cached documents from {cache_path}")
        start: float = time.time()
        with open(cache_path, "rb") as f:
            documents: list[list[YamlNode]] = pickle.load(f)
        print(f"Loaded {len(documents):,} documents in {time.time() - start:.1f}s")
        return documents

    from datasets import load_dataset

    print(f"Building document cache from {dataset_name}")
    ds = load_dataset(dataset_name, split="train")

    total: int = len(ds) if max_docs is None else min(max_docs, len(ds))
    print(f"Linearizing {total:,} documents with {num_workers or cpu_count()} workers...")

    # Extract YAML content strings
    yaml_contents: list[str] = [ds[i]["content"] for i in range(total)]

    # Parallel linearization with progress bar
    from tqdm import tqdm

    start = time.time()
    workers: int = num_workers or cpu_count()
    with Pool(workers) as pool:
        results: list[list[YamlNode] | None] = list(tqdm(
            pool.imap(_linearize_one, yaml_contents, chunksize=500),
            total=len(yaml_contents),
            desc="Linearizing",
        ))

    documents = [doc for doc in results if doc is not None]
    skipped: int = sum(1 for doc in results if doc is None)
    elapsed: float = time.time() - start

    print(f"Linearized {len(documents):,} documents in {elapsed:.1f}s ({skipped} skipped)")

    # Save cache
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(documents, f)

    cache_size: float = os.path.getsize(cache_path) / 1024 / 1024
    print(f"Cache saved: {cache_path} ({cache_size:.1f} MB)")

    return documents
