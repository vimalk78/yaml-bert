"""Build a 2D UMAP projection of v8 doc_vecs for the manifest galaxy demo.

Loads the 276K doc_vecs + doc_cache produced by v8 training, samples a 10K
subset, extracts (kind, name, namespace) per doc, projects to 2D with UMAP,
and writes hf-space/galaxy_data.json for the Space to render via Plotly.

Run once, commit the JSON. Re-run only if doc_vecs change.
"""

from __future__ import annotations

import json
import pickle
import random
import sys
from collections import Counter
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from yaml_bert.types import NodeType, YamlNode


DOC_VECS_PATH = "output_v8_276K_recon_seed42/doc_vecs_epoch_20.pt"
CACHE_PATH = "output_v8_276K_recon_seed42/doc_cache.pkl"
OUT_PATH = "hf-space/galaxy_data.json"
N_SUBSET = 10_000
SEED = 42


def extract_name_namespace(nodes: list[YamlNode]) -> tuple[str, str]:
    """Return (name, namespace) for a single doc."""
    name = "(no name)"
    namespace = "(default)"
    in_metadata = False
    for i, n in enumerate(nodes):
        if n.depth == 0 and n.node_type == NodeType.KEY:
            in_metadata = n.token == "metadata"
        elif in_metadata and n.depth == 1 and n.node_type == NodeType.KEY:
            if i + 1 >= len(nodes):
                continue
            nxt = nodes[i + 1]
            if nxt.node_type != NodeType.VALUE:
                continue
            if n.token == "name":
                name = nxt.token
            elif n.token == "namespace":
                namespace = nxt.token
    return name, namespace


def main() -> None:
    print(f"Loading doc_vecs from {DOC_VECS_PATH}...")
    payload = torch.load(DOC_VECS_PATH, weights_only=True, map_location="cpu")
    doc_vecs = payload["doc_vecs"]
    all_kinds = payload["kinds"]
    print(f"  shape: {tuple(doc_vecs.shape)}, kinds: {len(all_kinds):,}")

    print(f"Loading cache from {CACHE_PATH}...")
    with open(CACHE_PATH, "rb") as f:
        cache = pickle.load(f)
    print(f"  {len(cache):,} docs")
    assert len(cache) == doc_vecs.shape[0] == len(all_kinds), "length mismatch"

    random.seed(SEED)
    indices = sorted(random.sample(range(len(cache)), N_SUBSET))
    print(f"Sampling {N_SUBSET:,} docs (seed={SEED})...")

    subset_vecs = doc_vecs[indices].numpy()
    kinds = [all_kinds[i] or "(unknown)" for i in indices]
    nn_pairs = [extract_name_namespace(cache[i]) for i in indices]
    names = [p[0] for p in nn_pairs]
    namespaces = [p[1] for p in nn_pairs]

    print("Kind distribution (top 25):")
    kind_counts = Counter(kinds)
    for k, c in kind_counts.most_common(25):
        print(f"  {k:35s} {c:5d}")
    print(f"  ... {len(kind_counts)} distinct kinds total")

    print("Computing UMAP projection (2D, cosine, this takes ~1-2 min)...")
    import umap
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.1,
        metric="cosine",
        random_state=SEED,
    )
    xy = reducer.fit_transform(subset_vecs)

    print(f"Writing {OUT_PATH}...")
    data = {
        "x": [round(float(v), 3) for v in xy[:, 0]],
        "y": [round(float(v), 3) for v in xy[:, 1]],
        "kind": kinds,
        "name": names,
        "namespace": namespaces,
        "n": N_SUBSET,
        "source": DOC_VECS_PATH,
        "seed": SEED,
    }
    with open(OUT_PATH, "w") as f:
        json.dump(data, f)
    size_mb = Path(OUT_PATH).stat().st_size / 1024 / 1024
    print(f"  wrote {N_SUBSET:,} points, {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
