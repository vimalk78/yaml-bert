"""Run smoke-test + finer-grained probes on doc_vec dumps from
train_v8_phase1_recon.py.

Reads doc_vecs_epoch_<N>.pt files + the raw doc_cache.pkl, builds labels
from cached YamlNode lists, runs:
  - 4 original smoke probes: kind, has-containers, has-initContainers,
    has-volume-mounts (sklearn LogisticRegression)
  - 5 finer binary structural probes: has-tolerations, has-affinity,
    has-multiple-containers, has-resource-limits, has-readiness-probe
  - 2 multi-class kind-filtered probes: service-type (Service only),
    update-strategy-type (Deployment/StatefulSet/DaemonSet only)
  - 2 retrieval probes (cosine over doc_vecs, no sklearn):
    triplet-accuracy@same-kind, knn-purity@5

Labels are derived directly from the cached YamlNode lists — no raw YAML text
or HF re-fetch required.
"""
from __future__ import annotations
import _setup_path  # noqa: F401

import argparse
import os
import pickle
import re
from collections import Counter

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from yaml_bert.dataset import _extract_kind
from yaml_bert.types import NodeType, YamlNode


# Structural positions where 'containers' / 'initContainers' are workload specs.
_CONTAINERS_PARENT_PATHS = frozenset({
    "spec",                                  # Pod
    "spec.template.spec",                    # Deployment, StatefulSet, DaemonSet, Job
    "spec.jobTemplate.spec.template.spec",   # CronJob
})

_VOLUME_MOUNTS_PATH_RE = re.compile(
    # Bare Pod: spec.containers.N or spec.initContainers.N
    r"^spec\.(containers|initContainers)\.\d+$"
    # Workload template (Deployment, StatefulSet, Job, etc.):
    r"|^spec\.template\.spec\.(containers|initContainers|ephemeralContainers)\.\d+$"
    # CronJob:
    r"|^spec\.jobTemplate\.spec\.template\.spec\.(containers|initContainers|ephemeralContainers)\.\d+$"
)

# A single container item's parent_path: spec[.template.spec].containers.N
_CONTAINER_ITEM_PATH_RE = re.compile(
    r"^spec(\.template\.spec)?\.containers\.\d+$"
)

# A container's resources sub-block: <container-item>.resources
_CONTAINER_RESOURCES_PATH_RE = re.compile(
    r"^spec(\.template\.spec)?\.containers\.\d+\.resources$"
)


def _label_has_workload_field(nodes: list[YamlNode], field_name: str) -> bool:
    """True if any KEY node has token=field_name at a workload spec position.

    Accepted parent_paths: 'spec' (Pod) or 'spec.template.spec' (Deployment /
    StatefulSet / DaemonSet) or CronJob's nested spec.
    """
    for n in nodes:
        if n.node_type not in (NodeType.KEY, NodeType.LIST_KEY):
            continue
        if n.token != field_name:
            continue
        if n.parent_path in _CONTAINERS_PARENT_PATHS:
            return True
    return False


def _label_has_containers(nodes: list[YamlNode]) -> bool:
    return _label_has_workload_field(nodes, "containers")


def _label_has_init_containers(nodes: list[YamlNode]) -> bool:
    return _label_has_workload_field(nodes, "initContainers")


def _label_has_volume_mounts(nodes: list[YamlNode]) -> bool:
    """True if any KEY node 'volumeMounts' sits inside a container list item."""
    for n in nodes:
        if n.node_type not in (NodeType.KEY, NodeType.LIST_KEY):
            continue
        if n.token != "volumeMounts":
            continue
        if _VOLUME_MOUNTS_PATH_RE.match(n.parent_path):
            return True
    return False


# ----- New finer binary probes ------------------------------------------------

def _label_has_tolerations(nodes: list[YamlNode]) -> bool:
    """True if KEY 'tolerations' exists at a workload spec position."""
    return _label_has_workload_field(nodes, "tolerations")


def _label_has_affinity(nodes: list[YamlNode]) -> bool:
    """True if KEY 'affinity' exists at a workload spec position."""
    return _label_has_workload_field(nodes, "affinity")


def _label_has_multiple_containers(nodes: list[YamlNode]) -> bool:
    """True if there are >=2 distinct container items in the workload spec.

    Each container is a list item under spec[.template.spec].containers; we
    count distinct numeric indices appearing as parent_path's terminal segment.
    """
    container_indices: set[str] = set()
    for n in nodes:
        if n.node_type not in (NodeType.KEY, NodeType.LIST_KEY):
            continue
        m = _CONTAINER_ITEM_PATH_RE.match(n.parent_path)
        if m:
            container_indices.add(n.parent_path)
    return len(container_indices) >= 2


def _label_has_resource_limits(nodes: list[YamlNode]) -> bool:
    """True if any container has a 'limits' KEY under its 'resources' block."""
    for n in nodes:
        if n.node_type not in (NodeType.KEY, NodeType.LIST_KEY):
            continue
        if n.token != "limits":
            continue
        if _CONTAINER_RESOURCES_PATH_RE.match(n.parent_path):
            return True
    return False


def _label_has_readiness_probe(nodes: list[YamlNode]) -> bool:
    """True if any container has a 'readinessProbe' KEY directly under it."""
    for n in nodes:
        if n.node_type not in (NodeType.KEY, NodeType.LIST_KEY):
            continue
        if n.token != "readinessProbe":
            continue
        if _CONTAINER_ITEM_PATH_RE.match(n.parent_path):
            return True
    return False


# ----- New multi-class probes (kind-filtered) ---------------------------------

# Service-type values. Missing → ClusterIP (K8s default).
_SERVICE_TYPES = ("ClusterIP", "NodePort", "LoadBalancer", "ExternalName")


def _label_service_type(nodes: list[YamlNode]) -> str | None:
    """For a Service doc, return one of _SERVICE_TYPES (default ClusterIP).
    For non-Services, return None (filtered out)."""
    if _extract_kind(nodes) != "Service":
        return None
    for n in nodes:
        if n.node_type != NodeType.VALUE:
            continue
        if n.parent_path != "spec.type":
            continue
        # Found explicit spec.type value
        if n.token in _SERVICE_TYPES:
            return n.token
        return None  # unknown / weird value — drop from probe
    return "ClusterIP"  # default when spec.type is absent


# Update-strategy values across Deployment/StatefulSet/DaemonSet.
_UPDATE_STRATEGY_TYPES = ("RollingUpdate", "Recreate", "OnDelete")
_UPDATE_STRATEGY_KINDS = frozenset({"Deployment", "StatefulSet", "DaemonSet"})
# Per-kind default per K8s API:
#   Deployment    → RollingUpdate (under spec.strategy.type)
#   StatefulSet   → RollingUpdate (under spec.updateStrategy.type)
#   DaemonSet     → RollingUpdate (under spec.updateStrategy.type)
_UPDATE_STRATEGY_PARENT_PATHS = frozenset({
    "spec.strategy", "spec.updateStrategy",
})


def _label_update_strategy_type(nodes: list[YamlNode]) -> str | None:
    """For Deployment/StatefulSet/DaemonSet, return strategy type."""
    if _extract_kind(nodes) not in _UPDATE_STRATEGY_KINDS:
        return None
    for n in nodes:
        if n.node_type != NodeType.VALUE:
            continue
        if n.parent_path not in {"spec.strategy.type", "spec.updateStrategy.type"}:
            continue
        if n.token in _UPDATE_STRATEGY_TYPES:
            return n.token
        return None
    return "RollingUpdate"  # default


def _extract_apiversion(nodes: list[YamlNode]) -> str:
    """Extract apiVersion VALUE from a doc, or empty string if absent."""
    for i, node in enumerate(nodes):
        if (node.token == "apiVersion"
                and node.depth == 0
                and node.node_type == NodeType.KEY
                and i + 1 < len(nodes)
                and nodes[i + 1].node_type == NodeType.VALUE):
            return nodes[i + 1].token
    return ""


def _build_labels(cached_docs: list[list[YamlNode]], top_k_kinds: int = 10) -> dict:
    """Build label arrays for all classification probes from cached YamlNode lists."""
    kinds = [_extract_kind(nodes) for nodes in cached_docs]
    counter = Counter(k for k in kinds if k)
    top_kinds = [k for k, _ in counter.most_common(top_k_kinds)]
    kind_to_idx = {k: i for i, k in enumerate(top_kinds)}
    kind_labels = np.array(
        [kind_to_idx.get(k, -1) for k in kinds], dtype=int,
    )

    # apiVersion probe: multi-class over the top-K most common apiVersions
    apivers = [_extract_apiversion(nodes) for nodes in cached_docs]
    av_counter = Counter(a for a in apivers if a)
    # Cap at 10 to keep the probe tractable; covers the long tail with -1
    top_apivers = [a for a, _ in av_counter.most_common(10)]
    av_to_idx = {a: i for i, a in enumerate(top_apivers)}
    apiversion_labels = np.array(
        [av_to_idx.get(a, -1) for a in apivers], dtype=int,
    )

    # apiVersion+Kind combined probe: K8s GroupVersionKind (GVK) — the canonical
    # resource type identifier. Multi-class over top-K combos.
    av_kind = [
        f"{a}|{k}" if (a and k) else ""
        for a, k in zip(apivers, kinds)
    ]
    avk_counter = Counter(c for c in av_kind if c)
    # Top 15: covers most workload + RBAC + storage combos in typical K8s corpora
    top_av_kinds = [c for c, _ in avk_counter.most_common(15)]
    avk_to_idx = {c: i for i, c in enumerate(top_av_kinds)}
    apiversion_kind_labels = np.array(
        [avk_to_idx.get(c, -1) for c in av_kind], dtype=int,
    )

    # Multi-class probes: build string labels, then encode + mask
    service_strs = [_label_service_type(nodes) for nodes in cached_docs]
    service_idx_map = {t: i for i, t in enumerate(_SERVICE_TYPES)}
    service_labels = np.array(
        [service_idx_map[s] if s is not None else -1 for s in service_strs],
        dtype=int,
    )

    strategy_strs = [_label_update_strategy_type(nodes) for nodes in cached_docs]
    strategy_idx_map = {t: i for i, t in enumerate(_UPDATE_STRATEGY_TYPES)}
    strategy_labels = np.array(
        [strategy_idx_map[s] if s is not None else -1 for s in strategy_strs],
        dtype=int,
    )

    return {
        # Existing
        "kind_labels": kind_labels,
        "kind_names": top_kinds,
        "has_containers": np.array(
            [_label_has_containers(nodes) for nodes in cached_docs], dtype=int,
        ),
        "has_init_containers": np.array(
            [_label_has_init_containers(nodes) for nodes in cached_docs], dtype=int,
        ),
        "has_volume_mounts": np.array(
            [_label_has_volume_mounts(nodes) for nodes in cached_docs], dtype=int,
        ),
        # New binary
        "has_tolerations": np.array(
            [_label_has_tolerations(nodes) for nodes in cached_docs], dtype=int,
        ),
        "has_affinity": np.array(
            [_label_has_affinity(nodes) for nodes in cached_docs], dtype=int,
        ),
        "has_multiple_containers": np.array(
            [_label_has_multiple_containers(nodes) for nodes in cached_docs], dtype=int,
        ),
        "has_resource_limits": np.array(
            [_label_has_resource_limits(nodes) for nodes in cached_docs], dtype=int,
        ),
        "has_readiness_probe": np.array(
            [_label_has_readiness_probe(nodes) for nodes in cached_docs], dtype=int,
        ),
        # New multi-class (label_filter applied per-probe via -1 sentinel)
        "service_type_labels": service_labels,
        "update_strategy_labels": strategy_labels,
        # apiVersion + apiVersion-kind combined
        "apiversion_labels": apiversion_labels,
        "apiversion_names": top_apivers,
        "apiversion_kind_labels": apiversion_kind_labels,
        "apiversion_kind_names": top_av_kinds,
        # All kinds (for retrieval probes — not just top-K)
        "all_kinds": np.array(
            [k if k else "_unknown" for k in kinds], dtype=object,
        ),
    }


def _probe_accuracy(
    X: np.ndarray,
    y: np.ndarray,
    label_filter: np.ndarray | None = None,
) -> float:
    """Fit LogisticRegression on 80% of (X, y), report accuracy on 20%."""
    if label_filter is not None:
        X = X[label_filter]
        y = y[label_filter]
    # Drop classes with <2 members (can't stratify, can't meaningfully probe)
    if len(y) > 0:
        unique, counts = np.unique(y, return_counts=True)
        keep_classes = unique[counts >= 2]
        keep_mask = np.isin(y, keep_classes)
        X = X[keep_mask]
        y = y[keep_mask]
    if len(np.unique(y)) < 2:
        return float("nan")
    if len(y) < 10:
        return float("nan")  # too few samples for a meaningful split
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y,
    )
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(X_tr, y_tr)
    return float(clf.score(X_te, y_te))


def _triplet_accuracy(
    X: np.ndarray,
    kinds: np.ndarray,
    n_triplets: int = 1000,
    seed: int = 42,
) -> float:
    """Sample n_triplets (A, B, C) where B has same kind as A and C has different.
    Return fraction where cos(A,B) > cos(A,C)."""
    rng = np.random.RandomState(seed)
    unique_kinds = list({k for k in kinds if k != "_unknown"})
    if len(unique_kinds) < 2:
        return float("nan")

    # Pre-index docs by kind for sampling
    kind_to_indices: dict[str, list[int]] = {}
    for i, k in enumerate(kinds):
        kind_to_indices.setdefault(k, []).append(i)

    # Normalize doc_vecs for cosine
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    Xn = X / norms

    passes = 0
    attempted = 0
    for _ in range(n_triplets):
        a_kind = unique_kinds[rng.randint(len(unique_kinds))]
        same_idxs = kind_to_indices.get(a_kind, [])
        if len(same_idxs) < 2:
            continue
        diff_kinds = [k for k in unique_kinds if k != a_kind
                      and len(kind_to_indices.get(k, [])) > 0]
        if not diff_kinds:
            continue
        a_i = same_idxs[rng.randint(len(same_idxs))]
        # Re-sample b until different from a
        for _ in range(10):
            b_i = same_idxs[rng.randint(len(same_idxs))]
            if b_i != a_i:
                break
        else:
            continue
        c_kind = diff_kinds[rng.randint(len(diff_kinds))]
        c_pool = kind_to_indices[c_kind]
        c_i = c_pool[rng.randint(len(c_pool))]

        # cosine: since normalized, just dot product
        sim_ab = float(Xn[a_i] @ Xn[b_i])
        sim_ac = float(Xn[a_i] @ Xn[c_i])
        attempted += 1
        if sim_ab > sim_ac:
            passes += 1

    if attempted == 0:
        return float("nan")
    return passes / attempted


def _knn_purity_at_5(
    X: np.ndarray,
    kinds: np.ndarray,
    k: int = 5,
    n_queries: int = 5000,
    chunk_size: int = 500,
    seed: int = 42,
) -> float:
    """For a random sample of query docs, find top-k nearest by cosine across
    the full corpus. Return mean fraction sharing the query's kind.

    Sampled + chunked to avoid the O(N^2) pairwise-similarity memory blowup
    at large N (276K docs would need 285 GB for the full matrix).
    """
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    Xn = X / norms

    n_docs = Xn.shape[0]
    # Sample query indices (excluding _unknown kinds for cleaner signal)
    valid_idx = np.where(kinds != "_unknown")[0]
    if len(valid_idx) == 0:
        return float("nan")
    n_sample = min(n_queries, len(valid_idx))
    rng = np.random.RandomState(seed)
    query_idx = rng.choice(valid_idx, size=n_sample, replace=False)

    purities: list[float] = []
    for start in range(0, n_sample, chunk_size):
        end = min(start + chunk_size, n_sample)
        chunk_q_idx = query_idx[start:end]
        # (chunk, d_model) @ (d_model, N) -> (chunk, N)
        chunk_sims = Xn[chunk_q_idx] @ Xn.T  # ~500 * 276K * 4B = 553 MB peak
        # Mask self-similarity by setting diagonal-like entries to -inf
        for local_i, doc_i in enumerate(chunk_q_idx):
            chunk_sims[local_i, doc_i] = -np.inf
        # Top-k indices per row
        topk_idx = np.argpartition(-chunk_sims, kth=k - 1, axis=1)[:, :k]
        for local_i, doc_i in enumerate(chunk_q_idx):
            query_kind = kinds[doc_i]
            neighbours = topk_idx[local_i]
            match = sum(1 for j in neighbours if kinds[j] == query_kind)
            purities.append(match / k)

    return float(np.mean(purities)) if purities else float("nan")


def _eval_one_dump(doc_vecs_path: str, labels: dict) -> dict:
    """Run all probes on one doc_vec dump file."""
    data = torch.load(doc_vecs_path, map_location="cpu", weights_only=False)
    X = data["doc_vecs"].numpy()  # (D, d_model)
    n = X.shape[0]

    kind_mask = labels["kind_labels"][:n] >= 0
    service_mask = labels["service_type_labels"][:n] >= 0
    strategy_mask = labels["update_strategy_labels"][:n] >= 0
    av_mask = labels["apiversion_labels"][:n] >= 0
    avk_mask = labels["apiversion_kind_labels"][:n] >= 0

    return {
        # Original 4
        "kind": _probe_accuracy(X, labels["kind_labels"][:n], label_filter=kind_mask),
        "has_containers": _probe_accuracy(X, labels["has_containers"][:n]),
        "has_init_containers": _probe_accuracy(X, labels["has_init_containers"][:n]),
        "has_volume_mounts": _probe_accuracy(X, labels["has_volume_mounts"][:n]),
        # New 5 binary
        "has_tolerations": _probe_accuracy(X, labels["has_tolerations"][:n]),
        "has_affinity": _probe_accuracy(X, labels["has_affinity"][:n]),
        "has_multi_containers": _probe_accuracy(X, labels["has_multiple_containers"][:n]),
        "has_resource_limits": _probe_accuracy(X, labels["has_resource_limits"][:n]),
        "has_readiness_probe": _probe_accuracy(X, labels["has_readiness_probe"][:n]),
        # New 2 multi-class (kind-filtered)
        "service_type": _probe_accuracy(
            X, labels["service_type_labels"][:n], label_filter=service_mask,
        ),
        "update_strategy": _probe_accuracy(
            X, labels["update_strategy_labels"][:n], label_filter=strategy_mask,
        ),
        # NEW: apiVersion + apiVersion-kind multi-class
        "apiversion": _probe_accuracy(
            X, labels["apiversion_labels"][:n], label_filter=av_mask,
        ),
        "apiversion_kind": _probe_accuracy(
            X, labels["apiversion_kind_labels"][:n], label_filter=avk_mask,
        ),
        # 2 retrieval
        "triplet": _triplet_accuracy(X, labels["all_kinds"][:n]),
        "knn5": _knn_purity_at_5(X, labels["all_kinds"][:n]),
    }


def _fmt_pct(v: float) -> str:
    if v != v:  # NaN
        return "  n/a"
    return f"{v * 100:>5.1f}%"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True,
                        help="dir containing doc_vecs_epoch_*.pt and doc_cache.pkl")
    parser.add_argument("--top-k-kinds", type=int, default=10)
    args = parser.parse_args()

    cache_path = os.path.join(args.output_dir, "doc_cache.pkl")
    print(f"Loading doc_cache from {cache_path}")
    with open(cache_path, "rb") as f:
        cached: list[list[YamlNode]] = pickle.load(f)
    print(f"Loaded {len(cached):,} cached documents")

    print(f"Building labels for {len(cached)} docs from linearized nodes...")
    labels = _build_labels(cached, top_k_kinds=args.top_k_kinds)
    print(f"Top kinds: {labels['kind_names']}")
    print(f"Top apiVersions ({len(labels['apiversion_names'])}): "
          f"{labels['apiversion_names']}")
    print(f"Top apiVersion+Kind ({len(labels['apiversion_kind_names'])}): "
          f"{labels['apiversion_kind_names']}")
    print(
        f"Positive counts: "
        f"containers={int(labels['has_containers'].sum())} | "
        f"init={int(labels['has_init_containers'].sum())} | "
        f"vol_mounts={int(labels['has_volume_mounts'].sum())} | "
        f"tol={int(labels['has_tolerations'].sum())} | "
        f"aff={int(labels['has_affinity'].sum())} | "
        f"multi_ctr={int(labels['has_multiple_containers'].sum())} | "
        f"res_lim={int(labels['has_resource_limits'].sum())} | "
        f"readyP={int(labels['has_readiness_probe'].sum())}"
    )
    print(
        f"Multi-class N: "
        f"service_type={int((labels['service_type_labels'] >= 0).sum())} "
        f"({len(_SERVICE_TYPES)}-class) | "
        f"update_strategy={int((labels['update_strategy_labels'] >= 0).sum())} "
        f"({len(_UPDATE_STRATEGY_TYPES)}-class) | "
        f"apiVersion={int((labels['apiversion_labels'] >= 0).sum())} "
        f"({len(labels['apiversion_names'])}-class) | "
        f"apiVersion+Kind={int((labels['apiversion_kind_labels'] >= 0).sum())} "
        f"({len(labels['apiversion_kind_names'])}-class)"
    )

    # Find all per-epoch dumps
    dumps = sorted(
        [
            f for f in os.listdir(args.output_dir)
            if f.startswith("doc_vecs_epoch_") and f.endswith(".pt")
        ],
        key=lambda f: int(f.split("_")[3].split(".")[0]),
    )
    if not dumps:
        print(f"No per-epoch dumps in {args.output_dir}; trying doc_vecs.pt")
        candidate = os.path.join(args.output_dir, "doc_vecs.pt")
        if os.path.exists(candidate):
            dumps = ["doc_vecs.pt"]
        else:
            print(f"ERROR: no dump files found in {args.output_dir}")
            return

    # Header: epoch | 4 original | 5 binary | 2 multi | 2 GVK | 2 retrieval
    header_cols = [
        ("ep",     "%4s"),
        ("kind",   "%6s"),
        ("ctr",    "%6s"),
        ("init",   "%6s"),
        ("volM",   "%6s"),
        ("tol",    "%6s"),
        ("aff",    "%6s"),
        ("multiC", "%6s"),
        ("resL",   "%6s"),
        ("readyP", "%6s"),
        ("svcTy",  "%6s"),
        ("updSt",  "%6s"),
        ("apiV",   "%6s"),
        ("apiVK",  "%6s"),
        ("trip",   "%6s"),
        ("knn5",   "%6s"),
    ]
    header_line = " | ".join(fmt % name for name, fmt in header_cols)
    print(f"\n{header_line}")
    print("-" * len(header_line))

    result_keys = [
        "_epoch", "kind", "has_containers", "has_init_containers",
        "has_volume_mounts", "has_tolerations", "has_affinity",
        "has_multi_containers", "has_resource_limits", "has_readiness_probe",
        "service_type", "update_strategy", "triplet", "knn5",
    ]

    for fn in dumps:
        epoch_label: str = (
            str(int(fn.split("_")[3].split(".")[0]))
            if "epoch_" in fn
            else "fin"
        )
        results = _eval_one_dump(os.path.join(args.output_dir, fn), labels)
        row = [
            f"{epoch_label:>4}",
            _fmt_pct(results["kind"]),
            _fmt_pct(results["has_containers"]),
            _fmt_pct(results["has_init_containers"]),
            _fmt_pct(results["has_volume_mounts"]),
            _fmt_pct(results["has_tolerations"]),
            _fmt_pct(results["has_affinity"]),
            _fmt_pct(results["has_multi_containers"]),
            _fmt_pct(results["has_resource_limits"]),
            _fmt_pct(results["has_readiness_probe"]),
            _fmt_pct(results["service_type"]),
            _fmt_pct(results["update_strategy"]),
            _fmt_pct(results["apiversion"]),
            _fmt_pct(results["apiversion_kind"]),
            _fmt_pct(results["triplet"]),
            _fmt_pct(results["knn5"]),
        ]
        print(" | ".join(row))


if __name__ == "__main__":
    main()
