"""V8 missing-field suggester using V8Model (atomic-vocab prediction head).

API is identical to yaml_bert/suggest.py's suggest_missing_fields, but uses
V8Model + atomic vocabulary instead of YamlBertModel + compound vocabulary.

Key differences from v7:
- Single atomic head (no kind_head / simple_head routing)
- Decoding uses vocab.atomic_target_vocab reverse map
- Building the input batch via V8Dataset + v8_collate_fn (precomputes tree tensors)
- No path stripping: output IS the atomic key (e.g., "image", not "containers::image")
"""
from __future__ import annotations

import sys
from typing import Any

import torch
import torch.nn.functional as F

import yaml as _yaml

from yaml_bert.annotator import DomainAnnotator
from yaml_bert.config import YamlBertConfig
from yaml_bert.linearizer import YamlLinearizer
from yaml_bert.types import NodeType, YamlNode, _extract_kind  # noqa: F401
from yaml_bert.v8_dataset import V8Dataset, v8_collate_fn
from yaml_bert.v8_model import V8Model
from yaml_bert.vocab import Vocabulary

# Keys managed by the cluster, not written by users
_CLUSTER_MANAGED_KEYS: set[str] = {
    "status", "creationTimestamp", "generation", "resourceVersion",
    "selfLink", "uid", "managedFields",
}


def _find_empty_mapping_paths(yaml_text: str) -> list[tuple[str, str, int]]:
    """Walk the YAML and find keys whose value is an empty mapping ({}).
    These are not represented as 'parents' in the linearized output because
    they have no child KEY nodes, but they are valid probe positions.

    Returns list of (full_path, key_name, child_depth) tuples.
    full_path: dotted path that becomes parent_path of the would-be child.
    key_name: the empty parent's key (used as parent_key_name during probe).
    child_depth: depth at which children would be emitted by the linearizer.
    """
    try:
        data = _yaml.safe_load(yaml_text)
    except Exception:
        return []
    if data is None:
        return []
    results: list[tuple[str, str, int]] = []

    def walk(d: Any, depth: int, parent_path: str) -> None:
        if isinstance(d, dict):
            for k, v in d.items():
                k_str = str(k)
                child_path = f"{parent_path}.{k_str}" if parent_path else k_str
                if isinstance(v, dict):
                    if not v:
                        results.append((child_path, k_str, depth + 1))
                    else:
                        walk(v, depth + 1, child_path)
                elif isinstance(v, list):
                    walk_list(v, depth + 1, child_path)
                elif v is None:
                    # `key:` with no value (parses as None in PyYAML). Treat as
                    # potential empty mapping — probe under it. If the position
                    # is truly scalar (e.g. `image:`), the model's predictions
                    # will be diffuse/low-confidence and nothing surfaces.
                    results.append((child_path, k_str, depth + 1))

    def walk_list(lst: list, depth: int, parent_path: str) -> None:
        for i, item in enumerate(lst):
            item_path = f"{parent_path}.{i}"
            if isinstance(item, dict):
                walk(item, depth, item_path)  # list items don't increment depth
            elif isinstance(item, list):
                walk_list(item, depth, item_path)

    walk(data, 0, "")
    return results


_NODE_TYPE_INDEX: dict[NodeType, int] = {
    NodeType.KEY: 0,
    NodeType.VALUE: 1,
    NodeType.LIST_KEY: 2,
    NodeType.LIST_VALUE: 3,
}


def suggest_missing_fields_v8(
    model: V8Model,
    vocab: Vocabulary,
    yaml_text: str,
    threshold: float = 0.3,
    top_k: int = 10,
    verbose: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Suggest missing fields using V8Model (atomic-vocab head).

    Args:
        model: Trained V8Model
        vocab: Vocabulary with atomic_target_vocab populated
        yaml_text: Raw YAML text
        threshold: Minimum confidence to report a missing field
        top_k: Number of predictions per masked position
        verbose: Print per-probe debug output to stderr

    Returns:
        (suggestions, skipped_by_parent) where:
          suggestions: list of {"parent_path", "missing_key", "confidence"} sorted by -confidence
          skipped_by_parent: empty dict (API parity with v7; no parent-routing in v8)
    """
    linearizer = YamlLinearizer()
    annotator = DomainAnnotator()
    nodes: list[YamlNode] = linearizer.linearize(yaml_text)
    if not nodes:
        return [], {}
    annotator.annotate(nodes)

    mask_id: int = vocab.special_tokens["[MASK]"]

    # Build atomic reverse map for decoding
    id_to_atomic: dict[int, str] = {v: k for k, v in vocab.atomic_target_vocab.items()}
    id_to_special: dict[int, str] = {v: k for k, v in vocab.special_tokens.items()}

    # Group key nodes by parent_path
    keys_by_parent: dict[str, set[str]] = {}
    key_positions_by_parent: dict[str, list[int]] = {}
    for i, node in enumerate(nodes):
        if node.node_type in (NodeType.KEY, NodeType.LIST_KEY):
            keys_by_parent.setdefault(node.parent_path, set()).add(node.token)
            key_positions_by_parent.setdefault(node.parent_path, []).append(i)

    all_root_keys: set[str] = {
        n.token for n in nodes
        if n.node_type in (NodeType.KEY, NodeType.LIST_KEY) and n.depth == 0
    }

    model.eval()
    predicted_keys_by_parent: dict[str, dict[str, float]] = {}
    skipped_by_parent: dict[str, Any] = {}

    # Build probe specs: same two-source logic as v7
    # (a) Non-empty parents: probe as "next sibling of last child"
    # (b) Empty mappings: probe as "first child"
    probe_specs: list[dict[str, Any]] = []

    for parent_path, positions in key_positions_by_parent.items():
        last_pos: int = positions[-1]
        last_node: YamlNode = nodes[last_pos]
        insert_pos = last_pos + 1
        while insert_pos < len(nodes) and nodes[insert_pos].depth > last_node.depth:
            insert_pos += 1
        probe_specs.append({
            "parent_path": parent_path,
            "insert_pos": insert_pos,
            "fake_depth": last_node.depth,
            "fake_sibling": min(last_node.sibling_index + 1, 31),
            "fake_node_type": last_node.node_type,
            "ref_depth": last_node.depth,
            "is_empty_parent": False,
        })

    for empty_full_path, empty_key_name, child_depth in _find_empty_mapping_paths(yaml_text):
        last_dot = empty_full_path.rfind(".")
        ep_parent_path = empty_full_path[:last_dot] if last_dot > 0 else ""
        ep_token = empty_full_path[last_dot + 1:] if last_dot > 0 else empty_full_path
        empty_node_idx = -1
        for i, n in enumerate(nodes):
            if (n.node_type in (NodeType.KEY, NodeType.LIST_KEY)
                    and n.parent_path == ep_parent_path
                    and n.token == ep_token):
                empty_node_idx = i
                break
        if empty_node_idx < 0:
            continue
        probe_specs.append({
            "parent_path": empty_full_path,
            "insert_pos": empty_node_idx + 1,
            "fake_depth": min(child_depth, 15),
            "fake_sibling": 0,
            "fake_node_type": NodeType.KEY,
            "ref_depth": child_depth,
            "is_empty_parent": True,
        })

    # Dataset config: mask_prob=0.0 so dataset doesn't randomly mask.
    # recon_enabled=False — we don't need subtree tensors at inference.
    infer_config = YamlBertConfig(v8_mode=True, mask_prob=0.0, recon_enabled=False)

    for spec in probe_specs:
        predicted, candidates_log = _run_probe_v8(
            model=model,
            vocab=vocab,
            nodes=nodes,
            spec=spec,
            mask_id=mask_id,
            infer_config=infer_config,
            id_to_atomic=id_to_atomic,
            id_to_special=id_to_special,
            all_root_keys=all_root_keys,
            top_k=top_k,
            verbose=verbose,
        )
        predicted_keys_by_parent[spec["parent_path"]] = predicted

        if verbose:
            existing_set: set[str] = keys_by_parent.get(spec["parent_path"], set())
            for c in candidates_log:
                if c["status"] != "PASS":
                    continue
                if c["key"] in existing_set:
                    c["status"], c["reason"] = "EXISTS", "already present in YAML"
                elif c["key"] in _CLUSTER_MANAGED_KEYS:
                    c["status"], c["reason"] = "MGMT", "cluster-managed field"
                elif c["prob"] < threshold:
                    c["status"], c["reason"] = "BELOW", f"below threshold {threshold:.0%}"
                else:
                    c["status"], c["reason"] = "KEEP", None
            tag = "  EMPTY-PARENT" if spec["is_empty_parent"] else ""
            print(
                f"\n[{spec['parent_path'] or '(root)'}]  depth={spec['ref_depth']}{tag}",
                file=sys.stderr,
            )
            existing_set_display = (
                "[] (empty mapping in YAML)" if spec["is_empty_parent"] else f"{sorted(existing_set)}"
            )
            print(f"  existing: {existing_set_display}", file=sys.stderr)
            print(f"  top-{len(candidates_log)} candidates:", file=sys.stderr)
            for c in candidates_log:
                marker = {
                    "KEEP": "✓ KEEP ",
                    "EXISTS": "· EXIST",
                    "BELOW": "↓ BELOW",
                    "MGMT": "· MGMT ",
                    "DROP": "✗ DROP ",
                    "PASS": "· PASS ",
                }[c["status"]]
                reason = f"  ({c['reason']})" if c.get("reason") else ""
                print(f"    {marker} {c['prob']:6.2%}  {c['key']}{reason}", file=sys.stderr)

    # Collect suggestions
    suggestions: list[dict[str, Any]] = []
    for parent_path, predicted in predicted_keys_by_parent.items():
        existing: set[str] = keys_by_parent.get(parent_path, set())
        for key_name, confidence in predicted.items():
            if (key_name not in existing
                    and key_name not in _CLUSTER_MANAGED_KEYS
                    and confidence >= threshold):
                suggestions.append({
                    "parent_path": parent_path,
                    "missing_key": key_name,
                    "confidence": confidence,
                })

    suggestions.sort(key=lambda s: -s["confidence"])
    return suggestions, skipped_by_parent


def _run_probe_v8(
    *,
    model: V8Model,
    vocab: Vocabulary,
    nodes: list[YamlNode],
    spec: dict[str, Any],
    mask_id: int,
    infer_config: YamlBertConfig,
    id_to_atomic: dict[int, str],
    id_to_special: dict[int, str],
    all_root_keys: set[str],
    top_k: int,
    verbose: bool,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    """Run one probe: splice a fake [MASK] node, forward through V8Model, decode top-k.

    Returns:
        predicted: {key_name: prob} that passed all filters
        candidates_log: per-candidate trace for verbose output
    """
    parent_path = spec["parent_path"]
    insert_pos = spec["insert_pos"]
    fake_depth = spec["fake_depth"]
    fake_sibling = spec["fake_sibling"]
    fake_node_type = spec["fake_node_type"]
    ref_depth = spec["ref_depth"]

    # Build a fake node for the MASK position
    fake_node = YamlNode(
        token="[MASK]",
        node_type=fake_node_type,
        depth=fake_depth,
        sibling_index=fake_sibling,
        parent_path=parent_path,
    )

    # Splice fake node into the node list
    fake_nodes: list[YamlNode] = nodes[:insert_pos] + [fake_node] + nodes[insert_pos:]

    # Build dataset item: V8Dataset encodes all nodes and computes children_info
    ds = V8Dataset([fake_nodes], vocab, infer_config)
    item = ds[0]

    # Apply [MASK] at insert_pos (dataset has mask_prob=0.0, so token is intact)
    item["token_ids"] = item["token_ids"].clone()
    item["token_ids"][insert_pos] = mask_id

    batch = v8_collate_fn([item])

    with torch.no_grad():
        out = model(
            token_ids=batch["token_ids"],
            node_types=batch["node_types"],
            depths=batch["depths"],
            sibling_indices=batch["sibling_indices"],
            batch_info=batch["batch_info"],
            padding_mask=batch["padding_mask"],
            parent_of_tensor=batch["parent_of_tensor"],
            top_level_key_mask=batch["top_level_key_mask"],
            edges_by_depth=batch["edges_by_depth"],
            parents_by_depth=batch["parents_by_depth"],
        )

    # V8Model returns (logits, doc_vec) or (logits, doc_vec, recon_logits)
    logits = out[0]  # (1, N, V_atomic)
    probs = F.softmax(logits[0, insert_pos], dim=-1)
    topk = probs.topk(top_k + 5)

    predicted: dict[str, float] = {}
    candidates_log: list[dict[str, Any]] = []

    for j in range(topk.indices.shape[0]):
        target_id = topk.indices[j].item()
        prob = topk.values[j].item()

        # Decode: special tokens take priority
        if target_id in id_to_special:
            key_name = id_to_special[target_id]
        else:
            key_name = id_to_atomic.get(target_id, "[UNK]")

        # Filter: drop special tokens
        if key_name in ("[PAD]", "[UNK]", "[MASK]"):
            if verbose:
                candidates_log.append({"prob": prob, "key": key_name, "status": "DROP", "reason": "special token"})
            continue

        # Filter: drop self-reference (mask is probing inside this parent)
        parent_key_name = _parent_key_name(parent_path)
        if key_name == parent_key_name:
            if verbose:
                candidates_log.append({"prob": prob, "key": key_name, "status": "DROP", "reason": "self-reference"})
            continue

        # Filter: drop root-level keys predicted at non-root depth
        if parent_path and key_name in all_root_keys and ref_depth > 0:
            if verbose:
                candidates_log.append({"prob": prob, "key": key_name, "status": "DROP", "reason": "root-key at deep position"})
            continue

        predicted[key_name] = prob
        if verbose:
            candidates_log.append({"prob": prob, "key": key_name, "status": "PASS", "reason": None})

    return predicted, candidates_log


def _parent_key_name(parent_path: str) -> str:
    """Extract last non-numeric segment from a parent_path.

    E.g., 'spec.containers.0' -> 'containers'
    """
    if not parent_path:
        return ""
    for part in reversed(parent_path.split(".")):
        if not part.isdigit():
            return part
    return ""
