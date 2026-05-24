from __future__ import annotations

import sys
from typing import Any

import torch
import torch.nn.functional as F
import yaml as _yaml

from yaml_bert.annotator import DomainAnnotator
from yaml_bert.dataset import _extract_kind
from yaml_bert.linearizer import YamlLinearizer
from yaml_bert.model import YamlBertModel
from yaml_bert.types import NodeType, YamlNode
from yaml_bert.vocab import Vocabulary, compute_target


# Keys managed by the cluster, not written by users
_CLUSTER_MANAGED_KEYS: set[str] = {
    "status", "creationTimestamp", "generation", "resourceVersion",
    "selfLink", "uid", "managedFields",
}

_NODE_TYPE_INDEX: dict[NodeType, int] = {
    NodeType.KEY: 0,
    NodeType.VALUE: 1,
    NodeType.LIST_KEY: 2,
    NodeType.LIST_VALUE: 3,
}


def suggest_missing_fields(
    model: YamlBertModel,
    vocab: Vocabulary,
    yaml_text: str,
    threshold: float = 0.3,
    top_k: int = 10,
    kind_conditioning: bool = True,
    verbose: bool = False,
) -> list[dict[str, Any]]:
    """Suggest missing fields in a YAML document based on model conventions.

    Uses the dual-head (simple + kind-specific) prediction approach.
    For each parent level, a fake [MASK] node is appended and both heads
    are consulted based on which head would apply at that tree position.

    Args:
        model: Trained YAML-BERT model
        vocab: Vocabulary
        yaml_text: Raw YAML text
        threshold: Minimum confidence to report a missing field
        top_k: Number of predictions per masked position
        kind_conditioning: If True (default), mask kind_head logits to
            keep only targets whose kind prefix matches the document's
            kind. Set False to bypass for A/B comparison.

    Returns:
        List of suggestions sorted by confidence (highest first).
        Each suggestion: {"parent_path": str, "missing_key": str, "confidence": float}
    """
    linearizer: YamlLinearizer = YamlLinearizer()
    annotator: DomainAnnotator = DomainAnnotator()
    nodes: list[YamlNode] = linearizer.linearize(yaml_text)
    if not nodes:
        return [], {}
    annotator.annotate(nodes)

    token_ids, node_types, depths, siblings = _encode_nodes(nodes, vocab)

    kind: str = _extract_kind(nodes)
    mask_id: int = vocab.special_tokens["[MASK]"]

    # Build reverse lookups for decoding predictions
    id_to_simple: dict[int, str] = {v: k for k, v in vocab.simple_target_vocab.items()}
    id_to_kind: dict[int, str] = {v: k for k, v in vocab.kind_target_vocab.items()}
    id_to_special: dict[int, str] = {v: k for k, v in vocab.special_tokens.items()}

    # Group key nodes by parent_path
    keys_by_parent: dict[str, set[str]] = {}
    key_positions_by_parent: dict[str, list[int]] = {}

    for i, node in enumerate(nodes):
        if node.node_type in (NodeType.KEY, NodeType.LIST_KEY):
            keys_by_parent.setdefault(node.parent_path, set()).add(node.token)
            key_positions_by_parent.setdefault(node.parent_path, []).append(i)

    # For each parent level, append a fake masked node as the "next sibling"
    # to discover missing keys. Filter out noise: self-references, keys that
    # already exist at root level, and special tokens.
    model.eval()
    predicted_keys_by_parent: dict[str, dict[str, float]] = {}
    skipped_by_parent: dict[str, list[dict[str, Any]]] = {}

    # Collect all existing keys across the entire document for cross-reference filtering
    all_root_keys: set[str] = {
        n.token for n in nodes
        if n.node_type in (NodeType.KEY, NodeType.LIST_KEY) and n.depth == 0
    }

    t = lambda x: torch.tensor([x])

    # Build the list of probe specs: each describes where to insert a fake [MASK]
    # node and what tree-PE coordinates to give it. Two sources:
    #   (a) Parents that have at least one child → probe as "next sibling of last child"
    #       at the same depth, sibling_index = last + 1.
    #   (b) Empty-mapping parents (e.g., `securityContext: {}`) that have no children →
    #       probe as "first child of this empty parent" at depth+1, sibling_index = 0.
    # The rest of the pipeline (forward, head routing, mask, decode, filter, log)
    # is identical for both — see _run_probe below.
    probe_specs: list[dict[str, Any]] = []

    for parent_path, positions in key_positions_by_parent.items():
        last_pos: int = positions[-1]
        last_node: YamlNode = nodes[last_pos]
        # Walk past last_node's subtree to find the insertion point.
        insert_pos = last_pos + 1
        while insert_pos < len(token_ids) and nodes[insert_pos].depth > last_node.depth:
            insert_pos += 1
        probe_specs.append({
            "parent_path": parent_path,
            "parent_key_name": _parent_key_name(parent_path),
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
            "parent_key_name": empty_key_name,
            "insert_pos": empty_node_idx + 1,
            "fake_depth": min(child_depth, 15),
            "fake_sibling": 0,
            "fake_node_type": NodeType.KEY,
            "ref_depth": child_depth,
            "is_empty_parent": True,
        })

    # Single probe loop, shared by both sibling-style and empty-parent-style probes.
    for spec in probe_specs:
        predicted, candidates_log, head_type = _run_probe(
            model=model, vocab=vocab, kind=kind,
            token_ids=token_ids, node_types=node_types, depths=depths, siblings=siblings,
            spec=spec,
            mask_id=mask_id, kind_conditioning=kind_conditioning, top_k=top_k,
            id_to_simple=id_to_simple, id_to_kind=id_to_kind, id_to_special=id_to_special,
            all_root_keys=all_root_keys,
            skipped_by_parent=skipped_by_parent,
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
            print(f"\n[{spec['parent_path'] or '(root)'}]  depth={spec['ref_depth']}  head={head_type}{tag}", file=sys.stderr)
            existing_display = "[] (empty mapping in YAML)" if spec["is_empty_parent"] else f"{sorted(existing_set)}"
            print(f"  existing: {existing_display}", file=sys.stderr)
            print(f"  top-{len(candidates_log)} candidates:", file=sys.stderr)
            for c in candidates_log:
                marker = {"KEEP": "✓ KEEP ", "EXISTS": "· EXIST", "BELOW": "↓ BELOW", "MGMT": "· MGMT ", "DROP": "✗ DROP "}[c["status"]]
                reason = f"  ({c['reason']})" if c['reason'] else ""
                print(f"    {marker} {c['prob']:6.2%}  {c['target']}{reason}", file=sys.stderr)

    # Find missing keys
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


def _parent_key_name(parent_path: str) -> str:
    """Extract the key name for a parent_path, skipping numeric list indices.
    E.g., "spec.containers.0" → "containers"."""
    if not parent_path:
        return ""
    for part in reversed(parent_path.split(".")):
        if not part.isdigit():
            return part
    return ""


def _run_probe(
    *,
    model: YamlBertModel,
    vocab: Vocabulary,
    kind: str,
    token_ids: list[int],
    node_types: list[int],
    depths: list[int],
    siblings: list[int],
    spec: dict[str, Any],
    mask_id: int,
    kind_conditioning: bool,
    top_k: int,
    id_to_simple: dict[int, str],
    id_to_kind: dict[int, str],
    id_to_special: dict[int, str],
    all_root_keys: set[str],
    skipped_by_parent: dict[str, list[dict[str, Any]]],
    verbose: bool,
) -> tuple[dict[str, float], list[dict[str, Any]], str]:
    """Run a single probe at the position described by `spec`.

    Returns:
        predicted: {key_name: prob} that passed the inner-loop filters.
        candidates_log: per-candidate trace (only meaningful when verbose=True).
        head_type: which head was used ("simple" or "kind_specific").
    """
    parent_path = spec["parent_path"]
    parent_key_name = spec["parent_key_name"]
    insert_pos = spec["insert_pos"]
    fake_depth = spec["fake_depth"]
    fake_sibling = spec["fake_sibling"]
    fake_node_type = spec["fake_node_type"]
    ref_depth = spec["ref_depth"]
    is_empty_parent = spec["is_empty_parent"]

    # Splice fake [MASK] into the encoded sequences.
    fake_token_ids = token_ids[:insert_pos] + [mask_id] + token_ids[insert_pos:]
    fake_node_types = node_types[:insert_pos] + [_NODE_TYPE_INDEX[fake_node_type]] + node_types[insert_pos:]
    fake_depths = depths[:insert_pos] + [fake_depth] + depths[insert_pos:]
    fake_siblings = siblings[:insert_pos] + [fake_sibling] + siblings[insert_pos:]

    # Route to head (simple vs kind_specific) by replaying the training-time decision.
    # The fake [MASK]'s parent_path is `spec["parent_path"]` in both cases:
    # sibling-style → parent_path of the last child (same parent we're probing);
    # empty-parent  → the empty parent's full path (which becomes the would-be child's parent_path).
    fake_node = YamlNode(
        token="__probe__",
        node_type=fake_node_type,
        depth=ref_depth,
        sibling_index=fake_sibling,
        parent_path=parent_path,
    )
    _, head_type = compute_target(fake_node, kind)

    t = lambda x: torch.tensor([x])
    with torch.no_grad():
        simple_logits, kind_logits = model(
            t(fake_token_ids), t(fake_node_types), t(fake_depths), t(fake_siblings),
        )

    if head_type == "simple":
        probs = F.softmax(simple_logits[0, insert_pos], dim=-1)
        id_to_target = id_to_simple
    else:
        kind_logits_pos = kind_logits[0, insert_pos].clone()
        if kind and kind_conditioning:
            kind_prefix = f"{kind}::"
            allowed_ids = [idx for ts, idx in vocab.kind_target_vocab.items() if ts.startswith(kind_prefix)]
            if allowed_ids:
                mask = torch.full_like(kind_logits_pos, float("-inf"))
                for idx in allowed_ids:
                    mask[idx] = 0.0
                for idx in vocab.special_tokens.values():
                    mask[idx] = 0.0
                kind_logits_pos = kind_logits_pos + mask
        probs = F.softmax(kind_logits_pos, dim=-1)
        id_to_target = id_to_kind

    topk = probs.topk(top_k + 5)
    predicted: dict[str, float] = {}
    candidates_log: list[dict[str, Any]] = []

    for j in range(topk.indices.shape[0]):
        target_id = topk.indices[j].item()
        prob = topk.values[j].item()
        if target_id in id_to_special:
            target_name = id_to_special[target_id]
        else:
            target_name = id_to_target.get(target_id, "[UNK]")
        parts = target_name.split("::")
        key_name = parts[-1]

        log_status, log_reason = "PASS", None
        if len(parts) >= 2 and parts[-2] != parent_key_name:
            skipped_by_parent.setdefault(parent_path, []).append({
                "target": target_name, "key": key_name,
                "predicted_parent": parts[-2], "actual_parent": parent_key_name,
                "confidence": prob,
            })
            log_status, log_reason = "DROP", f"wrong-parent (predicted '{parts[-2]}', actual '{parent_key_name}')"
        elif key_name in ("[PAD]", "[UNK]", "[MASK]"):
            log_status, log_reason = "DROP", "special token"
        elif key_name == parent_key_name:
            log_status, log_reason = "DROP", "self-reference"
        elif parent_path and key_name in all_root_keys and ref_depth > 0:
            log_status, log_reason = "DROP", "root-key at deep position"

        if log_status == "DROP":
            if verbose:
                candidates_log.append({"prob": prob, "target": target_name, "key": key_name, "status": "DROP", "reason": log_reason})
            continue

        predicted[key_name] = prob
        if verbose:
            candidates_log.append({"prob": prob, "target": target_name, "key": key_name, "status": "PASS", "reason": None})

    return predicted, candidates_log, head_type


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


def _encode_nodes(
    nodes: list[YamlNode],
    vocab: Vocabulary,
) -> tuple[list[int], list[int], list[int], list[int]]:
    """Encode nodes to integer lists for model input."""
    token_ids: list[int] = []
    node_types: list[int] = []
    depths: list[int] = []
    siblings: list[int] = []

    for node in nodes:
        if node.node_type in (NodeType.KEY, NodeType.LIST_KEY):
            token_ids.append(vocab.encode_key(node.token))
        else:
            token_ids.append(vocab.encode_value(node.token))
        node_types.append(_NODE_TYPE_INDEX[node.node_type])
        depths.append(min(node.depth, 15))
        siblings.append(min(node.sibling_index, 31))

    return token_ids, node_types, depths, siblings
