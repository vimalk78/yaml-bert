from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

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

    Returns:
        List of suggestions sorted by confidence (highest first).
        Each suggestion: {"parent_path": str, "missing_key": str, "confidence": float}
    """
    linearizer: YamlLinearizer = YamlLinearizer()
    annotator: DomainAnnotator = DomainAnnotator()
    nodes: list[YamlNode] = linearizer.linearize(yaml_text)
    if not nodes:
        return []
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

    # Collect all existing keys across the entire document for cross-reference filtering
    all_root_keys: set[str] = {
        n.token for n in nodes
        if n.node_type in (NodeType.KEY, NodeType.LIST_KEY) and n.depth == 0
    }

    t = lambda x: torch.tensor([x])

    for parent_path, positions in key_positions_by_parent.items():
        predicted: dict[str, float] = {}
        parent_key_name: str = parent_path.split(".")[-1] if parent_path else ""

        # Use the last key node at this level as a template for the fake node
        last_pos: int = positions[-1]
        last_node: YamlNode = nodes[last_pos]
        next_sibling: int = min(last_node.sibling_index + 1, 31)

        # Append a fake [MASK] node at the end of the sequence
        fake_token_ids: list[int] = token_ids + [mask_id]
        fake_node_types: list[int] = node_types + [_NODE_TYPE_INDEX[last_node.node_type]]
        fake_depths: list[int] = depths + [last_node.depth]
        fake_siblings: list[int] = siblings + [next_sibling]
        fake_pos: int = len(token_ids)

        # Determine which head applies for this position using compute_target logic.
        # Create a fake YamlNode to probe which head to use.
        fake_node = YamlNode(
            token="__probe__",
            node_type=last_node.node_type,
            depth=last_node.depth,
            sibling_index=next_sibling,
            parent_path=last_node.parent_path,
        )
        _, head_type = compute_target(fake_node, kind)

        with torch.no_grad():
            simple_logits, kind_logits = model(
                t(fake_token_ids), t(fake_node_types), t(fake_depths),
                t(fake_siblings),
            )

        if head_type == "simple":
            probs: torch.Tensor = F.softmax(simple_logits[0, fake_pos], dim=-1)
            id_to_target = id_to_simple
        else:
            probs = F.softmax(kind_logits[0, fake_pos], dim=-1)
            id_to_target = id_to_kind

        topk = probs.topk(top_k + 5)

        for j in range(topk.indices.shape[0]):
            target_id: int = topk.indices[j].item()
            prob: float = topk.values[j].item()

            # Decode target string
            if target_id in id_to_special:
                target_name: str = id_to_special[target_id]
            else:
                target_name = id_to_target.get(target_id, "[UNK]")

            # Extract the raw key name from composite target strings
            # Simple targets: "parent::key" or just "key"
            # Kind targets: "Kind::parent::key"
            key_name: str = target_name.split("::")[-1] if "::" in target_name else target_name

            # Skip special tokens
            if key_name in ("[PAD]", "[UNK]", "[MASK]"):
                continue
            # Skip self-referencing (key same as parent key name)
            if key_name == parent_key_name:
                continue
            # Skip root-level keys suggested as children (e.g., roleRef under subjects)
            if parent_path and key_name in all_root_keys and last_node.depth > 0:
                continue

            predicted[key_name] = prob

        predicted_keys_by_parent[parent_path] = predicted

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
    return suggestions


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
