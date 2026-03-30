from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from yaml_bert.annotator import DomainAnnotator
from yaml_bert.dataset import _extract_kind
from yaml_bert.linearizer import YamlLinearizer
from yaml_bert.model import YamlBertModel
from yaml_bert.types import NodeType, YamlNode
from yaml_bert.vocab import Vocabulary


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

    token_ids, node_types, depths, siblings, parent_keys = _encode_nodes(nodes, vocab)

    kind: str = _extract_kind(nodes)
    kind_id: int = vocab.encode_kind(kind)
    kind_ids: list[int] = [kind_id] * len(nodes)

    mask_id: int = vocab.special_tokens["[MASK]"]

    # Group key nodes by parent_path
    keys_by_parent: dict[str, set[str]] = {}
    key_positions_by_parent: dict[str, list[int]] = {}

    for i, node in enumerate(nodes):
        if node.node_type in (NodeType.KEY, NodeType.LIST_KEY):
            keys_by_parent.setdefault(node.parent_path, set()).add(node.token)
            key_positions_by_parent.setdefault(node.parent_path, []).append(i)

    # For each parent level, mask each key and collect predictions
    model.eval()
    predicted_keys_by_parent: dict[str, dict[str, float]] = {}

    t = lambda x: torch.tensor([x])

    for parent_path, positions in key_positions_by_parent.items():
        predicted: dict[str, float] = {}

        for pos in positions:
            masked_ids: list[int] = token_ids.copy()
            masked_ids[pos] = mask_id

            with torch.no_grad():
                key_logits, _, _ = model(
                    t(masked_ids), t(node_types), t(depths), t(siblings), t(parent_keys),
                    kind_ids=t(kind_ids),
                )

            probs: torch.Tensor = F.softmax(key_logits[0, pos], dim=-1)
            topk = probs.topk(top_k)

            for j in range(top_k):
                key_name: str = vocab.decode_key(topk.indices[j].item())
                prob: float = topk.values[j].item()
                if key_name in ("[PAD]", "[UNK]", "[MASK]"):
                    continue
                if key_name not in predicted or prob > predicted[key_name]:
                    predicted[key_name] = prob

        predicted_keys_by_parent[parent_path] = predicted

    # Find missing keys
    suggestions: list[dict[str, Any]] = []

    for parent_path, predicted in predicted_keys_by_parent.items():
        existing: set[str] = keys_by_parent.get(parent_path, set())
        for key_name, confidence in predicted.items():
            if key_name not in existing and confidence >= threshold:
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
) -> tuple[list[int], list[int], list[int], list[int], list[int]]:
    """Encode nodes to integer lists for model input."""
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
        node_types.append(_NODE_TYPE_INDEX[node.node_type])
        depths.append(min(node.depth, 15))
        siblings.append(min(node.sibling_index, 31))
        parent_keys.append(vocab.encode_key(Vocabulary.extract_parent_key(node.parent_path)))

    return token_ids, node_types, depths, siblings, parent_keys
