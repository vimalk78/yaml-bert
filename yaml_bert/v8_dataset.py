"""v8 dataset extensions: children-info precompute for tree aggregator."""
from __future__ import annotations

from yaml_bert.types import NodeType, YamlNode


def compute_children_info(nodes: list[YamlNode]) -> dict:
    """Compute per-position parent/child relationships.

    For each KEY/LIST_KEY position, its children are KEY/LIST_KEY positions
    whose parent_path equals this key's full_path. (VALUE nodes are leaves —
    their hidden states are used in the aggregator without being treated as
    "children" of any KEY for subtree purposes.)

    Returns a dict with:
        children_of: list[list[int]] — children's positions per node
        parent_of:   list[int]       — parent position (or -1)
        key_positions: list[int]     — positions that are KEY/LIST_KEY
        depth_of:    list[int]       — depth per position
        full_path_of: list[str]      — full path per position
    """
    n = len(nodes)
    full_path_of: list[str] = []
    for node in nodes:
        if node.parent_path:
            full_path_of.append(f"{node.parent_path}.{node.token}")
        else:
            full_path_of.append(node.token)

    # Index KEY positions by their full_path
    key_positions: list[int] = [
        i for i, node in enumerate(nodes)
        if node.node_type in (NodeType.KEY, NodeType.LIST_KEY)
    ]
    path_to_key_pos: dict[str, int] = {
        full_path_of[p]: p for p in key_positions
    }

    children_of: list[list[int]] = [[] for _ in range(n)]
    parent_of: list[int] = [-1] * n
    depth_of: list[int] = [node.depth for node in nodes]

    for p in key_positions:
        parent_path = nodes[p].parent_path
        if parent_path and parent_path in path_to_key_pos:
            parent_pos = path_to_key_pos[parent_path]
            parent_of[p] = parent_pos
            children_of[parent_pos].append(p)

    return {
        "children_of": children_of,
        "parent_of": parent_of,
        "key_positions": key_positions,
        "depth_of": depth_of,
        "full_path_of": full_path_of,
    }
