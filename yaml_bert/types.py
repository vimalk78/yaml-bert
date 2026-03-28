from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class NodeType(Enum):
    KEY = "KEY"
    VALUE = "VALUE"
    LIST_KEY = "LIST_KEY"
    LIST_VALUE = "LIST_VALUE"


@dataclass
class YamlNode:
    token: str
    node_type: NodeType
    depth: int
    sibling_index: int
    parent_path: str
    annotations: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"YamlNode({self.token!r}, {self.node_type.value}, "
            f"depth={self.depth}, sibling={self.sibling_index}, "
            f"path={self.parent_path!r})"
        )
