from __future__ import annotations

from yaml_bert.types import NodeType, YamlNode


class DomainAnnotator:
    ORDERED_LISTS = {"initContainers"}

    def annotate(self, nodes: list[YamlNode]) -> list[YamlNode]:
        list_parent_ids = self._find_list_parent_ids(nodes)
        for node in nodes:
            if id(node) in list_parent_ids:
                node.annotations["list_ordered"] = (
                    node.token in self.ORDERED_LISTS
                )
        return nodes

    def _find_list_parent_ids(self, nodes: list[YamlNode]) -> set[int]:
        list_parent_ids: set[int] = set()
        parent_paths_with_list_items: set[str] = set()

        for node in nodes:
            if node.node_type in (NodeType.LIST_KEY, NodeType.LIST_VALUE):
                parts = node.parent_path.split(".")
                for i, part in enumerate(parts):
                    if part.isdigit():
                        list_parent_path = ".".join(parts[:i])
                        parent_paths_with_list_items.add(list_parent_path)
                        break

        for node in nodes:
            if node.node_type in (NodeType.KEY, NodeType.LIST_KEY):
                node_full_path = (
                    f"{node.parent_path}.{node.token}" if node.parent_path else node.token
                )
                if node_full_path in parent_paths_with_list_items:
                    list_parent_ids.add(id(node))

        return list_parent_ids
