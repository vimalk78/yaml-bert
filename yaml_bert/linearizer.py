from __future__ import annotations

import yaml

from yaml_bert.types import NodeType, YamlNode


class YamlLinearizer:
    def linearize(self, yaml_string: str) -> list[YamlNode]:
        data = yaml.safe_load(yaml_string)
        if data is None:
            return []
        nodes: list[YamlNode] = []
        self._walk(data, depth=0, parent_path="", nodes=nodes, in_list=False)
        return nodes

    def _walk(
        self,
        data,
        depth: int,
        parent_path: str,
        nodes: list[YamlNode],
        in_list: bool,
    ) -> None:
        if isinstance(data, dict):
            for sibling_index, (key, value) in enumerate(data.items()):
                key_str = str(key)
                key_type = NodeType.LIST_KEY if in_list else NodeType.KEY
                nodes.append(
                    YamlNode(
                        token=key_str,
                        node_type=key_type,
                        depth=depth,
                        sibling_index=sibling_index,
                        parent_path=parent_path,
                    )
                )
                if isinstance(value, dict):
                    child_path = f"{parent_path}.{key_str}" if parent_path else key_str
                    self._walk(value, depth + 1, child_path, nodes, in_list=False)
                elif isinstance(value, list):
                    child_path = f"{parent_path}.{key_str}" if parent_path else key_str
                    self._walk_list(value, depth + 1, child_path, nodes)
                else:
                    value_path = f"{parent_path}.{key_str}" if parent_path else key_str
                    value_type = NodeType.LIST_VALUE if in_list else NodeType.VALUE
                    nodes.append(
                        YamlNode(
                            token=str(value),
                            node_type=value_type,
                            depth=depth,
                            sibling_index=sibling_index,
                            parent_path=value_path,
                        )
                    )

    def linearize_file(self, path: str) -> list[YamlNode]:
        with open(path) as f:
            content = f.read()
        nodes: list[YamlNode] = []
        for doc in yaml.safe_load_all(content):
            if doc is None:
                continue
            self._walk(doc, depth=0, parent_path="", nodes=nodes, in_list=False)
        return nodes

    def linearize_multi_doc(self, yaml_string: str) -> list[list[YamlNode]]:
        result = []
        for doc in yaml.safe_load_all(yaml_string):
            if doc is None:
                continue
            nodes: list[YamlNode] = []
            self._walk(doc, depth=0, parent_path="", nodes=nodes, in_list=False)
            result.append(nodes)
        return result

    def _walk_list(
        self,
        data: list,
        depth: int,
        parent_path: str,
        nodes: list[YamlNode],
    ) -> None:
        for item_index, item in enumerate(data):
            item_path = f"{parent_path}.{item_index}"
            if isinstance(item, dict):
                self._walk(item, depth, item_path, nodes, in_list=True)
            elif isinstance(item, list):
                self._walk_list(item, depth, item_path, nodes)
            else:
                nodes.append(
                    YamlNode(
                        token=str(item),
                        node_type=NodeType.LIST_VALUE,
                        depth=depth,
                        sibling_index=item_index,
                        parent_path=item_path,
                    )
                )
