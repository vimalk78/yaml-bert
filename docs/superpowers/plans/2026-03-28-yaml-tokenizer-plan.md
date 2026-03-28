# YAML-BERT Tokenizer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a structural tokenizer that converts Kubernetes YAML into linearized, annotated node sequences with separate key/value vocabularies.

**Architecture:** Three independent modules — YamlLinearizer (YAML to flat node sequence via DFS), VocabBuilder (corpus-driven vocabulary with separate key/value namespaces), DomainAnnotator (pluggable annotation layer for domain knowledge like list ordering). Shared data structures in types.py.

**Tech Stack:** Python 3.10+, PyYAML, pytest

---

### Task 1: Project Scaffolding

**Files:**
- Create: `yaml_bert/__init__.py`
- Create: `yaml_bert/types.py`
- Create: `tests/__init__.py`
- Create: `requirements.txt`

- [ ] **Step 1: Create requirements.txt**

```
pyyaml>=6.0
pytest>=7.0
```

- [ ] **Step 2: Create yaml_bert package init**

```python
"""YAML-BERT: Attention on Kubernetes Structured Data."""
```

- [ ] **Step 3: Create tests package init**

Empty `tests/__init__.py`.

- [ ] **Step 4: Create types.py with NodeType and YamlNode**

File: `yaml_bert/types.py`

```python
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
```

- [ ] **Step 5: Install dependencies and verify imports**

Run: `pip install -r requirements.txt`
Run: `python -c "from yaml_bert.types import NodeType, YamlNode; print('OK')"`
Expected: `OK`

- [ ] **Step 6: Commit**

```bash
git add yaml_bert/__init__.py yaml_bert/types.py tests/__init__.py requirements.txt
git commit -m "feat: project scaffolding with NodeType and YamlNode"
```

---

### Task 2: YamlLinearizer — Key-Value Pairs

**Files:**
- Create: `yaml_bert/linearizer.py`
- Create: `tests/test_linearizer.py`

- [ ] **Step 1: Write failing test for simple key-value pairs**

File: `tests/test_linearizer.py`

```python
from yaml_bert.linearizer import YamlLinearizer
from yaml_bert.types import NodeType


def test_simple_key_value_pairs():
    yaml_str = "app: redis\nrole: replica\ntier: backend\n"
    linearizer = YamlLinearizer()
    nodes = linearizer.linearize(yaml_str)

    assert len(nodes) == 6

    assert nodes[0].token == "app"
    assert nodes[0].node_type == NodeType.KEY
    assert nodes[0].depth == 0
    assert nodes[0].sibling_index == 0
    assert nodes[0].parent_path == ""

    assert nodes[1].token == "redis"
    assert nodes[1].node_type == NodeType.VALUE
    assert nodes[1].depth == 0
    assert nodes[1].sibling_index == 0
    assert nodes[1].parent_path == "app"

    assert nodes[2].token == "role"
    assert nodes[2].node_type == NodeType.KEY
    assert nodes[2].sibling_index == 1

    assert nodes[3].token == "replica"
    assert nodes[3].node_type == NodeType.VALUE
    assert nodes[3].parent_path == "role"

    assert nodes[4].token == "tier"
    assert nodes[4].sibling_index == 2

    assert nodes[5].token == "backend"
    assert nodes[5].parent_path == "tier"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_linearizer.py::test_simple_key_value_pairs -v`
Expected: FAIL with `ModuleNotFoundError` or `ImportError`

- [ ] **Step 3: Implement YamlLinearizer with key-value support**

File: `yaml_bert/linearizer.py`

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_linearizer.py::test_simple_key_value_pairs -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add yaml_bert/linearizer.py tests/test_linearizer.py
git commit -m "feat: YamlLinearizer with key-value pair support"
```

---

### Task 3: YamlLinearizer — Nested Mappings

**Files:**
- Modify: `tests/test_linearizer.py`

- [ ] **Step 1: Write failing test for nested mappings**

Append to `tests/test_linearizer.py`:

```python
def test_nested_mapping():
    yaml_str = "metadata:\n  name: nginx\n  namespace: default\n"
    linearizer = YamlLinearizer()
    nodes = linearizer.linearize(yaml_str)

    assert len(nodes) == 5

    assert nodes[0].token == "metadata"
    assert nodes[0].node_type == NodeType.KEY
    assert nodes[0].depth == 0
    assert nodes[0].sibling_index == 0
    assert nodes[0].parent_path == ""

    assert nodes[1].token == "name"
    assert nodes[1].node_type == NodeType.KEY
    assert nodes[1].depth == 1
    assert nodes[1].sibling_index == 0
    assert nodes[1].parent_path == "metadata"

    assert nodes[2].token == "nginx"
    assert nodes[2].node_type == NodeType.VALUE
    assert nodes[2].depth == 1
    assert nodes[2].sibling_index == 0
    assert nodes[2].parent_path == "metadata.name"

    assert nodes[3].token == "namespace"
    assert nodes[3].node_type == NodeType.KEY
    assert nodes[3].depth == 1
    assert nodes[3].sibling_index == 1
    assert nodes[3].parent_path == "metadata"

    assert nodes[4].token == "default"
    assert nodes[4].node_type == NodeType.VALUE
    assert nodes[4].depth == 1
    assert nodes[4].sibling_index == 1
    assert nodes[4].parent_path == "metadata.namespace"
```

- [ ] **Step 2: Run test to verify it passes (should pass with existing implementation)**

Run: `pytest tests/test_linearizer.py::test_nested_mapping -v`
Expected: PASS (nested dicts already handled by `_walk`)

- [ ] **Step 3: Commit**

```bash
git add tests/test_linearizer.py
git commit -m "test: add nested mapping linearization test"
```

---

### Task 4: YamlLinearizer — Lists of Maps

**Files:**
- Modify: `tests/test_linearizer.py`

- [ ] **Step 1: Write failing test for lists of maps**

Append to `tests/test_linearizer.py`:

```python
def test_list_of_maps():
    yaml_str = (
        "containers:\n"
        "- name: webserver1\n"
        "  image: nginx:1.6\n"
        "- name: database-server\n"
        "  image: mysql-3.2\n"
    )
    linearizer = YamlLinearizer()
    nodes = linearizer.linearize(yaml_str)

    assert len(nodes) == 9

    # containers key
    assert nodes[0].token == "containers"
    assert nodes[0].node_type == NodeType.KEY
    assert nodes[0].depth == 0
    assert nodes[0].parent_path == ""

    # first item: name
    assert nodes[1].token == "name"
    assert nodes[1].node_type == NodeType.LIST_KEY
    assert nodes[1].depth == 1
    assert nodes[1].sibling_index == 0
    assert nodes[1].parent_path == "containers.0"

    # first item: webserver1
    assert nodes[2].token == "webserver1"
    assert nodes[2].node_type == NodeType.LIST_VALUE
    assert nodes[2].depth == 1
    assert nodes[2].sibling_index == 0
    assert nodes[2].parent_path == "containers.0.name"

    # first item: image
    assert nodes[3].token == "image"
    assert nodes[3].node_type == NodeType.LIST_KEY
    assert nodes[3].depth == 1
    assert nodes[3].sibling_index == 1
    assert nodes[3].parent_path == "containers.0"

    # first item: nginx:1.6
    assert nodes[4].token == "nginx:1.6"
    assert nodes[4].node_type == NodeType.LIST_VALUE
    assert nodes[4].depth == 1
    assert nodes[4].sibling_index == 1
    assert nodes[4].parent_path == "containers.0.image"

    # second item: name
    assert nodes[5].token == "name"
    assert nodes[5].node_type == NodeType.LIST_KEY
    assert nodes[5].depth == 1
    assert nodes[5].sibling_index == 0
    assert nodes[5].parent_path == "containers.1"

    # second item: database-server
    assert nodes[6].token == "database-server"
    assert nodes[6].node_type == NodeType.LIST_VALUE
    assert nodes[6].depth == 1
    assert nodes[6].sibling_index == 0
    assert nodes[6].parent_path == "containers.1.name"

    # second item: image
    assert nodes[7].token == "image"
    assert nodes[7].node_type == NodeType.LIST_KEY
    assert nodes[7].depth == 1
    assert nodes[7].sibling_index == 1
    assert nodes[7].parent_path == "containers.1"

    # second item: mysql-3.2
    assert nodes[8].token == "mysql-3.2"
    assert nodes[8].node_type == NodeType.LIST_VALUE
    assert nodes[8].depth == 1
    assert nodes[8].sibling_index == 1
    assert nodes[8].parent_path == "containers.1.image"
```

- [ ] **Step 2: Run test to verify it passes**

Run: `pytest tests/test_linearizer.py::test_list_of_maps -v`
Expected: PASS (list handling already implemented in `_walk_list`)

- [ ] **Step 3: Commit**

```bash
git add tests/test_linearizer.py
git commit -m "test: add list-of-maps linearization test"
```

---

### Task 5: YamlLinearizer — Scalar Lists and Nested Lists

**Files:**
- Modify: `tests/test_linearizer.py`

- [ ] **Step 1: Write failing test for scalar lists**

Append to `tests/test_linearizer.py`:

```python
def test_scalar_list():
    yaml_str = "args:\n- --config\n- /etc/app.yaml\n"
    linearizer = YamlLinearizer()
    nodes = linearizer.linearize(yaml_str)

    assert len(nodes) == 3

    assert nodes[0].token == "args"
    assert nodes[0].node_type == NodeType.KEY
    assert nodes[0].depth == 0

    assert nodes[1].token == "--config"
    assert nodes[1].node_type == NodeType.LIST_VALUE
    assert nodes[1].depth == 1
    assert nodes[1].sibling_index == 0
    assert nodes[1].parent_path == "args.0"

    assert nodes[2].token == "/etc/app.yaml"
    assert nodes[2].node_type == NodeType.LIST_VALUE
    assert nodes[2].depth == 1
    assert nodes[2].sibling_index == 1
    assert nodes[2].parent_path == "args.1"
```

- [ ] **Step 2: Write test for nested list inside list item (ports inside containers)**

Append to `tests/test_linearizer.py`:

```python
def test_nested_list_in_list_item():
    yaml_str = (
        "containers:\n"
        "- name: webserver1\n"
        "  ports:\n"
        "  - containerPort: 80\n"
        "  - containerPort: 443\n"
    )
    linearizer = YamlLinearizer()
    nodes = linearizer.linearize(yaml_str)

    assert len(nodes) == 7

    # containers
    assert nodes[0].token == "containers"
    assert nodes[0].node_type == NodeType.KEY
    assert nodes[0].depth == 0

    # name: webserver1
    assert nodes[1].token == "name"
    assert nodes[1].node_type == NodeType.LIST_KEY
    assert nodes[1].depth == 1
    assert nodes[1].parent_path == "containers.0"

    assert nodes[2].token == "webserver1"
    assert nodes[2].node_type == NodeType.LIST_VALUE
    assert nodes[2].parent_path == "containers.0.name"

    # ports key
    assert nodes[3].token == "ports"
    assert nodes[3].node_type == NodeType.LIST_KEY
    assert nodes[3].depth == 1
    assert nodes[3].parent_path == "containers.0"

    # containerPort: 80
    assert nodes[4].token == "containerPort"
    assert nodes[4].node_type == NodeType.LIST_KEY
    assert nodes[4].depth == 2
    assert nodes[4].parent_path == "containers.0.ports.0"

    assert nodes[5].token == "80"
    assert nodes[5].node_type == NodeType.LIST_VALUE
    assert nodes[5].depth == 2
    assert nodes[5].parent_path == "containers.0.ports.0.containerPort"

    # containerPort: 443
    assert nodes[6].token == "containerPort"
    assert nodes[6].node_type == NodeType.LIST_KEY
    assert nodes[6].depth == 2
    assert nodes[6].parent_path == "containers.0.ports.1"
```

- [ ] **Step 3: Run tests**

Run: `pytest tests/test_linearizer.py -v`
Expected: ALL PASS

- [ ] **Step 4: Commit**

```bash
git add tests/test_linearizer.py
git commit -m "test: add scalar list and nested list linearization tests"
```

---

### Task 6: YamlLinearizer — Multi-Document and File Support

**Files:**
- Modify: `yaml_bert/linearizer.py`
- Modify: `tests/test_linearizer.py`

- [ ] **Step 1: Write failing test for multi-document YAML**

Append to `tests/test_linearizer.py`:

```python
def test_multi_document():
    yaml_str = "---\nkind: Deployment\n---\nkind: Service\n"
    linearizer = YamlLinearizer()
    docs = linearizer.linearize_multi_doc(yaml_str)

    assert len(docs) == 2

    assert len(docs[0]) == 2
    assert docs[0][0].token == "kind"
    assert docs[0][1].token == "Deployment"

    assert len(docs[1]) == 2
    assert docs[1][0].token == "kind"
    assert docs[1][1].token == "Service"
```

- [ ] **Step 2: Write failing test for file loading using real YAML templates**

Append to `tests/test_linearizer.py`:

```python
import os

TEMPLATES_DIR = os.path.join(
    os.path.dirname(__file__), "..", "kubernetes-yaml-templates"
)


def test_linearize_file():
    linearizer = YamlLinearizer()
    path = os.path.join(TEMPLATES_DIR, "deployment", "deployment-nginx.yaml")
    nodes = linearizer.linearize_file(path)

    assert len(nodes) > 0

    # Should start with apiVersion
    assert nodes[0].token == "apiVersion"
    assert nodes[0].node_type == NodeType.KEY
    assert nodes[1].token == "apps/v1"
    assert nodes[1].node_type == NodeType.VALUE

    # Should contain kind: Deployment
    kind_node = next(n for n in nodes if n.token == "kind")
    assert kind_node.node_type == NodeType.KEY
    deployment_node = next(n for n in nodes if n.token == "Deployment")
    assert deployment_node.node_type == NodeType.VALUE


def test_linearize_file_service():
    linearizer = YamlLinearizer()
    path = os.path.join(TEMPLATES_DIR, "service", "service-clusterip-nginx.yaml")
    nodes = linearizer.linearize_file(path)

    assert len(nodes) > 0
    kind_node = next(n for n in nodes if n.token == "kind")
    service_node = next(n for n in nodes if n.token == "Service")
    assert kind_node.node_type == NodeType.KEY
    assert service_node.node_type == NodeType.VALUE
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/test_linearizer.py::test_multi_document tests/test_linearizer.py::test_linearize_file -v`
Expected: FAIL with `AttributeError: 'YamlLinearizer' object has no attribute 'linearize_multi_doc'`

- [ ] **Step 4: Implement multi-doc and file methods**

Add to `yaml_bert/linearizer.py` inside the `YamlLinearizer` class:

```python
    def linearize_file(self, path: str) -> list[YamlNode]:
        with open(path) as f:
            return self.linearize(f.read())

    def linearize_multi_doc(self, yaml_string: str) -> list[list[YamlNode]]:
        result = []
        for doc in yaml.safe_load_all(yaml_string):
            if doc is None:
                continue
            nodes: list[YamlNode] = []
            self._walk(doc, depth=0, parent_path="", nodes=nodes, in_list=False)
            result.append(nodes)
        return result
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_linearizer.py -v`
Expected: ALL PASS

- [ ] **Step 6: Commit**

```bash
git add yaml_bert/linearizer.py tests/test_linearizer.py
git commit -m "feat: add multi-doc and file support for linearizer"
```

---

### Task 7: YamlLinearizer — Edge Cases and Real Corpus Smoke Test

**Files:**
- Modify: `tests/test_linearizer.py`
- Modify: `yaml_bert/linearizer.py` (if fixes needed)

- [ ] **Step 1: Write tests for edge cases**

Append to `tests/test_linearizer.py`:

```python
def test_empty_yaml():
    linearizer = YamlLinearizer()
    assert linearizer.linearize("") == []
    assert linearizer.linearize("---") == []


def test_boolean_and_null_values():
    yaml_str = "enabled: true\ncount: 0\nmissing: null\n"
    linearizer = YamlLinearizer()
    nodes = linearizer.linearize(yaml_str)

    assert nodes[1].token == "True"
    assert nodes[3].token == "0"
    assert nodes[5].token == "None"


def test_deeply_nested():
    yaml_str = "a:\n  b:\n    c:\n      d: leaf\n"
    linearizer = YamlLinearizer()
    nodes = linearizer.linearize(yaml_str)

    assert nodes[0].token == "a"
    assert nodes[0].depth == 0
    assert nodes[0].parent_path == ""

    assert nodes[1].token == "b"
    assert nodes[1].depth == 1
    assert nodes[1].parent_path == "a"

    assert nodes[2].token == "c"
    assert nodes[2].depth == 2
    assert nodes[2].parent_path == "a.b"

    assert nodes[3].token == "d"
    assert nodes[3].depth == 3
    assert nodes[3].parent_path == "a.b.c"

    assert nodes[4].token == "leaf"
    assert nodes[4].depth == 3
    assert nodes[4].parent_path == "a.b.c.d"


def test_integer_and_float_values():
    yaml_str = "replicas: 3\ncpu: 0.5\n"
    linearizer = YamlLinearizer()
    nodes = linearizer.linearize(yaml_str)

    assert nodes[1].token == "3"
    assert nodes[1].node_type == NodeType.VALUE

    assert nodes[3].token == "0.5"
    assert nodes[3].node_type == NodeType.VALUE
```

- [ ] **Step 2: Write smoke test that linearizes all 52 YAML files from the real corpus**

Append to `tests/test_linearizer.py`:

```python
import glob


def test_linearize_all_kubernetes_templates():
    """Smoke test: linearize every YAML file in kubernetes-yaml-templates/.
    Ensures no crashes on real-world K8s manifests."""
    linearizer = YamlLinearizer()
    yaml_files = glob.glob(
        os.path.join(TEMPLATES_DIR, "**", "*.yaml"), recursive=True
    )
    assert len(yaml_files) > 40, f"Expected 40+ YAML files, found {len(yaml_files)}"

    total_nodes = 0
    for path in yaml_files:
        nodes = linearizer.linearize_file(path)
        assert len(nodes) > 0, f"Empty linearization for {path}"
        for node in nodes:
            assert node.token is not None
            assert node.node_type is not None
            assert node.depth >= 0
            assert node.sibling_index >= 0
        total_nodes += len(nodes)

    # Sanity: across 52 files we should have hundreds of nodes
    assert total_nodes > 500, f"Expected 500+ total nodes, got {total_nodes}"
```

- [ ] **Step 3: Run tests**

Run: `pytest tests/test_linearizer.py -v`
Expected: ALL PASS (or identify fixes needed)

- [ ] **Step 4: Fix any failures and re-run**

Common issues:
- PyYAML parses `true` → Python `True`, `null` → Python `None`. The `str()` conversion handles this, but verify the exact string representations match the test expectations.
- Some YAML files may have multi-line string values (e.g., ConfigMap data blocks). Ensure `str()` conversion handles these without crashing.

- [ ] **Step 5: Commit**

```bash
git add tests/test_linearizer.py yaml_bert/linearizer.py
git commit -m "test: add edge case and real corpus smoke tests for linearizer"
```

---

### Task 8: VocabBuilder and Vocabulary

**Files:**
- Create: `yaml_bert/vocab.py`
- Create: `tests/test_vocab.py`

- [ ] **Step 1: Write failing test for vocabulary building**

File: `tests/test_vocab.py`

```python
from yaml_bert.linearizer import YamlLinearizer
from yaml_bert.vocab import Vocabulary, VocabBuilder


def test_build_vocab_from_simple_yaml():
    yaml_str = "apiVersion: v1\nkind: Pod\nmetadata:\n  name: nginx\n"
    linearizer = YamlLinearizer()
    nodes = linearizer.linearize(yaml_str)

    builder = VocabBuilder()
    vocab = builder.build(nodes)

    # Keys: apiVersion, kind, metadata, name
    assert "apiVersion" in vocab.key_vocab
    assert "kind" in vocab.key_vocab
    assert "metadata" in vocab.key_vocab
    assert "name" in vocab.key_vocab

    # Values: v1, Pod, nginx
    assert "v1" in vocab.value_vocab
    assert "Pod" in vocab.value_vocab
    assert "nginx" in vocab.value_vocab

    # Keys should NOT be in value vocab and vice versa
    assert "apiVersion" not in vocab.value_vocab
    assert "v1" not in vocab.key_vocab


def test_special_tokens_present():
    builder = VocabBuilder()
    vocab = builder.build([])

    assert "[UNK]" in vocab.special_tokens
    assert "[PAD]" in vocab.special_tokens
    assert "[MASK]" in vocab.special_tokens

    # Special tokens should have the lowest IDs
    assert vocab.special_tokens["[PAD]"] == 0
    assert vocab.special_tokens["[UNK]"] == 1
    assert vocab.special_tokens["[MASK]"] == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_vocab.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement VocabBuilder and Vocabulary**

File: `yaml_bert/vocab.py`

```python
from __future__ import annotations

import json
from yaml_bert.types import NodeType, YamlNode


SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[MASK]"]


class Vocabulary:
    def __init__(
        self,
        key_vocab: dict[str, int],
        value_vocab: dict[str, int],
        special_tokens: dict[str, int],
    ) -> None:
        self.key_vocab = key_vocab
        self.value_vocab = value_vocab
        self.special_tokens = special_tokens
        self._id_to_key = {v: k for k, v in key_vocab.items()}
        self._id_to_value = {v: k for k, v in value_vocab.items()}
        self._id_to_special = {v: k for k, v in special_tokens.items()}

    def encode_key(self, token: str) -> int:
        return self.key_vocab.get(token, self.special_tokens["[UNK]"])

    def encode_value(self, token: str) -> int:
        return self.value_vocab.get(token, self.special_tokens["[UNK]"])

    def decode_key(self, id: int) -> str:
        if id in self._id_to_special:
            return self._id_to_special[id]
        return self._id_to_key.get(id, "[UNK]")

    def decode_value(self, id: int) -> str:
        if id in self._id_to_special:
            return self._id_to_special[id]
        return self._id_to_value.get(id, "[UNK]")

    @property
    def key_vocab_size(self) -> int:
        return len(self.key_vocab) + len(self.special_tokens)

    @property
    def value_vocab_size(self) -> int:
        return len(self.value_vocab) + len(self.special_tokens)

    def save(self, path: str) -> None:
        data = {
            "key_vocab": self.key_vocab,
            "value_vocab": self.value_vocab,
            "special_tokens": self.special_tokens,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str) -> Vocabulary:
        with open(path) as f:
            data = json.load(f)
        return cls(
            key_vocab=data["key_vocab"],
            value_vocab=data["value_vocab"],
            special_tokens=data["special_tokens"],
        )


class VocabBuilder:
    def build(self, nodes: list[YamlNode], min_freq: int = 1) -> Vocabulary:
        special_tokens = {tok: i for i, tok in enumerate(SPECIAL_TOKENS)}
        offset = len(special_tokens)

        key_counts: dict[str, int] = {}
        value_counts: dict[str, int] = {}

        for node in nodes:
            if node.node_type in (NodeType.KEY, NodeType.LIST_KEY):
                key_counts[node.token] = key_counts.get(node.token, 0) + 1
            elif node.node_type in (NodeType.VALUE, NodeType.LIST_VALUE):
                value_counts[node.token] = value_counts.get(node.token, 0) + 1

        key_vocab = {
            token: i + offset
            for i, (token, count) in enumerate(
                sorted(key_counts.items())
            )
            if count >= min_freq
        }

        value_vocab = {
            token: i + offset
            for i, (token, count) in enumerate(
                sorted(value_counts.items())
            )
            if count >= min_freq
        }

        return Vocabulary(key_vocab, value_vocab, special_tokens)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_vocab.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add yaml_bert/vocab.py tests/test_vocab.py
git commit -m "feat: VocabBuilder and Vocabulary with separate key/value namespaces"
```

---

### Task 9: Vocabulary — Encoding, Decoding, and Persistence

**Files:**
- Modify: `tests/test_vocab.py`

- [ ] **Step 1: Write tests for encode/decode and save/load**

Append to `tests/test_vocab.py`:

```python
def test_encode_decode_roundtrip():
    yaml_str = "apiVersion: v1\nkind: Pod\n"
    from yaml_bert.linearizer import YamlLinearizer
    linearizer = YamlLinearizer()
    nodes = linearizer.linearize(yaml_str)

    builder = VocabBuilder()
    vocab = builder.build(nodes)

    # Encode and decode keys
    key_id = vocab.encode_key("apiVersion")
    assert vocab.decode_key(key_id) == "apiVersion"

    # Encode and decode values
    value_id = vocab.encode_value("v1")
    assert vocab.decode_value(value_id) == "v1"

    # Unknown tokens return [UNK] id
    unk_id = vocab.special_tokens["[UNK]"]
    assert vocab.encode_key("nonexistent_key") == unk_id
    assert vocab.encode_value("nonexistent_value") == unk_id


def test_save_and_load(tmp_path):
    yaml_str = "apiVersion: v1\nkind: Pod\n"
    from yaml_bert.linearizer import YamlLinearizer
    linearizer = YamlLinearizer()
    nodes = linearizer.linearize(yaml_str)

    builder = VocabBuilder()
    vocab = builder.build(nodes)

    vocab_path = str(tmp_path / "vocab.json")
    vocab.save(vocab_path)

    loaded = Vocabulary.load(vocab_path)

    assert loaded.key_vocab == vocab.key_vocab
    assert loaded.value_vocab == vocab.value_vocab
    assert loaded.special_tokens == vocab.special_tokens
    assert loaded.encode_key("apiVersion") == vocab.encode_key("apiVersion")
    assert loaded.encode_value("v1") == vocab.encode_value("v1")


def test_min_freq_filtering():
    yaml_str = "a: x\nb: y\na: x\n"
    from yaml_bert.linearizer import YamlLinearizer
    linearizer = YamlLinearizer()
    # Build from two documents to get repeated tokens
    nodes1 = linearizer.linearize("a: x\nb: y\n")
    nodes2 = linearizer.linearize("a: x\nc: z\n")
    all_nodes = nodes1 + nodes2

    builder = VocabBuilder()
    vocab = builder.build(all_nodes, min_freq=2)

    # "a" appears twice, should be in vocab
    assert "a" in vocab.key_vocab
    # "b" and "c" appear once, should be filtered out
    assert "b" not in vocab.key_vocab
    assert "c" not in vocab.key_vocab
    # "x" appears twice, should be in vocab
    assert "x" in vocab.value_vocab
    # "y" and "z" appear once, should be filtered out
    assert "y" not in vocab.value_vocab
```

- [ ] **Step 2: Run tests**

Run: `pytest tests/test_vocab.py -v`
Expected: ALL PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_vocab.py
git commit -m "test: add encode/decode, save/load, min_freq vocab tests"
```

---

### Task 10: DomainAnnotator

**Files:**
- Create: `yaml_bert/annotator.py`
- Create: `tests/test_annotator.py`

- [ ] **Step 1: Write failing test for list ordering annotation**

File: `tests/test_annotator.py`

```python
from yaml_bert.annotator import DomainAnnotator
from yaml_bert.linearizer import YamlLinearizer
from yaml_bert.types import NodeType


def test_annotate_unordered_list():
    yaml_str = (
        "spec:\n"
        "  containers:\n"
        "  - name: nginx\n"
        "  - name: sidecar\n"
    )
    linearizer = YamlLinearizer()
    annotator = DomainAnnotator()

    nodes = linearizer.linearize(yaml_str)
    annotated = annotator.annotate(nodes)

    # Find the "containers" key node
    containers_node = next(n for n in annotated if n.token == "containers")
    assert containers_node.annotations["list_ordered"] is False


def test_annotate_ordered_list():
    yaml_str = (
        "spec:\n"
        "  initContainers:\n"
        "  - name: init-db\n"
        "  - name: init-cache\n"
    )
    linearizer = YamlLinearizer()
    annotator = DomainAnnotator()

    nodes = linearizer.linearize(yaml_str)
    annotated = annotator.annotate(nodes)

    init_node = next(n for n in annotated if n.token == "initContainers")
    assert init_node.annotations["list_ordered"] is True


def test_non_list_keys_have_no_annotation():
    yaml_str = "apiVersion: v1\nkind: Pod\n"
    linearizer = YamlLinearizer()
    annotator = DomainAnnotator()

    nodes = linearizer.linearize(yaml_str)
    annotated = annotator.annotate(nodes)

    for node in annotated:
        assert "list_ordered" not in node.annotations
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_annotator.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement DomainAnnotator**

File: `yaml_bert/annotator.py`

```python
from __future__ import annotations

from yaml_bert.types import NodeType, YamlNode


class DomainAnnotator:
    ORDERED_LISTS = {"initContainers"}

    def annotate(self, nodes: list[YamlNode]) -> list[YamlNode]:
        list_parents = self._find_list_parents(nodes)
        for node in nodes:
            if node in list_parents:
                node.annotations["list_ordered"] = (
                    node.token in self.ORDERED_LISTS
                )
        return nodes

    def _find_list_parents(self, nodes: list[YamlNode]) -> set[YamlNode]:
        list_parents: set[YamlNode] = set()
        parent_paths_with_list_items: set[str] = set()

        # Collect parent paths that contain list items
        for node in nodes:
            if node.node_type in (NodeType.LIST_KEY, NodeType.LIST_VALUE):
                # parent_path looks like "spec.containers.0" or "spec.containers.0.name"
                # The list parent's path is everything before the numeric index
                parts = node.parent_path.split(".")
                # Walk backwards to find the first numeric part
                for i, part in enumerate(parts):
                    if part.isdigit():
                        list_parent_path = ".".join(parts[:i])
                        parent_paths_with_list_items.add(list_parent_path)
                        break

        # Find the KEY nodes that are list parents
        for node in nodes:
            if node.node_type in (NodeType.KEY, NodeType.LIST_KEY):
                node_full_path = (
                    f"{node.parent_path}.{node.token}" if node.parent_path else node.token
                )
                if node_full_path in parent_paths_with_list_items:
                    list_parents.add(node)

        return list_parents
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_annotator.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add yaml_bert/annotator.py tests/test_annotator.py
git commit -m "feat: DomainAnnotator with list ordering annotation"
```

---

### Task 11: Full Pipeline Integration Test with Real Corpus

**Files:**
- Create: `tests/test_integration.py`

- [ ] **Step 1: Write integration test using real YAML templates**

File: `tests/test_integration.py`

```python
import glob
import os

from yaml_bert.linearizer import YamlLinearizer
from yaml_bert.annotator import DomainAnnotator
from yaml_bert.vocab import VocabBuilder
from yaml_bert.types import NodeType


TEMPLATES_DIR = os.path.join(
    os.path.dirname(__file__), "..", "kubernetes-yaml-templates"
)


def _load_all_nodes():
    """Helper: linearize all YAML files from the real corpus."""
    linearizer = YamlLinearizer()
    yaml_files = glob.glob(
        os.path.join(TEMPLATES_DIR, "**", "*.yaml"), recursive=True
    )
    all_nodes = []
    for path in yaml_files:
        all_nodes.extend(linearizer.linearize_file(path))
    return all_nodes


def test_full_pipeline_on_corpus():
    """End-to-end: linearize → annotate → build vocab from all 52 YAML files."""
    linearizer = YamlLinearizer()
    annotator = DomainAnnotator()
    builder = VocabBuilder()

    yaml_files = glob.glob(
        os.path.join(TEMPLATES_DIR, "**", "*.yaml"), recursive=True
    )

    all_nodes = []
    for path in yaml_files:
        nodes = linearizer.linearize_file(path)
        annotated = annotator.annotate(nodes)
        all_nodes.extend(annotated)

    # Build vocabulary from entire corpus
    vocab = builder.build(all_nodes)

    # Common K8s keys should be in key_vocab
    for key in ["apiVersion", "kind", "metadata", "name", "spec"]:
        assert key in vocab.key_vocab, f"Expected '{key}' in key_vocab"

    # Common K8s values should be in value_vocab
    for value in ["v1", "Pod"]:
        assert value in vocab.value_vocab, f"Expected '{value}' in value_vocab"

    # Vocab sizes should be reasonable
    assert len(vocab.key_vocab) > 20, "Expected 20+ unique keys across corpus"
    assert len(vocab.value_vocab) > 20, "Expected 20+ unique values across corpus"

    # Encoding roundtrip
    for key in ["apiVersion", "kind", "metadata"]:
        assert vocab.decode_key(vocab.encode_key(key)) == key


def test_full_pipeline_single_deployment():
    """Pipeline test using a specific real deployment file."""
    linearizer = YamlLinearizer()
    annotator = DomainAnnotator()
    builder = VocabBuilder()

    path = os.path.join(TEMPLATES_DIR, "deployment", "deployment-nginx.yaml")
    nodes = linearizer.linearize_file(path)
    annotated = annotator.annotate(nodes)
    vocab = builder.build(annotated)

    # containers should be marked as unordered
    containers_node = next(n for n in annotated if n.token == "containers")
    assert containers_node.annotations["list_ordered"] is False

    # Verify key/value separation
    assert "apiVersion" in vocab.key_vocab
    assert "Deployment" in vocab.value_vocab
    assert "containers" in vocab.key_vocab

    # Verify encoding works
    assert vocab.encode_key("apiVersion") >= 3  # after special tokens
    assert vocab.encode_value("Deployment") >= 3


def test_spec_at_two_depths():
    """'spec' at two different depths gets the same token ID but different parent_paths.
    Tree positional encoding (Phase 2) will differentiate them."""
    linearizer = YamlLinearizer()
    path = os.path.join(TEMPLATES_DIR, "deployment", "deployment-nginx.yaml")
    nodes = linearizer.linearize_file(path)

    spec_nodes = [n for n in nodes if n.token == "spec"]
    assert len(spec_nodes) == 2

    # Same token
    assert spec_nodes[0].token == spec_nodes[1].token

    # Different depths and parent paths
    assert spec_nodes[0].depth != spec_nodes[1].depth
    assert spec_nodes[0].parent_path != spec_nodes[1].parent_path


def test_vocab_save_load_roundtrip_on_corpus(tmp_path):
    """Build vocab from real corpus, save, reload, verify identical."""
    all_nodes = _load_all_nodes()

    builder = VocabBuilder()
    vocab = builder.build(all_nodes)

    vocab_path = str(tmp_path / "corpus_vocab.json")
    vocab.save(vocab_path)

    from yaml_bert.vocab import Vocabulary
    loaded = Vocabulary.load(vocab_path)

    assert loaded.key_vocab == vocab.key_vocab
    assert loaded.value_vocab == vocab.value_vocab
    assert loaded.special_tokens == vocab.special_tokens
```

- [ ] **Step 2: Run all tests**

Run: `pytest tests/ -v`
Expected: ALL PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_integration.py
git commit -m "test: add full pipeline integration tests using real K8s corpus"
```

---

### Task 12: Update Package Exports

**Files:**
- Modify: `yaml_bert/__init__.py`

- [ ] **Step 1: Update __init__.py with public API**

File: `yaml_bert/__init__.py`

```python
"""YAML-BERT: Attention on Kubernetes Structured Data."""

from yaml_bert.types import NodeType, YamlNode
from yaml_bert.linearizer import YamlLinearizer
from yaml_bert.vocab import Vocabulary, VocabBuilder
from yaml_bert.annotator import DomainAnnotator

__all__ = [
    "NodeType",
    "YamlNode",
    "YamlLinearizer",
    "Vocabulary",
    "VocabBuilder",
    "DomainAnnotator",
]
```

- [ ] **Step 2: Verify imports work**

Run: `python -c "from yaml_bert import YamlLinearizer, VocabBuilder, DomainAnnotator, NodeType, YamlNode, Vocabulary; print('All exports OK')"`
Expected: `All exports OK`

- [ ] **Step 3: Run full test suite one final time**

Run: `pytest tests/ -v`
Expected: ALL PASS

- [ ] **Step 4: Commit**

```bash
git add yaml_bert/__init__.py
git commit -m "feat: update package exports with public API"
```
