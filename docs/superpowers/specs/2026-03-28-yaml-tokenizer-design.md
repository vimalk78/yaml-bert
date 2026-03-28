# YAML-BERT Tokenizer Design

## Overview

Phase 1 tokenizer for the YAML-BERT project. Converts Kubernetes YAML manifests into a linearized sequence of annotated nodes suitable for transformer input.

This is **not** a text tokenizer (BPE/WordPiece). It is a structural tokenizer that:
1. Parses YAML into a tree
2. Linearizes via DFS into a flat sequence of typed nodes
3. Builds a vocabulary from a corpus
4. Annotates nodes with domain knowledge

## Architecture

Three independent modules with clear interfaces:

```
YAML string
    |
    v
YamlLinearizer  -->  list[YamlNode]
    |
    v
DomainAnnotator -->  list[YamlNode] (with annotations populated)

Separately:
Corpus of YamlNodes --> VocabBuilder --> Vocabulary (key_vocab + value_vocab)
```

## Data Structures

### NodeType

```python
class NodeType(Enum):
    KEY = "KEY"             # key in a regular mapping
    VALUE = "VALUE"         # scalar value in a regular mapping
    LIST_KEY = "LIST_KEY"   # key inside a list item
    LIST_VALUE = "LIST_VALUE"  # scalar value inside a list item
```

No synthetic `LIST_ITEM` boundary tokens. List item boundaries are implicit via numeric indices in `parent_path`.

"Map" is not a separate node type — it is emergent from nesting (a KEY whose children are more KEYs).

### YamlNode

```python
@dataclass
class YamlNode:
    token: str                              # the key or value string
    node_type: NodeType                     # KEY, VALUE, LIST_KEY, LIST_VALUE
    depth: int                              # 0-based depth in tree
    sibling_index: int                      # position among siblings
    parent_path: str                        # dot-joined path from root
    annotations: dict[str, Any] = field(default_factory=dict)  # domain knowledge
```

- `annotations` is a generic dict, kept empty by the linearizer. Populated by `DomainAnnotator`.
- `parent_path` for VALUE/LIST_VALUE nodes includes the key name (e.g., `metadata.name` not just `metadata`).

## Module Interfaces

### YamlLinearizer

```python
class YamlLinearizer:
    def linearize(self, yaml_string: str) -> list[YamlNode]
    def linearize_file(self, path: str) -> list[YamlNode]
    def linearize_multi_doc(self, yaml_string: str) -> list[list[YamlNode]]
```

- Uses PyYAML for parsing, then DFS traversal.
- `linearize_multi_doc` handles `---` separated documents (common in kubectl output).
- Pure structural conversion — no domain knowledge.

### VocabBuilder and Vocabulary

```python
class VocabBuilder:
    def build(self, nodes: list[YamlNode], min_freq: int = 1) -> Vocabulary

class Vocabulary:
    key_vocab: dict[str, int]
    value_vocab: dict[str, int]
    special_tokens: dict[str, int]  # [UNK], [PAD], [MASK]

    def encode_key(self, token: str) -> int
    def encode_value(self, token: str) -> int
    def decode_key(self, id: int) -> str
    def decode_value(self, id: int) -> str
    def save(self, path: str)
    def load(self, path: str) -> "Vocabulary"
```

- **Separate vocabularies** for keys and values. Keys are a small closed set; values are a large open set.
- Both project to the same embedding dimension in Phase 2 (separate `nn.Embedding` tables, same hidden_dim).
- `node_type` routes lookups: KEY/LIST_KEY use `key_vocab`, VALUE/LIST_VALUE use `value_vocab`.
- Unseen tokens map to `[UNK]`. Key `[UNK]` (potentially malformed YAML) and value `[UNK]` (unseen value) have different semantics.
- Vocabulary is built from data, not hardcoded. Rebuilt when corpus grows.

### DomainAnnotator

```python
class DomainAnnotator:
    ORDERED_LISTS = {"initContainers"}

    def annotate(self, nodes: list[YamlNode]) -> list[YamlNode]
```

- Tags list parent nodes with `annotations["list_ordered"] = True/False`.
- Domain knowledge is explicit and pluggable — a simple lookup table that grows over time.
- Separate from the linearizer to keep structural parsing clean.

## Linearization Rules

### Key-Value Pairs

```yaml
app: redis
role: replica
```

```
(app,     KEY,   depth=0, sibling=0, parent_path="")
(redis,   VALUE, depth=0, sibling=0, parent_path="app")
(role,    KEY,   depth=0, sibling=1, parent_path="")
(replica, VALUE, depth=0, sibling=1, parent_path="role")
```

### Nested Mappings

```yaml
metadata:
  name: nginx
  namespace: default
```

```
(metadata,  KEY,   depth=0, sibling=0, parent_path="")
(name,      KEY,   depth=1, sibling=0, parent_path="metadata")
(nginx,     VALUE, depth=1, sibling=0, parent_path="metadata.name")
(namespace, KEY,   depth=1, sibling=1, parent_path="metadata")
(default,   VALUE, depth=1, sibling=1, parent_path="metadata.namespace")
```

### Lists of Maps

```yaml
containers:
- name: webserver1
  image: nginx:1.6
  ports:
  - containerPort: 80
- name: database-server
  image: mysql-3.2
  ports:
  - containerPort: 3306
```

```
(containers,      KEY,        depth=0, sibling=0, parent_path="")
(name,            LIST_KEY,   depth=1, sibling=0, parent_path="containers.0")
(webserver1,      LIST_VALUE, depth=1, sibling=0, parent_path="containers.0.name")
(image,           LIST_KEY,   depth=1, sibling=1, parent_path="containers.0")
(nginx:1.6,       LIST_VALUE, depth=1, sibling=1, parent_path="containers.0.image")
(ports,           LIST_KEY,   depth=1, sibling=2, parent_path="containers.0")
(containerPort,   LIST_KEY,   depth=2, sibling=0, parent_path="containers.0.ports.0")
(80,              LIST_VALUE, depth=2, sibling=0, parent_path="containers.0.ports.0.containerPort")
(name,            LIST_KEY,   depth=1, sibling=0, parent_path="containers.1")
(database-server, LIST_VALUE, depth=1, sibling=0, parent_path="containers.1.name")
(image,           LIST_KEY,   depth=1, sibling=1, parent_path="containers.1")
(mysql-3.2,       LIST_VALUE, depth=1, sibling=1, parent_path="containers.1.image")
(ports,           LIST_KEY,   depth=1, sibling=2, parent_path="containers.1")
(containerPort,   LIST_KEY,   depth=2, sibling=0, parent_path="containers.1.ports.0")
(3306,            LIST_VALUE, depth=2, sibling=0, parent_path="containers.1.ports.0.containerPort")
```

### Scalar Lists

```yaml
args:
- --config
- /etc/app.yaml
```

```
(args,          KEY,        depth=0, sibling=0, parent_path="")
(--config,      LIST_VALUE, depth=1, sibling=0, parent_path="args.0")
(/etc/app.yaml, LIST_VALUE, depth=1, sibling=1, parent_path="args.1")
```

## Key Design Decisions

### Separate key and value vocabularies
Keys are a small, closed set (few hundred). Values are a large, open set. They have fundamentally different frequency distributions and will need different handling when sub-tokenization is added. Separate embedding tables in Phase 2, both projecting to the same hidden dimension.

### Atomic leaf values
Leaf values are treated as opaque strings for now. Compound values like `nginx:1.21`, `500Mi`, `app.kubernetes.io/name=frontend` are single tokens. **Known limitation:** the model cannot generalize across similar values (e.g., `3` vs `4`, `nginx:1.21` vs `nginx:1.22`). Sub-tokenization is planned as the next phase.

### No LIST_ITEM synthetic nodes
List item boundaries are represented by numeric indices in `parent_path` and `LIST_KEY`/`LIST_VALUE` node types, not by synthetic boundary tokens. This keeps sequences shorter and avoids an artificial tree level.

### List ordering as annotation
Whether a list is ordered or unordered is domain knowledge, not structural information. Captured in `annotations["list_ordered"]` by `DomainAnnotator`, not baked into the core data structure. Defaults to unordered; `initContainers` is ordered.

### VALUE parent_path includes key name
A VALUE node's `parent_path` includes the key it belongs to (e.g., `metadata.name` not `metadata`). This makes values self-locating in the tree without needing to examine neighboring nodes.

## Project Structure

```
yaml-bert/
├── yaml_bert/
│   ├── __init__.py
│   ├── linearizer.py
│   ├── vocab.py
│   ├── annotator.py
│   └── types.py
├── tests/
│   ├── test_linearizer.py
│   ├── test_vocab.py
│   └── test_annotator.py
├── data/
│   └── samples/
├── docs/
├── YAML_BERT_PLAN.md
└── requirements.txt        # pyyaml, pytest
```

## Dependencies

- Python 3.10+
- PyYAML (parsing)
- pytest (testing)

## What This Does NOT Cover

- Tree positional encoding (Phase 2)
- Sub-tokenization of compound values (next phase after this)
- Masking strategy (Phase 3)
- HuggingFace tokenizers integration (deferred until base results are available)

## References

- [CNCF: How to write YAML file for Kubernetes](https://www.cncf.io/blog/2022/03/03/how-to-write-yaml-file-for-kubernetes/) — YAML basic structure types (key-value pairs, lists, maps)
- YAML_BERT_PLAN.md — overall project plan
