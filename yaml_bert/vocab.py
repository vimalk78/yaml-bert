from __future__ import annotations

import json
from yaml_bert.types import NodeType, YamlNode


SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[MASK]"]

UNIVERSAL_ROOT_KEYS: set[str] = {"apiVersion", "kind", "metadata"}


def compute_target(node: YamlNode, kind: str) -> tuple[str, str]:
    """Compute hybrid prediction target.

    Returns (target_string, head_type) where head_type is "simple" or "kind_specific".
    """
    if node.depth == 0:
        return node.token, "simple"

    parent_key: str = Vocabulary.extract_parent_key(node.parent_path)

    if node.depth == 1 and parent_key not in UNIVERSAL_ROOT_KEYS and parent_key != "":
        return f"{kind}::{parent_key}::{node.token}", "kind_specific"

    return (f"{parent_key}::{node.token}" if parent_key else node.token), "simple"


class Vocabulary:
    def __init__(
        self,
        key_vocab: dict[str, int],
        value_vocab: dict[str, int],
        special_tokens: dict[str, int],
        kind_vocab: dict[str, int] | None = None,
        simple_target_vocab: dict[str, int] | None = None,
        kind_target_vocab: dict[str, int] | None = None,
    ) -> None:
        self.key_vocab = key_vocab
        self.value_vocab = value_vocab
        self.special_tokens = special_tokens
        self.kind_vocab = kind_vocab or {"[NO_KIND]": 0}
        self.simple_target_vocab = simple_target_vocab or {}
        self.kind_target_vocab = kind_target_vocab or {}
        self._id_to_key = {v: k for k, v in key_vocab.items()}
        self._id_to_value = {v: k for k, v in value_vocab.items()}
        self._id_to_special = {v: k for k, v in special_tokens.items()}

    def encode_key(self, token: str) -> int:
        return self.key_vocab.get(token, self.special_tokens["[UNK]"])

    def encode_value(self, token: str) -> int:
        return self.value_vocab.get(token, self.special_tokens["[UNK]"])

    def encode_kind(self, kind: str) -> int:
        if not kind:
            return self.kind_vocab["[NO_KIND]"]
        return self.kind_vocab.get(kind, self.kind_vocab["[NO_KIND]"])

    def decode_key(self, id: int) -> str:
        if id in self._id_to_special:
            return self._id_to_special[id]
        return self._id_to_key.get(id, "[UNK]")

    def decode_value(self, id: int) -> str:
        if id in self._id_to_special:
            return self._id_to_special[id]
        return self._id_to_value.get(id, "[UNK]")

    def encode_simple_target(self, target: str) -> int:
        return self.simple_target_vocab.get(target, self.special_tokens["[UNK]"])

    def encode_kind_target(self, target: str) -> int:
        return self.kind_target_vocab.get(target, self.special_tokens["[UNK]"])

    @staticmethod
    def extract_parent_key(parent_path: str) -> str:
        """Extract the last non-numeric component from a parent_path.

        Examples:
            "spec.template.spec.containers.0" -> "containers"
            "metadata" -> "metadata"
            "" -> ""
        """
        if not parent_path:
            return ""
        parts = parent_path.split(".")
        for part in reversed(parts):
            if not part.isdigit():
                return part
        return ""

    @property
    def key_vocab_size(self) -> int:
        return len(self.key_vocab) + len(self.special_tokens)

    @property
    def value_vocab_size(self) -> int:
        return len(self.value_vocab) + len(self.special_tokens)

    @property
    def kind_vocab_size(self) -> int:
        return len(self.kind_vocab)

    @property
    def simple_target_vocab_size(self) -> int:
        return len(self.simple_target_vocab) + len(self.special_tokens)

    @property
    def kind_target_vocab_size(self) -> int:
        return len(self.kind_target_vocab) + len(self.special_tokens)

    def save(self, path: str) -> None:
        data = {
            "key_vocab": self.key_vocab,
            "value_vocab": self.value_vocab,
            "special_tokens": self.special_tokens,
            "kind_vocab": self.kind_vocab,
            "simple_target_vocab": self.simple_target_vocab,
            "kind_target_vocab": self.kind_target_vocab,
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
            kind_vocab=data.get("kind_vocab"),
            simple_target_vocab=data.get("simple_target_vocab", {}),
            kind_target_vocab=data.get("kind_target_vocab", {}),
        )


class VocabBuilder:
    def build(self, nodes: list[YamlNode], min_freq: int = 1) -> Vocabulary:
        key_counts: dict[str, int] = {}
        value_counts: dict[str, int] = {}
        kind_set: set[str] = set()
        simple_target_set: set[str] = set()
        kind_target_set: set[str] = set()

        current_kind: str = ""
        prev_was_kind_key: bool = False
        for node in nodes:
            if node.node_type in (NodeType.KEY, NodeType.LIST_KEY):
                key_counts[node.token] = key_counts.get(node.token, 0) + 1
                prev_was_kind_key = (node.token == "kind" and node.depth == 0)
                target, head_type = compute_target(node, current_kind)
                if head_type == "kind_specific":
                    kind_target_set.add(target)
                else:
                    simple_target_set.add(target)
            elif node.node_type in (NodeType.VALUE, NodeType.LIST_VALUE):
                value_counts[node.token] = value_counts.get(node.token, 0) + 1
                if prev_was_kind_key:
                    kind_set.add(node.token)
                    current_kind = node.token
                prev_was_kind_key = False

        return self.build_from_counts(
            key_counts, value_counts, min_freq, kind_set,
            simple_target_set, kind_target_set,
        )

    @staticmethod
    def build_from_counts(
        key_counts: dict[str, int],
        value_counts: dict[str, int],
        min_freq: int = 1,
        kind_set: set[str] | None = None,
        simple_target_set: set[str] | None = None,
        kind_target_set: set[str] | None = None,
    ) -> Vocabulary:
        """Build vocabulary from pre-computed token counts."""
        special_tokens = {tok: i for i, tok in enumerate(SPECIAL_TOKENS)}
        offset = len(special_tokens)

        key_vocab = {
            token: i + offset
            for i, token in enumerate(
                sorted(t for t, c in key_counts.items() if c >= min_freq)
            )
        }

        value_vocab = {
            token: i + offset
            for i, token in enumerate(
                sorted(t for t, c in value_counts.items() if c >= min_freq)
            )
        }

        kind_vocab: dict[str, int] = {"[NO_KIND]": 0}
        for i, kind in enumerate(sorted(kind_set or [])):
            kind_vocab[kind] = i + 1

        simple_target_vocab = {
            target: i + offset
            for i, target in enumerate(sorted(simple_target_set or []))
        }

        kind_target_vocab = {
            target: i + offset
            for i, target in enumerate(sorted(kind_target_set or []))
        }

        return Vocabulary(key_vocab, value_vocab, special_tokens, kind_vocab,
                          simple_target_vocab, kind_target_vocab)

    @staticmethod
    def save_counts(
        key_counts: dict[str, int],
        value_counts: dict[str, int],
        path: str,
    ) -> None:
        """Save raw token counts to a JSON file for reuse."""
        with open(path, "w") as f:
            json.dump({"key_counts": key_counts, "value_counts": value_counts}, f)
        print(f"Token counts saved: {path} ({len(key_counts)} keys, {len(value_counts)} values)")

    @staticmethod
    def load_counts(path: str) -> tuple[dict[str, int], dict[str, int]]:
        """Load raw token counts from a JSON file."""
        with open(path) as f:
            data = json.load(f)
        key_counts: dict[str, int] = data["key_counts"]
        value_counts: dict[str, int] = data["value_counts"]
        print(f"Token counts loaded: {path} ({len(key_counts)} keys, {len(value_counts)} values)")
        return key_counts, value_counts

    def build_from_huggingface(
        self,
        dataset_name: str,
        linearizer: "YamlLinearizer",
        annotator: "DomainAnnotator",
        max_docs: int | None = None,
        min_freq: int = 1,
        counts_path: str | None = None,
    ) -> Vocabulary:
        """Build vocabulary by scanning a HuggingFace dataset.

        Args:
            dataset_name: HuggingFace dataset ID
            linearizer: YamlLinearizer instance
            annotator: DomainAnnotator instance
            max_docs: Scan at most this many docs for vocab (None = all)
            min_freq: Minimum frequency to include a token
            counts_path: If set, save raw counts here (and load if file exists)
        """
        import os

        # Reuse saved counts if available
        if counts_path and os.path.exists(counts_path):
            key_counts, value_counts = self.load_counts(counts_path)
            return self.build_from_counts(key_counts, value_counts, min_freq, None)

        from datasets import load_dataset
        from yaml_bert.types import YamlNode

        print(f"Building vocabulary from: {dataset_name}")
        ds = load_dataset(dataset_name, split="train")

        total: int = len(ds) if max_docs is None else min(max_docs, len(ds))
        print(f"Scanning {total:,} / {len(ds):,} documents for vocabulary...")

        all_nodes: list[YamlNode] = []
        skipped: int = 0
        for i in range(total):
            try:
                nodes = linearizer.linearize(ds[i]["content"])
            except Exception:
                skipped += 1
                continue
            if nodes:
                annotator.annotate(nodes)
                all_nodes.extend(nodes)

            if (i + 1) % 10000 == 0:
                print(f"  {i + 1:,} / {total:,} scanned ({len(all_nodes):,} nodes)")

        print(f"Scanned {total - skipped:,} docs, {len(all_nodes):,} nodes ({skipped} skipped)")

        # Compute counts
        key_counts: dict[str, int] = {}
        value_counts: dict[str, int] = {}
        kind_set: set[str] = set()
        prev_was_kind_key: bool = False
        for node in all_nodes:
            if node.node_type in (NodeType.KEY, NodeType.LIST_KEY):
                key_counts[node.token] = key_counts.get(node.token, 0) + 1
                prev_was_kind_key = (node.token == "kind" and node.depth == 0)
            elif node.node_type in (NodeType.VALUE, NodeType.LIST_VALUE):
                value_counts[node.token] = value_counts.get(node.token, 0) + 1
                if prev_was_kind_key:
                    kind_set.add(node.token)
                prev_was_kind_key = False

        # Save counts for reuse
        if counts_path:
            self.save_counts(key_counts, value_counts, counts_path)

        return self.build_from_counts(key_counts, value_counts, min_freq, kind_set)
