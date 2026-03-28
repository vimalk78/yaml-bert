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

        return Vocabulary(key_vocab, value_vocab, special_tokens)

    def build_from_huggingface(
        self,
        dataset_name: str,
        linearizer: "YamlLinearizer",
        annotator: "DomainAnnotator",
        max_docs: int | None = None,
        min_freq: int = 1,
    ) -> Vocabulary:
        """Build vocabulary by scanning a HuggingFace dataset.

        Args:
            dataset_name: HuggingFace dataset ID
            linearizer: YamlLinearizer instance
            annotator: DomainAnnotator instance
            max_docs: Scan at most this many docs for vocab (None = all)
            min_freq: Minimum frequency to include a token
        """
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
        return self.build(all_nodes, min_freq=min_freq)
