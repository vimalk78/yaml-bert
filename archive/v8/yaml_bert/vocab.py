from __future__ import annotations

import json
from yaml_bert.types import NodeType, YamlNode


SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[MASK]"]

# Canonical casing for known Kubernetes kinds.
# Maps lowercase → canonical form. Populated by VocabBuilder.build().
_KIND_CANONICAL: dict[str, str] = {}


def normalize_kind(kind: str) -> str:
    """Normalize kind value to canonical casing.

    'configmap' → 'ConfigMap', 'pod' → 'Pod', etc.
    Unknown kinds are returned as-is.
    """
    return _KIND_CANONICAL.get(kind.lower(), kind)


class Vocabulary:
    def __init__(
        self,
        key_vocab: dict[str, int],
        value_vocab: dict[str, int],
        special_tokens: dict[str, int],
        atomic_target_vocab: dict[str, int] | None = None,
    ) -> None:
        self.key_vocab = key_vocab
        self.value_vocab = value_vocab
        self.special_tokens = special_tokens
        self.atomic_target_vocab = atomic_target_vocab or {}
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

    def encode_atomic_target(self, target: str) -> int:
        """Encode a single key token as its atomic target id.

        Returns the [UNK] id if `target` is not in the atomic target vocab.
        The Token Head predicts single keys drawn from the same token universe
        as `key_vocab`.
        """
        return self.atomic_target_vocab.get(target, self.special_tokens["[UNK]"])

    @property
    def key_vocab_size(self) -> int:
        return len(self.key_vocab) + len(self.special_tokens)

    @property
    def value_vocab_size(self) -> int:
        return len(self.value_vocab) + len(self.special_tokens)

    @property
    def atomic_target_vocab_size(self) -> int:
        return len(self.atomic_target_vocab) + len(self.special_tokens)

    def save(self, path: str) -> None:
        data = {
            "key_vocab": self.key_vocab,
            "value_vocab": self.value_vocab,
            "special_tokens": self.special_tokens,
            "atomic_target_vocab": self.atomic_target_vocab,
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
            atomic_target_vocab=data.get("atomic_target_vocab", {}),
        )


class VocabBuilder:
    def build(
        self,
        nodes: list[YamlNode],
        min_freq: int = 1,
        *,
        key_min_freq: int | None = None,
        value_min_freq: int | None = None,
    ) -> Vocabulary:
        """Build vocab with optional per-category min_freq thresholds.

        Per-category thresholds default to the global `min_freq` for backward
        compatibility.
        """
        key_min_freq = key_min_freq if key_min_freq is not None else min_freq
        value_min_freq = value_min_freq if value_min_freq is not None else min_freq
        key_counts: dict[str, int] = {}
        value_counts: dict[str, int] = {}
        kind_set: set[str] = set()

        # First pass: collect all kind values to build canonical casing map.
        # The casing map (_KIND_CANONICAL) is needed by normalize_kind() which
        # is used downstream by _extract_kind in the types module.
        raw_kinds: dict[str, dict[str, int]] = {}  # lowercase → {variant: count}
        prev_was_kind_key = False
        for node in nodes:
            if node.node_type in (NodeType.KEY, NodeType.LIST_KEY):
                prev_was_kind_key = (node.token == "kind" and node.depth == 0)
            elif node.node_type in (NodeType.VALUE, NodeType.LIST_VALUE):
                if prev_was_kind_key:
                    lower = node.token.lower()
                    raw_kinds.setdefault(lower, {})
                    raw_kinds[lower][node.token] = raw_kinds[lower].get(node.token, 0) + 1
                prev_was_kind_key = False

        # Build canonical map: most frequent casing wins
        _KIND_CANONICAL.clear()
        for lower, variants in raw_kinds.items():
            canonical = max(variants, key=variants.get)
            _KIND_CANONICAL[lower] = canonical

        # Second pass: count tokens with normalized kinds
        prev_was_kind_key = False
        for node in nodes:
            if node.node_type in (NodeType.KEY, NodeType.LIST_KEY):
                key_counts[node.token] = key_counts.get(node.token, 0) + 1
                prev_was_kind_key = (node.token == "kind" and node.depth == 0)
            elif node.node_type in (NodeType.VALUE, NodeType.LIST_VALUE):
                value_counts[node.token] = value_counts.get(node.token, 0) + 1
                if prev_was_kind_key:
                    kind_set.add(normalize_kind(node.token))
                prev_was_kind_key = False

        # Atomic target vocab: single key tokens. Uses the same filter
        # threshold as keys since the entries ARE keys.
        atomic_target_set: set[str] = {
            t for t, c in key_counts.items() if c >= key_min_freq
        }

        return self.build_from_counts(
            key_counts, value_counts, key_min_freq, kind_set,
            value_min_freq=value_min_freq,
            atomic_target_set=atomic_target_set,
        )

    @staticmethod
    def build_from_counts(
        key_counts: dict[str, int],
        value_counts: dict[str, int],
        min_freq: int = 1,
        kind_set: set[str] | None = None,
        *,
        value_min_freq: int | None = None,
        atomic_target_set: set[str] | None = None,
    ) -> Vocabulary:
        """Build vocabulary from pre-computed token counts.

        `min_freq` filters keys (and values if value_min_freq is None).
        `value_min_freq` overrides for values only.
        """
        if value_min_freq is None:
            value_min_freq = min_freq
        # Derive atomic_target_set from key_counts when not supplied. The atomic
        # set IS-A subset of keys by definition, so this is self-consistent and
        # prevents callers (e.g. build_from_huggingface) from silently producing
        # an empty atomic_target_vocab — which would break training (output head
        # would collapse to ~3 classes and every target would map to [UNK]).
        if atomic_target_set is None:
            atomic_target_set = {t for t, c in key_counts.items() if c >= min_freq}
        special_tokens = {tok: i for i, tok in enumerate(SPECIAL_TOKENS)}
        offset = len(special_tokens)

        key_vocab = {
            token: i + offset
            for i, token in enumerate(
                sorted(t for t, c in key_counts.items() if c >= min_freq)
            )
        }

        # Build value vocab: min_freq filtered, but always include kind values
        value_tokens = set(t for t, c in value_counts.items() if c >= value_min_freq)
        if kind_set:
            value_tokens.update(kind_set)  # kinds always in value vocab
        value_vocab = {
            token: i + offset
            for i, token in enumerate(sorted(value_tokens))
        }

        atomic_target_vocab = {
            target: i + offset
            for i, target in enumerate(sorted(atomic_target_set or []))
        }

        return Vocabulary(
            key_vocab, value_vocab, special_tokens,
            atomic_target_vocab=atomic_target_vocab,
        )

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
