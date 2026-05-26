"""v9 Vocabulary: subword tokenizer + atomic_target_vocab.

The v8 key_vocab + value_vocab were merged into a single subword vocabulary
exposed via SubwordTokenizer. The Token Head still predicts over an atomic
target vocab built from frequent KEY tokens in the training corpus.
"""
from __future__ import annotations

import json
from yaml_bert.tokenizer import SubwordTokenizer

# Canonical casing for known Kubernetes kinds.
# Maps lowercase → canonical form. Populated by VocabBuilder.build_atomic_target_vocab.
_KIND_CANONICAL: dict[str, str] = {}


def normalize_kind(kind: str) -> str:
    return _KIND_CANONICAL.get(kind.lower(), kind)


class Vocabulary:
    """Holds a subword tokenizer + an atomic target vocab.

    `atomic_target_vocab` maps whole KEY strings (e.g. "containers",
    "restartPolicy") to integer class ids for the Token Head's output.
    The 4 special tokens (pad/unk/mask/long_value) get the first 4 ids.
    """

    def __init__(
        self,
        tokenizer: SubwordTokenizer,
        atomic_target_vocab: dict[str, int],
        tokenizer_path: str | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.atomic_target_vocab = atomic_target_vocab
        self.tokenizer_path = tokenizer_path
        # Convenience: expose the 4 special-token ids at the vocab level
        self.pad_id = tokenizer.pad_id
        self.unk_id = tokenizer.unk_id
        self.mask_id = tokenizer.mask_id
        self.long_value_id = tokenizer.long_value_id

    @classmethod
    def from_tokenizer_path(
        cls,
        tokenizer_path: str,
        atomic_target_vocab: dict[str, int],
    ) -> "Vocabulary":
        return cls(
            tokenizer=SubwordTokenizer.load(tokenizer_path),
            atomic_target_vocab=atomic_target_vocab,
            tokenizer_path=tokenizer_path,
        )

    def encode_token(self, token: str, *, is_value: bool) -> list[int]:
        return self.tokenizer.encode_token(token, is_value=is_value)

    def encode_atomic_target(self, key: str) -> int:
        return self.atomic_target_vocab.get(key, self.unk_id)

    @property
    def subword_vocab_size(self) -> int:
        return self.tokenizer.vocab_size

    @property
    def atomic_target_vocab_size(self) -> int:
        # +4 to reserve ids 0-3 for special tokens (consistent with v8 layout)
        return len(self.atomic_target_vocab) + 4

    def save(self, path: str) -> None:
        if self.tokenizer_path is None:
            raise ValueError(
                "Vocabulary.save requires tokenizer_path to be set "
                "(use Vocabulary.from_tokenizer_path)."
            )
        with open(path, "w") as f:
            json.dump({
                "tokenizer_path": self.tokenizer_path,
                "atomic_target_vocab": self.atomic_target_vocab,
            }, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "Vocabulary":
        with open(path) as f:
            data = json.load(f)
        return cls.from_tokenizer_path(
            tokenizer_path=data["tokenizer_path"],
            atomic_target_vocab=data["atomic_target_vocab"],
        )


class VocabBuilder:
    """Builds the atomic_target_vocab from a corpus.

    The subword tokenizer is built separately by scripts/train_unified_tokenizer.py.
    """

    @staticmethod
    def build_atomic_target_vocab(
        nodes_per_doc: list[list],
        min_freq: int,
    ) -> dict[str, int]:
        """Scan KEY tokens across docs, return {token: id} for keys appearing >= min_freq times.

        IDs start at 4 (0-3 are reserved for special tokens).
        """
        from yaml_bert.types import NodeType
        counts: dict[str, int] = {}
        for nodes in nodes_per_doc:
            for n in nodes:
                if n.node_type in (NodeType.KEY, NodeType.LIST_KEY):
                    counts[n.token] = counts.get(n.token, 0) + 1

        # Also populate kind-canonical map as a side effect (used by suggest.py)
        prev_kind_key = False
        for nodes in nodes_per_doc:
            for n in nodes:
                if n.node_type in (NodeType.KEY, NodeType.LIST_KEY):
                    prev_kind_key = (n.token == "kind" and n.depth == 0)
                elif n.node_type in (NodeType.VALUE, NodeType.LIST_VALUE):
                    if prev_kind_key:
                        lower = n.token.lower()
                        existing = _KIND_CANONICAL.get(lower)
                        if existing is None:
                            _KIND_CANONICAL[lower] = n.token
                    prev_kind_key = False

        kept = sorted(t for t, c in counts.items() if c >= min_freq)
        return {t: 4 + i for i, t in enumerate(kept)}
