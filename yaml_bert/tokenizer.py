"""Subword tokenizer wrapper for YAML-BERT v9.

Wraps the HF `tokenizers` library so the rest of yaml_bert depends on
this stable interface, not on HF internals.

Vocabulary semantics:
  - Special tokens reserved at training time: [PAD], [UNK], [MASK], [LONG_VALUE]
  - Otherwise byte-level BPE; any string can be encoded.

Long-value rule (values only; keys never get this treatment):
  - value length >= LONG_VALUE_THRESHOLD chars → single [LONG_VALUE] token
  - MAX_CHARS_VALUE < value length < LONG_VALUE_THRESHOLD → truncate to MAX_CHARS_VALUE chars, then BPE
  - shorter → BPE in full
"""
from __future__ import annotations

from tokenizers import Tokenizer

PAD_TOKEN = "[PAD]"
UNK_TOKEN = "[UNK]"
MASK_TOKEN = "[MASK]"
LONG_VALUE_TOKEN = "[LONG_VALUE]"

MAX_CHARS_VALUE = 64
LONG_VALUE_THRESHOLD = 256


class SubwordTokenizer:
    """Wraps an HF Tokenizer with YAML-BERT-specific value-length rules."""

    def __init__(self, hf_tokenizer: Tokenizer) -> None:
        self._tok = hf_tokenizer
        self.pad_id = hf_tokenizer.token_to_id(PAD_TOKEN)
        self.unk_id = hf_tokenizer.token_to_id(UNK_TOKEN)
        self.mask_id = hf_tokenizer.token_to_id(MASK_TOKEN)
        self.long_value_id = hf_tokenizer.token_to_id(LONG_VALUE_TOKEN)
        for name, val in (
            (PAD_TOKEN, self.pad_id), (UNK_TOKEN, self.unk_id),
            (MASK_TOKEN, self.mask_id), (LONG_VALUE_TOKEN, self.long_value_id),
        ):
            if val is None:
                raise ValueError(
                    f"SubwordTokenizer: required special token {name!r} not "
                    f"in tokenizer vocab — was the tokenizer trained with "
                    f"this special token reserved?"
                )

    @classmethod
    def load(cls, path: str) -> "SubwordTokenizer":
        return cls(Tokenizer.from_file(path))

    @property
    def vocab_size(self) -> int:
        return self._tok.get_vocab_size()

    def encode_token(self, token: str, *, is_value: bool) -> list[int]:
        """Encode one linearizer-node token to a list of subword ids.

        See module docstring for the value-length rule. Keys are always
        BPE-encoded in full regardless of length.
        """
        if is_value:
            if len(token) >= LONG_VALUE_THRESHOLD:
                return [self.long_value_id]
            if len(token) > MAX_CHARS_VALUE:
                token = token[:MAX_CHARS_VALUE]
        return self._tok.encode(token).ids

    def decode(self, ids: list[int]) -> str:
        return self._tok.decode(ids)
