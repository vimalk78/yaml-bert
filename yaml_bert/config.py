from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class YamlBertConfig:
    """Central configuration for YAML-BERT hyperparameters."""

    # Model architecture
    d_model: int = 256
    num_layers: int = 6
    num_heads: int = 8
    d_ff: int = 0  # 0 means "auto: 4 * d_model"
    max_depth: int = 16
    max_sibling: int = 32

    # Training
    mask_prob: float = 0.15
    lr: float = 1e-4
    batch_size: int = 32
    num_epochs: int = 30
    max_seq_len: int = 512

    # Auxiliary loss weights
    aux_kind_weight: float = 0.1     # α: kind classification loss weight
    aux_parent_weight: float = 0.1   # β: parent_key classification loss weight

    def __post_init__(self) -> None:
        if self.d_ff == 0:
            self.d_ff = 4 * self.d_model
