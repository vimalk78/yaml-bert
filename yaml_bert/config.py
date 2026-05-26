from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class TreePosVariant(str, Enum):
    """Tree positional encoding ablation variants."""

    FULL = "full"             # depth + sibling + node_type (default)
    NO_DEPTH = "no_depth"     # sibling + node_type
    NO_SIBLING = "no_sibling" # depth + node_type
    SEQUENTIAL = "sequential" # learned pos[seq_idx] + node_type


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
    tree_pos_variant: TreePosVariant = TreePosVariant.FULL

    # Training
    mask_prob: float = 0.15
    lr: float = 1e-4
    batch_size: int = 32
    num_epochs: int = 30
    max_seq_len: int = 768  # was 512 in v8; v9 BPE-expands sequences ~2.3x

    # Reconstruction objective (subtree masking + bag-of-keys prediction)
    recon_enabled: bool = False
    recon_loss_weight: float = 0.5

    def __post_init__(self) -> None:
        if self.d_ff == 0:
            self.d_ff = 4 * self.d_model
