from __future__ import annotations

from dataclasses import dataclass, field
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
    # tree-bias attention: per-(distance, head) bias added to attention logits.
    # DISABLED by default in v7 because PyTorch's nn.TransformerEncoder takes a
    # slow path whenever attn_mask is non-None (~15× slowdown observed). The
    # module + integration are kept for future use once a custom encoder layer
    # using F.scaled_dot_product_attention (which supports fast-path attn_mask)
    # is built. Enable manually if you've made that change.
    tree_bias_enabled: bool = False

    # Training
    mask_prob: float = 0.15
    lr: float = 1e-4
    batch_size: int = 32
    num_epochs: int = 30
    max_seq_len: int = 512
    skip_unk_targets: bool = True  # v6: don't supervise on positions whose target is [UNK]

    # Auxiliary loss weights
    aux_kind_weight: float = 0.1     # α: kind classification loss weight
    aux_parent_weight: float = 0.1   # β: parent_key classification loss weight

    def __post_init__(self) -> None:
        if self.d_ff == 0:
            self.d_ff = 4 * self.d_model
