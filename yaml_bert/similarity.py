from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from yaml_bert.annotator import DomainAnnotator
from yaml_bert.dataset import _extract_kind
from yaml_bert.linearizer import YamlLinearizer
from yaml_bert.model import YamlBertModel
from yaml_bert.pooling import DocumentPooling
from yaml_bert.types import NodeType, YamlNode
from yaml_bert.vocab import Vocabulary


_NODE_TYPE_INDEX: dict[NodeType, int] = {
    NodeType.KEY: 0, NodeType.VALUE: 1,
    NodeType.LIST_KEY: 2, NodeType.LIST_VALUE: 3,
}


@torch.no_grad()
def extract_hidden_states(
    model: YamlBertModel,
    vocab: Vocabulary,
    yaml_text: str,
) -> tuple[torch.Tensor, int]:
    linearizer: YamlLinearizer = YamlLinearizer()
    annotator: DomainAnnotator = DomainAnnotator()
    # Use linearize_multi_doc to handle --- separators, take the first doc
    docs: list[list[YamlNode]] = linearizer.linearize_multi_doc(yaml_text)
    nodes: list[YamlNode] = docs[0] if docs else []
    if not nodes:
        return torch.empty(0), -1
    annotator.annotate(nodes)

    token_ids: list[int] = []
    node_types: list[int] = []
    depths: list[int] = []
    siblings: list[int] = []
    parent_keys: list[int] = []
    kind_pos: int = -1

    for i, node in enumerate(nodes):
        if node.node_type in (NodeType.KEY, NodeType.LIST_KEY):
            token_ids.append(vocab.encode_key(node.token))
        else:
            token_ids.append(vocab.encode_value(node.token))
        node_types.append(_NODE_TYPE_INDEX[node.node_type])
        depths.append(min(node.depth, 15))
        siblings.append(min(node.sibling_index, 31))
        parent_keys.append(vocab.encode_key(Vocabulary.extract_parent_key(node.parent_path)))

        if node.token == "kind" and node.depth == 0 and node.node_type == NodeType.KEY and kind_pos == -1:
            kind_pos = i

    kind: str = _extract_kind(nodes)
    kind_id: int = vocab.encode_kind(kind)
    kind_ids: list[int] = [kind_id] * len(nodes)

    t = lambda x: torch.tensor([x])
    model.eval()

    x: torch.Tensor = model.embedding(
        t(token_ids), t(node_types), t(depths), t(siblings), t(parent_keys),
        kind_ids=t(kind_ids),
    )
    for layer in model.encoder.layers:
        x = layer(x)

    return x.squeeze(0), kind_pos


@torch.no_grad()
def get_document_embedding(
    model: YamlBertModel,
    pooling: DocumentPooling,
    vocab: Vocabulary,
    yaml_text: str,
) -> torch.Tensor:
    hidden, kind_pos = extract_hidden_states(model, vocab, yaml_text)
    if hidden.shape[0] == 0 or kind_pos < 0:
        return torch.zeros(hidden.shape[1] if hidden.dim() > 1 else 1)

    pooling.eval()
    kind_hidden: torch.Tensor = hidden[kind_pos].unsqueeze(0).unsqueeze(0)
    all_hidden: torch.Tensor = hidden.unsqueeze(0)
    return pooling(kind_hidden, all_hidden).squeeze(0)


def cosine_similarity_matrix(embeddings: torch.Tensor) -> torch.Tensor:
    normed: torch.Tensor = F.normalize(embeddings, dim=1)
    return normed @ normed.T
