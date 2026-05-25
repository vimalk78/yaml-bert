"""Test document embedding similarity.

Compare cosine similarity across different resource types.
Target: < 0.7 (vs v1's 0.84-0.92).

Usage:
    PYTHONPATH=. python scripts/test_similarity.py output_v4_quick/checkpoints/yaml_bert_v4_epoch_10.pt --vocab output_v4_quick/vocab.json
"""
from __future__ import annotations
import _setup_path  # noqa: F401

import argparse

import torch
import torch.nn.functional as F

from yaml_bert.config import YamlBertConfig
from yaml_bert.embedding import YamlBertEmbedding
from yaml_bert.model import YamlBertModel
from yaml_bert.linearizer import YamlLinearizer
from yaml_bert.annotator import DomainAnnotator
from yaml_bert.vocab import Vocabulary
from yaml_bert.types import NodeType


YAMLS: list[tuple[str, str]] = [
    ("Deployment", """\
apiVersion: apps/v1
kind: Deployment
metadata:
  name: web
spec:
  replicas: 3
"""),
    ("Service", """\
apiVersion: v1
kind: Service
metadata:
  name: svc
spec:
  ports:
  - port: 80
"""),
    ("Pod", """\
apiVersion: v1
kind: Pod
metadata:
  name: pod
spec:
  containers:
  - name: app
    image: nginx
"""),
    ("ConfigMap", """\
apiVersion: v1
kind: ConfigMap
metadata:
  name: cm
data:
  key: value
"""),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=str)
    parser.add_argument("--vocab", type=str, required=True)
    args = parser.parse_args()

    torch.manual_seed(42)
    vocab: Vocabulary = Vocabulary.load(args.vocab)
    config: YamlBertConfig = YamlBertConfig()
    emb = YamlBertEmbedding(
        config=config,
        key_vocab_size=vocab.key_vocab_size,
        value_vocab_size=vocab.value_vocab_size,
    )
    model = YamlBertModel(
        config=config, embedding=emb,
        simple_vocab_size=vocab.simple_target_vocab_size,
        kind_vocab_size=vocab.kind_target_vocab_size,
    )
    cp: dict = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(cp["model_state_dict"])
    model.eval()
    print(f"Loaded epoch {cp['epoch']}")

    linearizer: YamlLinearizer = YamlLinearizer()
    annotator: DomainAnnotator = DomainAnnotator()
    type_map = {NodeType.KEY: 0, NodeType.VALUE: 1, NodeType.LIST_KEY: 2, NodeType.LIST_VALUE: 3}

    means: list[torch.Tensor] = []
    labels: list[str] = []

    for label, yaml_text in YAMLS:
        nodes = linearizer.linearize(yaml_text)
        annotator.annotate(nodes)
        token_ids, node_types, depths, siblings = [], [], [], []
        for n in nodes:
            if n.node_type in (NodeType.KEY, NodeType.LIST_KEY):
                token_ids.append(vocab.encode_key(n.token))
            else:
                token_ids.append(vocab.encode_value(n.token))
            node_types.append(type_map[n.node_type])
            depths.append(min(n.depth, 15))
            siblings.append(min(n.sibling_index, 31))

        t = lambda x: torch.tensor([x])
        with torch.no_grad():
            x = model.embedding(t(token_ids), t(node_types), t(depths), t(siblings))
            for layer in model.encoder.layers:
                x = layer(x)

        means.append(x.squeeze(0).mean(dim=0))
        labels.append(label)

    stacked: torch.Tensor = torch.stack(means)
    normed: torch.Tensor = F.normalize(stacked, dim=1)
    sim: torch.Tensor = normed @ normed.T

    print("\nPairwise cosine similarity:")
    print(f"{'':>12}", end="")
    for l in labels:
        print(f"{l:>12}", end="")
    print()
    for i in range(len(labels)):
        print(f"{labels[i]:>12}", end="")
        for j in range(len(labels)):
            print(f"{sim[i,j]:>12.4f}", end="")
        print()

    avg_off_diag: float = (sim.sum() - sim.trace()) / (len(labels) * (len(labels) - 1))
    print(f"\nAverage off-diagonal similarity: {avg_off_diag:.4f}")
    print(f"Target: < 0.70 (v1 was 0.84-0.92)")


if __name__ == "__main__":
    main()
