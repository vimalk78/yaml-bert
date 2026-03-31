from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from yaml_bert.dataset import YamlDataset
from yaml_bert.model import YamlBertModel
from yaml_bert.vocab import Vocabulary


class YamlBertEvaluator:
    """Post-training evaluation for YAML-BERT."""

    def __init__(
        self,
        model: YamlBertModel,
        dataset: YamlDataset,
        vocab: Vocabulary,
    ) -> None:
        self.model: YamlBertModel = model
        self.dataset: YamlDataset = dataset
        self.vocab: Vocabulary = vocab
        self.device: torch.device = next(model.parameters()).device
        self._id_to_simple: dict[int, str] = {
            v: k for k, v in vocab.simple_target_vocab.items()
        }
        self._id_to_kind: dict[int, str] = {
            v: k for k, v in vocab.kind_target_vocab.items()
        }

    def _decode_simple(self, id: int) -> str:
        if id in self.vocab._id_to_special:
            return self.vocab._id_to_special[id]
        return self._id_to_simple.get(id, "[UNK]")

    def _decode_kind(self, id: int) -> str:
        if id in self.vocab._id_to_special:
            return self.vocab._id_to_special[id]
        return self._id_to_kind.get(id, "[UNK]")

    @torch.no_grad()
    def evaluate_prediction_accuracy(self) -> dict[str, float]:
        """Compute top-1 and top-5 masked key prediction accuracy over the dataset.

        Evaluates both simple and kind-specific heads independently.
        """
        self.model.eval()

        simple_total: int = 0
        simple_top1: int = 0
        simple_top5: int = 0
        kind_total: int = 0
        kind_top1: int = 0
        kind_top5: int = 0

        for idx in range(len(self.dataset)):
            item: dict[str, torch.Tensor] = self.dataset[idx]
            simple_labels: torch.Tensor = item["simple_labels"]
            kind_labels: torch.Tensor = item["kind_labels"]

            simple_masked: torch.Tensor = simple_labels != -100
            kind_masked: torch.Tensor = kind_labels != -100
            if not simple_masked.any() and not kind_masked.any():
                continue

            simple_logits, kind_logits = self.model(
                token_ids=item["token_ids"].unsqueeze(0).to(self.device),
                node_types=item["node_types"].unsqueeze(0).to(self.device),
                depths=item["depths"].unsqueeze(0).to(self.device),
                sibling_indices=item["sibling_indices"].unsqueeze(0).to(self.device),
            )

            s_logits: torch.Tensor = simple_logits[0]
            for pos in simple_masked.nonzero(as_tuple=True)[0]:
                true_id: int = simple_labels[pos].item()
                pos_logits: torch.Tensor = s_logits[pos]
                top5_ids: torch.Tensor = pos_logits.topk(5).indices

                if top5_ids[0].item() == true_id:
                    simple_top1 += 1
                if true_id in top5_ids.tolist():
                    simple_top5 += 1
                simple_total += 1

            k_logits: torch.Tensor = kind_logits[0]
            for pos in kind_masked.nonzero(as_tuple=True)[0]:
                true_id = kind_labels[pos].item()
                pos_logits = k_logits[pos]
                top5_ids = pos_logits.topk(5).indices

                if top5_ids[0].item() == true_id:
                    kind_top1 += 1
                if true_id in top5_ids.tolist():
                    kind_top5 += 1
                kind_total += 1

        total_masked: int = simple_total + kind_total
        total_top1: int = simple_top1 + kind_top1
        total_top5: int = simple_top5 + kind_top5

        return {
            "top1_accuracy": total_top1 / max(total_masked, 1),
            "top5_accuracy": total_top5 / max(total_masked, 1),
            "total_masked": total_masked,
            "simple_top1_accuracy": simple_top1 / max(simple_total, 1),
            "kind_top1_accuracy": kind_top1 / max(kind_total, 1),
        }

    @torch.no_grad()
    def analyze_embeddings(self) -> list[dict[str, Any]]:
        """Compare embeddings of the same key at different tree positions."""
        self.model.eval()
        results: list[dict[str, Any]] = []

        test_pairs: list[dict[str, Any]] = [
            {
                "key": "spec",
                "position_a": {"depth": 0},
                "position_b": {"depth": 2},
            },
            {
                "key": "name",
                "position_a": {"depth": 1},
                "position_b": {"depth": 1},
            },
        ]

        for pair in test_pairs:
            key_id: int = self.vocab.encode_key(pair["key"])

            token_ids: torch.Tensor = torch.tensor(
                [[key_id, key_id]], device=self.device
            )
            node_types: torch.Tensor = torch.tensor(
                [[0, 0]], device=self.device
            )
            depths: torch.Tensor = torch.tensor(
                [[pair["position_a"]["depth"], pair["position_b"]["depth"]]],
                device=self.device,
            )
            siblings: torch.Tensor = torch.tensor(
                [[0, 0]], device=self.device
            )

            embeddings: torch.Tensor = self.model.embedding(
                token_ids, node_types, depths, siblings
            )

            cosine_sim: float = F.cosine_similarity(
                embeddings[0, 0].unsqueeze(0),
                embeddings[0, 1].unsqueeze(0),
            ).item()

            results.append({
                "key": pair["key"],
                "position_a": pair["position_a"],
                "position_b": pair["position_b"],
                "cosine_similarity": cosine_sim,
            })

        return results

    @torch.no_grad()
    def top_k_predictions(
        self, doc_idx: int, k: int = 5
    ) -> list[dict[str, Any]]:
        """Show top-k predicted keys for each masked position in a document.

        Reports predictions from both the simple and kind-specific heads.
        """
        self.model.eval()

        item: dict[str, torch.Tensor] = self.dataset[doc_idx]
        simple_labels: torch.Tensor = item["simple_labels"]
        kind_labels: torch.Tensor = item["kind_labels"]

        simple_masked: torch.Tensor = simple_labels != -100
        kind_masked: torch.Tensor = kind_labels != -100
        if not simple_masked.any() and not kind_masked.any():
            return []

        simple_logits, kind_logits = self.model(
            token_ids=item["token_ids"].unsqueeze(0).to(self.device),
            node_types=item["node_types"].unsqueeze(0).to(self.device),
            depths=item["depths"].unsqueeze(0).to(self.device),
            sibling_indices=item["sibling_indices"].unsqueeze(0).to(self.device),
        )

        predictions: list[dict[str, Any]] = []

        # Simple head predictions
        s_logits: torch.Tensor = simple_logits[0]
        for pos in simple_masked.nonzero(as_tuple=True)[0]:
            true_id: int = simple_labels[pos].item()
            pos_logits: torch.Tensor = s_logits[pos]
            probs: torch.Tensor = F.softmax(pos_logits, dim=-1)
            topk: torch.return_types.topk = probs.topk(k)

            predicted_keys: list[dict[str, Any]] = [
                {
                    "key": self._decode_simple(topk.indices[i].item()),
                    "probability": topk.values[i].item(),
                }
                for i in range(k)
            ]

            predictions.append({
                "position": pos.item(),
                "head": "simple",
                "true_key": self._decode_simple(true_id),
                "predicted_keys": predicted_keys,
            })

        # Kind-specific head predictions
        k_logits: torch.Tensor = kind_logits[0]
        for pos in kind_masked.nonzero(as_tuple=True)[0]:
            true_id = kind_labels[pos].item()
            pos_logits = k_logits[pos]
            probs = F.softmax(pos_logits, dim=-1)
            topk = probs.topk(k)

            predicted_keys = [
                {
                    "key": self._decode_kind(topk.indices[i].item()),
                    "probability": topk.values[i].item(),
                }
                for i in range(k)
            ]

            predictions.append({
                "position": pos.item(),
                "head": "kind",
                "true_key": self._decode_kind(true_id),
                "predicted_keys": predicted_keys,
            })

        predictions.sort(key=lambda p: p["position"])
        return predictions
