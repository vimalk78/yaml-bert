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

    @torch.no_grad()
    def evaluate_prediction_accuracy(self) -> dict[str, float]:
        """Compute top-1 and top-5 masked key prediction accuracy over the dataset."""
        self.model.eval()

        total_masked: int = 0
        top1_correct: int = 0
        top5_correct: int = 0

        for idx in range(len(self.dataset)):
            item: dict[str, torch.Tensor] = self.dataset[idx]
            labels: torch.Tensor = item["labels"]

            masked_positions: torch.Tensor = labels != -100
            if not masked_positions.any():
                continue

            batch: dict[str, torch.Tensor] = {
                k: v.unsqueeze(0).to(self.device) for k, v in item.items()
            }

            key_logits: torch.Tensor = self.model(
                token_ids=batch["token_ids"],
                node_types=batch["node_types"],
                depths=batch["depths"],
                sibling_indices=batch["sibling_indices"],
                parent_key_ids=batch["parent_key_ids"],
            )

            logits: torch.Tensor = key_logits[0]
            for pos in masked_positions.nonzero(as_tuple=True)[0]:
                true_id: int = labels[pos].item()
                pos_logits: torch.Tensor = logits[pos]
                top5_ids: torch.Tensor = pos_logits.topk(5).indices

                if top5_ids[0].item() == true_id:
                    top1_correct += 1
                if true_id in top5_ids.tolist():
                    top5_correct += 1
                total_masked += 1

        return {
            "top1_accuracy": top1_correct / max(total_masked, 1),
            "top5_accuracy": top5_correct / max(total_masked, 1),
            "total_masked": total_masked,
        }

    @torch.no_grad()
    def analyze_embeddings(self) -> list[dict[str, Any]]:
        """Compare embeddings of the same key at different tree positions."""
        self.model.eval()
        results: list[dict[str, Any]] = []

        test_pairs: list[dict[str, Any]] = [
            {
                "key": "spec",
                "position_a": {"depth": 0, "parent_key": ""},
                "position_b": {"depth": 2, "parent_key": "template"},
            },
            {
                "key": "name",
                "position_a": {"depth": 1, "parent_key": "metadata"},
                "position_b": {"depth": 1, "parent_key": "containers"},
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
            parent_a_id: int = self.vocab.encode_key(
                pair["position_a"]["parent_key"]
            )
            parent_b_id: int = self.vocab.encode_key(
                pair["position_b"]["parent_key"]
            )
            parent_keys: torch.Tensor = torch.tensor(
                [[parent_a_id, parent_b_id]], device=self.device
            )

            embeddings: torch.Tensor = self.model.embedding(
                token_ids, node_types, depths, siblings, parent_keys
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
        """Show top-k predicted keys for each masked position in a document."""
        self.model.eval()

        item: dict[str, torch.Tensor] = self.dataset[doc_idx]
        labels: torch.Tensor = item["labels"]

        masked_positions: torch.Tensor = labels != -100
        if not masked_positions.any():
            return []

        batch: dict[str, torch.Tensor] = {
            k_: v.unsqueeze(0).to(self.device) for k_, v in item.items()
        }

        key_logits: torch.Tensor = self.model(
            token_ids=batch["token_ids"],
            node_types=batch["node_types"],
            depths=batch["depths"],
            sibling_indices=batch["sibling_indices"],
            parent_key_ids=batch["parent_key_ids"],
        )

        logits: torch.Tensor = key_logits[0]
        predictions: list[dict[str, Any]] = []

        for pos in masked_positions.nonzero(as_tuple=True)[0]:
            true_id: int = labels[pos].item()
            pos_logits: torch.Tensor = logits[pos]
            probs: torch.Tensor = F.softmax(pos_logits, dim=-1)
            topk: torch.return_types.topk = probs.topk(k)

            predicted_keys: list[dict[str, Any]] = [
                {
                    "key": self.vocab.decode_key(topk.indices[i].item()),
                    "probability": topk.values[i].item(),
                }
                for i in range(k)
            ]

            predictions.append({
                "position": pos.item(),
                "true_key": self.vocab.decode_key(true_id),
                "predicted_keys": predicted_keys,
            })

        return predictions
