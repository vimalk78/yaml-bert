from __future__ import annotations

import os

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from yaml_bert.config import YamlBertConfig
from yaml_bert.dataset import YamlDataset, collate_fn
from yaml_bert.model import YamlBertModel


class YamlBertTrainer:
    """Training loop with hybrid loss from two prediction heads."""

    def __init__(
        self,
        config: YamlBertConfig,
        model: YamlBertModel,
        dataset: YamlDataset,
        checkpoint_dir: str | None = None,
        checkpoint_every: int = 1,
        resume_from: str | None = None,
    ) -> None:
        self.config = config
        self.model = model
        self.dataset = dataset
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_every = checkpoint_every
        self.resume_from = resume_from
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def train(self) -> list[float]:
        from datetime import datetime
        self.model.to(self.device)
        self.model.train()

        print(f"Training started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        num_params = sum(p.numel() for p in self.model.parameters())
        print(f"Model parameters: {num_params:,}")
        print(f"Config: d_model={self.config.d_model}, layers={self.config.num_layers}, heads={self.config.num_heads}")
        print(f"Device: {self.device}")

        optimizer = AdamW(self.model.parameters(), lr=self.config.lr, weight_decay=0.01)
        start_epoch = 0

        if self.resume_from:
            checkpoint = torch.load(self.resume_from, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_epoch = checkpoint["epoch"]
            print(f"Resumed from epoch {start_epoch}")

        dataloader = DataLoader(
            self.dataset, batch_size=self.config.batch_size,
            shuffle=True, collate_fn=collate_fn,
        )

        epoch_losses: list[float] = []
        for epoch in range(start_epoch, self.config.num_epochs):
            total_loss: float = 0.0
            num_batches: int = 0
            running_breakdown: dict[str, float] = {}

            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{self.config.num_epochs}", leave=True)
            for batch in pbar:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                optimizer.zero_grad()

                simple_logits, kind_logits = self.model(
                    token_ids=batch["token_ids"],
                    node_types=batch["node_types"],
                    depths=batch["depths"],
                    sibling_indices=batch["sibling_indices"],
                    padding_mask=batch["padding_mask"],
                )

                loss, breakdown = self.model.compute_loss(
                    simple_logits, batch["simple_labels"],
                    kind_logits, batch["kind_labels"],
                )

                if torch.isnan(loss):
                    continue  # skip bad batches

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

                total_loss += loss.item()
                for k, v in breakdown.items():
                    running_breakdown[k] = running_breakdown.get(k, 0.0) + v
                num_batches += 1

                postfix = {"loss": f"{total_loss/num_batches:.4f}"}
                for k in ["simple", "kind"]:
                    if k in running_breakdown:
                        postfix[k] = f"{running_breakdown[k]/num_batches:.4f}"
                pbar.set_postfix(**postfix)

            avg_loss: float = total_loss / max(num_batches, 1)
            epoch_losses.append(avg_loss)
            breakdown_str: str = " | ".join(
                f"{k}: {v/max(num_batches,1):.4f}" for k, v in sorted(running_breakdown.items())
            )
            print(f"Epoch {epoch+1}/{self.config.num_epochs} — loss: {avg_loss:.4f} ({breakdown_str})")

            if self.checkpoint_dir and (epoch+1) % self.checkpoint_every == 0:
                self._save_checkpoint(epoch+1, optimizer)

        if self.checkpoint_dir:
            self._save_checkpoint(self.config.num_epochs, optimizer)

        return epoch_losses

    def _save_checkpoint(self, epoch: int, optimizer: AdamW) -> None:
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        path = os.path.join(self.checkpoint_dir, f"yaml_bert_v4_epoch_{epoch}.pt")
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }, path)
        print(f"Checkpoint saved: {path}")
