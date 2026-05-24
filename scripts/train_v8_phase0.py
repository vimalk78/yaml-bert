"""v8 Phase 0 benchmark: train on 5K-doc subset, measure perf + loss.

Mirrors v7's quick-mode training (5K docs, 10 epochs) for direct comparison.
Saves: vocab.json, checkpoints, training.log, doc_vec dump (for kind probe).
"""
from __future__ import annotations
import _setup_path  # noqa: F401

import argparse
import json
import os
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from yaml_bert.cache import build_or_load_cache
from yaml_bert.config import YamlBertConfig
from yaml_bert.embedding import YamlBertEmbedding
from yaml_bert.v8_dataset import V8Dataset, v8_collate_fn
from yaml_bert.v8_model import V8Model
from yaml_bert.vocab import VocabBuilder

DATASET_NAME = "substratusai/the-stack-yaml-k8s"


def _forward_v8(model, batch, device):
    """Forward V8Model with vectorized path active. Returns (logits, doc_vec)."""
    return model(
        token_ids=batch["token_ids"].to(device),
        node_types=batch["node_types"].to(device),
        depths=batch["depths"].to(device),
        sibling_indices=batch["sibling_indices"].to(device),
        batch_info=batch["batch_info"],
        padding_mask=batch["padding_mask"].to(device),
        parent_of_tensor=batch["parent_of_tensor"].to(device),
        top_level_key_mask=batch["top_level_key_mask"].to(device),
        edges_by_depth={
            d: t.to(device) for d, t in batch["edges_by_depth"].items()
        },
        parents_by_depth={
            d: t.to(device) for d, t in batch["parents_by_depth"].items()
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-docs", type=int, default=5000)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output-dir", type=str, default="output_v8_phase0")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    import random
    import numpy as np
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    cache_path = os.path.join(args.output_dir, "doc_cache.pkl")
    print(f"Step 0: Linearize → {cache_path}")
    cached = build_or_load_cache(DATASET_NAME, cache_path=cache_path,
                                 max_docs=args.max_docs)

    print("Step 1: Build vocab (v8 mode — atomic targets)")
    all_nodes = [n for doc in cached for n in doc]
    vocab = VocabBuilder().build(
        all_nodes,
        key_min_freq=10,
        value_min_freq=10,
        simple_target_min_freq=5,
        kind_target_min_freq=2,
    )
    vocab.save(os.path.join(args.output_dir, "vocab.json"))
    print(f"  key vocab: {vocab.key_vocab_size}")
    print(f"  atomic vocab: {vocab.atomic_target_vocab_size}")

    print("Step 2: Build dataset")
    config = YamlBertConfig(num_epochs=args.epochs, batch_size=args.batch_size,
                            v8_mode=True)
    dataset = V8Dataset(cached, vocab, config)

    print("Step 3: Build model")
    emb = YamlBertEmbedding(config=config,
                            key_vocab_size=vocab.key_vocab_size,
                            value_vocab_size=vocab.value_vocab_size)
    model = V8Model(config=config, embedding=emb,
                    atomic_vocab_size=vocab.atomic_target_vocab_size)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  device: {device}")
    print(f"  params: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)

    num_workers = min(8, max(2, (os.cpu_count() or 4) // 2))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                       collate_fn=v8_collate_fn, num_workers=num_workers,
                       persistent_workers=True, pin_memory=True)

    print("Step 4: Training")
    train_start = time.time()
    epoch_losses: list[float] = []
    for epoch in range(args.epochs):
        epoch_start = time.time()
        model.train()
        total_loss = 0.0
        n_batches = 0
        for batch in loader:
            optimizer.zero_grad()
            logits, doc_vec = _forward_v8(model, batch, device)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                batch["atomic_labels"].to(device).view(-1),
                ignore_index=-100,
            )
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
            if not torch.isfinite(loss):
                print(f"  !! NaN/Inf loss at batch {n_batches}; stopping early")
                return

        avg_loss = total_loss / max(1, n_batches)
        epoch_losses.append(avg_loss)
        epoch_dur = time.time() - epoch_start
        print(f"  Epoch {epoch+1}/{args.epochs} — loss: {avg_loss:.4f}  "
              f"({n_batches} batches, {epoch_dur:.1f}s, "
              f"{n_batches/epoch_dur:.2f} it/s)")

    total_dur = time.time() - train_start
    print(f"Step 5: Save checkpoint")
    ckpt_path = os.path.join(args.output_dir, "v8_phase0.pt")
    torch.save({"model_state_dict": model.state_dict(),
                "epoch_losses": epoch_losses,
                "n_params": n_params,
                "total_train_sec": total_dur}, ckpt_path)

    print(f"Step 6: Dump per-doc doc vectors for probe")
    model.eval()
    doc_vecs: list[torch.Tensor] = []
    doc_kinds: list[str] = []
    from yaml_bert.dataset import _extract_kind
    eval_loader = DataLoader(dataset, batch_size=args.batch_size,
                            shuffle=False, collate_fn=v8_collate_fn,
                            num_workers=num_workers)
    with torch.no_grad():
        for batch_idx, batch in enumerate(eval_loader):
            _, dvec = _forward_v8(model, batch, device)
            doc_vecs.append(dvec.cpu())
            for doc_idx_in_batch in range(dvec.size(0)):
                global_idx = batch_idx * args.batch_size + doc_idx_in_batch
                if global_idx < len(cached):
                    doc_kinds.append(_extract_kind(cached[global_idx]))

    doc_vecs_t = torch.cat(doc_vecs, dim=0)
    torch.save({"doc_vecs": doc_vecs_t, "kinds": doc_kinds},
               os.path.join(args.output_dir, "doc_vecs.pt"))
    print(f"  saved {doc_vecs_t.shape[0]} doc vectors")

    print(f"Done. Total: {total_dur:.1f}s")


if __name__ == "__main__":
    main()
