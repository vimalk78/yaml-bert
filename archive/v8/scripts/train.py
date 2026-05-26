"""v8 Phase 1 reconstruction benchmark: train on 5K-doc subset, MLM-only OR
MLM+reconstruction. Per-epoch loss + val + doc_vec dumps for probe trajectory.
"""
from __future__ import annotations
import _setup_path  # noqa: F401

import argparse
import os
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

from yaml_bert.cache import build_or_load_cache
from yaml_bert.config import YamlBertConfig
from yaml_bert.embedding import YamlBertEmbedding
from yaml_bert.dataset import YamlBertDataset, collate_fn
from yaml_bert.model import YamlBertModel
from yaml_bert.vocab import VocabBuilder

DATASET_NAME = "substratusai/the-stack-yaml-k8s"


def _forward_v8(model, batch, device, recon_enabled: bool):
    """Forward YamlBertModel. Returns (logits, doc_vec, recon_logits|None)."""
    kwargs = dict(
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
    if recon_enabled and "subtree_mask" in batch:
        kwargs["subtree_mask"] = batch["subtree_mask"].to(device)
        kwargs["subtree_roots_flat"] = batch["subtree_roots_flat"].to(device)
        out = model(**kwargs)
        if len(out) == 3:
            return out  # (logits, doc_vec, recon_logits)
        return (*out, None)  # (logits, doc_vec, None) — no subtrees this batch
    out = model(**kwargs)
    return (*out, None)


def _compute_losses(out, batch, device, recon_enabled: bool, recon_weight: float):
    logits, _, recon_logits = out
    labels = batch["atomic_labels"].to(device)
    zero = torch.tensor(0.0, device=device)
    # When every position is -100 (ignored), cross_entropy returns NaN.
    # We replace it with a constant zero — this batch contributes no gradient.
    # The downstream `total_loss.requires_grad` guard in the training loop
    # will then skip backward+step for this batch.
    if (labels != -100).any():
        mlm_loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=-100,
        )
    else:
        mlm_loss = zero
    if not recon_enabled or recon_logits is None:
        return mlm_loss, mlm_loss, zero
    recon_target = batch["bag_of_keys_targets_flat"].to(device)
    recon_loss = F.binary_cross_entropy_with_logits(recon_logits, recon_target)
    total = mlm_loss + recon_weight * recon_loss
    return total, mlm_loss, recon_loss


def _dump_doc_vecs(model, dataset, batch_size, device, recon_enabled,
                   output_path, cached, num_workers):
    """One pass over the FULL 5K corpus dumping doc_vecs to disk."""
    from yaml_bert.types import _extract_kind
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                       collate_fn=collate_fn, num_workers=num_workers)
    doc_vecs: list[torch.Tensor] = []
    doc_kinds: list[str] = []
    dump_iter = tqdm(
        loader,
        desc=f"dump → {os.path.basename(output_path)}",
        mininterval=2,
        dynamic_ncols=True,
    )
    with torch.no_grad():
        for batch_idx, batch in enumerate(dump_iter):
            _, dvec, _ = _forward_v8(model, batch, device, recon_enabled)
            doc_vecs.append(dvec.cpu())
            for j in range(dvec.size(0)):
                gi = batch_idx * batch_size + j
                if gi < len(cached):
                    doc_kinds.append(_extract_kind(cached[gi]))
    torch.save({
        "doc_vecs": torch.cat(doc_vecs, dim=0),
        "kinds": doc_kinds,
    }, output_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-docs", type=int, default=5000)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reconstruction", choices=["on", "off"], default="off")
    parser.add_argument("--recon-weight", type=float, default=0.5)
    parser.add_argument("--dump-every-n-epochs", type=int, default=1,
                        help="dump doc_vecs every N epochs (final epoch always "
                             "dumped). 1=every epoch; 5=every 5 epochs + final.")
    args = parser.parse_args()

    recon_enabled = args.reconstruction == "on"

    import random
    import numpy as np
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    cache_path = os.path.join(args.output_dir, "doc_cache.pkl")
    # Convention: --max-docs 0 means "use the full corpus" (matches scripts/train.py)
    max_docs = None if args.max_docs == 0 else args.max_docs
    print(f"Step 0: Linearize → {cache_path} (max_docs={max_docs or 'all'})")
    cached = build_or_load_cache(DATASET_NAME, cache_path=cache_path,
                                 max_docs=max_docs)

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
    print(f"  reconstruction: {args.reconstruction} (weight={args.recon_weight})")

    print("Step 2: Build dataset (train: 90%, val: 10%)")
    config = YamlBertConfig(num_epochs=args.epochs, batch_size=args.batch_size,
                            recon_enabled=recon_enabled,
                            recon_loss_weight=args.recon_weight)
    full_dataset = YamlBertDataset(cached, vocab, config)
    # Val size: ~10% capped at 2000 (prevents 30K val passes from dominating
    # full-corpus runs). At 5K corpus → 500; at 276K corpus → 2000.
    val_size = max(1, min(2000, len(cached) // 10))
    train_indices = list(range(len(cached) - val_size))
    val_indices = list(range(len(cached) - val_size, len(cached)))
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    print(f"  train: {len(train_indices)}, val: {len(val_indices)}")

    print("Step 3: Build model")
    emb = YamlBertEmbedding(config=config,
                            key_vocab_size=vocab.key_vocab_size,
                            value_vocab_size=vocab.value_vocab_size)
    model = YamlBertModel(config=config, embedding=emb,
                    atomic_vocab_size=vocab.atomic_target_vocab_size)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  device: {device}")
    print(f"  params: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)

    num_workers = min(8, max(2, (os.cpu_count() or 4) // 2))
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, collate_fn=collate_fn,
                              num_workers=num_workers, persistent_workers=True,
                              pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, collate_fn=collate_fn,
                            num_workers=num_workers, pin_memory=True)

    print("Step 4: Training")
    train_start = time.time()
    epoch_log: list[dict] = []
    for epoch in range(args.epochs):
        epoch_start = time.time()
        model.train()
        sums = {"total": 0.0, "mlm": 0.0, "recon": 0.0}
        n_batches = 0
        # tqdm with mininterval=30 keeps jl-logs readable on long runs —
        # one line every 30 seconds, not per batch.
        train_iter = tqdm(
            train_loader,
            desc=f"epoch {epoch+1}/{args.epochs} train",
            mininterval=2,
            dynamic_ncols=True,
        )
        for batch in train_iter:
            optimizer.zero_grad()
            out = _forward_v8(model, batch, device, recon_enabled)
            total_loss, mlm_loss, recon_loss = _compute_losses(
                out, batch, device, recon_enabled, args.recon_weight,
            )
            if not torch.isfinite(total_loss):
                print(f"  !! NaN/Inf loss at batch {n_batches + 1}; "
                      f"stopping before backward to avoid corrupting weights")
                return
            if total_loss.requires_grad:
                total_loss.backward()
                optimizer.step()
            sums["total"] += total_loss.item()
            sums["mlm"] += mlm_loss.item()
            sums["recon"] += recon_loss.item()
            n_batches += 1
            # Live loss in tqdm postfix
            if n_batches % 50 == 0:
                train_iter.set_postfix(
                    mlm=f"{sums['mlm']/n_batches:.3f}",
                    recon=f"{sums['recon']/n_batches:.3f}",
                )

        avg = {k: v / max(1, n_batches) for k, v in sums.items()}

        # Validation pass (no_grad)
        model.eval()
        val_sums = {"total": 0.0, "mlm": 0.0, "recon": 0.0}
        n_val = 0
        val_iter = tqdm(
            val_loader,
            desc=f"epoch {epoch+1}/{args.epochs} val",
            mininterval=2,
            dynamic_ncols=True,
        )
        with torch.no_grad():
            for vb in val_iter:
                out = _forward_v8(model, vb, device, recon_enabled)
                tl, ml, rl = _compute_losses(
                    out, vb, device, recon_enabled, args.recon_weight,
                )
                val_sums["total"] += tl.item()
                val_sums["mlm"] += ml.item()
                val_sums["recon"] += rl.item()
                n_val += 1
        val_avg = {k: v / max(1, n_val) for k, v in val_sums.items()}

        epoch_dur = time.time() - epoch_start
        print(
            f"  Epoch {epoch+1}/{args.epochs} — "
            f"train total {avg['total']:.4f} mlm {avg['mlm']:.4f} "
            f"recon {avg['recon']:.4f}  |  "
            f"val total {val_avg['total']:.4f} mlm {val_avg['mlm']:.4f} "
            f"recon {val_avg['recon']:.4f}  "
            f"({n_batches} batches, {epoch_dur:.1f}s, "
            f"{n_batches/epoch_dur:.2f} it/s)"
        )
        epoch_log.append({"epoch": epoch + 1, "train": avg, "val": val_avg,
                          "dur_sec": epoch_dur, "n_batches": n_batches})

        # Doc_vec dump: every N epochs (default 1 = every epoch). Final epoch
        # always dumped regardless of cadence.
        is_dump_epoch = (
            (epoch + 1) % args.dump_every_n_epochs == 0
            or (epoch + 1) == args.epochs
        )
        if is_dump_epoch:
            dump_path = os.path.join(
                args.output_dir, f"doc_vecs_epoch_{epoch+1}.pt",
            )
            _dump_doc_vecs(model, full_dataset, args.batch_size, device,
                           recon_enabled, dump_path, cached, num_workers)

    total_dur = time.time() - train_start
    print(f"Step 5: Save final checkpoint")
    ckpt_path = os.path.join(args.output_dir, "v8_phase1_recon.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "epoch_log": epoch_log,
        "n_params": n_params,
        "total_train_sec": total_dur,
        "reconstruction": args.reconstruction,
        "recon_weight": args.recon_weight,
    }, ckpt_path)

    print(f"Done. Total: {total_dur:.1f}s")


if __name__ == "__main__":
    main()
