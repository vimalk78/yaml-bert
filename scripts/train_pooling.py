"""Train the document pooling layer on the frozen encoder.

Usage:
    python scripts/train_pooling.py output_v1/yaml_bert_v1_final.pt --vocab output_v1/vocab.json --epochs 10
"""
from __future__ import annotations
import _setup_path  # noqa: F401

import argparse
import os
import time

import torch
from torch.optim import Adam

from yaml_bert.config import YamlBertConfig
from yaml_bert.embedding import YamlBertEmbedding
from yaml_bert.model import YamlBertModel
from yaml_bert.pooling import DocumentPooling, supervised_contrastive_loss
from yaml_bert.similarity import extract_hidden_states
from yaml_bert.vocab import Vocabulary
from yaml_bert.dataset import _extract_kind
from yaml_bert.linearizer import YamlLinearizer
from yaml_bert.annotator import DomainAnnotator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train document pooling layer")
    parser.add_argument("checkpoint", type=str, help="Frozen encoder checkpoint")
    parser.add_argument("--vocab", type=str, required=True)
    parser.add_argument("--data-dir", type=str, default=None, help="Local YAML dir")
    parser.add_argument("--hf-dataset", type=str, default="substratusai/the-stack-yaml-k8s")
    parser.add_argument("--max-docs", type=int, default=5000)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--output", type=str, default="pooling_layer.pt")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    torch.manual_seed(42)

    print("Loading frozen encoder...")
    vocab: Vocabulary = Vocabulary.load(args.vocab)
    config: YamlBertConfig = YamlBertConfig()
    emb = YamlBertEmbedding(
        config=config,
        key_vocab_size=vocab.key_vocab_size,
        value_vocab_size=vocab.value_vocab_size,
        kind_vocab_size=vocab.kind_vocab_size,
    )
    model = YamlBertModel(
        config=config, embedding=emb,
        key_vocab_size=vocab.key_vocab_size,
        kind_vocab_size=vocab.kind_vocab_size,
    )
    cp: dict = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(cp["model_state_dict"], strict=False)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    print(f"Encoder frozen (epoch {cp.get('epoch', '?')})")

    # Load documents
    print("Loading documents...")
    linearizer: YamlLinearizer = YamlLinearizer()
    annotator: DomainAnnotator = DomainAnnotator()

    yaml_texts: list[str] = []
    if args.data_dir:
        import glob
        for path in sorted(glob.glob(os.path.join(args.data_dir, "**", "*.yaml"), recursive=True)):
            with open(path) as f:
                yaml_texts.append(f.read())
    else:
        from datasets import load_dataset
        ds = load_dataset(args.hf_dataset, split="train")
        total: int = min(args.max_docs, len(ds))
        yaml_texts = [ds[i]["content"] for i in range(total)]

    # Pre-extract hidden states and kind labels (encoder is frozen)
    print(f"Extracting hidden states from {len(yaml_texts)} documents...")
    all_hidden: list[torch.Tensor] = []
    all_kind_hidden: list[torch.Tensor] = []
    all_kind_labels: list[int] = []
    skipped: int = 0

    for i, yaml_text in enumerate(yaml_texts):
        hidden, kind_pos = extract_hidden_states(model, vocab, yaml_text)
        if hidden.shape[0] == 0 or kind_pos < 0:
            skipped += 1
            continue

        nodes = linearizer.linearize(yaml_text)
        if not nodes:
            skipped += 1
            continue
        annotator.annotate(nodes)
        kind: str = _extract_kind(nodes)
        kind_id: int = vocab.encode_kind(kind)

        all_hidden.append(hidden)
        all_kind_hidden.append(hidden[kind_pos])
        all_kind_labels.append(kind_id)

        if (i + 1) % 1000 == 0:
            print(f"  {i + 1}/{len(yaml_texts)} extracted")

    print(f"Extracted {len(all_hidden)} documents ({skipped} skipped)")

    # Create pooling layer
    pooling: DocumentPooling = DocumentPooling(d_model=args.d_model, num_heads=args.num_heads)
    optimizer: Adam = Adam(pooling.parameters(), lr=args.lr)

    num_params: int = sum(p.numel() for p in pooling.parameters())
    print(f"Pooling layer: {num_params:,} parameters")

    # Training loop
    print(f"\nTraining for {args.epochs} epochs...")
    n: int = len(all_hidden)

    for epoch in range(args.epochs):
        pooling.train()
        # Shuffle
        perm: list[int] = torch.randperm(n).tolist()
        total_loss: float = 0.0
        num_batches: int = 0

        for start in range(0, n, args.batch_size):
            batch_idx: list[int] = perm[start:start + args.batch_size]
            if len(batch_idx) < 2:
                continue

            # Pad hidden states to same length
            batch_hidden: list[torch.Tensor] = [all_hidden[i] for i in batch_idx]
            max_len: int = max(h.shape[0] for h in batch_hidden)
            padded: torch.Tensor = torch.zeros(len(batch_idx), max_len, args.d_model)
            for j, h in enumerate(batch_hidden):
                padded[j, :h.shape[0]] = h

            kind_h: torch.Tensor = torch.stack([all_kind_hidden[i] for i in batch_idx]).unsqueeze(1)
            labels: torch.Tensor = torch.tensor([all_kind_labels[i] for i in batch_idx])

            optimizer.zero_grad()
            doc_embs: torch.Tensor = pooling(kind_h, padded)
            loss: torch.Tensor = supervised_contrastive_loss(doc_embs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss: float = total_loss / max(num_batches, 1)
        print(f"Epoch {epoch + 1}/{args.epochs} — loss: {avg_loss:.4f}")

    # Save
    torch.save({
        "pooling_state_dict": pooling.state_dict(),
        "d_model": args.d_model,
        "num_heads": args.num_heads,
    }, args.output)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
