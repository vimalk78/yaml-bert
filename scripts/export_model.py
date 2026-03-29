"""Export a training checkpoint to a clean inference-ready model file.

Strips optimizer state, adds config metadata.

Usage:
    python scripts/export_model.py output_v1/checkpoints/yaml_bert_epoch_15.pt
    python scripts/export_model.py output_v1/checkpoints/yaml_bert_epoch_15.pt --output output_v1/yaml_bert_v1_final.pt
"""
from __future__ import annotations
import _setup_path  # noqa: F401

import argparse
import os

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description="Export checkpoint to clean model file")
    parser.add_argument("checkpoint", type=str, help="Path to training checkpoint")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path (default: same dir as checkpoint, named yaml_bert_final.pt)")
    args = parser.parse_args()

    print(f"Loading checkpoint: {args.checkpoint}")
    cp: dict = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    epoch: int = cp["epoch"]
    model_state: dict = cp["model_state_dict"]
    num_params: int = sum(p.numel() for p in model_state.values())

    # Print kind embedding info if present
    kind_emb_key: str = "embedding.kind_embedding.weight"
    if kind_emb_key in model_state:
        kind_vocab_size: int = model_state[kind_emb_key].shape[0]
        print(f"Kind embedding: {kind_vocab_size} kinds")

    # Determine output path
    if args.output:
        output_path: str = args.output
    else:
        output_dir: str = os.path.dirname(args.checkpoint)
        output_path = os.path.join(output_dir, "yaml_bert_final.pt")

    # Save clean model
    torch.save({
        "model_state_dict": model_state,
        "epoch": epoch,
        "parameters": num_params,
    }, output_path)

    checkpoint_size: float = os.path.getsize(args.checkpoint) / 1024 / 1024
    final_size: float = os.path.getsize(output_path) / 1024 / 1024

    print(f"Epoch: {epoch}")
    print(f"Parameters: {num_params:,}")
    print(f"Checkpoint size: {checkpoint_size:.1f} MB")
    print(f"Final model size: {final_size:.1f} MB")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
