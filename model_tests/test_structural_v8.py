"""V8 structural tests: same 9 tests as test_structural.py, adapted to V8Model.

V8 uses atomic-vocab prediction (~427 tokens) instead of v7's compound-vocab.
Atomic predictions are already raw key names — no path stripping needed.

Usage:
    python model_tests/test_structural_v8.py \\
        output_v8_phase1_control/v8_phase1_recon.pt \\
        --vocab output_v8_phase1_control/vocab.json
"""
from __future__ import annotations
import _setup_path  # noqa: F401

import argparse

import torch
import torch.nn.functional as F

from yaml_bert.annotator import DomainAnnotator
from yaml_bert.config import YamlBertConfig
from yaml_bert.embedding import YamlBertEmbedding
from yaml_bert.linearizer import YamlLinearizer
from yaml_bert.v8_dataset import V8Dataset, v8_collate_fn
from yaml_bert.v8_model import V8Model
from yaml_bert.vocab import Vocabulary

from model_tests._cases_structural import run_tests, print_predictions  # noqa: F401


def predict_masked_key_v8(
    model: V8Model,
    vocab: Vocabulary,
    config: YamlBertConfig,
    yaml_text: str,
    mask_position: int,
    k: int = 10,
) -> list[tuple[str, float]]:
    """Mask key at position, return top-k (atomic_key, prob) tuples."""
    linearizer = YamlLinearizer()
    annotator = DomainAnnotator()

    nodes = linearizer.linearize(yaml_text)
    annotator.annotate(nodes)

    # Build single-doc batch via V8Dataset + v8_collate_fn so we get all
    # precomputed tensors (parent_of_tensor, edges_by_depth, etc.) that
    # the vectorised V8Model.forward expects.
    ds = V8Dataset([nodes], vocab, config)
    item = ds[0]

    # Apply mask AFTER dataset construction (mask_prob=0.0 means no random masking)
    mask_id = vocab.special_tokens["[MASK]"]
    item["token_ids"] = item["token_ids"].clone()
    item["token_ids"][mask_position] = mask_id

    batch = v8_collate_fn([item])

    model.eval()
    with torch.no_grad():
        out = model(
            token_ids=batch["token_ids"],
            node_types=batch["node_types"],
            depths=batch["depths"],
            sibling_indices=batch["sibling_indices"],
            batch_info=batch["batch_info"],
            padding_mask=batch["padding_mask"],
            parent_of_tensor=batch["parent_of_tensor"],
            top_level_key_mask=batch["top_level_key_mask"],
            edges_by_depth=batch["edges_by_depth"],
            parents_by_depth=batch["parents_by_depth"],
        )
    # V8Model returns (logits, doc_vec) or (logits, doc_vec, recon_logits)
    logits = out[0]  # (1, N, V_atomic)

    probs = F.softmax(logits[0, mask_position], dim=-1)
    topk = probs.topk(k)

    # Atomic-vocab reverse map — predictions are already raw key names
    id_to_atomic: dict[int, str] = {v: k for k, v in vocab.atomic_target_vocab.items()}
    for tok, tok_id in vocab.special_tokens.items():
        id_to_atomic[tok_id] = tok

    return [
        (id_to_atomic.get(topk.indices[i].item(), f"[ID:{topk.indices[i].item()}]"),
         topk.values[i].item())
        for i in range(k)
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=str)
    parser.add_argument("--vocab", type=str, required=True)
    args = parser.parse_args()

    vocab = Vocabulary.load(args.vocab)
    config = YamlBertConfig(v8_mode=True, recon_enabled=False, mask_prob=0.0)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    torch.manual_seed(42)
    emb = YamlBertEmbedding(
        config=config,
        key_vocab_size=vocab.key_vocab_size,
        value_vocab_size=vocab.value_vocab_size,
    )
    model = V8Model(
        config=config,
        embedding=emb,
        atomic_vocab_size=vocab.atomic_target_vocab_size,
    )
    state = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state)
    model.eval()

    print(f"V8 Model loaded. Atomic vocab size: {vocab.atomic_target_vocab_size}\n")

    def predict_fn(yaml_text: str, mask_position: int) -> list[tuple[str, float]]:
        return predict_masked_key_v8(model, vocab, config, yaml_text, mask_position)

    run_tests(predict_fn)


if __name__ == "__main__":
    main()
