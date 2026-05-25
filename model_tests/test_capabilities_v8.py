"""V8 capability tests: same 93 test cases as v7's test_capabilities.py,
adapted to v8's V8Model + atomic-prediction head.

V7 uses compound bigram/trigram prediction (vocab ~28K). V8 uses atomic
prediction (vocab ~427). The test cases' `expect_in_top5` lists are already
atomic keys (after v7's `_extract_key_from_target` strips paths), so the
expectations work as-is — we just need a v8-specific predict path.

Usage:
    python model_tests/test_capabilities_v8.py \\
        output_v8_276K_seed42/v8_phase1_recon.pt \\
        --vocab output_v8_276K_seed42/vocab.json
"""
from __future__ import annotations
import _setup_path  # noqa: F401

import argparse

import torch
import torch.nn.functional as F

from model_tests.test_capabilities import (
    build_capabilities,
    _check_assertions,
)
from yaml_bert.annotator import DomainAnnotator
from yaml_bert.config import YamlBertConfig
from yaml_bert.embedding import YamlBertEmbedding
from yaml_bert.linearizer import YamlLinearizer
from yaml_bert.types import NodeType
from yaml_bert.v8_dataset import V8Dataset, v8_collate_fn
from yaml_bert.v8_model import V8Model
from yaml_bert.vocab import Vocabulary


def run_v8_test(model: V8Model, vocab: Vocabulary, config: YamlBertConfig,
                test):
    """Run a single capability test against V8Model. Returns TestResult."""
    from model_tests.test_capabilities import TestResult

    linearizer = YamlLinearizer()
    annotator = DomainAnnotator()

    nodes = linearizer.linearize(test.yaml_text)
    if not nodes:
        return TestResult(test.name, False, "Failed to parse YAML", [])
    annotator.annotate(nodes)

    # Find mask position by first occurrence of mask_token
    mask_pos = -1
    for i, n in enumerate(nodes):
        if n.token == test.mask_token and mask_pos == -1:
            mask_pos = i
            break
    if mask_pos == -1:
        return TestResult(test.name, False,
                          f"Token '{test.mask_token}' not found", [])

    # Build a single-doc batch via V8Dataset + v8_collate_fn so we get all
    # the precompute tensors (parent_of_tensor, edges_by_depth, etc.) the
    # vectorized V8Model.forward expects.
    ds = V8Dataset([nodes], vocab, config)
    item = ds[0]

    # Apply mask AFTER dataset construction (dataset's MLM is disabled via
    # config.mask_prob=0.0; we mask exactly the test's target position)
    mask_id = vocab.special_tokens["[MASK]"]
    item["token_ids"] = item["token_ids"].clone()
    item["token_ids"][mask_pos] = mask_id

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

    probs = F.softmax(logits[0, mask_pos], dim=-1)
    topk = probs.topk(10)

    # Build atomic-vocab reverse map
    id_to_atomic: dict[int, str] = {
        v: k for k, v in vocab.atomic_target_vocab.items()
    }
    for tok, tok_id in vocab.special_tokens.items():
        id_to_atomic[tok_id] = tok

    predictions: list[tuple[str, float]] = []
    for i in range(10):
        idx = topk.indices[i].item()
        key = id_to_atomic.get(idx, f"[ID:{idx}]")
        predictions.append((key, topk.values[i].item()))

    return _check_assertions(test, predictions)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=str)
    parser.add_argument("--vocab", type=str, required=True)
    parser.add_argument("--verbose", "-v", action="store_true")
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

    capabilities = build_capabilities()

    print("=" * 70)
    print(f"V8 Capability Tests — checkpoint: {args.checkpoint}")
    print(f"Atomic vocab size: {vocab.atomic_target_vocab_size}")
    print(f"Total capabilities: {len(capabilities)} "
          f"(pretrain: {sum(1 for c in capabilities if c.phase == 'pretrain')}, "
          f"finetune: {sum(1 for c in capabilities if c.phase == 'finetune')})")
    print(f"Total test cases: {sum(len(c.tests) for c in capabilities)}")
    print("=" * 70)

    pretrain_pass = pretrain_total = 0
    finetune_pass = finetune_total = 0
    pretrain_caps_pass = pretrain_caps_total = 0
    finetune_caps_pass = finetune_caps_total = 0

    for cap in capabilities:
        cap_results = [run_v8_test(model, vocab, config, t) for t in cap.tests]
        n_pass = sum(1 for r in cap_results if r.passed)
        n_total = len(cap_results)
        all_pass = n_pass == n_total
        partial = 0 < n_pass < n_total
        status = "   PASS" if all_pass else ("PARTIAL" if partial else "   FAIL")
        # Phase: filter pretrain by skipping finetune-only capabilities
        if cap.phase == "pretrain":
            pretrain_pass += n_pass
            pretrain_total += n_total
            pretrain_caps_total += 1
            if all_pass:
                pretrain_caps_pass += 1
        else:
            finetune_pass += n_pass
            finetune_total += n_total
            finetune_caps_total += 1
            if all_pass:
                finetune_caps_pass += 1

        print(f"    [{status}] {cap.name}: {n_pass}/{n_total} "
              f"({100*n_pass/n_total:.0f}%)")
        if args.verbose or not all_pass:
            for r in cap_results:
                if not r.passed or args.verbose:
                    print(f"        {'OK ' if r.passed else 'FAIL'} "
                          f"{r.test_name}: {r.details}")

    print()
    print("=" * 70)
    print(f"Pre-training: {pretrain_caps_pass}/{pretrain_caps_total} "
          f"capabilities, {pretrain_pass}/{pretrain_total} tests "
          f"({100*pretrain_pass/max(1,pretrain_total):.1f}%)")
    print(f"Fine-tuning:  {finetune_caps_pass}/{finetune_caps_total} "
          f"capabilities, {finetune_pass}/{finetune_total} tests "
          f"({100*finetune_pass/max(1,finetune_total):.1f}%)")
    print("=" * 70)


if __name__ == "__main__":
    main()
