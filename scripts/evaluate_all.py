"""Run all evaluations on a YAML-BERT checkpoint.

Combines: accuracy evaluation, structural tests, attention visualization,
tree embedding visualization, and embedding analysis.

Usage:
    python evaluate_all.py output_v1/checkpoints/yaml_bert_epoch_10.pt
    python evaluate_all.py output_v1/checkpoints/yaml_bert_epoch_15.pt --output-dir output_v1/eval_epoch15
"""
from __future__ import annotations
import _setup_path  # noqa: F401

import argparse
import os
import sys
import time


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run all YAML-BERT evaluations")
    parser.add_argument("checkpoint", type=str, help="Path to checkpoint .pt file")
    parser.add_argument("--vocab", type=str, default="output_v1/vocab.json")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: output_v1/eval_epoch_N)")
    parser.add_argument("--max-eval-docs", type=int, default=500,
                        help="Max docs for accuracy evaluation")
    parser.add_argument("--skip-accuracy", action="store_true",
                        help="Skip accuracy eval (slow on large datasets)")
    parser.add_argument("--skip-attention", action="store_true",
                        help="Skip attention visualizations")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    import torch
    checkpoint: dict = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    epoch: int = checkpoint["epoch"]

    if args.output_dir is None:
        args.output_dir = f"output_v1/eval_epoch_{epoch}"
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"{'=' * 70}")
    print(f"YAML-BERT Evaluation Suite — Epoch {epoch}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Output: {args.output_dir}")
    print(f"{'=' * 70}")

    from yaml_bert.config import YamlBertConfig
    from yaml_bert.annotator import DomainAnnotator
    from yaml_bert.dataset import YamlDataset, _extract_kind
    from yaml_bert.embedding import YamlBertEmbedding
    from yaml_bert.evaluate import YamlBertEvaluator
    from yaml_bert.linearizer import YamlLinearizer
    from yaml_bert.model import YamlBertModel
    from yaml_bert.visualize import plot_accuracy, plot_embedding_similarity
    from yaml_bert.vocab import Vocabulary
    from yaml_bert.types import NodeType
    import torch.nn.functional as F

    vocab: Vocabulary = Vocabulary.load(args.vocab)
    config: YamlBertConfig = YamlBertConfig()

    emb = YamlBertEmbedding(config=config, key_vocab_size=vocab.key_vocab_size, value_vocab_size=vocab.value_vocab_size, kind_vocab_size=vocab.kind_vocab_size)
    model = YamlBertModel(config=config, embedding=emb, key_vocab_size=vocab.key_vocab_size, kind_vocab_size=vocab.kind_vocab_size)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.eval()

    linearizer: YamlLinearizer = YamlLinearizer()
    annotator: DomainAnnotator = DomainAnnotator()

    results_file = open(os.path.join(args.output_dir, "results.txt"), "w")

    def log(msg: str) -> None:
        print(msg)
        results_file.write(msg + "\n")

    # ==============================================================
    # 1. PREDICTION ACCURACY
    # ==============================================================
    if not args.skip_accuracy:
        log(f"\n{'=' * 70}")
        log("1. PREDICTION ACCURACY")
        log(f"{'=' * 70}")
        start: float = time.time()

        dataset: YamlDataset = YamlDataset.from_huggingface(
            "substratusai/the-stack-yaml-k8s",
            vocab=vocab, linearizer=linearizer, annotator=annotator,
            config=config, max_docs=args.max_eval_docs,
        )

        evaluator: YamlBertEvaluator = YamlBertEvaluator(model=model, dataset=dataset, vocab=vocab)
        accuracy = evaluator.evaluate_prediction_accuracy()

        log(f"Top-1 accuracy: {accuracy['top1_accuracy']:.2%}")
        log(f"Top-5 accuracy: {accuracy['top5_accuracy']:.2%}")
        log(f"Total masked: {accuracy['total_masked']}")
        log(f"Eval time: {time.time() - start:.1f}s")

        plot_accuracy(accuracy, output_path=os.path.join(args.output_dir, "accuracy.png"))

        # Embedding analysis
        embeddings = evaluator.analyze_embeddings()
        for entry in embeddings:
            log(f"  {entry['key']}: ({entry['position_a']}) vs ({entry['position_b']}) "
                f"cosine_sim={entry['cosine_similarity']:.4f}")
        plot_embedding_similarity(embeddings, output_path=os.path.join(args.output_dir, "embedding_similarity.png"))

        # Sample predictions
        log("\nSample predictions:")
        for doc_idx in range(min(3, len(dataset))):
            predictions = evaluator.top_k_predictions(doc_idx=doc_idx, k=5)
            if predictions:
                log(f"\n  Document {doc_idx}:")
                for pred in predictions[:3]:
                    log(f"    Position {pred['position']}: true='{pred['true_key']}'")
                    for i, pk in enumerate(pred["predicted_keys"]):
                        marker = " <--" if pk["key"] == pred["true_key"] else ""
                        log(f"      {i+1}. '{pk['key']}' ({pk['probability']:.2%}){marker}")

    # ==============================================================
    # 2. STRUCTURAL TESTS
    # ==============================================================
    log(f"\n{'=' * 70}")
    log("2. STRUCTURAL TESTS")
    log(f"{'=' * 70}")

    def predict_masked(yaml_text: str, mask_token: str, k: int = 5) -> list[tuple[str, float]]:
        nodes = linearizer.linearize(yaml_text)
        annotator.annotate(nodes)
        type_map = {NodeType.KEY: 0, NodeType.VALUE: 1, NodeType.LIST_KEY: 2, NodeType.LIST_VALUE: 3}
        token_ids, node_types, depths, siblings, parent_keys = [], [], [], [], []
        mask_pos: int = -1

        for i, node in enumerate(nodes):
            if node.node_type in (NodeType.KEY, NodeType.LIST_KEY):
                token_ids.append(vocab.encode_key(node.token))
            else:
                token_ids.append(vocab.encode_value(node.token))
            node_types.append(type_map[node.node_type])
            depths.append(min(node.depth, 15))
            siblings.append(min(node.sibling_index, 31))
            parent_keys.append(vocab.encode_key(Vocabulary.extract_parent_key(node.parent_path)))

            if node.token == mask_token and mask_pos == -1:
                mask_pos = i

        if mask_pos == -1:
            return []

        token_ids[mask_pos] = vocab.special_tokens["[MASK]"]
        kind: str = _extract_kind(nodes)
        kind_id: int = vocab.encode_kind(kind)
        kind_ids: list[int] = [kind_id] * len(nodes)
        t = lambda x: torch.tensor([x])
        with torch.no_grad():
            logits, _, _ = model(t(token_ids), t(node_types), t(depths), t(siblings), t(parent_keys), kind_ids=t(kind_ids))
        probs = F.softmax(logits[0, mask_pos], dim=-1)
        topk = probs.topk(k)
        return [(vocab.decode_key(topk.indices[i].item()), topk.values[i].item()) for i in range(k)]

    tests: list[dict] = [
        {
            "name": "Kind conditioning: Deployment replicas",
            "yaml": "apiVersion: apps/v1\nkind: Deployment\nmetadata:\n  name: test\nspec:\n  replicas: 3\n  selector:\n    matchLabels:\n      app: test\n",
            "mask": "replicas",
            "expected": "replicas",
        },
        {
            "name": "Kind conditioning: Service type",
            "yaml": "apiVersion: v1\nkind: Service\nmetadata:\n  name: test\nspec:\n  type: ClusterIP\n  ports:\n  - port: 80\n",
            "mask": "type",
            "expected": "type",
        },
        {
            "name": "Wrong parent: containers under metadata",
            "yaml": "apiVersion: v1\nkind: Pod\nmetadata:\n  name: test\n  containers:\n  - name: nginx\nspec:\n  containers:\n  - name: nginx\n",
            "mask": "containers",
            "expected_not": "containers",
        },
        {
            "name": "Depth awareness: kind at depth 0",
            "yaml": "apiVersion: apps/v1\nkind: Deployment\nmetadata:\n  name: test\n",
            "mask": "kind",
            "expected": "kind",
        },
        {
            "name": "spec vs status: replicas under spec",
            "yaml": "apiVersion: apps/v1\nkind: Deployment\nmetadata:\n  name: test\nspec:\n  replicas: 3\n",
            "mask": "replicas",
            "expected": "replicas",
        },
        {
            "name": "Nonsense confidence drop",
            "yaml_valid": "apiVersion: v1\nkind: Pod\nmetadata:\n  name: test\nspec:\n  containers:\n  - name: nginx\n",
            "yaml_nonsense": "apiVersion: v1\nkind: Pod\nspec:\n  metadata:\n    containers:\n      replicas:\n        kind:\n          apiVersion: wrong\n",
            "mask": "containers",
            "type": "confidence_compare",
        },
    ]

    struct_passed: int = 0
    struct_total: int = 0

    for test in tests:
        struct_total += 1

        if test.get("type") == "confidence_compare":
            preds_valid = predict_masked(test["yaml_valid"], test["mask"])
            preds_nonsense = predict_masked(test["yaml_nonsense"], test["mask"])
            if preds_valid and preds_nonsense:
                v_conf = preds_valid[0][1]
                n_conf = preds_nonsense[0][1]
                passed = v_conf > n_conf
                if passed:
                    struct_passed += 1
                status = "PASS" if passed else "FAIL"
                log(f"\n  {test['name']}: {status}")
                log(f"    Valid: '{preds_valid[0][0]}' ({v_conf:.2%})")
                log(f"    Nonsense: '{preds_nonsense[0][0]}' ({n_conf:.2%})")
            continue

        preds = predict_masked(test["yaml"], test["mask"])
        if not preds:
            log(f"\n  {test['name']}: SKIP (token not found)")
            continue

        if "expected" in test:
            top5 = [k for k, _ in preds[:5]]
            passed = test["expected"] in top5
            if passed:
                struct_passed += 1
            status = "PASS" if passed else "FAIL"
            log(f"\n  {test['name']}: {status}")
        elif "expected_not" in test:
            passed = preds[0][0] != test["expected_not"]
            if passed:
                struct_passed += 1
            status = "PASS" if passed else "FAIL"
            log(f"\n  {test['name']}: {status}")

        for i, (key, prob) in enumerate(preds[:5]):
            log(f"    {i+1}. '{key}' ({prob:.2%})")

    log(f"\nStructural tests: {struct_passed}/{struct_total} passed")

    # ==============================================================
    # 3. TREE EMBEDDING VISUALIZATION
    # ==============================================================
    log(f"\n{'=' * 70}")
    log("3. TREE EMBEDDING VISUALIZATION")
    log(f"{'=' * 70}")

    tree_dir: str = os.path.join(args.output_dir, "tree_viz")
    os.makedirs(tree_dir, exist_ok=True)

    # Import and run tree visualization
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from visualize_tree import compute_embeddings, embeddings_to_colors, draw_tree

    deployment_yaml: str = """\
apiVersion: apps/v1
kind: Deployment
metadata:
  name: nginx
  labels:
    app: web
spec:
  replicas: 3
  selector:
    matchLabels:
      app: web
  template:
    metadata:
      labels:
        app: web
    spec:
      containers:
      - name: nginx
        image: nginx:1.21
        ports:
        - containerPort: 80
"""
    nodes = linearizer.linearize(deployment_yaml)
    annotator.annotate(nodes)
    all_embs = compute_embeddings(model, vocab, nodes)

    mode_titles = {
        "tree_pos": "Tree Positional Encoding (depth + sibling + type + parent_key)",
        "full": "Full Embedding (token + tree position)",
        "token_only": "Token Embedding Only",
        "depth": "Component: depth",
        "parent_key": "Component: parent_key",
        "node_type": "Component: node_type",
    }

    for mode, title in mode_titles.items():
        colors = embeddings_to_colors(all_embs[mode])
        output_path = os.path.join(tree_dir, f"{mode}.png")
        draw_tree(nodes, colors, title=title, output_path=output_path)

    log(f"Tree visualizations saved to: {tree_dir}/")

    # ==============================================================
    # 4. ATTENTION VISUALIZATION
    # ==============================================================
    if not args.skip_attention:
        log(f"\n{'=' * 70}")
        log("4. ATTENTION VISUALIZATION")
        log(f"{'=' * 70}")

        attn_dir: str = os.path.join(args.output_dir, "attention")
        os.makedirs(attn_dir, exist_ok=True)

        # Compute attention on the deployment
        from yaml_bert.types import NodeType as NT
        type_map = {NT.KEY: 0, NT.VALUE: 1, NT.LIST_KEY: 2, NT.LIST_VALUE: 3}
        token_ids, node_types, depths, siblings, parent_keys = [], [], [], [], []
        token_labels: list[str] = []

        for node in nodes:
            if node.node_type in (NT.KEY, NT.LIST_KEY):
                token_ids.append(vocab.encode_key(node.token))
            else:
                token_ids.append(vocab.encode_value(node.token))
            node_types.append(type_map[node.node_type])
            depths.append(min(node.depth, 15))
            siblings.append(min(node.sibling_index, 31))
            parent_keys.append(vocab.encode_key(Vocabulary.extract_parent_key(node.parent_path)))
            prefix = "=" if node.node_type in (NT.VALUE, NT.LIST_VALUE) else ""
            token_labels.append(f"{prefix}{node.token[:20]}")

        t = lambda x: torch.tensor([x])
        kind: str = _extract_kind(nodes)
        kind_id: int = vocab.encode_kind(kind)
        kind_ids: list[int] = [kind_id] * len(nodes)
        attn_weights = model.get_attention_weights(
            t(token_ids), t(node_types), t(depths), t(siblings), t(parent_keys), kind_ids=t(kind_ids)
        )

        # Find top patterns
        log("\nTop attention patterns (Deployment):")
        results_attn: list[tuple[float, int, int, int, int]] = []
        for layer_idx, layer_w in enumerate(attn_weights):
            w = layer_w[0]
            for head_idx in range(w.shape[0]):
                h = w[head_idx]
                off_diag = h.clone()
                off_diag.fill_diagonal_(0)
                max_val = off_diag.max().item()
                max_idx = off_diag.argmax().item()
                from_idx = max_idx // len(nodes)
                to_idx = max_idx % len(nodes)
                results_attn.append((max_val, layer_idx, head_idx, from_idx, to_idx))

        results_attn.sort(reverse=True)
        for val, li, hi, fi, ti in results_attn[:10]:
            log(f"  L{li}H{hi}: {token_labels[fi]} -> {token_labels[ti]} (attn={val:.3f})")

        # Save top 3 head plots
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        seen: set[tuple[int, int]] = set()
        for val, li, hi, fi, ti in results_attn[:5]:
            if (li, hi) in seen:
                continue
            seen.add((li, hi))
            weights = attn_weights[li][0][hi].cpu()
            fig, ax = plt.subplots(figsize=(12, 10))
            ax.imshow(weights.numpy(), cmap="Blues", vmin=0)
            ax.set_title(f"Layer {li}, Head {hi}", fontsize=14)
            ax.set_xticks(range(len(token_labels)))
            ax.set_yticks(range(len(token_labels)))
            ax.set_xticklabels(token_labels, rotation=45, ha="right", fontsize=8)
            ax.set_yticklabels(token_labels, fontsize=8)
            ax.set_xlabel("Attending to")
            ax.set_ylabel("Attending from")
            fig.tight_layout()
            fig.savefig(os.path.join(attn_dir, f"L{li}H{hi}.png"), dpi=150)
            plt.close(fig)

        log(f"Attention plots saved to: {attn_dir}/")

    # ==============================================================
    # SUMMARY
    # ==============================================================
    log(f"\n{'=' * 70}")
    log(f"EVALUATION COMPLETE — Epoch {epoch}")
    log(f"{'=' * 70}")
    if not args.skip_accuracy:
        log(f"Accuracy: Top-1={accuracy['top1_accuracy']:.2%}, Top-5={accuracy['top5_accuracy']:.2%}")
    log(f"Structural tests: {struct_passed}/{struct_total}")
    log(f"All outputs: {args.output_dir}/")

    results_file.close()
    print(f"\nResults saved to: {os.path.join(args.output_dir, 'results.txt')}")


if __name__ == "__main__":
    main()
