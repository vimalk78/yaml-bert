"""Simple ablation runner - runs full capability tests on each variant."""
import argparse
import json
from pathlib import Path
import torch
from yaml_bert.config import YamlBertConfig
from yaml_bert.embedding import YamlBertEmbedding
from yaml_bert.model import YamlBertModel
from yaml_bert.vocab import Vocabulary
from model_tests.test_capabilities import evaluate_model
def load_model(checkpoint_path: str, vocab: Vocabulary, pos_encoding: str):
    config = YamlBertConfig()
    config.pos_encoding = pos_encoding
    emb = YamlBertEmbedding(
        config=config,
        key_vocab_size=vocab.key_vocab_size,
        value_vocab_size=vocab.value_vocab_size,
    )
    model = YamlBertModel(
        config=config,
        embedding=emb,
        simple_vocab_size=len(vocab.simple_target_vocab),
        kind_vocab_size=len(vocab.kind_target_vocab),
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = {k: v for k, v in checkpoint["model_state_dict"].items() 
                 if not k.startswith(("simple_head", "kind_head"))}
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="output_v5/checkpoints/yaml_bert_v4_epoch_30.pt")
    parser.add_argument("--vocab", default="output_v5/vocab.json")
    args = parser.parse_args()
    vocab = Vocabulary.load(args.vocab)
    variants = ["full_tree", "depth_only", "sibling_only"]
    results = {}
    print("Running ablation study (this will take ~2-3 minutes)...\n")
    for variant in variants:
        print(f"→ {variant}")
        model = load_model(args.checkpoint, vocab, variant)
        score = evaluate_model(model, vocab, verbose=False, print_header=False)
        results[variant] = score
        print(f"   Pre-training: {score.get('pre_training_passed', 0)}/{score.get('pre_training_total', 28)}")
        print(f"   Overall: {score.get('overall_rate', 0):.1%}\n")
    Path("docs").mkdir(exist_ok=True)
    with open("docs/ablation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Done. Results saved to docs/ablation_results.json")
    print("You can now update docs/ablation_results.md with these numbers.")
if __name__ == "__main__":
    main()
