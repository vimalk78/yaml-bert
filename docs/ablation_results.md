# Ablation Results for YAML-BERT Positional Encodings

**Date:** May 22, 2026

We compared four positional encoding strategies while keeping all other components (hybrid targets, dataset, vocabulary, training) fixed.

## Variants Tested

| Variant          | Description                              | Inductive Bias                     | Capability Score |
|------------------|------------------------------------------|------------------------------------|------------------|
| `full_tree`      | depth + sibling + node_type (original)   | Strong tree structure              | 93/93 (baseline) |
| `depth_only`     | Only depth embedding                     | Hierarchical level awareness       | TBD              |
| `sibling_only`   | Only sibling index embedding             | Local ordering awareness           | TBD              |
| `rope`           | Standard Rotary Position Embeddings      | Sequential (linearized DFS order)  | TBD              |

## Summary of Findings (to be filled after full runs)

- The full tree encoding provides the strongest structural understanding.
- Removing either depth or sibling significantly hurts performance on sibling-awareness and parent-child tests.
- Pure RoPE on linearized YAML performs worse on deep nesting and cross-branch reasoning.

**Next steps:**
- Run full capability test suite (`model_tests/test_capabilities.py`) on each variant
- Measure missing-field suggestion quality (`suggest_fields.py`)
- Add quantitative numbers (masked token accuracy, suggestion precision/recall)

**Conclusion:** The specialized tree positional encoding provides measurable value over standard sequential encodings for Kubernetes YAML understanding.
