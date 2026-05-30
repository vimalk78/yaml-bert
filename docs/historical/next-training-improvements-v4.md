# Next Training Run Improvements

Issues and ideas discovered during evaluation of the v4 model (epoch 15, 276K docs). Each item is a concrete change to make before the next training run.

---

## ~~1. Don't Apply min_freq to Kind Values~~ DONE

**Problem:** Rare kinds like `Binding`, `CSINode`, `Event`, `Eviction`, `FlowSchema`, `Lease`, `TokenReview` map to `[UNK]` in the value vocabulary because they appear fewer than 100 times. The model can't distinguish these kinds from each other — they all share the same `[UNK]` embedding.

**Why it matters:** Kind values are a small, closed set (~50-80 Kubernetes resource types). Every kind is meaningful. Unlike keys, there are no "junk" kinds to filter out.

**Fix:** Exempt kind values from min_freq filtering in the value vocabulary. During vocab building, collect all tokens that appear as values of the `kind` key at depth 0, and always include them in the value vocab regardless of frequency.

**Impact:** ~30 additional entries in the value vocab. Negligible memory cost. The model can now distinguish all kinds through their value embeddings.

---

## ~~2. Case Normalization for Kind Values~~ DONE

**Problem:** The value vocabulary has separate entries for `ConfigMap`, `configmap`, and `configMap`. These are the same kind but get different embeddings. The model can't connect them — a YAML with `kind: configmap` gets a completely different representation than `kind: ConfigMap`.

Other affected kinds: `Pod`/`pod`/`POD`, `Deployment`/`deployment`, `Service`/`service`, `Secret`/`secret`, `Job`/`job`, `Namespace`/`namespace`.

**Fix:** Normalize kind values to their canonical form during linearization or encoding. Map `configmap` → `ConfigMap`, `pod` → `Pod`, etc. This can be a lookup table of known kinds, or simply title-case the kind value.

**Impact:** Reduces value vocab size slightly. More importantly, all case variants of a kind now share one embedding and one set of trigram targets.

---

## 3. Sine/Cosine Initialization for Depth Embeddings

**Problem:** Learned depth embeddings are nearly orthogonal — adjacent depths (0 and 1, 2 and 3) have no more similarity than distant depths (0 and 10). The model treats each depth as an independent category with no notion of "nearby in the tree."

This may contribute to the wrong-sibling prediction weakness: the model can't distinguish children of `template` from children of `selector` when both are at the same depth, partly because depth embeddings carry no relational information.

**Experiment:** Initialize `depth_embedding.weight` with sine/cosine positional encoding (same as "Attention Is All You Need") instead of random. The model starts with "nearby depths are related" and can refine from there. Compare after same number of epochs:
- Embedding structure test: do adjacent depths stay more similar?
- Capability tests: same or better?
- Suggest tool: fewer wrong-sibling predictions?

**Implementation:** Replace random init with:
```python
def sincos_init(max_depth, d_model):
    pos = torch.arange(max_depth).unsqueeze(1).float()
    dim = torch.arange(0, d_model, 2).float()
    angles = pos / (10000 ** (dim / d_model))
    emb = torch.zeros(max_depth, d_model)
    emb[:, 0::2] = torch.sin(angles)
    emb[:, 1::2] = torch.cos(angles)
    return emb
```

---

## 4. More Training Epochs

**Problem:** Loss was still decreasing at epoch 15 (0.59), dropping ~0.01 per epoch. The model likely hasn't converged.

**Experiment:** Train for 30 epochs. Compare loss curve, capability tests, and suggest quality. If loss plateaus before 30, we know the optimal stopping point.

---

## 5. Expand Kind-Specific Targets Beyond Depth 1

**Problem:** The kind head only activates at depth 1 under non-universal root keys (`spec`, `data`, etc.). Deeper positions use the structure head with kind-independent bigrams. This means the model can't distinguish kind-specific patterns at depth 2+ (e.g., `Deployment::spec::template::spec` vs `StatefulSet::spec::template::spec`).

**Consideration:** Expanding trigrams to depth 2 would increase the kind target vocabulary significantly. Need to analyze whether kind-specific patterns actually differ at depth 2+ or whether bigrams are sufficient there. Run `analyze_kind_overlap.py` at depth 2 to check.

---

## 6. Tree-Aware Attention Bias

**Problem:** The model sometimes predicts keys for the wrong sibling branch (e.g., `selector::matchLabels` when probing under `template`). The model relies on attention to figure out parent-child relationships, but attention has no built-in tree structure.

**Experiment:** Add a tree distance bias to attention scores:
```
attention_score(i, j) = Q_i · K_j / √d + tree_bias(i, j)
```
Where `tree_bias` is a learned bias based on the tree relationship between positions i and j (parent, sibling, cousin, etc.). This gives the model a structural prior — nearby nodes in the tree attend to each other more strongly by default.

**Note:** This is a DeBERTa-style disentangled attention adaptation for tree structures. More complex to implement — defer until other improvements are tested.

---

## Priority Order

1. ~~**Kind value fixes** (items 1 and 2) — easy, high impact, vocab rebuild only~~ DONE (v5 training)
2. ~~**More epochs** (item 4) — just change a number~~ DONE (30 epochs on L4)
3. **Sine/cosine depth init** (item 3) — easy to implement, retrain needed
4. **Deeper trigrams** (item 5) — needs analysis first
5. **Tree attention bias** (item 6) — significant architecture change, defer
