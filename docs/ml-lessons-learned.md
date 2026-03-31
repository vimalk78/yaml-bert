# Practical ML Lessons From Building a Transformer

Lessons from building a transformer model on structured data. Written for someone who knows the basics (embeddings, attention, backpropagation, gradient descent) and wants to understand the things you only learn by building something real.

---

## 1. Your Prediction Target Shapes What the Model Learns

This is the single most important lesson.

A transformer model optimizes ONE thing: minimize the loss on the prediction target. Everything it learns is in service of that goal. If the prediction target doesn't require certain knowledge, the model won't learn it — even if you put that knowledge directly in the input.

### Example: Simple vs Compound Targets

Suppose you're predicting a masked token in a document. If the target is just the token name:

```
Input: [context...] [MASK] [context...]
Target: "price"
```

Two documents from different domains (finance vs e-commerce) both have the token "price." The target is the same. The model has **zero incentive** to produce different internal representations for "price" in finance vs "price" in e-commerce — because both predict the same target correctly with the same hidden state.

If the target is compound:

```
Document A target: "finance::price"
Document B target: "ecommerce::price"
```

Now the model **must** produce different hidden states — because a single `Linear(d_model, vocab_size)` layer can only map different outputs from different inputs. The hidden states diverge by mathematical necessity.

### The Mathematical Argument

A prediction head is `y = Wx + b` where `W` is `(vocab_size, d_model)`. For two different target classes `y_a ≠ y_b` to get the highest score, the input vectors `x_a ≠ x_b` must differ. The model has no choice.

### Practical Implication

When designing your training objective, ask: "Does my target require the model to encode the information I care about?" If two different situations produce the same target, the model will learn to treat them identically — regardless of how much distinguishing information you put in the input.

---

## 2. Auxiliary Losses Can Be Trivially Easy

Adding extra losses to force the model to preserve information sounds principled. But if the auxiliary task can be solved from the residual stream with near-zero effort, it teaches nothing.

### How This Happens

In a transformer, the input embedding flows through residual connections:

```
input → Layer 0 → Layer 1 → ... → Layer N → output
  └──────────────────────────────────────────→ (residual)
```

Each layer ADDS to the residual stream. The original input information is never explicitly removed — it's still there, mixed with learned representations.

If your auxiliary loss asks "predict property X from the final hidden state," and property X was encoded in the input embedding, the model can solve it by reading X from the residual signal. No restructuring needed. Loss drops to near-zero in the first few batches.

### The Numbers

```
Auxiliary kind classification loss:  0.0007  (trivially zero)
Auxiliary parent classification loss: 0.0002  (trivially zero)
Main key prediction loss:            0.1093  (the real work)
```

The auxiliary losses contributed 0.07% of total gradient. They might as well not exist.

### When Auxiliary Losses DO Work

Auxiliary losses work when:
1. The task is **not solvable** from the existing representation (forces restructuring)
2. The information is **not directly available** in the input embedding
3. The task requires **combining information** that the model might otherwise discard

If you can solve the auxiliary task with a linear probe on the input embedding, the auxiliary loss won't teach the model anything new.

### Why Masking Prevents This for the Main Task

The main prediction target (masked token prediction) does NOT suffer from this problem. The token at the masked position is **replaced** with a `[MASK]` embedding — the real answer is removed from the input. There is nothing to leak through the residual. The model must attend to surrounding context to reconstruct the missing token.

Auxiliary losses on **unmasked** metadata (kind, parent key) are the ones that leak — the answer is sitting right there in the input embedding, carried through by the residual connection, ready to be read at the output.

The key distinction: masked inputs force real learning. Unmasked inputs enable residual shortcuts.

---

## 3. Information Degrades Through Transformer Layers

What you put in the input embedding may not survive to the final layer. The transformer overwrites the input representations with learned features it finds more useful for the training objective.

### Probing Reveals This

A linear probe is a small linear classifier trained on hidden states to check if specific information is encoded:

```
Layer    Depth info  Parent info  Type info
Embedding   87%        79%         95%     ← you put this in
Layer 0     80%        77%         81%
Layer 1     79%        73%         88%
Layer 2     71%        65%         68%
...
Layer 5     67%        63%         79%     ← the model kept this
```

Depth drops from 87% → 67%. Parent drops from 79% → 63%. The transformer is gradually replacing the explicit positional encoding with its own learned representation that works for the prediction task but isn't linearly decodable as "depth" anymore.

### Why This Happens

The model optimizes the prediction loss. If knowing depth explicitly doesn't help prediction (because the model found a shortcut that combines depth with other features), the dedicated "depth signal" gets overwritten with something more useful.

This is not a bug — it's the model finding efficient representations. But it means you can't assume the model "knows" what you told it in the input.

### Practical Implication

If you need the final representation to preserve specific input features, you must add a loss that penalizes their absence. But see Lesson 2 — make sure that loss isn't trivially easy.

---

## 4. LayerNorm Explains Similar Norms, Not Similar Directions

When you observe that all document embeddings have similar magnitudes AND similar directions (high cosine similarity), these are two different phenomena with different causes.

### LayerNorm → Similar Norms

LayerNorm normalizes each vector to mean≈0, std≈1 across the d_model dimensions. This means:

```
||v|| ≈ √d_model
```

For d_model=256: all vectors have norm ≈ 16. This is a mathematical consequence of normalization, not a learned behavior.

### Why √d_model? The Geometry

After LayerNorm, each of the d_model components has mean≈0 and variance≈1. The squared norm of the vector is the sum of squared components:

```
||v||² = v₁² + v₂² + ... + v_d²
```

Each vᵢ² has expected value ≈ 1 (variance=1, mean=0). So `||v||² ≈ d_model`, giving `||v|| ≈ √d_model`. Every vector lands at roughly the same distance from the origin — on the surface of a hypersphere of radius √d_model.

Note: the √dₖ divisor in the attention formula `softmax(QKᵀ/√dₖ)` from "Attention Is All You Need" is a **different concept**. That scaling prevents dot products from growing too large with dimensionality (which would saturate softmax and kill gradients). Both involve √dimensions, but for different reasons — one is a consequence of normalization, the other is deliberate scaling to stabilize training.

### Training Objective → Similar Directions

High cosine similarity (0.85-0.92) between different documents means the vectors point in similar directions. This is caused by the training objective:

If the prediction "the masked token here is `name`" is correct regardless of which document you're in, then the model produces similar hidden states for the `name` position across all documents. Why make them different if the target is the same?

The mean-pooled document embedding inherits this: shared tokens (common across documents) dominate the mean, pulling all document embeddings toward the same direction.

### Why Mean Pooling Makes Documents Look Similar

Consider two documents:

```
Deployment: [apiVersion, kind, metadata, name, spec, replicas, template, containers, image]
ConfigMap:  [apiVersion, kind, metadata, name, spec, data, key, value]
```

The first 5 tokens are shared and produce similar hidden states in both documents (Lesson 1 — same prediction target, same representation). The remaining tokens are unique to each document type.

```
Deployment mean = (5 shared vectors + 4 unique vectors) / 9
ConfigMap  mean = (5 shared vectors + 3 unique vectors) / 8
```

The shared vectors pull both means toward the same direction. The unique vectors pull in different directions but are outnumbered. Result: cosine similarity is high (0.89).

If documents had 50 unique tokens and only 5 shared, the unique tokens would dominate the mean and similarity would drop. The problem is specific to domains with a lot of shared structure (like Kubernetes YAMLs, where every resource has `apiVersion`, `kind`, `metadata`, `spec`).

### The Distinction Matters

If you only see similar norms → LayerNorm, not a problem.
If you see similar directions → the model isn't discriminating, which is a training objective problem (Lesson 1).

---

## 5. Per-Head Probing Is Misleading

When probing transformer attention heads individually, the accuracy is often very low. This does NOT mean the model doesn't encode that information.

### The Numbers

```
Best per-head accuracy for parent info:    4.5%  (looks terrible)
Full residual stream accuracy for parent:  79%   (actually fine)
```

### Why Per-Head is Low

Each head projects the d_model vector to d_model/num_heads dimensions. With d_model=256 and 8 heads, each head sees only 32 dimensions. Classifying 1600+ parent keys from 32 dimensions is nearly impossible for a linear probe.

The information is distributed ACROSS heads. Head 3 might carry 5% of the parent signal, head 7 another 5%, with the rest encoded in cross-head interactions. No single head is responsible.

### Why Residual Stream is High

The residual stream is the full d_model-dimensional vector. All head outputs are concatenated and projected back to d_model before being added to the residual. A linear probe on the full vector sees the combined signal from all heads.

### Practical Rule

- Per-head probing: useful for understanding **specialization** (does any head focus on one feature?)
- Residual stream probing: useful for understanding **information content** (does the model know X?)

Never conclude "the model doesn't know X" from per-head probing alone.

---

## 6. Vocabulary Size Directly Impacts GPU Memory

The most common OOM surprise. Model parameters are usually small. The logits tensor is what kills you.

### The Math

```
logits = model_output @ prediction_head_weight.T
shape: (batch_size, seq_len, vocab_size)
memory: batch_size × seq_len × vocab_size × 4 bytes (fp32)
```

Example:
```
batch=32, seq=512, vocab=1,664   →  103 MB  ✓ fits
batch=32, seq=512, vocab=72,000  → 4,500 MB  ✗ OOM on 4GB GPU
```

The model weights (7M params = 28 MB) are tiny compared to the logits tensor. Doubling the vocabulary from 5K to 10K doubles the logits memory.

### Parameters vs Activations: Where Memory Actually Goes

GPU memory holds two very different things:

1. **Parameters** — the weight matrices stored in the model. Fixed size, independent of batch. A `Linear(256, 200000)` layer stores a `(200000, 256)` matrix = 200 MB. Always in memory.
2. **Activations** — intermediate tensors created during the forward pass. Every matrix multiplication produces one. They scale with `batch_size × seq_len` and must be kept in memory until backward pass computes gradients through them.

Every layer produces activations: `(batch, seq, d_model)`. These are moderate (d_model=256). But the final logits tensor is `(batch, seq, vocab_size)` — and when vocab_size is 200K instead of 256, that one tensor can be 100x larger than all other activations combined.

This is why a model can "fit" in memory (parameters load fine) but OOM when you actually run a training batch through it. The parameters are the small, permanent residents. The activations are the temporary guests that trash the place.

### Backward Pass Is Worse

`loss.backward()` needs to store gradients for the logits tensor. Memory roughly doubles during backward. A model that fits in forward pass can OOM during backward.

### Variable Sequence Lengths

With dynamic batching (pad to longest sequence in batch), memory varies per batch. Most batches are fine, but one batch with a long document spikes the memory:

```
Batch 1-465: sequences 30-80 tokens  → 2.0 GB  ✓
Batch 466:   one sequence 200 tokens → 3.5 GB  ✗ OOM
```

This causes **sporadic OOM** — training runs fine for minutes then crashes randomly.

### Practical Rules

1. Estimate memory as `batch × seq_len × vocab × 4 × 2` (forward + backward)
2. If you add a new prediction head, the vocab of that head multiplies memory
3. Always test with the largest possible sequence in your dataset, not average
4. Reducing batch_size doesn't always help if one long sequence alone exceeds memory

---

## 7. Batch Size Doesn't Always Speed Up Training

Doubling the batch size halves the iterations per epoch. But if the GPU is already fully utilized, each iteration takes proportionally longer.

### The Math

```
batch_size=16: 6.0 it/sec × 16 docs/batch = 96 docs/sec
batch_size=24: 4.0 it/sec × 24 docs/batch = 96 docs/sec
batch_size=32: 3.0 it/sec × 32 docs/batch = 96 docs/sec
```

Same throughput. The GPU processes the same number of documents per second because it was already the bottleneck. Larger batches just mean fewer, larger matrix multiplications — same total FLOPS.

### When Batch Size DOES Help

- **CPU-bound data loading**: larger batch = fewer loader calls
- **GPU not fully utilized**: small models with small sequences leave GPU idle between batches
- **Multi-GPU**: distributed training needs larger batches to keep all GPUs busy

### When Batch Size Hurts

- **Memory**: larger batch = larger activations, gradients, logits
- **Generalization**: very large batches can converge to sharp minima that generalize poorly (debated, but observed in practice)

### Practical Rule

Measure throughput in **docs/sec** or **tokens/sec**, not iterations/sec. If doubling batch_size doesn't increase docs/sec, you're already GPU-bound.

---

## 8. Frozen Encoder Embeddings May Not Work for Downstream Tasks

A model trained for task A (e.g., masked token prediction) doesn't automatically produce good representations for task B (e.g., document similarity). This is a well-known problem, but easy to forget.

### Why It Fails

The encoder was optimized so that each **position** produces a hidden state useful for predicting the masked token at that position. It was NOT optimized to produce hidden states that, when averaged across positions, distinguish one document from another.

Mean-pooling hidden states gives you a vector dominated by common patterns. If all documents share common tokens (like boilerplate headers), the mean is similar across documents:

```
Cosine similarity between different document types:
  Document A vs Document B: 0.89  (should be < 0.5 for different types)
  Document A vs Document C: 0.91  (barely different)
```

### Contrastive Fine-Tuning Helps

This is what Sentence-BERT/Sentence-Transformers do: fine-tune the encoder (or a pooling layer on top) with a contrastive loss that pulls similar documents together and pushes dissimilar ones apart.

But: if the encoder's hidden states are too similar to begin with (cosine > 0.85 between different document types), even a pooling layer can't separate them — there's not enough variation in the input for the pooling layer to exploit.

### Fix at the Source

The better approach: change the training objective so the encoder MUST produce discriminative representations. If the prediction target differs by document type (compound targets), the hidden states will naturally diverge.

---

## 9. Non-Determinism Across Processes

A model can be perfectly deterministic within one Python process but produce different results across separate runs. This is confusing because the model weights are loaded from the same checkpoint.

### The Cause: `strict=False` Loading

When loading a checkpoint into a model with extra layers the checkpoint doesn't have:

```python
model.load_state_dict(checkpoint["model_state_dict"], strict=False)
```

The existing layers get correct weights from the checkpoint. The NEW layers (not in checkpoint) keep their **random initialization** — which differs every time the model is constructed, because PyTorch seeds differ across processes.

### The Fix

Set the random seed BEFORE constructing the model:

```python
torch.manual_seed(42)
model = MyModel(...)  # random layers get deterministic init
model.load_state_dict(checkpoint, strict=False)  # existing layers loaded
```

Now the "random" initialization is the same every process.

### Other Sources of Non-Determinism

1. **Multi-threaded BLAS**: parallel matrix operations can reduce in different orders → slightly different floating point results. Fix: `torch.set_num_threads(1)` (slow but deterministic)
2. **GPU atomicAdd**: CUDA operations accumulate gradients non-deterministically. Fix: `torch.use_deterministic_algorithms(True)`
3. **Data shuffling**: different random order per epoch. Fix: `torch.manual_seed(42)` before each epoch

---

## 10. When to Add Input Features vs Let the Model Learn

A key architecture decision: should you encode feature X explicitly in the input (as an embedding table), or let the model discover X from context via attention?

### Add It When:

- The feature is **not inferrable** from context (e.g., absolute position in a sequence — nearby tokens don't tell you your position number)
- The feature is **cheap to compute** but **expensive to learn** (e.g., tree depth — easy to count from data, hard to learn from attention patterns)
- You want the model to use it **from the start** of training (speeds up convergence)

### Don't Add It When:

- The feature is **redundant** with the prediction target (the model gets the same gradient signal from the loss)
- The feature is **always available** in the context (e.g., a document type token that's always unmasked — the model can attend to it)
- Adding it creates a **shortcut** that prevents the model from learning deeper patterns (the model copies from the input instead of understanding the structure)

### The Residual Problem

When you add feature X as an input embedding, the residual connections carry it through all layers. The model can "read" X from the residual at any layer without doing any work. If you also have a loss that asks "do you know X?" — the model solves it trivially from the residual (Lesson 2).

This means: explicit input features can make auxiliary losses useless. The input gives the answer away.

### Practical Rule

If you want the model to learn feature X from context (to develop deep understanding), DON'T put X in the input embedding AND put X in the prediction target. The target forces learning. The input would short-circuit it.

---

## 11. Testing ML Models: Capability-Based Testing

Unit tests verify code correctness. Capability tests verify model behavior. They're fundamentally different.

### The CheckList Approach

Define **capabilities** your model should have. Each capability has multiple test cases. Track pass/fail per capability:

```
Pre-training capabilities:
  [PASS] Parent-child structure: 8/8
  [PASS] Depth sensitivity: 3/3
  [PASS] Sibling awareness: 3/3
  ...

Fine-tuning capabilities (requires fine-tuned model):
  [PARTIAL] Invalid structure rejection: 6/11
```

### Why This Matters

A single accuracy number (95%) hides WHERE the model fails. Capability tests reveal:
- The model knows depth (100%) but can't distinguish resource types (40%)
- The model predicts common keys perfectly but fails on rare ones
- The model works for one domain (Deployments) but not another (CRDs)

### Separating Pre-Training from Fine-Tuning Tests

Some tests are UNFAIR for a pre-trained model. If the model was trained to predict valid structure, testing "can you reject invalid structure?" is testing a different task. Mark these as fine-tuning tests:

```
Pre-training: 28/28 capabilities passing  ← evaluate the model on these
Fine-tuning:  0/2 capabilities passing    ← save for after fine-tuning
```

### Ablation Tests

Replace a component with garbage and measure accuracy drop:

```
Normal (values present):   100% accuracy
Ablated (values = [UNK]):   97% accuracy
Impact:                      3% — values barely contribute
```

This tells you what the model actually uses, not what you think it uses.

---

## 12. The Pooling Problem: From Token Representations to Document Representations

Transformers produce per-token hidden states. Many downstream tasks need a single fixed-size vector per document. Getting from N token vectors to 1 document vector is the "pooling problem."

### Mean Pooling

Average all hidden states: `doc_emb = mean(hidden_states)`

Simple but treats every token equally. Common tokens (headers, boilerplate) dominate the mean, making all documents look similar.

### [CLS] Token

BERT prepends a special [CLS] (Classification) token. Its final hidden state is the document embedding. But it only works if the model was trained with a **document-level objective** that forces information into the CLS position.

In BERT, this objective is Next Sentence Prediction (NSP): given `[CLS] sentence_A [SEP] sentence_B [SEP]`, predict whether B actually follows A in the original text. The CLS hidden state feeds a `Linear(d_model, 2)` head for this binary classification. Because the loss backpropagates through CLS, the model learns to aggregate document-level information into that position via attention.

Without a document-level objective (e.g., MLM-only training), the CLS hidden state has no reason to summarize anything — it's just another token optimizing for local predictions.

### Attention Pooling

Mean pooling gives equal weight to every token. Attention pooling learns **which tokens matter** via a weighted average:

```python
query = nn.Parameter(torch.randn(1, d_model))     # fixed learned vector
keys = W_K @ hidden_states                         # input-dependent
values = W_V @ hidden_states                       # input-dependent
weights = softmax(query @ keys.T / √d_model)       # (1, seq_len)
doc_embedding = weights @ values                    # (1, d_model)
```

The learned query is fixed — it asks the same question of every document: "what's informative here?" But the keys and values are input-dependent, so each document produces different attention weights. Shared boilerplate tokens (`apiVersion`, `metadata`) can be downweighted while distinguishing tokens (`replicas`, `data`) get upweighted.

### Why a Fixed Query, Not a W_Q Matrix?

In self-attention, each of N tokens generates its own query via `W_Q` — the matrix has N different inputs. In attention pooling, there is only **one** query for the whole document. Applying `W_Q` to one fixed vector just produces another fixed vector: `W_Q @ learned_vector = another_learned_vector`. No added expressiveness — the result can be absorbed into the learned vector itself.

This is also why attention pooling is O(N), not O(N²) like self-attention. It computes just one row of the attention matrix.

The training is lightweight — just the query vector, `W_K`, `W_V`, and a contrastive loss (e.g., "Deployment docs should be closer to each other than to ConfigMap docs").

### The Fundamental Issue

If the encoder wasn't trained to produce discriminative per-token representations (Lesson 8), no pooling method can produce discriminative document embeddings. Fix the encoder first, then pool.

---

## Summary

| Lesson | One-Line Summary |
|--------|-----------------|
| 1 | Prediction target = what the model learns. Same target = same representation. |
| 2 | Auxiliary losses are useless if solvable from the residual stream. |
| 3 | Input features degrade through transformer layers. Don't assume preservation. |
| 4 | LayerNorm → similar norms. Training objective → similar directions. Different causes. |
| 5 | Per-head probing underestimates. Use residual stream probing for information content. |
| 6 | vocab_size × batch × seq_len = OOM. Vocabulary is the memory killer, not parameters. |
| 7 | Same docs/sec regardless of batch size if GPU-bound. Measure throughput, not iterations. |
| 8 | Token-level training ≠ good document embeddings. Fine-tune or fix the objective. |
| 9 | strict=False + random init = non-deterministic across processes. Seed before model creation. |
| 10 | Input features can short-circuit learning. Put info in the target, not the input, if you want the model to learn it. |
| 11 | Capability tests > accuracy numbers. Know WHERE the model fails, not just how often. |
| 12 | Pooling can't fix a non-discriminative encoder. Fix the encoder first. |
