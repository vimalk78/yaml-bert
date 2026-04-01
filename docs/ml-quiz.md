# Practical ML Quiz: Transformers and Training

12 questions testing practical understanding of transformer models. Each question describes a scenario and asks you to reason about what happens and why. No code required — just conceptual understanding.

Prerequisites: basic understanding of embeddings, attention, backpropagation, and gradient descent.

---

### Q1: Residual Connections and Auxiliary Losses

You're training a multilingual transformer. The input embedding includes a language ID embedding (English=0, French=1, Spanish=2). You add an auxiliary loss: "predict the language from the final hidden state."

The auxiliary loss drops to 0.001 within the first 100 steps.

**a)** Why did the auxiliary loss drop so fast?

**b)** A linear probe on the input embedding (before any transformer layers) achieves 99.5% language classification accuracy. What does this tell you about the auxiliary loss?

**c)** If you removed the language ID from the input but kept it in the auxiliary loss target, what would change?

---

### Q2: Information Degradation Through Layers

You train a 6-layer transformer on a text corpus. You run linear probes at each layer to check how well part-of-speech (POS) tags can be predicted from hidden states:

```
Embedding: 91%
Layer 0:   88%
Layer 1:   82%
Layer 2:   74%
Layer 3:   71%
Layer 4:   68%
Layer 5:   65%
```

**a)** POS tags were part of the input embedding. Why does accuracy drop through layers?

**b)** Does this mean the model "forgot" POS information? Or something else?

**c)** Your teammate adds a POS auxiliary loss to preserve POS info in the final layer. The probe accuracy at Layer 5 jumps to 94%. Has the model learned POS better, or is something else happening?

---

### Q3: LayerNorm and Document Similarity

You compute document embeddings by averaging all token hidden states (mean pooling). You notice two things:

1. Every document embedding has magnitude ≈ 22.6
2. Cosine similarity between unrelated documents is 0.87

Your model has `d_model=512`.

**a)** Explain why all embeddings have the same magnitude. What is the relationship between 22.6 and 512?

**b)** Are these two observations (same magnitude, high similarity) caused by the same thing?

**c)** Which one is a real problem, and which is a harmless artifact?

---

### Q4: Linear Probes — What Are They Actually Measuring?

You train a linear probe on the final hidden states of your model to predict whether a token is a named entity. The probe achieves 85% accuracy on the test set.

**a)** What does "linear probe" mean concretely? What is its architecture?

**b)** Why do we use a linear probe instead of a 3-layer MLP probe? Both could achieve higher accuracy.

**c)** The same probe on the input embedding (before any transformer layer) achieves 84%. What does this tell you about whether the model learned named entity information?

---

### Q5: GPU Memory — Where Does It Actually Go?

Your model has 10M parameters (40 MB in fp32). You load it on a 6 GB GPU — plenty of room. You start training with batch_size=32, sequence_length=512, and vocab_size=5,000. Training works fine.

You then increase the vocabulary to 200,000 (you forgot to filter rare tokens). The model parameters grow to 60M (240 MB) — still well within 6 GB. But training immediately OOMs.

**a)** The model weights fit in 240 MB. Where is the remaining memory going?

**b)** Write the formula for the size of the tensor that's causing the OOM.

**c)** Why does the backward pass make this even worse?

---

### Q6: Batch Size and Throughput

You're training a model and measure:
- batch_size=8: 12 iterations/second
- batch_size=16: 6 iterations/second
- batch_size=32: 3 iterations/second

Your manager says "batch_size=8 is fastest, look at the iterations."

**a)** Calculate documents/second for each batch size. Is your manager right?

**b)** Why doesn't doubling batch size double the throughput?

**c)** When would increasing batch size actually help throughput?

---

### Q7: Mean Pooling for Document Embeddings

You train a BERT-like model on legal documents using masked word prediction. You then mean-pool the final hidden states to get document embeddings for a contract similarity search.

All contracts have cosine similarity > 0.9 with each other, even contracts about completely different topics (employment vs real estate).

**a)** Why are the embeddings so similar? Think about what tokens legal documents share.

**b)** Would training longer fix this?

**c)** Name two alternative pooling strategies and explain why they might help.

---

### Q8: Non-Determinism Across Runs

You save a model checkpoint after training. You write a script that loads it and runs inference on the same input. You run the script twice and get different outputs.

**a)** Name three possible causes of non-determinism when loading and running the same checkpoint.

**b)** One common cause involves `strict=False` when loading checkpoints. Explain how this causes non-determinism even though the loaded weights are identical.

**c)** How would you fix the `strict=False` issue?

---

### Q9: Input Features vs Learning From Context

You're building a model that processes structured data. You have a feature called "category" (one of 50 categories) that you know is important.

Option A: Add category as an input embedding
Option B: Don't add category to the input, but include it in the prediction target

**a)** With Option A, what happens to the category information through the transformer layers? (Think about residual connections.)

**b)** With Option B, how does the model learn category if it's not in the input?

**c)** You try Option A and add an auxiliary loss "predict category from the final hidden state." The loss drops to 0.0003 instantly. Why? (Connect this to Q1.)

**d)** You switch to Option B. The same auxiliary loss now starts at 2.1 and gradually drops to 0.8 over 10 epochs. What's different?

---

### Q10: Capability Testing vs Accuracy Metrics

You report 95% accuracy on your test set. Your colleague deploys the model and it fails on a common use case.

**a)** How can a model with 95% accuracy fail on common cases?

**b)** What is "capability-based testing" and how does it differ from measuring accuracy?

**c)** Give an example of how capability testing would have caught the deployment failure.

---

### Q11: The Pooling Problem

You need to convert variable-length sequences into fixed-size vectors for a classification task. You consider:

1. Mean pooling (average all hidden states)
2. Using a special [CLS] token's hidden state
3. Attention pooling with a learned query vector

**a)** The CLS token is just a token prepended to the input. Why would its hidden state represent the whole document?

**b)** Your CLS-based embeddings are terrible. The model was trained with masked token prediction only, no other objective. Why doesn't CLS work?

**c)** In attention pooling, why is the query a fixed learned vector instead of being computed from input (via a W_Q matrix)?

**d)** Attention pooling is O(N) while self-attention is O(N^2). Why?

---

### Q12: Dynamic Sequence Lengths

Your model processes batches where each input has a different length. Batch 1 has sequences of lengths [30, 45, 38]. Batch 2 has sequences of lengths [120, 80, 95].

**a)** Within a batch, all sequences must be the same length for matrix multiplication. How is this handled?

**b)** Why doesn't the different sequence length between batches cause a shape mismatch in the model's weight matrices?

**c)** Your model trains fine for 1000 batches then suddenly OOMs on batch 1001. Most sequences are ~50 tokens but batch 1001 has one sequence of 400 tokens. Explain why this happens.

---

## Answer Key

<details>
<summary>Click to expand answers</summary>

### A1
**a)** The language ID embedding is in the input. The residual connection carries it through all layers untouched. The model reads it directly from the residual stream — zero computation needed.

**b)** It tells you the auxiliary loss is useless. The information is already linearly decodable from the input before any transformer processing. The auxiliary loss doesn't force the model to learn anything new.

**c)** The auxiliary loss would become meaningful. The model would have to discover the language from context (vocabulary patterns, grammar, word order) through attention. The loss would start high and gradually decrease as the model learns language identification through computation, not copying.

### A2
**a)** The model overwrites the POS embedding with learned features more useful for the prediction task. Each layer adds to the residual stream, gradually repurposing dimensions that stored POS info.

**b)** Something else. The model likely still "knows" POS but encodes it nonlinearly — entangled with other features in a way a linear probe can't extract. The model found a more efficient combined representation.

**c)** Something else is happening. The POS auxiliary loss is trivially easy because POS was in the input embedding and leaked through the residual (same as Q1). The 94% probe accuracy is reading the input embedding from the residual, not measuring learned understanding.

### A3
**a)** LayerNorm normalizes each vector to mean≈0, std≈1 across d_model dimensions. The expected squared norm is `d_model` (sum of d_model components each with variance≈1). So `||v|| ≈ √512 ≈ 22.6`.

**b)** No. Same magnitude is caused by LayerNorm (mathematical artifact). High similarity is caused by the training objective — shared tokens across documents produce similar hidden states, dominating the mean pool.

**c)** Same magnitude is harmless. High similarity is the real problem — the model isn't producing discriminative document representations.

### A4
**a)** A single linear layer: `y = Wx + b` where `x` is the hidden state (e.g., 256-dim), `W` is `(num_classes, 256)`, and `y` is a score per class. Trained with cross-entropy on frozen hidden states. Evaluated on a held-out test set.

**b)** A linear probe only succeeds if the information is linearly encoded in the vector — cleanly represented in the directions and magnitudes. A nonlinear MLP could memorize or discover complex patterns that aren't really "there" in a usable way. The linear constraint is what makes it a valid test of representation quality.

**c)** The model didn't learn named entity information — it was already in the input embedding. The 1% improvement (84→85%) is negligible. The transformer layers didn't add meaningful NER capability.

### A5
**a)** Activations — intermediate tensors created during the forward pass. The biggest one is the logits tensor, which has one entry per vocabulary item, per sequence position, per batch element.

**b)** `batch_size × seq_len × vocab_size × 4 bytes = 32 × 512 × 200,000 × 4 = 13.1 GB`

**c)** `loss.backward()` must store gradients for the logits tensor — roughly the same size. So memory approximately doubles: ~26 GB total for logits + their gradients.

### A6
**a)** batch=8: 12 × 8 = 96 docs/sec. batch=16: 6 × 16 = 96 docs/sec. batch=32: 3 × 32 = 96 docs/sec. All the same. Manager is wrong.

**b)** The GPU is already the bottleneck. Larger batches mean fewer but larger matrix multiplications — same total FLOPS. The GPU was already fully utilized.

**c)** When the GPU isn't fully utilized (small model, short sequences, CPU-bound data loading) or with multi-GPU distributed training.

### A7
**a)** Legal contracts share extensive boilerplate: "hereinafter referred to as," "party of the first part," "terms and conditions," etc. These common tokens dominate the mean-pooled embedding, pulling all contracts toward the same direction.

**b)** No. Longer training makes the model better at predicting masked tokens, but the per-token representations are still optimized for word prediction, not document discrimination. The similarity problem comes from the pooling method and shared vocabulary, not insufficient training.

**c)** (1) Attention pooling — learns which tokens to weight, can downweight boilerplate. (2) CLS token with a document-level objective (e.g., contrastive loss) — forces information aggregation into one position.

### A8
**a)** (1) `strict=False` loading leaving extra layers with random init. (2) GPU non-determinism — CUDA operations like atomicAdd accumulate floats in non-deterministic order. (3) Multi-threaded math libraries reducing in different orders.

**b)** The model class may have layers that the checkpoint doesn't have (e.g., a new classification head added after training). `strict=False` silently ignores the mismatch — the new layers keep random initialization. PyTorch's random seed differs each process, so the random init differs each run.

**c)** Set `torch.manual_seed(42)` before constructing the model. The random initialization becomes deterministic.

### A9
**a)** The residual connection carries the category embedding through all layers. The category is available at every layer without any computation.

**b)** The model must attend to context tokens that reveal the category. It learns through the prediction target — to produce the correct category-specific prediction, it must figure out the category from surrounding tokens.

**c)** Same as Q2 — category is in the input embedding, leaked through residual, trivially read at the output. The auxiliary loss teaches nothing.

**d)** Category is NOT in the input now. The model must actually learn to identify category from context. The loss is meaningful — it decreases as the model builds genuine category understanding through attention.

### A10
**a)** The 95% accuracy is averaged across all test cases. The model might be 99% accurate on common easy cases and 20% on rare but important cases. The average hides the failure.

**b)** Capability testing defines specific behaviors the model should have (e.g., "understands negation," "handles long inputs") and tests each separately. Instead of one number, you get a profile of strengths and weaknesses.

**c)** If the deployment failure was "model can't handle negation," a capability test for negation would show 0/5 passing — clearly flagging the problem before deployment, even though overall accuracy was 95%.

### A11
**a)** Only if the model was trained with a document-level objective (like Next Sentence Prediction) that forces information aggregation into the CLS position. The loss backpropagates through CLS, teaching it to summarize the input via attention.

**b)** With masked token prediction only, CLS optimizes for predicting local masked tokens, not summarizing the document. Its hidden state has no reason to contain document-level information.

**c)** In attention pooling there's only ONE query for the whole document. `W_Q @ fixed_vector = another_fixed_vector` — no added expressiveness. The result can be absorbed into the learned vector itself.

**d)** Self-attention: N tokens each attend to N tokens → N×N scores. Attention pooling: 1 query attends to N tokens → 1×N scores. Just one row of the attention matrix.

### A12
**a)** Padding — shorter sequences are padded to the length of the longest sequence in the batch. A padding mask tells the model to ignore padded positions (attention weights set to zero).

**b)** Weight matrices have shapes determined by `d_model` (e.g., `Linear(256, 256)`), not by sequence length. Sequence length only affects the number of rows in activation tensors, which are computed fresh each batch.

**c)** Dynamic padding pads to batch max, not global max. Most batches pad to ~50, using moderate memory. Batch 1001 pads to 400 — activations scale with sequence length, so memory spikes. The logits tensor alone: `batch × 400 × vocab_size × 4 bytes` — much larger than `batch × 50 × vocab_size × 4 bytes`.

</details>
