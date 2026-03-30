# YAML Similarity and Clustering

## Problem

Given a collection of Kubernetes manifests, practitioners need to:
- Find manifests structurally similar to a given one ("show me Deployments like this")
- Detect outliers in a fleet ("this Deployment looks nothing like the others")
- Cluster resources by structural patterns, not just kind

The pre-trained encoder produces per-node hidden states, but there's no document-level embedding. Mean pooling is naive — it treats every node equally and wasn't optimized for similarity.

## Solution

Add a **Pooling by Multi-head Attention (PMA)** layer on top of the frozen encoder. The `kind` node queries all other nodes through cross-attention, producing a single document embedding that aggregates the full document through the kind node's perspective.

Train with **supervised contrastive loss** — same-kind documents are pulled together, different-kind documents are pushed apart. The encoder stays frozen. Only the small pooling layer is trained.

## Architecture

```
YAML document
    |
    v
Frozen encoder (v1/v3)  →  per-node hidden states (256 dims each)
    |                              |
    v                              v
Pooling attention layer        Key prediction head (existing, unchanged)
    |                              |
    v                              v
Document embedding (256 dims)  Key suggestions (unchanged)
```

### Pooling Attention Layer

```python
class DocumentPooling(nn.Module):
    def __init__(self, d_model: int, num_heads: int = 4):
        self.query_proj = nn.Linear(d_model, d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)

    def forward(self, kind_hidden: torch.Tensor, all_hidden: torch.Tensor):
        # kind_hidden: (batch, 1, d_model) — the kind node's hidden state
        # all_hidden: (batch, seq_len, d_model) — all nodes
        query = self.query_proj(kind_hidden)
        doc_emb, _ = self.cross_attn(query, all_hidden, all_hidden)
        return doc_emb.squeeze(1)  # (batch, d_model)
```

The `kind` node acts as a learned query. It asks every other node "what do you contain?" through cross-attention. The result is a 256-dim document embedding.

### Why the Kind Node

- Every K8s document has exactly one `kind` node
- It's the document's identity — semantically the right aggregation point
- Its hidden state already contains document-level context from 6 layers of self-attention
- Using it as the query (not a random learned vector) grounds the aggregation in the actual document

### Contrastive Training

Supervised contrastive loss (SupCon, Khosla et al., 2020):

For a batch of N documents with kind labels:
- Documents of the same kind are positive pairs
- Documents of different kinds are negative pairs
- Loss pulls same-kind embeddings together, pushes different-kind apart

```python
def sup_contrastive_loss(embeddings, kind_labels, temperature=0.1):
    # Normalize embeddings
    embeddings = F.normalize(embeddings, dim=1)
    # Pairwise cosine similarity
    sim = embeddings @ embeddings.T / temperature
    # Mask: same kind = positive
    mask = kind_labels.unsqueeze(0) == kind_labels.unsqueeze(1)
    # InfoNCE-style loss
    ...
```

### What Gets Trained

| Component | Trainable | Parameters |
|-----------|-----------|------------|
| Encoder (6 layers) | Frozen | 0 |
| Key prediction head | Frozen | 0 |
| Kind/parent classifiers | Frozen | 0 |
| **Pooling attention layer** | **Yes** | **~260K** |

~260K parameters — tiny. Trains in minutes, not hours.

## Files

| File | Purpose |
|------|---------|
| `yaml_bert/pooling.py` | DocumentPooling module + contrastive loss |
| `yaml_bert/similarity.py` | Embedding extraction, similarity search, clustering |
| `scripts/train_pooling.py` | Train the pooling layer |
| `scripts/cluster_yamls.py` | CLI tool for clustering and similarity |
| `tests/test_pooling.py` | Tests |
| `tests/test_similarity.py` | Tests |

## Training

```bash
python scripts/train_pooling.py \
    --encoder-checkpoint output_v3_full/yaml_bert_v3_final.pt \
    --vocab output_v3_full/vocab.json \
    --epochs 10 \
    --batch-size 64 \
    --output pooling_layer.pt
```

Uses the cached linearized documents. Encoder is frozen — only the pooling layer parameters receive gradients. Should converge in a few epochs.

## CLI Usage

```bash
# Embed a single YAML and find similar ones in a collection
python scripts/cluster_yamls.py \
    --encoder output_v3_full/yaml_bert_v3_final.pt \
    --pooling pooling_layer.pt \
    --query my-deployment.yaml \
    --corpus ./manifests/

# Cluster all YAMLs in a directory
python scripts/cluster_yamls.py \
    --encoder output_v3_full/yaml_bert_v3_final.pt \
    --pooling pooling_layer.pt \
    --corpus ./manifests/ \
    --cluster --n-clusters 10

# Find outliers
python scripts/cluster_yamls.py \
    --encoder output_v3_full/yaml_bert_v3_final.pt \
    --pooling pooling_layer.pt \
    --corpus ./manifests/ \
    --outliers
```

## Output

### Similarity search
```
Query: my-deployment.yaml (Deployment/web)

Most similar:
  1. [0.95] nginx-deployment.yaml (Deployment/nginx)
  2. [0.91] api-deployment.yaml (Deployment/api)
  3. [0.87] worker-deployment.yaml (Deployment/worker)

Least similar:
  1. [0.12] redis-service.yaml (Service/redis)
  2. [0.15] monitoring-daemonset.yaml (DaemonSet/monitor)
```

### Clustering
```
Cluster 0 (15 docs): Deployments with single container
Cluster 1 (8 docs): Deployments with multiple containers
Cluster 2 (12 docs): Services
Cluster 3 (5 docs): ConfigMaps
Cluster 4 (3 docs): StatefulSets
```

### Outlier detection
```
Outliers (distance > 2σ from cluster center):
  - weird-deployment.yaml (Deployment/legacy) — score: 3.2σ
  - test-pod.yaml (Pod/debug) — score: 2.8σ
```

## Success Criteria

1. Same-kind documents cluster together
2. Within a kind, structurally similar documents are closer than dissimilar ones
3. Outliers (structurally unusual manifests) are detectable
4. The pooling layer trains in under 10 minutes on the cached corpus

## What This Does NOT Do

- Cross-resource validation (checking if Service matches Deployment) — planned
- Value-level similarity (doesn't compare actual values, only structural patterns)
- Semantic similarity of purpose (two different apps with the same structure are "similar")
