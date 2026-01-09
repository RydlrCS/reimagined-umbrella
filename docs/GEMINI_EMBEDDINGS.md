# Gemini Embeddings Integration Guide

This document explains how Kinetic Ledger uses Gemini's native embedding API for motion analysis and semantic search.

## Overview

We use **`gemini-embedding-001`** for generating vector embeddings from motion descriptors. This provides:
- **768-dimensional embeddings** (recommended balance of accuracy and cost)
- **Task-specific optimization** for different use cases
- **Batch embedding support** (50% cheaper, higher throughput)
- **Free storage and query-time embeddings** (only pay for initial indexing: $0.15 per 1M tokens)

## Why Gemini Embeddings?

**Previously:** We used SentenceTransformers (`all-MiniLM-L6-v2`) with 384-dimensional embeddings.

**Now:** Native Gemini embeddings provide:
1. **Better integration** - Same SDK for analysis and embeddings (google-genai)
2. **Optimized dimensions** - 768-dim recommended (higher accuracy than 384-dim)
3. **Task type optimization** - Different strategies for indexing vs search vs similarity
4. **Cost efficiency** - Free storage/query, cheaper batch processing
5. **Single vendor** - Reduces dependencies and complexity

## Task Types

Gemini embeddings support 5 task types for optimization:

### 1. `RETRIEVAL_DOCUMENT` (Indexing)
**When to use:** Storing motion descriptors in Elasticsearch for future search

```python
from src.kinetic_ledger.services.embedding_service import embed_motion_descriptor

# Index a motion blend for search
embedding = embed_motion_descriptor(
    style_labels=["capoeira", "breakdance"],
    npc_tags=["warrior", "athletic"],
    summary="Dynamic blend with explosive kicks",
    task_type="RETRIEVAL_DOCUMENT"  # For indexing
)
```

### 2. `RETRIEVAL_QUERY` (Search)
**When to use:** Embedding user search queries to find similar motions

```python
from src.kinetic_ledger.services.embedding_service import embed_text

# Search for similar motions
query_embedding = embed_text(
    "martial arts blend with acrobatic moves",
    task_type="RETRIEVAL_QUERY"  # For search queries
)
```

### 3. `SEMANTIC_SIMILARITY` (Comparison)
**When to use:** Comparing two motions for similarity (k-NN, RkCNN validation)

```python
# Compare motion similarity
motion_a_embedding = embed_text(
    "capoeira backflip",
    task_type="SEMANTIC_SIMILARITY"
)

motion_b_embedding = embed_text(
    "breakdance freeze",
    task_type="SEMANTIC_SIMILARITY"
)

# Compute cosine similarity
similarity = cosine_similarity(motion_a_embedding, motion_b_embedding)
```

### 4. `CLASSIFICATION` (Categorization)
**When to use:** Classifying motion types (future feature)

```python
# Classify motion style
embedding = embed_text(
    "spin kick with 360 rotation",
    task_type="CLASSIFICATION"
)
```

### 5. `CLUSTERING` (Grouping)
**When to use:** Grouping similar motions (future feature)

```python
# Cluster similar motion patterns
embedding = embed_text(
    "low sweep kick",
    task_type="CLUSTERING"
)
```

## Batch Embedding

For bulk operations (indexing multiple motions), use batch embedding for 50% cost savings:

```python
from src.kinetic_ledger.services.embedding_service import EmbeddingService

service = EmbeddingService()

motion_descriptors = [
    "capoeira ginga with breakdance toprock",
    "kung fu crane stance transition",
    "taekwondo tornado kick blend"
]

# Batch embed (more efficient)
embeddings = service.batch_embed_text(
    motion_descriptors,
    task_type="RETRIEVAL_DOCUMENT"
)
```

## Normalization

Per Gemini documentation, embeddings with `output_dimensionality < 3072` should be normalized for accurate semantic similarity:

```python
# Automatic normalization (handled by EmbeddingService)
embedding = embed_text("motion descriptor")  # Already normalized

# Manual normalization (if needed)
import numpy as np

def normalize_embedding(embedding: List[float]) -> List[float]:
    embedding_np = np.array(embedding, dtype=np.float32)
    norm = np.linalg.norm(embedding_np)
    if norm > 0:
        embedding_np = embedding_np / norm
    return embedding_np.tolist()
```

## Configuration

### Environment Variables

```bash
# Required for Gemini embeddings
GEMINI_API_KEY=your_gemini_api_key_here

# Optional: Override defaults
EMBED_MODEL_NAME=gemini-embedding-001
VECTOR_DIMENSIONS=768  # or 1536, 3072
```

### Dimension Options

Gemini supports flexible dimensions (128-3072), but recommended sizes are:

- **768** (default) - Best balance of accuracy, speed, and cost
- **1536** - Higher accuracy for complex motion patterns
- **3072** - Maximum accuracy (no normalization needed)

## Integration with Elasticsearch

Our Elasticsearch index is configured for 768-dimensional Gemini embeddings:

```python
# elasticsearch_mappings.py
VECTOR_DIMENSIONS = 768  # Aligned with Gemini

# Index mapping
"motion_vector": {
    "type": "dense_vector",
    "dims": 768,
    "index": True,
    "similarity": "cosine"  # Matches normalized embeddings
}
```

## Cost Optimization

Gemini embeddings pricing (as of documentation):
- **Indexing:** $0.15 per 1M tokens (one-time when storing)
- **Storage:** Free
- **Query:** Free
- **Batch embedding:** 50% cheaper than individual calls

**Example cost calculation:**
- 10,000 motions × 50 tokens/motion = 500,000 tokens
- Cost: 500K / 1M × $0.15 = **$0.075** (one-time)
- Unlimited searches: **Free**

## Fallback Behavior

When Gemini API is unavailable (no API key, network issues), the service falls back to deterministic pseudo-embeddings:

```python
# Automatic fallback
embedding = embed_text("motion descriptor")
# Returns pseudo-embedding if Gemini unavailable
# Uses hash-based deterministic generation
```

This ensures tests and development work without API access.

## Best Practices

1. **Use appropriate task types:**
   - `RETRIEVAL_DOCUMENT` when indexing motions
   - `RETRIEVAL_QUERY` when searching
   - `SEMANTIC_SIMILARITY` when comparing

2. **Batch when possible:**
   - Use `batch_embed_text()` for bulk operations
   - Saves 50% on API costs

3. **Cache embeddings:**
   - Store embeddings in Elasticsearch
   - Don't re-embed the same motion descriptor

4. **Monitor dimensions:**
   - 768-dim is recommended default
   - Increase to 1536/3072 only if accuracy requires it

5. **Normalize for similarity:**
   - Always use normalized embeddings for cosine similarity
   - Service handles this automatically for dims < 3072

## Future: Gemini File Search

Gemini also offers a **File Search** tool with automatic chunking and indexing:

```python
# Alternative to manual Elasticsearch (future consideration)
from google import genai

client = genai.Client(api_key=api_key)

# Upload motion data to corpus
corpus = client.files.create_corpus(display_name="motion_library")

# Automatic embedding and indexing
file = client.files.upload(
    file_path="motion_sequence.fbx",
    corpus=corpus
)

# Search with natural language
results = client.models.generate_content(
    model="gemini-2.0-flash-exp",
    contents="Find martial arts blends with acrobatic elements",
    config=types.GenerateContentConfig(
        tools=[types.Tool(file_search=corpus)]
    )
)
```

This could complement or replace Elasticsearch for certain use cases.

## References

- [Gemini Embeddings API Documentation](https://ai.google.dev/gemini-api/docs/embeddings)
- [Gemini File Search Guide](https://ai.google.dev/gemini-api/docs/file-search)
- [google-genai Python SDK](https://github.com/googleapis/python-genai)
- [Elasticsearch k-NN Search](https://www.elastic.co/guide/en/elasticsearch/reference/current/knn-search.html)
