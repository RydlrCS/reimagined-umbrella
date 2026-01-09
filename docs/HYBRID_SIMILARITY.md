# Hybrid Similarity Architecture

## Overview

Kinetic Ledger uses a **hybrid architecture** combining Gemini File Search with traditional kNN/RkCNN similarity algorithms to provide both semantic discovery and precise novelty detection.

## Architecture Components

```
┌─────────────────────────────────────────────────────────────┐
│                    Similarity Pipeline                       │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  1. Motion Analysis                                          │
│     ├── Gemini Multimodal Analysis (gemini-2.0-flash-exp)   │
│     ├── Generate 768-dim embeddings (gemini-embedding-001)   │
│     └── Extract style labels, NPC tags, safety flags         │
│                                                               │
│  2. Storage (Hybrid)                                         │
│     ├── File Search: Documents + automatic indexing          │
│     ├── Embedding Cache: 768-dim vectors for kNN/RkCNN      │
│     └── Corpus Management: Organized motion collections      │
│                                                               │
│  3. Discovery (File Search)                                  │
│     ├── Natural language queries                             │
│     ├── Semantic search with grounding                       │
│     └── Free query-time embeddings                           │
│                                                               │
│  4. Novelty Detection (kNN + RkCNN)                          │
│     ├── kNN: Baseline similarity (k=15)                      │
│     ├── RkCNN: High-dimensional robustness (32 ensembles)   │
│     ├── Separation Score: Boundary distance [0, 1]           │
│     └── Vote Margin: Ensemble consensus [0, 1]               │
│                                                               │
│  5. Decision                                                 │
│     ├── MINT: separation >= 0.42, no safety flags            │
│     ├── REJECT: separation < 0.42 (too similar)              │
│     └── REVIEW: safety flags present (manual review)         │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

## Why Hybrid?

### File Search Strengths
- **Natural Language**: "Find aggressive parkour transitions"
- **Semantic Understanding**: Understands motion concepts
- **Zero Infrastructure**: No vector DB clusters required
- **Free Storage/Queries**: No per-vector costs
- **Grounding**: Citations to source documents

### File Search Limitations
- **No Raw Embeddings**: Can't access vectors directly
- **No Distance Metrics**: Can't compute exact k-NN distances
- **Limited Control**: Can't tune similarity algorithms

### kNN/RkCNN Strengths
- **Precise Distances**: Exact L2/cosine measurements
- **Novelty Scoring**: Separation score in [0, 1]
- **Ensemble Robustness**: 32-128 subspace samples
- **Tunable Thresholds**: Configure novelty_threshold
- **Proven Algorithms**: Well-understood mathematics

## Implementation

### 1. FileSearchConnector (Enhanced)

```python
class FileSearchConnector:
    """
    Gemini File Search with embedding cache for kNN/RkCNN.
    """
    _embedding_cache: Dict[str, np.ndarray] = {}
    
    def index_document(
        self,
        document: Dict[str, Any],
        embedding: Optional[np.ndarray] = None
    ) -> bool:
        """
        Index document in File Search AND cache embedding.
        
        - File Search: Natural language discovery
        - Cache: Precise kNN/RkCNN similarity
        """
        # Upload to File Search corpus
        file = self._client.files.create(
            corpus=self._corpus,
            display_name=document_id,
            mime_type="text/plain"
        )
        file.write(formatted_text)
        
        # Cache embedding for kNN/RkCNN
        if embedding is not None:
            self._embedding_cache[document_id] = embedding
    
    def get_all_embeddings(self) -> List[Tuple[str, np.ndarray]]:
        """
        Get all cached embeddings for kNN/RkCNN.
        """
        return list(self._embedding_cache.items())
```

### 2. AttestationOracle (Hybrid)

```python
class AttestationOracle:
    """
    Uses File Search cache for kNN/RkCNN, falls back to VectorStore.
    """
    
    def __init__(
        self,
        config: AttestationConfig,
        file_search: Optional[FileSearchConnector] = None,
    ):
        self.file_search = file_search or FileSearchConnector.get_instance()
    
    def validate_similarity(self, ...):
        """
        Hybrid similarity validation.
        """
        # Prefer File Search cache
        if self.file_search.is_available() and self.file_search.cache_size() > 0:
            items = self.file_search.get_all_embeddings()
        else:
            items = self.vector_store.get_all()  # Fallback
        
        # Run kNN
        knn_neighbors = knn(query, items, k=15)
        
        # Run RkCNN ensembles
        separation, vote_margin, ensembles = rkcnn(
            query, items, k=15, ensembles=32
        )
        
        # Decide: MINT, REJECT, or REVIEW
        if separation >= threshold:
            return "MINT"
        else:
            return "REJECT"
```

## Structured Outputs Integration

Gemini's [structured output feature](https://ai.google.dev/gemini-api/docs/structured-output) ensures type-safe responses:

```python
from pydantic import BaseModel, Field

class NoveltyAssessment(BaseModel):
    is_novel: bool
    novelty_score: float = Field(ge=0.0, le=1.0)
    knn_distance: float
    separation_score: float = Field(ge=0.0, le=1.0)
    vote_consensus: float = Field(ge=0.0, le=1.0)
    decision: Literal["MINT", "REJECT", "REVIEW"]
    reasoning: str

# Use with Gemini API
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=prompt,
    config={
        "response_mime_type": "application/json",
        "response_json_schema": NoveltyAssessment.model_json_schema(),
    },
)

assessment = NoveltyAssessment.model_validate_json(response.text)
```

## Use Cases

### 1. Discovery Phase
**Goal**: Find similar motions semantically

```python
# Natural language search via File Search
results = file_search.search(
    query="Find capoeira transitions with explosive power",
    model="gemini-2.0-flash-exp",
    max_results=10
)

# Returns: Semantically relevant motions with grounding metadata
```

### 2. Novelty Validation
**Goal**: Precise novelty scoring for minting

```python
# kNN + RkCNN with cached embeddings
similarity_check = oracle.validate_similarity(
    analysis_id="new-motion-001",
    tensor_hash="0xabc...",
    gemini_descriptors={"style": "capoeira-parkour blend"}
)

# Returns:
# - kNN neighbors: [("motion-042", 1.2), ("motion-089", 1.5), ...]
# - RkCNN separation: 0.65 (> 0.42 threshold = MINT)
# - Vote margin: 0.82 (strong consensus)
# - Decision: MINT
```

### 3. Structured Analysis
**Goal**: Type-safe motion classification

```python
from schemas.structured_outputs import MotionStyleClassification

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Classify this motion blend",
    config={
        "response_mime_type": "application/json",
        "response_json_schema": MotionStyleClassification.model_json_schema(),
    },
)

classification = MotionStyleClassification.model_validate_json(response.text)
# classification.primary_style = "capoeira"
# classification.energy_level = "explosive"
# classification.technical_difficulty = "advanced"
```

## Configuration

### AttestationConfig Parameters

```python
config = AttestationConfig(
    # kNN Configuration
    knn_k=15,                    # Number of neighbors
    distance_metric="euclidean", # or "cosine"
    
    # RkCNN Configuration
    rkcnn_k=15,                  # Neighbors per ensemble
    rkcnn_ensembles=32,          # Number of subspace samples (or None = auto)
    rkcnn_subspace_dim=128,      # Subspace dimensions (or None = auto)
    
    # Decision Thresholds
    novelty_threshold=0.42,      # Separation score threshold for MINT
    vote_margin_threshold=0.10,  # Minimum ensemble consensus
    
    # Embedding Configuration
    embedding_dim=768,           # Gemini embedding dimensions
    embedding_model_id="gemini-embedding-001",
)
```

### Auto-Computation

RkCNN can auto-compute optimal parameters:

```python
# Subspace dimension: min(d, max(16, round(4*sqrt(d))))
# For 768-dim: min(768, max(16, round(4*sqrt(768)))) ≈ 111

# Ensemble size: max(32, min(128, 8*ceil(log2(d))))
# For 768-dim: max(32, min(128, 8*ceil(log2(768)))) ≈ 80
```

## Testing

Comprehensive test suite in `tests/test_hybrid_similarity.py`:

```bash
pytest tests/test_hybrid_similarity.py -v
```

Tests cover:
- ✅ Embedding cache storage/retrieval
- ✅ kNN with File Search backend
- ✅ RkCNN ensembles with cached embeddings
- ✅ AttestationOracle hybrid selection
- ✅ Fallback to VectorStore
- ✅ End-to-end workflow
- ✅ Structured output validation

## Performance Characteristics

| Metric | File Search | kNN | RkCNN |
|--------|-------------|-----|-------|
| **Query Time** | ~500ms | ~10ms | ~100ms |
| **Storage Cost** | Free | RAM | RAM |
| **Accuracy** | Semantic | Exact | Robust |
| **Scale** | Unlimited | 10K vectors | 10K vectors |
| **Tuning** | Prompt | k, metric | k, E, m, metric |

## Best Practices

1. **Always cache embeddings** when indexing to File Search
2. **Use File Search for discovery**, kNN/RkCNN for validation
3. **Set appropriate thresholds** based on your novelty tolerance
4. **Monitor separation scores** to tune `novelty_threshold`
5. **Use structured outputs** for type safety
6. **Test with real data** to validate ensemble parameters

## Migration from Elasticsearch

If migrating from Elasticsearch:

```python
# OLD: Elasticsearch direct queries
es_client.search(
    index="motions",
    body={
        "query": {"knn": {"embedding": {"vector": query, "k": 15}}}
    }
)

# NEW: Hybrid approach
# 1. Discovery via File Search
results = file_search.search("Find similar capoeira motions")

# 2. Validation via kNN/RkCNN
similarity = oracle.validate_similarity(
    analysis_id=new_motion_id,
    tensor_hash=tensor_hash,
)
```

## Future Enhancements

- **Persistent cache**: Save embeddings to disk
- **Incremental indexing**: Update cache without full rebuild
- **Multi-corpus**: Separate corpora for different motion types
- **Hybrid search**: Combine File Search scores with kNN distances
- **Active learning**: Use REVIEW decisions to improve thresholds

## References

- [Gemini Structured Outputs](https://ai.google.dev/gemini-api/docs/structured-output)
- [Gemini File Search](https://ai.google.dev/gemini-api/docs/file-search)
- [Gemini Embeddings](./GEMINI_EMBEDDINGS.md)
- [RkCNN Paper](https://arxiv.org/abs/1808.04902)
