# RAG Configuration Guide

## Overview

This guide explains how to configure the RAG (Retrieval-Augmented Generation) system for optimal performance in different scenarios.

---

## Configuration Layers

### 1. RAG Chain Configuration

**File**: `app/services/rag/rag_chain.py`

**Class**: `RAGConfig`

```python
from app.services.rag.rag_chain import RAGConfig, RAGMode

config = RAGConfig(
    mode=RAGMode.CONVERSATIONAL,      # Operation mode
    use_memory=True,                  # Enable conversation memory
    use_query_rewriting=True,         # Rewrite queries with context
    use_hyde=False,                   # Hypothetical Document Embeddings
    use_compression=False,            # Contextual compression
    max_context_length=4000,          # Max tokens for context
    min_confidence_score=0.4,         # Minimum confidence threshold
    max_retries=3,                    # Retry attempts
    retry_delay=1.0,                  # Initial retry delay (seconds)
    timeout=30.0,                     # Operation timeout (seconds)
    include_sources=True,             # Include source attribution
    stream_response=True              # Enable streaming
)
```

### 2. Retrieval Configuration

**File**: `app/services/rag/retriever.py`

**Class**: `RetrievalConfig`

```python
from app.services.rag.retriever import (
    RetrievalConfig,
    RetrievalStrategy,
    RerankingMethod
)

config = RetrievalConfig(
    strategy=RetrievalStrategy.HYBRID,           # Retrieval strategy
    top_k=30,                                    # Documents to retrieve
    score_threshold=0.05,                        # Minimum similarity score
    fetch_k=60,                                  # Documents to fetch before reranking
    mmr_diversity=0.3,                           # MMR diversity (0-1)
    reranking=RerankingMethod.CROSS_ENCODER,     # Reranking method
    rerank_top_k=10,                             # Documents after reranking
    ensemble_weights={                           # Ensemble strategy weights
        "similarity": 0.4,
        "mmr": 0.1,
        "keyword": 0.5
    }
)
```

### 3. Chunking Configuration

**File**: `app/services/rag/text_splitter.py`

**Class**: `ChunkConfig`

```python
from app.services.rag.text_splitter import ChunkConfig, ChunkingStrategy

config = ChunkConfig(
    chunk_size=400,                              # Characters per chunk
    chunk_overlap=40,                            # Overlap between chunks
    strategy=ChunkingStrategy.CUSTOM_FAST,       # Chunking strategy
    min_chunk_size=50,                           # Minimum chunk size
    max_chunk_size=800,                          # Maximum chunk size
    strip_whitespace=True,                       # Remove extra whitespace
    metadata_level=0,                            # Metadata detail level (0-2)
    enable_timing=True                           # Enable performance timing
)
```

---

## Configuration Scenarios

### Scenario 1: High Accuracy (Slower)

**Use Case**: Legal documents, medical records, compliance

```python
RAGConfig(
    mode=RAGMode.STRICT,
    use_query_rewriting=True,
    use_hyde=True,                    # Better retrieval
    use_compression=True,             # Better context
    max_context_length=6000,          # More context
    min_confidence_score=0.7,         # Higher threshold
    include_sources=True
)

RetrievalConfig(
    strategy=RetrievalStrategy.ENSEMBLE,
    top_k=50,                         # More documents
    reranking=RerankingMethod.LLM,    # Best accuracy
    rerank_top_k=15
)
```

### Scenario 2: Balanced (Recommended)

**Use Case**: General chatbot, customer support

```python
RAGConfig(
    mode=RAGMode.CONVERSATIONAL,
    use_query_rewriting=True,
    use_hyde=False,
    use_compression=False,
    max_context_length=4000,
    min_confidence_score=0.4,
    include_sources=True
)

RetrievalConfig(
    strategy=RetrievalStrategy.HYBRID,
    top_k=30,
    reranking=RerankingMethod.CROSS_ENCODER,
    rerank_top_k=10
)
```

### Scenario 3: High Speed (Faster)

**Use Case**: Real-time chat, high-volume queries

```python
RAGConfig(
    mode=RAGMode.STANDARD,
    use_query_rewriting=False,
    use_hyde=False,
    use_compression=False,
    max_context_length=2000,
    min_confidence_score=0.3,
    include_sources=False             # Skip source formatting
)

RetrievalConfig(
    strategy=RetrievalStrategy.SIMILARITY,
    top_k=15,
    reranking=RerankingMethod.NONE,  # Skip reranking
    rerank_top_k=15
)
```

---

## Retrieval Strategies Explained

### SIMILARITY
- **Description**: Vector similarity search
- **Speed**: Fast (~50ms)
- **Use Case**: General queries
- **Pros**: Fast, simple
- **Cons**: May miss keyword matches

### MMR (Maximal Marginal Relevance)
- **Description**: Balances relevance and diversity
- **Speed**: Medium (~80ms)
- **Use Case**: Diverse results needed
- **Pros**: Reduces redundancy
- **Cons**: Slightly slower

### HYBRID
- **Description**: Vector + keyword search
- **Speed**: Medium (~100ms)
- **Use Case**: Comprehensive coverage
- **Pros**: Best recall
- **Cons**: More complex

### ENSEMBLE
- **Description**: Combines multiple strategies
- **Speed**: Slow (~150ms)
- **Use Case**: Maximum accuracy
- **Pros**: Best overall performance
- **Cons**: Highest latency

---

## Re-ranking Methods Explained

### CROSS_ENCODER
- **Model**: ms-marco-MiniLM-L-6-v2
- **Speed**: Fast (~50ms for 30 docs)
- **Accuracy**: High
- **Recommended**: ✅ Yes (best trade-off)

### COHERE
- **Model**: Cohere Rerank API
- **Speed**: Medium (~100ms)
- **Accuracy**: Very High
- **Recommended**: For production with API access

### LLM
- **Model**: Your LLM
- **Speed**: Slow (~500ms)
- **Accuracy**: Highest
- **Recommended**: For critical queries only

### RRF (Reciprocal Rank Fusion)
- **Model**: Algorithm-based
- **Speed**: Very Fast (~10ms)
- **Accuracy**: Medium
- **Recommended**: For ensemble strategies

---

## Tuning Parameters

### top_k (Documents to Retrieve)
- **Low (10-15)**: Fast, may miss relevant docs
- **Medium (20-30)**: Balanced (recommended)
- **High (40-50)**: Comprehensive, slower

### score_threshold (Minimum Similarity)
- **Low (0.05-0.2)**: High recall, may include irrelevant
- **Medium (0.3-0.5)**: Balanced (recommended)
- **High (0.6-0.8)**: High precision, may miss relevant

### chunk_size (Characters per Chunk)
- **Small (200-300)**: Fine-grained, more chunks
- **Medium (400-500)**: Balanced (recommended)
- **Large (600-800)**: Coarse-grained, fewer chunks

### chunk_overlap (Overlap Characters)
- **Low (20-30)**: Less redundancy
- **Medium (40-50)**: Balanced (recommended)
- **High (60-80)**: More context preservation

---

## Environment Variables

Add to `.env`:

```bash
# RAG Configuration
RAG_MODE=conversational
RAG_USE_MEMORY=true
RAG_USE_QUERY_REWRITING=true
RAG_MAX_CONTEXT_LENGTH=4000
RAG_MIN_CONFIDENCE_SCORE=0.4

# Retrieval Configuration
RETRIEVAL_STRATEGY=hybrid
RETRIEVAL_TOP_K=30
RETRIEVAL_SCORE_THRESHOLD=0.05
RERANKING_METHOD=cross_encoder
RERANK_TOP_K=10

# Chunking Configuration
CHUNK_SIZE=400
CHUNK_OVERLAP=40
CHUNKING_STRATEGY=custom_fast
```

---

## Best Practices

### 1. Start with Defaults
Use the balanced configuration and adjust based on metrics.

### 2. Monitor Performance
Track retrieval latency, accuracy, and user satisfaction.

### 3. A/B Test Changes
Test configuration changes with a subset of users first.

### 4. Document-Specific Tuning
Different document types may need different chunk sizes:
- **Code**: 600-800 chars (preserve function context)
- **Legal**: 400-600 chars (preserve clause context)
- **Chat logs**: 200-400 chars (preserve conversation turns)

### 5. Use Reranking
Always use reranking in production for better accuracy.

### 6. Enable Sources
Always include sources for transparency and debugging.

---

## Troubleshooting

### Issue: Low Retrieval Accuracy
**Solution**: 
- Increase `top_k` to 40-50
- Lower `score_threshold` to 0.03
- Enable reranking with CROSS_ENCODER
- Try HYBRID or ENSEMBLE strategy

### Issue: Slow Response Time
**Solution**:
- Decrease `top_k` to 15-20
- Disable HyDE and compression
- Use SIMILARITY strategy
- Disable reranking or use RRF

### Issue: Missing Relevant Documents
**Solution**:
- Lower `score_threshold`
- Increase `top_k`
- Use HYBRID strategy
- Check chunk size (may be too large)

### Issue: Too Many Irrelevant Results
**Solution**:
- Increase `score_threshold`
- Enable reranking
- Increase `min_confidence_score`
- Use CROSS_ENCODER or LLM reranking

---

## Conclusion

The RAG system is highly configurable. Start with the **Balanced** configuration and tune based on your specific needs and metrics.

**Recommended Production Config**: Hybrid retrieval + Cross-encoder reranking + 400-char chunks
