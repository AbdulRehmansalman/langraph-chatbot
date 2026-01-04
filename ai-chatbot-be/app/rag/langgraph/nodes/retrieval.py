"""
Document Retrieval Node
=======================

Hybrid document retrieval combining:
1. Vector Search (semantic understanding via embeddings)
2. Full-Text Search (keyword matching via PostgreSQL tsvector)
3. Reciprocal Rank Fusion (RRF) to merge results
4. BGE cross-encoder reranking for final relevance scoring

This approach ensures both semantic similarity AND keyword matches are captured.
"""

import asyncio
import hashlib
import logging
import os
import re
import time
from typing import Any, Optional

from app.rag.langgraph.state import (
    AgentState,
    track_node,
    add_documents,
    add_error,
)

logger = logging.getLogger(__name__)

# Configuration
RETRIEVAL_TIMEOUT = 10.0  # seconds
VECTOR_FETCH_COUNT = 15   # Candidates from vector search
KEYWORD_FETCH_COUNT = 15  # Candidates from keyword search
FINAL_RETURN_COUNT = 5    # Results after reranking
MATCH_THRESHOLD = float(os.getenv("MATCH_THRESHOLD", "0.1"))
RERANKER_ENABLED = os.getenv("RERANKER_ENABLED", "true").lower() == "true"
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-base")
HYBRID_SEARCH_ENABLED = os.getenv("HYBRID_SEARCH_ENABLED", "true").lower() == "true"
RRF_K = 60  # Reciprocal Rank Fusion constant (standard value)

# Stopwords to remove from keyword queries
STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "must", "shall", "can", "need", "dare",
    "ought", "used", "to", "of", "in", "for", "on", "with", "at", "by",
    "from", "as", "into", "through", "during", "before", "after", "above",
    "below", "between", "under", "again", "further", "then", "once", "here",
    "there", "when", "where", "why", "how", "all", "each", "few", "more",
    "most", "other", "some", "such", "no", "nor", "not", "only", "own",
    "same", "so", "than", "too", "very", "just", "and", "but", "if", "or",
    "because", "until", "while", "although", "though", "after", "before",
    "what", "which", "who", "whom", "this", "that", "these", "those", "am",
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you",
    "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself",
    "she", "her", "hers", "herself", "it", "its", "itself", "they", "them",
    "their", "theirs", "themselves", "about", "tell", "please", "help",
}

# Lazy-loaded singletons
_reranker: Optional[Any] = None
_reranker_lock = asyncio.Lock()
_embedding_service: Optional[Any] = None
_embedding_lock = asyncio.Lock()


async def _get_embedding_service():
    """Get or create singleton embedding service."""
    global _embedding_service
    if _embedding_service is not None:
        return _embedding_service

    async with _embedding_lock:
        if _embedding_service is not None:
            return _embedding_service

        try:
            from app.rag.embeddings.service import EmbeddingsService
            logger.info("Loading embedding service (singleton)...")
            _embedding_service = EmbeddingsService()
            logger.info(f"Embedding service loaded: {_embedding_service.model_name}")
            return _embedding_service
        except Exception as e:
            logger.error(f"Failed to load embedding service: {e}")
            raise


async def _get_reranker():
    """Lazy load reranker model (thread-safe singleton)."""
    global _reranker
    if _reranker is not None:
        return _reranker

    async with _reranker_lock:
        if _reranker is not None:  # Double-check after lock
            return _reranker

        try:
            from sentence_transformers import CrossEncoder
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Loading reranker {RERANKER_MODEL} on {device}")

            _reranker = await asyncio.to_thread(
                lambda: CrossEncoder(RERANKER_MODEL, device=device)
            )
            logger.info("Reranker loaded successfully")
            return _reranker
        except Exception as e:
            logger.error(f"Failed to load reranker: {e}")
            return None


async def _rerank_documents(
    query: str,
    documents: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """
    Rerank documents using BGE cross-encoder.
    Returns top FINAL_RETURN_COUNT documents with updated scores.
    """
    if not documents or not RERANKER_ENABLED:
        return documents[:FINAL_RETURN_COUNT]

    reranker = await _get_reranker()
    if reranker is None:
        logger.warning("Reranker unavailable, using original ranking")
        return documents[:FINAL_RETURN_COUNT]

    try:
        # Prepare query-document pairs
        pairs = [(query, doc.get("content", "")) for doc in documents]

        # Score pairs with cross-encoder
        rerank_start = time.time()
        scores = await asyncio.to_thread(reranker.predict, pairs)
        rerank_duration = (time.time() - rerank_start) * 1000

        logger.info(f"Reranking {len(documents)} docs took {rerank_duration:.1f}ms")

        # Add rerank scores and sort
        for doc, score in zip(documents, scores):
            doc["original_score"] = doc.get("score", 0.0)
            doc["rerank_score"] = float(score)
            doc["score"] = float(score)  # Use rerank score as primary

        # Sort by rerank score descending
        reranked = sorted(documents, key=lambda x: x["rerank_score"], reverse=True)

        # Log score changes for debugging
        for i, doc in enumerate(reranked[:FINAL_RETURN_COUNT]):
            logger.debug(
                f"Rank {i+1}: original={doc['original_score']:.3f}, "
                f"rerank={doc['rerank_score']:.3f}, source={doc.get('source', 'unknown')}"
            )

        return reranked[:FINAL_RETURN_COUNT]

    except Exception as e:
        logger.error(f"Reranking failed: {e}, using original ranking")
        return documents[:FINAL_RETURN_COUNT]


# =============================================================================
# QUERY PREPROCESSING
# =============================================================================

def preprocess_query(query: str) -> dict[str, Any]:
    """
    Preprocess query for better retrieval.

    Returns:
        Dict with:
        - original: original query
        - cleaned: lowercase, stripped
        - keywords: list of meaningful keywords (stopwords removed)
        - ts_query: PostgreSQL tsquery format string
    """
    original = query.strip()
    cleaned = original.lower()

    # Extract words, remove punctuation
    words = re.findall(r'\b\w+\b', cleaned)

    # Filter out stopwords and short words
    keywords = [w for w in words if w not in STOPWORDS and len(w) > 2]

    # Build PostgreSQL tsquery string (OR for recall, words joined with |)
    # Also include original words for exact matching
    ts_words = list(set(keywords))[:10]  # Limit to 10 keywords
    ts_query = " | ".join(ts_words) if ts_words else cleaned

    logger.debug(f"Query preprocessing: {len(words)} words -> {len(keywords)} keywords")

    return {
        "original": original,
        "cleaned": cleaned,
        "keywords": keywords,
        "ts_query": ts_query,
    }


# =============================================================================
# FULL-TEXT SEARCH (BM25-like keyword matching)
# =============================================================================

async def _fulltext_search(
    query_info: dict[str, Any],
    user_id: str = None,
    document_ids: list[str] = None,
    limit: int = KEYWORD_FETCH_COUNT
) -> list[dict[str, Any]]:
    """
    Full-text search using PostgreSQL tsvector/tsquery.

    Uses ts_rank for BM25-like scoring that considers:
    - Term frequency (TF)
    - Document length normalization
    - Word proximity
    """
    keywords = query_info.get("keywords", [])
    if not keywords:
        logger.debug("No keywords for full-text search")
        return []

    logger.info(f"Full-text search with keywords: {keywords[:5]}, user_id={user_id}")

    try:
        from app.database.connection import SessionLocal
        from sqlalchemy import text

        session = SessionLocal()
        try:
            # Build search query - use plainto_tsquery for robustness
            # Also try ILIKE as fallback for exact matches
            search_terms = " ".join(keywords[:5])
            ilike_pattern = "%" + "%".join(keywords[:3]) + "%"

            if user_id:
                # Use ts_rank with tsvector for proper ranking
                # Fallback to ILIKE for documents without tsvector
                # Cast user_id to UUID for proper comparison
                sql = text("""
                    WITH ranked_docs AS (
                        SELECT
                            dc.id,
                            dc.document_id,
                            dc.content,
                            dc.metadata,
                            COALESCE(
                                ts_rank_cd(
                                    to_tsvector('english', COALESCE(dc.content, '')),
                                    plainto_tsquery('english', :search_terms),
                                    32  -- Normalization: divide by document length
                                ),
                                0
                            ) as rank_score,
                            CASE
                                WHEN dc.content ILIKE :ilike_pattern THEN 0.3
                                ELSE 0
                            END as ilike_bonus
                        FROM document_chunks dc
                        JOIN documents d ON dc.document_id = d.id
                        WHERE d.user_id = CAST(:user_id AS uuid)
                          AND (
                              to_tsvector('english', COALESCE(dc.content, '')) @@
                              plainto_tsquery('english', :search_terms)
                              OR dc.content ILIKE :ilike_pattern
                          )
                    )
                    SELECT id, document_id, content, metadata,
                           (rank_score + ilike_bonus) as similarity
                    FROM ranked_docs
                    ORDER BY similarity DESC
                    LIMIT :limit
                """)
                result = await asyncio.to_thread(
                    lambda: session.execute(sql, {
                        "user_id": user_id,
                        "search_terms": search_terms,
                        "ilike_pattern": ilike_pattern,
                        "limit": limit
                    }).fetchall()
                )
            else:
                sql = text("""
                    WITH ranked_docs AS (
                        SELECT
                            dc.id,
                            dc.document_id,
                            dc.content,
                            dc.metadata,
                            COALESCE(
                                ts_rank_cd(
                                    to_tsvector('english', COALESCE(dc.content, '')),
                                    plainto_tsquery('english', :search_terms),
                                    32
                                ),
                                0
                            ) as rank_score,
                            CASE
                                WHEN dc.content ILIKE :ilike_pattern THEN 0.3
                                ELSE 0
                            END as ilike_bonus
                        FROM document_chunks dc
                        WHERE to_tsvector('english', COALESCE(dc.content, '')) @@
                              plainto_tsquery('english', :search_terms)
                              OR dc.content ILIKE :ilike_pattern
                    )
                    SELECT id, document_id, content, metadata,
                           (rank_score + ilike_bonus) as similarity
                    FROM ranked_docs
                    ORDER BY similarity DESC
                    LIMIT :limit
                """)
                result = await asyncio.to_thread(
                    lambda: session.execute(sql, {
                        "search_terms": search_terms,
                        "ilike_pattern": ilike_pattern,
                        "limit": limit
                    }).fetchall()
                )

            documents = []
            if result:
                logger.info(f"Full-text search found {len(result)} documents")
                for rank, row in enumerate(result):
                    chunk_id, doc_id, content, metadata, similarity = row

                    if document_ids and str(doc_id) not in document_ids:
                        continue

                    metadata = metadata or {}
                    documents.append({
                        "id": str(chunk_id),
                        "document_id": str(doc_id),
                        "content": content or "",
                        "source": metadata.get("source", "Unknown"),
                        "score": float(similarity) if similarity else 0.1,
                        "metadata": metadata,
                        "search_type": "fulltext",
                        "rank": rank + 1,  # 1-indexed rank for RRF
                    })

            return documents

        finally:
            session.close()

    except Exception as e:
        logger.error(f"Full-text search failed: {e}")
        return []


# =============================================================================
# VECTOR SEARCH (Semantic understanding)
# =============================================================================

async def _vector_search(
    query: str,
    user_id: str = None,
    document_ids: list[str] = None,
    limit: int = VECTOR_FETCH_COUNT
) -> list[dict[str, Any]]:
    """
    Vector similarity search using embeddings.
    Returns documents ranked by cosine similarity.
    """
    logger.info(f"Vector search for: {query[:50]}..., user_id={user_id}")

    try:
        from app.database.connection import SessionLocal
        from sqlalchemy import text

        # Get embedding
        embedding_service = await _get_embedding_service()
        query_embedding = await asyncio.to_thread(
            embedding_service.embed_query, query
        )
        embedding_str = "[" + ",".join(map(str, query_embedding)) + "]"

        session = SessionLocal()
        try:
            if user_id:
                # Cast user_id to UUID for proper comparison with database column
                sql = text(f"""
                    SELECT
                        dc.id,
                        dc.document_id,
                        dc.content,
                        dc.metadata,
                        1 - (dc.embedding <=> '{embedding_str}'::vector) AS similarity
                    FROM document_chunks dc
                    JOIN documents d ON dc.document_id = d.id
                    WHERE d.user_id = CAST(:user_id AS uuid)
                      AND dc.embedding IS NOT NULL
                    ORDER BY dc.embedding <=> '{embedding_str}'::vector
                    LIMIT :limit
                """)
                result = await asyncio.to_thread(
                    lambda: session.execute(sql, {"user_id": user_id, "limit": limit}).fetchall()
                )
            else:
                sql = text(f"""
                    SELECT
                        dc.id,
                        dc.document_id,
                        dc.content,
                        dc.metadata,
                        1 - (dc.embedding <=> '{embedding_str}'::vector) AS similarity
                    FROM document_chunks dc
                    WHERE dc.embedding IS NOT NULL
                    ORDER BY dc.embedding <=> '{embedding_str}'::vector
                    LIMIT :limit
                """)
                result = await asyncio.to_thread(
                    lambda: session.execute(sql, {"limit": limit}).fetchall()
                )

            documents = []
            filtered_count = 0
            if result:
                logger.info(f"Vector search raw results: {len(result)} documents")
                for rank, row in enumerate(result):
                    chunk_id, doc_id, content, metadata, similarity = row

                    if document_ids and str(doc_id) not in document_ids:
                        continue

                    # Log results with wording (INFO level for visibility)
                    if rank < 3:
                        wording = content[:150].replace("\n", " ") if content else "Empty"
                        logger.info(f"  Vector result {rank+1}: similarity={similarity:.4f} | Wording: {wording}...")

                    # RELAXED: Only filter very low similarity (< 0.01)
                    # Let the reranker handle quality filtering
                    if similarity < 0.01:
                        filtered_count += 1
                        continue

                    metadata = metadata or {}
                    documents.append({
                        "id": str(chunk_id),
                        "document_id": str(doc_id),
                        "content": content or "",
                        "source": metadata.get("source", "Unknown"),
                        "score": float(similarity),
                        "metadata": metadata,
                        "search_type": "vector",
                        "rank": rank + 1,  # 1-indexed rank for RRF
                    })

                if filtered_count > 0:
                    logger.info(f"  Filtered out {filtered_count} results with similarity < 0.01")

            logger.info(f"Vector search returning {len(documents)} documents after filtering")
            return documents

        finally:
            session.close()

    except Exception as e:
        logger.error(f"Vector search failed: {e}")
        return []


# =============================================================================
# RECIPROCAL RANK FUSION (RRF)
# =============================================================================

def reciprocal_rank_fusion(
    result_lists: list[list[dict[str, Any]]],
    k: int = RRF_K
) -> list[dict[str, Any]]:
    """
    Combine multiple ranked lists using Reciprocal Rank Fusion.

    RRF Score = Σ 1 / (k + rank_i) for each list where document appears

    This method:
    - Doesn't require score normalization
    - Handles missing documents gracefully
    - Balances recall from different retrieval methods

    Args:
        result_lists: List of result lists, each containing docs with 'id' and 'rank'
        k: Constant to prevent high scores for top-ranked docs (default 60)

    Returns:
        Fused list sorted by RRF score
    """
    # Track RRF scores and document data by ID
    rrf_scores: dict[str, float] = {}
    doc_data: dict[str, dict[str, Any]] = {}
    search_types: dict[str, list[str]] = {}

    for list_idx, result_list in enumerate(result_lists):
        for doc in result_list:
            doc_id = doc.get("id")
            if not doc_id:
                continue

            rank = doc.get("rank", 1)
            rrf_score = 1.0 / (k + rank)

            # Accumulate RRF score
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + rrf_score

            # Store document data (keep highest scoring version)
            if doc_id not in doc_data or doc.get("score", 0) > doc_data[doc_id].get("score", 0):
                doc_data[doc_id] = doc.copy()

            # Track which search types found this doc
            search_type = doc.get("search_type", f"list_{list_idx}")
            if doc_id not in search_types:
                search_types[doc_id] = []
            if search_type not in search_types[doc_id]:
                search_types[doc_id].append(search_type)

    # Build final results with RRF scores
    fused_results = []
    for doc_id, rrf_score in rrf_scores.items():
        doc = doc_data[doc_id]
        doc["rrf_score"] = rrf_score
        doc["original_score"] = doc.get("score", 0)
        doc["score"] = rrf_score  # Use RRF as primary score
        doc["found_by"] = search_types.get(doc_id, [])
        fused_results.append(doc)

    # Sort by RRF score descending
    fused_results.sort(key=lambda x: x["rrf_score"], reverse=True)

    # Log fusion stats
    both_count = sum(1 for d in fused_results if len(d.get("found_by", [])) > 1)
    logger.info(
        f"RRF fusion: {len(fused_results)} unique docs, "
        f"{both_count} found by multiple methods"
    )

    return fused_results


# =============================================================================
# LEGACY FALLBACK (when hybrid search is disabled)
# =============================================================================

async def _keyword_fallback_search(
    query: str,
    user_id: str = None,
    document_ids: list[str] = None,
    limit: int = 10
) -> list[dict[str, Any]]:
    """
    Simple keyword fallback when hybrid search is disabled.
    Uses basic ILIKE pattern matching.
    """
    logger.info(f"Using simple keyword fallback search for user_id={user_id}...")

    try:
        from app.database.connection import SessionLocal
        from sqlalchemy import text

        session = SessionLocal()
        try:
            keywords = [w.strip() for w in query.lower().split() if len(w.strip()) > 2]
            if not keywords:
                return []

            search_pattern = "%" + "%".join(keywords[:5]) + "%"

            if user_id:
                # Cast user_id to UUID for proper comparison
                sql = text("""
                    SELECT dc.id, dc.document_id, dc.content,
                           dc.metadata, 0.5 AS similarity
                    FROM document_chunks dc
                    JOIN documents d ON dc.document_id = d.id
                    WHERE d.user_id = CAST(:user_id AS uuid) AND dc.content ILIKE :pattern
                    LIMIT :limit
                """)
                result = session.execute(sql, {
                    "user_id": user_id, "pattern": search_pattern, "limit": limit
                }).fetchall()
            else:
                sql = text("""
                    SELECT dc.id, dc.document_id, dc.content,
                           dc.metadata, 0.5 AS similarity
                    FROM document_chunks dc
                    WHERE dc.content ILIKE :pattern
                    LIMIT :limit
                """)
                result = session.execute(sql, {
                    "pattern": search_pattern, "limit": limit
                }).fetchall()

            documents = []
            if result:
                for row in result:
                    chunk_id, doc_id, content, metadata, similarity = row
                    if document_ids and str(doc_id) not in document_ids:
                        continue
                    metadata = metadata or {}
                    documents.append({
                        "id": str(chunk_id),
                        "document_id": str(doc_id),
                        "content": content or "",
                        "source": metadata.get("source", "Unknown"),
                        "score": float(similarity),
                        "metadata": metadata,
                        "search_type": "keyword_fallback",
                    })
            return documents
        finally:
            session.close()
    except Exception as e:
        logger.error(f"Keyword fallback search failed: {e}")
        return []


def _compute_content_hash(content: str) -> str:
    """Compute hash for deduplication."""
    return hashlib.md5(content.encode()).hexdigest()[:16]


def _deduplicate_documents(documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove duplicate documents by content hash."""
    seen_hashes = set()
    unique_docs = []

    for doc in documents:
        content = doc.get("content", "")
        content_hash = _compute_content_hash(content)

        if content_hash not in seen_hashes:
            seen_hashes.add(content_hash)
            unique_docs.append(doc)

    return unique_docs


async def _fetch_any_user_documents(user_id: str, limit: int = 5) -> list[dict[str, Any]]:
    """
    Fetch ANY documents for a user without search scoring.
    This is a last resort when all searches return empty.
    Returns the most recent chunks.
    """
    try:
        from app.database.connection import SessionLocal
        from sqlalchemy import text

        session = SessionLocal()
        try:
            sql = text("""
                SELECT
                    dc.id,
                    dc.document_id,
                    dc.content,
                    dc.metadata,
                    0.5 AS similarity
                FROM document_chunks dc
                JOIN documents d ON dc.document_id = d.id
                WHERE d.user_id = CAST(:user_id AS uuid)
                  AND dc.content IS NOT NULL
                  AND LENGTH(dc.content) > 50
                ORDER BY dc.created_at DESC
                LIMIT :limit
            """)
            result = session.execute(sql, {"user_id": user_id, "limit": limit}).fetchall()

            documents = []
            if result:
                for row in result:
                    chunk_id, doc_id, content, metadata, similarity = row
                    metadata = metadata or {}
                    documents.append({
                        "id": str(chunk_id),
                        "document_id": str(doc_id),
                        "content": content or "",
                        "source": metadata.get("source", "Unknown"),
                        "score": float(similarity),
                        "metadata": metadata,
                        "search_type": "last_resort",
                    })
            return documents
        finally:
            session.close()
    except Exception as e:
        logger.error(f"Last resort document fetch failed: {e}")
        return []


async def _check_user_has_documents(user_id: str) -> dict[str, Any]:
    """
    Check if user has any documents with embeddings.
    Returns diagnostic info for debugging.
    """
    try:
        from app.database.connection import SessionLocal
        from sqlalchemy import text

        session = SessionLocal()
        try:
            sql = text("""
                SELECT
                    COUNT(DISTINCT d.id) as doc_count,
                    COUNT(dc.id) as chunk_count,
                    COUNT(dc.embedding) as embedded_count
                FROM documents d
                LEFT JOIN document_chunks dc ON d.id = dc.document_id
                WHERE d.user_id = CAST(:user_id AS uuid)
            """)
            result = session.execute(sql, {"user_id": user_id}).fetchone()

            if result:
                doc_count, chunk_count, embedded_count = result
                return {
                    "has_documents": doc_count > 0,
                    "document_count": doc_count,
                    "chunk_count": chunk_count,
                    "embedded_count": embedded_count,
                }
            return {"has_documents": False, "document_count": 0, "chunk_count": 0, "embedded_count": 0}
        finally:
            session.close()
    except Exception as e:
        logger.error(f"Error checking user documents: {e}")
        return {"has_documents": False, "error": str(e)}


async def document_retrieval_node(state: AgentState) -> dict[str, Any]:
    """
    Hybrid document retrieval combining vector search and full-text search.

    Pipeline:
    1. Check if query needs intent-aware retrieval (broad queries)
    2. Preprocess query (extract keywords, remove stopwords)
    3. Run vector search (semantic) and full-text search (keyword) in parallel
    4. Merge results using Reciprocal Rank Fusion (RRF)
    5. Deduplicate by content hash
    6. Rerank with BGE cross-encoder for final relevance scoring

    User Isolation: Only retrieves documents belonging to the specified user.

    Args:
        state: Current agent state

    Returns:
        Updated state with retrieved documents
    """
    start_time = time.time()
    logger.info("Starting hybrid document retrieval")

    # Check if this query needs intent-aware retrieval
    query = state.get("original_query", "")
    try:
        from app.rag.langgraph.nodes.intent_retrieval import (
            classify_query_intent,
            QueryScope,
            intent_aware_retrieval_node,
        )

        intent = classify_query_intent(query)

        # For BROAD scope queries, use intent-aware retrieval
        if intent.scope == QueryScope.BROAD:
            logger.info(f"Query is BROAD scope - using intent-aware retrieval")
            return await intent_aware_retrieval_node(state)

        logger.info(f"Query is {intent.scope.value} scope - using standard retrieval")

    except ImportError:
        logger.debug("Intent retrieval module not available, using standard retrieval")
    except Exception as e:
        logger.warning(f"Intent classification failed: {e}, using standard retrieval")

    query = state.get("original_query", "")
    user_id = state.get("user_id")
    document_ids = state.get("document_ids")

    updates = track_node(state, "document_retrieval")
    documents = []

    try:
        # Step 0: Check if user has any documents (for better error messaging)
        if user_id:
            user_doc_info = await _check_user_has_documents(user_id)
            logger.info(
                f"User {user_id} documents: {user_doc_info.get('document_count', 0)} docs, "
                f"{user_doc_info.get('chunk_count', 0)} chunks, "
                f"{user_doc_info.get('embedded_count', 0)} with embeddings"
            )

            if not user_doc_info.get("has_documents"):
                logger.warning(f"User {user_id} has no documents uploaded!")
                # Don't search - return early with helpful info
                updates.update(add_error(
                    state, "document_retrieval", "NO_DOCUMENTS",
                    "You haven't uploaded any documents yet. Please upload documents first.",
                    recoverable=True
                ))
                duration_ms = (time.time() - start_time) * 1000
                logger.info(f"Retrieval complete in {duration_ms:.1f}ms (no user documents)")
                return updates

            if user_doc_info.get("embedded_count", 0) == 0:
                logger.warning(f"User {user_id} has documents but no embeddings!")
                updates.update(add_error(
                    state, "document_retrieval", "NO_EMBEDDINGS",
                    "Your documents are being processed. Please try again shortly.",
                    recoverable=True
                ))
                duration_ms = (time.time() - start_time) * 1000
                logger.info(f"Retrieval complete in {duration_ms:.1f}ms (no embeddings)")
                return updates

        # Step 1: Preprocess query
        query_info = preprocess_query(query)
        logger.info(
            f"Query: '{query[:50]}...' -> {len(query_info['keywords'])} keywords: "
            f"{query_info['keywords'][:5]}"
        )

        if HYBRID_SEARCH_ENABLED:
            # Step 2: Run vector and full-text search in parallel
            logger.info(f"Running hybrid search (vector + full-text) for user_id={user_id}...")

            vector_task = asyncio.create_task(
                _vector_search(query, user_id, document_ids)
            )
            fulltext_task = asyncio.create_task(
                _fulltext_search(query_info, user_id, document_ids)
            )

            # Wait for both searches to complete
            vector_results, fulltext_results = await asyncio.gather(
                vector_task, fulltext_task, return_exceptions=True
            )

            # Handle exceptions gracefully
            if isinstance(vector_results, Exception):
                logger.error(f"Vector search failed: {vector_results}")
                vector_results = []
            if isinstance(fulltext_results, Exception):
                logger.error(f"Full-text search failed: {fulltext_results}")
                fulltext_results = []

            logger.info(
                f"Search results: vector={len(vector_results)}, "
                f"fulltext={len(fulltext_results)}"
            )

            # Step 3: Merge with Reciprocal Rank Fusion
            if vector_results or fulltext_results:
                documents = reciprocal_rank_fusion([vector_results, fulltext_results])

            # If no RRF results, try keyword fallback (still user-scoped)
            if not documents and user_id:
                logger.info("Hybrid search found no results, trying keyword fallback...")
                documents = await _keyword_fallback_search(query, user_id, document_ids)

            # LAST RESORT: If still no results but user HAS documents, fetch ANY chunks
            # This ensures users always get SOME context from their documents
            if not documents and user_id:
                logger.warning("All searches failed - fetching ANY user documents as last resort...")
                documents = await _fetch_any_user_documents(user_id, limit=5)
                if documents:
                    logger.info(f"Last resort: fetched {len(documents)} random documents")

        else:
            # Hybrid search disabled - use vector search only with fallback
            logger.info("Hybrid search disabled, using vector search only")
            documents = await _vector_search(query, user_id, document_ids)

            if not documents:
                logger.warning("Vector search returned no results, trying keyword fallback")
                documents = await _keyword_fallback_search(query, user_id, document_ids)

        # Step 4: Deduplicate by content
        documents = _deduplicate_documents(documents)
        logger.info(f"After deduplication: {len(documents)} unique documents")

        # Step 5: Rerank with cross-encoder
        if documents:
            documents = await _rerank_documents(query, documents)
            logger.info(f"Returning top {len(documents)} documents after reranking")

            # Log top results with wording for debugging (INFO level for visibility)
            for i, doc in enumerate(documents[:3]):
                found_by = doc.get("found_by", ["unknown"])
                content_preview = doc.get("content", "")[:250].replace("\n", " ")
                logger.info(
                    f"🔍 CHUNK {i+1} [{found_by}]: score={doc.get('score', 0):.4f}, "
                    f"source={doc.get('source', 'unknown')}"
                )
                logger.info(f"    - Wording: {content_preview}...")
        else:
            logger.warning(
                f"No documents found for query: {query[:50]}... "
                f"(user_id={user_id}). Query may not match document content."
            )

    except asyncio.TimeoutError:
        logger.error(f"Document retrieval timeout after {RETRIEVAL_TIMEOUT}s")
        updates.update(add_error(
            state, "document_retrieval", "TIMEOUT_ERROR",
            f"Retrieval timeout after {RETRIEVAL_TIMEOUT}s", recoverable=True
        ))

    except Exception as e:
        logger.error(f"Document retrieval error: {e}", exc_info=True)
        updates.update(add_error(
            state, "document_retrieval", "RETRIEVAL_ERROR",
            str(e), recoverable=True
        ))

    # Add documents to state
    if documents:
        updates.update(add_documents(state, documents))

    duration_ms = (time.time() - start_time) * 1000
    logger.info(f"Hybrid retrieval complete in {duration_ms:.1f}ms")

    return updates


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "document_retrieval_node",
    "preprocess_query",
    "reciprocal_rank_fusion",
    "HYBRID_SEARCH_ENABLED",
]
