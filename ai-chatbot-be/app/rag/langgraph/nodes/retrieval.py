"""
Document Retrieval Node
=======================

Production-ready simplified version.
Maintains exact same function signatures for compatibility.
"""

import asyncio
import hashlib
import logging
import os
import re
import time
from typing import Any, Optional, List, Dict

from app.core.config import settings
from app.rag.langgraph.state import (
    AgentState,
    track_node,
    add_documents,
    add_error,
)

logger = logging.getLogger(__name__)

# Configuration (fetched from central settings)
RETRIEVAL_TIMEOUT = settings.rag_timeout
VECTOR_FETCH_COUNT = settings.rag_top_k
KEYWORD_FETCH_COUNT = settings.rag_top_k
FINAL_RETURN_COUNT = settings.rag_rerank_top_k
MATCH_THRESHOLD = settings.rag_score_threshold
HYBRID_SEARCH_ENABLED = settings.rag_retrieval_strategy == "hybrid"

# Lazy-loaded components (same names)
_embedding_service: Optional[Any] = None
_embedding_lock = asyncio.Lock()

# Simple stopwords
STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "must", "shall", "can", "need", "dare",
}


# =============================================================================
# CORE FUNCTIONS (SAME NAMES, SIMPLER IMPLEMENTATIONS)
# =============================================================================

# if aleready implemented so return it 
async def _get_embedding_service():
    """Get embedding service - simplified."""
    global _embedding_service
    if _embedding_service is not None:
        return _embedding_service

    async with _embedding_lock:
        if _embedding_service is not None:
            return _embedding_service

        try:
            from app.rag.embeddings.service import EmbeddingsService
            logger.info("Loading embedding service...")
            _embedding_service = EmbeddingsService()
            return _embedding_service
        except Exception as e:
            logger.error(f"Failed to load embedding service: {e}")
            raise


def preprocess_query(query: str) -> Dict[str, Any]:
    """
    Simplified query preprocessing.
    Same signature, simpler implementation.
    """
    original = query.strip()
    cleaned = original.lower()
    
    # Extract keywords
    words = re.findall(r'\b\w+\b', cleaned)
    keywords = [w for w in words if w not in STOPWORDS and len(w) > 2]
    
    return {
        "original": original,
        "cleaned": cleaned,
        "keywords": keywords,
        "ts_query": " ".join(keywords[:5]),  # first fIVE TEXT AND JOIN IT
    }


async def _fulltext_search(
    query_info: Dict[str, Any],
    user_id: str = None,
    document_ids: List[str] = None,
    limit: int = KEYWORD_FETCH_COUNT
) -> List[Dict[str, Any]]:
    """
    Simplified keyword search.
    Same signature, simpler SQL.
    """
    keywords = query_info.get("keywords", [])
    if not keywords:
        return []

    try:
        from app.database.connection import SessionLocal
        from sqlalchemy import text

        session = SessionLocal()
        try:
            # Simple ILIKE search
            search_pattern = "%" + "%".join(keywords[:3]) + "%"
            
            if user_id:
                sql = text("""
                    SELECT dc.id, dc.document_id, dc.content, dc.metadata, 0.7 AS similarity
                    FROM document_chunks dc
                    JOIN documents d ON dc.document_id = d.id
                    WHERE d.user_id = CAST(:user_id AS uuid)
                      AND dc.content ILIKE :pattern
                    ORDER BY LENGTH(dc.content) DESC
                    LIMIT :limit
                """)
                result = session.execute(sql, {
                    "user_id": user_id,
                    "pattern": search_pattern,
                    "limit": limit
                }).fetchall()
            else:
                sql = text("""
                    SELECT dc.id, dc.document_id, dc.content, dc.metadata, 0.7 AS similarity
                    FROM document_chunks dc
                    WHERE dc.content ILIKE :pattern
                    ORDER BY LENGTH(dc.content) DESC
                    LIMIT :limit
                """)
                result = session.execute(sql, {
                    "pattern": search_pattern,
                    "limit": limit
                }).fetchall()

            documents = []
            for row in result:
                chunk_id, doc_id, content, metadata, similarity = row
                
                if document_ids and str(doc_id) not in document_ids:
                    continue
                
                documents.append({
                    "id": str(chunk_id),
                    "document_id": str(doc_id),
                    "content": content or "",
                    "source": (metadata or {}).get("source", "Unknown"),
                    "score": float(similarity),
                    "metadata": metadata or {},
                    "search_type": "fulltext",
                })

            logger.info(f"Keyword search found {len(documents)} documents")
            return documents

        finally:
            session.close()

    except Exception as e:
        logger.error(f"Keyword search failed: {e}")
        return []


async def _vector_search(
    query: str,
    user_id: str = None,
    document_ids: List[str] = None,
    limit: int = VECTOR_FETCH_COUNT
) -> List[Dict[str, Any]]:
    """
    Simplified vector search.
    Same signature, cleaner implementation.
    """
    try:
        from app.database.connection import SessionLocal
        from sqlalchemy import text

        # Get embedding
        embedding_service = await _get_embedding_service()
        query_embedding = embedding_service.embed_query(query)
        embedding_str = "[" + ",".join(map(str, query_embedding)) + "]"

        session = SessionLocal()
        try:
            if user_id:
                sql = text(f"""
                    SELECT dc.id, dc.document_id, dc.content, dc.metadata,
                           1 - (dc.embedding <=> '{embedding_str}'::vector) AS similarity
                    FROM document_chunks dc
                    JOIN documents d ON dc.document_id = d.id
                    WHERE d.user_id = CAST(:user_id AS uuid)
                      AND dc.embedding IS NOT NULL
                      AND (1 - (dc.embedding <=> '{embedding_str}'::vector)) >= :threshold
                    ORDER BY similarity DESC
                    LIMIT :limit
                """)
                result = session.execute(sql, {
                    "user_id": user_id,
                    "threshold": MATCH_THRESHOLD,
                    "limit": limit
                }).fetchall()
            else:
                sql = text(f"""
                    SELECT dc.id, dc.document_id, dc.content, dc.metadata,
                           1 - (dc.embedding <=> '{embedding_str}'::vector) AS similarity
                    FROM document_chunks dc
                    WHERE dc.embedding IS NOT NULL
                      AND (1 - (dc.embedding <=> '{embedding_str}'::vector)) >= :threshold
                    ORDER BY similarity DESC
                    LIMIT :limit
                """)
                result = session.execute(sql, {
                    "threshold": MATCH_THRESHOLD,
                    "limit": limit
                }).fetchall()

            documents = []
            for row in result:
                chunk_id, doc_id, content, metadata, similarity = row
                
                if document_ids and str(doc_id) not in document_ids:
                    continue
                
                documents.append({
                    "id": str(chunk_id),
                    "document_id": str(doc_id),
                    "content": content or "",
                    "source": (metadata or {}).get("source", "Unknown"),
                    "score": float(similarity),
                    "metadata": metadata or {},
                    "search_type": "vector",
                })

            logger.info(f"Vector search found {len(documents)} documents")
            return documents

        finally:
            session.close()

    except Exception as e:
        logger.error(f"Vector search failed: {e}")
        return []


def _compute_content_hash(content: str) -> str:
    """Same function, same implementation."""
    return hashlib.md5(content.encode()).hexdigest()[:16]


def _deduplicate_documents(documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Same function, same implementation."""
    seen_hashes = set()
    unique_docs = []

    for doc in documents:
        content = doc.get("content", "")
        content_hash = _compute_content_hash(content)

        if content_hash not in seen_hashes:
            seen_hashes.add(content_hash)
            unique_docs.append(doc)

    return unique_docs


async def _keyword_fallback_search(
    query: str,
    user_id: str = None,
    document_ids: List[str] = None,
    limit: int = 10
) -> List[Dict[str, Any]]:
    """
    Same fallback search - simplified.
    """
    try:
        from app.database.connection import SessionLocal
        from sqlalchemy import text

        session = SessionLocal()
        try:
            # Simple keyword matching
            words = [w.strip() for w in query.lower().split() if len(w.strip()) > 2]
            if not words:
                return []
            
            search_pattern = "%" + "%".join(words[:3]) + "%"

            if user_id:
                sql = text("""
                    SELECT dc.id, dc.document_id, dc.content, dc.metadata, 0.5 AS similarity
                    FROM document_chunks dc
                    JOIN documents d ON dc.document_id = d.id
                    WHERE d.user_id = CAST(:user_id AS uuid)
                      AND dc.content ILIKE :pattern
                    LIMIT :limit
                """)
                result = session.execute(sql, {
                    "user_id": user_id,
                    "pattern": search_pattern,
                    "limit": limit
                }).fetchall()
            else:
                sql = text("""
                    SELECT dc.id, dc.document_id, dc.content, dc.metadata, 0.5 AS similarity
                    FROM document_chunks dc
                    WHERE dc.content ILIKE :pattern
                    LIMIT :limit
                """)
                result = session.execute(sql, {
                    "pattern": search_pattern,
                    "limit": limit
                }).fetchall()

            documents = []
            for row in result:
                chunk_id, doc_id, content, metadata, similarity = row
                
                if document_ids and str(doc_id) not in document_ids:
                    continue
                
                documents.append({
                    "id": str(chunk_id),
                    "document_id": str(doc_id),
                    "content": content or "",
                    "source": (metadata or {}).get("source", "Unknown"),
                    "score": float(similarity),
                    "metadata": metadata or {},
                    "search_type": "keyword_fallback",
                })
            
            return documents
            
        finally:
            session.close()
            
    except Exception as e:
        logger.error(f"Fallback search failed: {e}")
        return []


# =============================================================================
# SIMPLIFIED MERGING (REPLACES COMPLEX RRF)
# =============================================================================

def _simple_merge_results(
    vector_results: List[Dict[str, Any]],
    keyword_results: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Simple merging instead of complex RRF.
    """
    # Combine all documents
    all_docs = {}
    
    # Add vector results first
    for doc in vector_results:
        doc_id = doc.get("id")
        if doc_id:
            all_docs[doc_id] = doc
    
    # Add keyword results (keep higher score)
    for doc in keyword_results:
        doc_id = doc.get("id")
        if doc_id:
            if doc_id not in all_docs or doc["score"] > all_docs[doc_id]["score"]:
                all_docs[doc_id] = doc
    
    # Sort by score
    merged = list(all_docs.values())
    merged.sort(key=lambda x: x["score"], reverse=True)
    
    logger.info(f"Merged {len(vector_results)} + {len(keyword_results)} -> {len(merged)} unique docs")
    return merged


# =============================================================================
# MAIN RETRIEVAL NODE (SAME SIGNATURE)
# =============================================================================

async def document_retrieval_node(state: AgentState) -> Dict[str, Any]:
    """
    Simplified document retrieval node.
    EXACT SAME SIGNATURE as before.
    """
    start_time = time.time()
    logger.info("Starting simplified document retrieval")
    
    query = state.get("original_query", "")
    user_id = state.get("user_id")
    document_ids = state.get("document_ids")
    
    updates = track_node(state, "document_retrieval")
    documents = []
    
    try:
        # Step 1: Preprocess query
        query_info = preprocess_query(query)
        
        if HYBRID_SEARCH_ENABLED:
            # Step 2: Run both searches in parallel
            vector_task = asyncio.create_task(
                _vector_search(query, user_id, document_ids)
            )
            keyword_task = asyncio.create_task(
                _fulltext_search(query_info, user_id, document_ids)
            )
            
            vector_results, keyword_results = await asyncio.gather(
                vector_task, keyword_task
            )
            
            # Step 3: Simple merge
            if vector_results or keyword_results:
                documents = _simple_merge_results(vector_results, keyword_results)
            
            # Fallback if no results
            if not documents and user_id:
                documents = await _keyword_fallback_search(query, user_id, document_ids)
                
        else:
            # Vector search only
            documents = await _vector_search(query, user_id, document_ids)
            
            if not documents:
                documents = await _keyword_fallback_search(query, user_id, document_ids)
        
        # Step 4: Deduplicate
        documents = _deduplicate_documents(documents)
        
        # Step 5: Limit results
        documents = documents[:FINAL_RETURN_COUNT]
        
        # Add to state
        if documents:
            updates.update(add_documents(state, documents))
            logger.info(f"Retrieved {len(documents)} documents")
        else:
            logger.warning(f"No documents found for query: '{query[:50]}...'")
        
    except asyncio.TimeoutError:
        logger.error(f"Retrieval timeout after {RETRIEVAL_TIMEOUT}s")
        updates.update(add_error(
            state, "document_retrieval", "TIMEOUT_ERROR",
            f"Retrieval timeout after {RETRIEVAL_TIMEOUT}s", recoverable=True
        ))
        
    except Exception as e:
        logger.error(f"Retrieval error: {e}", exc_info=True)
        updates.update(add_error(
            state, "document_retrieval", "RETRIEVAL_ERROR",
            str(e), recoverable=True
        ))
    
    duration_ms = (time.time() - start_time) * 1000
    logger.info(f"Retrieval completed in {duration_ms:.1f}ms")
    
    return updates


# =============================================================================
# EXPORTS (SAME AS BEFORE)
# =============================================================================

__all__ = [
    "document_retrieval_node",
    "preprocess_query",
    "HYBRID_SEARCH_ENABLED",
]