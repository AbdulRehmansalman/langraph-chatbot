"""
Document Retrieval Node
=======================

Simple document retrieval for LangGraph agent.
Uses vector store for semantic search.
"""

import asyncio
import hashlib
import logging
import time
from typing import Any

from app.rag.langgraph.state import (
    AgentState,
    track_node,
    add_documents,
    add_error,
)

logger = logging.getLogger(__name__)

RETRIEVAL_TIMEOUT = 10.0  # seconds


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


async def document_retrieval_node(state: AgentState) -> dict[str, Any]:
    """
    Retrieve relevant documents for the query.

    Args:
        state: Current agent state

    Returns:
        Updated state with retrieved documents
    """
    start_time = time.time()
    logger.info("Starting document retrieval")

    query = state.get("original_query", "")
    user_id = state.get("user_id")
    document_ids = state.get("document_ids")

    updates = track_node(state, "document_retrieval")
    documents = []

    try:
        from app.rag.embeddings.service import EmbeddingService
        from app.services.supabase_client import get_supabase_client

        supabase = await get_supabase_client()
        embedding_service = EmbeddingService()

        # Generate embedding for query
        query_embedding = await asyncio.to_thread(
            embedding_service.embed_query, query
        )

        # Build query for vector search
        query_builder = supabase.rpc(
            "match_documents",
            {
                "query_embedding": query_embedding,
                "match_threshold": 0.5,
                "match_count": 10,
            }
        )

        # Add user filter if provided
        if user_id:
            query_builder = query_builder.eq("user_id", user_id)

        # Add document ID filter if provided
        if document_ids:
            query_builder = query_builder.in_("document_id", document_ids)

        # Execute with timeout
        result = await asyncio.wait_for(
            asyncio.to_thread(query_builder.execute),
            timeout=RETRIEVAL_TIMEOUT
        )

        if result.data:
            for i, doc in enumerate(result.data):
                documents.append({
                    "id": doc.get("id", f"doc_{i}"),
                    "content": doc.get("content", ""),
                    "source": doc.get("metadata", {}).get("source", "Unknown"),
                    "score": doc.get("similarity", 0.0),
                    "metadata": doc.get("metadata", {}),
                })

        # Deduplicate
        documents = _deduplicate_documents(documents)

        logger.info(f"Retrieved {len(documents)} documents")

    except asyncio.TimeoutError:
        logger.error(f"Document retrieval timeout after {RETRIEVAL_TIMEOUT}s")
        updates.update(add_error(
            state, "document_retrieval", "TIMEOUT_ERROR",
            f"Retrieval timeout after {RETRIEVAL_TIMEOUT}s", recoverable=True
        ))

    except Exception as e:
        logger.error(f"Document retrieval error: {e}")
        updates.update(add_error(
            state, "document_retrieval", "RETRIEVAL_ERROR",
            str(e), recoverable=True
        ))

    # Add documents to state
    if documents:
        updates.update(add_documents(state, documents))

    duration_ms = (time.time() - start_time) * 1000
    logger.info(f"Document retrieval complete in {duration_ms:.1f}ms")

    return updates


__all__ = ["document_retrieval_node"]
