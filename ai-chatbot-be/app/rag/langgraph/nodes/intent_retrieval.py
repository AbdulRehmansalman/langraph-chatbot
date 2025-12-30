"""
Intent-Aware Retrieval System
=============================

Semantic understanding BEFORE retrieval:
1. Intent Classification - understand query scope and type
2. Query Expansion - generate semantic variations
3. Multi-Angle Retrieval - search from multiple perspectives
4. Aggregation - combine and rank for complete coverage

Optimized for broad, intent-based questions like:
- "What services do you provide?"
- "Tell me about your offerings"
- "What can you help me with?"
"""

import asyncio
import hashlib
import logging
import os
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# INTENT CLASSIFICATION
# =============================================================================

class QueryScope(Enum):
    """Query scope classification."""
    BROAD = "broad"      # High-level, needs multi-section retrieval
    MEDIUM = "medium"    # Moderate scope
    NARROW = "narrow"    # Specific, single-answer query


class QueryType(Enum):
    """Query type classification."""
    ENUMERATION = "enumeration"    # List all X (services, features, etc.)
    EXPLANATION = "explanation"    # Explain how/what/why
    COMPARISON = "comparison"      # Compare X vs Y
    SPECIFIC = "specific"          # Specific fact lookup
    PROCEDURAL = "procedural"      # How to do X


@dataclass
class QueryIntent:
    """Parsed query intent."""
    original_query: str
    scope: QueryScope
    query_type: QueryType
    domain: str  # services, policies, procedures, etc.
    expanded_queries: list[str]
    target_chunk_count: int  # How many chunks to retrieve
    confidence: float


# Semantic concept mappings - words that mean the same thing
CONCEPT_SYNONYMS = {
    "services": [
        "services", "offerings", "capabilities", "solutions",
        "products", "what we do", "what you do", "what we offer",
        "what you offer", "features", "service offerings",
        "service catalog", "our services", "available services"
    ],
    "policies": [
        "policies", "policy", "rules", "guidelines", "procedures",
        "regulations", "requirements", "standards", "protocols"
    ],
    "pricing": [
        "pricing", "prices", "cost", "costs", "rates", "fees",
        "charges", "payment", "billing", "how much"
    ],
    "contact": [
        "contact", "reach", "get in touch", "phone", "email",
        "address", "location", "support", "help desk"
    ],
    "about": [
        "about", "who are you", "company", "organization",
        "background", "history", "mission", "vision", "team"
    ],
    "benefits": [
        "benefits", "advantages", "perks", "compensation",
        "insurance", "health", "retirement", "401k", "pto",
        "vacation", "leave", "time off"
    ],
    "procedures": [
        "procedures", "process", "how to", "steps", "guide",
        "instructions", "workflow", "method"
    ],
}

# Broad query patterns that need multi-section retrieval
BROAD_QUERY_PATTERNS = [
    r"^what\s+(services?|offerings?|capabilities?|solutions?)",
    r"^(tell|show)\s+me\s+(about|all)",
    r"^(list|enumerate|describe)\s+(all|the|your)",
    r"^what\s+(do you|can you)\s+(do|offer|provide|help)",
    r"^(give|provide)\s+me\s+(an?\s+)?(overview|summary|list)",
    r"^what\s+are\s+(your|the|all)",
    r"^(explain|describe)\s+(your|the|all)",
]

# Enumeration patterns - queries asking for lists
ENUMERATION_PATTERNS = [
    r"^(what|which)\s+(are|is)\s+(your|the|all)",
    r"^(list|enumerate|name)\s+",
    r"^(tell|show)\s+me\s+(all|about)",
    r"\b(all|every|each|list\s+of)\b",
]


def classify_query_intent(query: str) -> QueryIntent:
    """
    Classify the query's intent, scope, and domain.

    This is the FIRST step - understanding what the user wants
    BEFORE we search.
    """
    query_lower = query.lower().strip()
    query_words = set(re.findall(r'\b\w+\b', query_lower))

    # Determine scope
    scope = QueryScope.NARROW  # Default
    for pattern in BROAD_QUERY_PATTERNS:
        if re.search(pattern, query_lower):
            scope = QueryScope.BROAD
            break

    # Check word count - longer queries are often more specific
    if scope != QueryScope.BROAD and len(query_words) > 10:
        scope = QueryScope.NARROW
    elif scope != QueryScope.BROAD and len(query_words) <= 5:
        scope = QueryScope.MEDIUM

    # Determine query type
    query_type = QueryType.SPECIFIC  # Default
    for pattern in ENUMERATION_PATTERNS:
        if re.search(pattern, query_lower):
            query_type = QueryType.ENUMERATION
            break

    if query_type != QueryType.ENUMERATION:
        if re.search(r"^how\s+(do|can|to|does)", query_lower):
            query_type = QueryType.PROCEDURAL
        elif re.search(r"(vs\.?|versus|compare|difference)", query_lower):
            query_type = QueryType.COMPARISON
        elif re.search(r"^(what|why|explain|describe)", query_lower):
            query_type = QueryType.EXPLANATION

    # Determine domain by matching concept synonyms
    domain = "general"
    max_matches = 0
    for concept, synonyms in CONCEPT_SYNONYMS.items():
        matches = sum(1 for syn in synonyms if syn in query_lower)
        if matches > max_matches:
            max_matches = matches
            domain = concept

    # Expand query based on domain
    expanded_queries = expand_query(query, domain, scope)

    # Determine target chunk count based on scope
    if scope == QueryScope.BROAD:
        target_chunks = 15  # Get more chunks for broad queries
    elif scope == QueryScope.MEDIUM:
        target_chunks = 10
    else:
        target_chunks = 5

    # For enumeration queries, always get more
    if query_type == QueryType.ENUMERATION:
        target_chunks = max(target_chunks, 12)

    intent = QueryIntent(
        original_query=query,
        scope=scope,
        query_type=query_type,
        domain=domain,
        expanded_queries=expanded_queries,
        target_chunk_count=target_chunks,
        confidence=0.8 if max_matches > 0 else 0.5,
    )

    logger.info(
        f"Intent classified: scope={scope.value}, type={query_type.value}, "
        f"domain={domain}, expansions={len(expanded_queries)}, "
        f"target_chunks={target_chunks}"
    )

    return intent


def expand_query(query: str, domain: str, scope: QueryScope) -> list[str]:
    """
    Expand query into semantic variations.

    For "What services do you provide?", generates:
    - "What services do you provide?"
    - "What offerings do you have?"
    - "What are your capabilities?"
    - "What solutions do you offer?"
    - "List of available services"
    """
    expanded = [query]  # Always include original
    query_lower = query.lower()

    # Get synonyms for the domain
    synonyms = CONCEPT_SYNONYMS.get(domain, [])

    if scope == QueryScope.BROAD and synonyms:
        # For broad queries, create variations using synonyms
        base_patterns = [
            "What {word} do you provide?",
            "What {word} do you offer?",
            "Tell me about your {word}",
            "List of {word}",
            "All available {word}",
            "Describe your {word}",
            "{word} overview",
        ]

        # Use top synonyms to create variations
        for synonym in synonyms[:5]:
            for pattern in base_patterns[:3]:  # Limit patterns
                variation = pattern.format(word=synonym)
                if variation.lower() != query_lower:
                    expanded.append(variation)

        # Add domain-specific expansions
        if domain == "services":
            expanded.extend([
                "What can you help me with?",
                "How can you assist me?",
                "What do you specialize in?",
            ])
        elif domain == "benefits":
            expanded.extend([
                "What are the employee benefits?",
                "Tell me about compensation and benefits",
                "What perks are available?",
            ])

    # Limit total expansions
    return expanded[:8]


# =============================================================================
# MULTI-ANGLE RETRIEVAL
# =============================================================================

async def intent_aware_retrieval(
    query: str,
    user_id: str,
    document_ids: list[str] = None,
) -> tuple[list[dict[str, Any]], QueryIntent]:
    """
    Intent-aware retrieval that understands semantic meaning.

    1. Classify intent
    2. Expand query
    3. Multi-angle search
    4. Aggregate results
    """
    # Step 1: Classify intent
    intent = classify_query_intent(query)

    logger.info(f"Starting intent-aware retrieval: {intent.scope.value} scope, {intent.domain} domain")

    # Step 2: Multi-angle retrieval based on scope
    if intent.scope == QueryScope.BROAD:
        documents = await broad_scope_retrieval(intent, user_id, document_ids)
    elif intent.scope == QueryScope.MEDIUM:
        documents = await medium_scope_retrieval(intent, user_id, document_ids)
    else:
        documents = await narrow_scope_retrieval(intent, user_id, document_ids)

    logger.info(f"Intent-aware retrieval complete: {len(documents)} documents")

    return documents, intent


async def broad_scope_retrieval(
    intent: QueryIntent,
    user_id: str,
    document_ids: list[str] = None,
) -> list[dict[str, Any]]:
    """
    Retrieval strategy for BROAD scope queries.

    - Search with ALL expanded queries
    - Search by section headings/metadata
    - Lower similarity threshold
    - Aggregate and deduplicate
    - Return more chunks (10-15)
    """
    from app.rag.embeddings.service import EmbeddingsService
    from app.database.connection import SessionLocal
    from sqlalchemy import text

    all_results = []
    seen_ids = set()

    try:
        embedding_service = EmbeddingsService()
        session = SessionLocal()

        try:
            # Strategy 1: Vector search with expanded queries
            logger.info(f"Broad retrieval: searching with {len(intent.expanded_queries)} query variations")

            for i, expanded_query in enumerate(intent.expanded_queries[:5]):
                query_embedding = await asyncio.to_thread(
                    embedding_service.embed_query, expanded_query
                )
                embedding_str = "[" + ",".join(map(str, query_embedding)) + "]"

                # Lower threshold for broad queries (0.2 instead of default)
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
                      AND LENGTH(dc.content) > 50
                    ORDER BY dc.embedding <=> '{embedding_str}'::vector
                    LIMIT 10
                """)

                result = await asyncio.to_thread(
                    lambda: session.execute(sql, {"user_id": user_id}).fetchall()
                )

                for row in result:
                    chunk_id, doc_id, content, metadata, similarity = row

                    if str(chunk_id) in seen_ids:
                        continue

                    # Lower threshold for broad queries
                    if similarity < 0.15:
                        continue

                    seen_ids.add(str(chunk_id))
                    metadata = metadata or {}

                    all_results.append({
                        "id": str(chunk_id),
                        "document_id": str(doc_id),
                        "content": content or "",
                        "source": metadata.get("source", "Unknown"),
                        "score": float(similarity),
                        "metadata": metadata,
                        "search_type": f"expanded_query_{i}",
                        "query_used": expanded_query,
                    })

            # Strategy 2: Metadata/heading search for the domain
            logger.info(f"Broad retrieval: searching by headings for domain '{intent.domain}'")

            domain_keywords = CONCEPT_SYNONYMS.get(intent.domain, [intent.domain])
            heading_pattern = "|".join(domain_keywords[:5])

            sql = text(f"""
                SELECT
                    dc.id,
                    dc.document_id,
                    dc.content,
                    dc.metadata,
                    0.6 AS similarity
                FROM document_chunks dc
                JOIN documents d ON dc.document_id = d.id
                WHERE d.user_id = CAST(:user_id AS uuid)
                  AND dc.content IS NOT NULL
                  AND LENGTH(dc.content) > 50
                  AND (
                      dc.metadata->>'heading' ~* :pattern
                      OR dc.metadata->>'section' ~* :pattern
                      OR dc.content ~* :pattern
                  )
                LIMIT 15
            """)

            result = await asyncio.to_thread(
                lambda: session.execute(sql, {
                    "user_id": user_id,
                    "pattern": heading_pattern,
                }).fetchall()
            )

            for row in result:
                chunk_id, doc_id, content, metadata, similarity = row

                if str(chunk_id) in seen_ids:
                    continue

                seen_ids.add(str(chunk_id))
                metadata = metadata or {}

                all_results.append({
                    "id": str(chunk_id),
                    "document_id": str(doc_id),
                    "content": content or "",
                    "source": metadata.get("source", "Unknown"),
                    "score": float(similarity),
                    "metadata": metadata,
                    "search_type": "heading_match",
                })

            # Strategy 3: Full-text search with domain keywords
            logger.info("Broad retrieval: full-text search with domain keywords")

            search_terms = " | ".join(domain_keywords[:5])

            sql = text(f"""
                SELECT
                    dc.id,
                    dc.document_id,
                    dc.content,
                    dc.metadata,
                    ts_rank_cd(
                        to_tsvector('english', COALESCE(dc.content, '')),
                        to_tsquery('english', :search_terms)
                    ) AS similarity
                FROM document_chunks dc
                JOIN documents d ON dc.document_id = d.id
                WHERE d.user_id = CAST(:user_id AS uuid)
                  AND to_tsvector('english', COALESCE(dc.content, '')) @@
                      to_tsquery('english', :search_terms)
                ORDER BY similarity DESC
                LIMIT 10
            """)

            result = await asyncio.to_thread(
                lambda: session.execute(sql, {
                    "user_id": user_id,
                    "search_terms": search_terms,
                }).fetchall()
            )

            for row in result:
                chunk_id, doc_id, content, metadata, similarity = row

                if str(chunk_id) in seen_ids:
                    continue

                seen_ids.add(str(chunk_id))
                metadata = metadata or {}

                all_results.append({
                    "id": str(chunk_id),
                    "document_id": str(doc_id),
                    "content": content or "",
                    "source": metadata.get("source", "Unknown"),
                    "score": float(similarity) + 0.3,  # Boost FTS results
                    "metadata": metadata,
                    "search_type": "fulltext_domain",
                })

        finally:
            session.close()

    except Exception as e:
        logger.error(f"Broad scope retrieval error: {e}", exc_info=True)

    # Sort by score and return top N
    all_results.sort(key=lambda x: x["score"], reverse=True)

    # For broad queries, return more chunks
    target = intent.target_chunk_count
    final_results = all_results[:target]

    logger.info(
        f"Broad retrieval complete: {len(all_results)} total found, "
        f"returning top {len(final_results)}"
    )

    return final_results


async def medium_scope_retrieval(
    intent: QueryIntent,
    user_id: str,
    document_ids: list[str] = None,
) -> list[dict[str, Any]]:
    """
    Retrieval strategy for MEDIUM scope queries.

    - Search with original + 2-3 expanded queries
    - Normal similarity threshold
    - Return 8-10 chunks
    """
    from app.rag.embeddings.service import EmbeddingsService
    from app.database.connection import SessionLocal
    from sqlalchemy import text

    all_results = []
    seen_ids = set()

    try:
        embedding_service = EmbeddingsService()
        session = SessionLocal()

        try:
            # Use original + top 2 expansions
            queries_to_use = intent.expanded_queries[:3]

            for expanded_query in queries_to_use:
                query_embedding = await asyncio.to_thread(
                    embedding_service.embed_query, expanded_query
                )
                embedding_str = "[" + ",".join(map(str, query_embedding)) + "]"

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
                    LIMIT 8
                """)

                result = await asyncio.to_thread(
                    lambda: session.execute(sql, {"user_id": user_id}).fetchall()
                )

                for row in result:
                    chunk_id, doc_id, content, metadata, similarity = row

                    if str(chunk_id) in seen_ids:
                        continue

                    if similarity < 0.1:
                        continue

                    seen_ids.add(str(chunk_id))
                    metadata = metadata or {}

                    all_results.append({
                        "id": str(chunk_id),
                        "document_id": str(doc_id),
                        "content": content or "",
                        "source": metadata.get("source", "Unknown"),
                        "score": float(similarity),
                        "metadata": metadata,
                        "search_type": "medium_scope",
                    })

        finally:
            session.close()

    except Exception as e:
        logger.error(f"Medium scope retrieval error: {e}", exc_info=True)

    all_results.sort(key=lambda x: x["score"], reverse=True)
    return all_results[:intent.target_chunk_count]


async def narrow_scope_retrieval(
    intent: QueryIntent,
    user_id: str,
    document_ids: list[str] = None,
) -> list[dict[str, Any]]:
    """
    Retrieval strategy for NARROW scope queries.

    - Search with original query only
    - Higher similarity threshold
    - Return 5 chunks
    """
    from app.rag.embeddings.service import EmbeddingsService
    from app.database.connection import SessionLocal
    from sqlalchemy import text

    results = []

    try:
        embedding_service = EmbeddingsService()
        session = SessionLocal()

        try:
            query_embedding = await asyncio.to_thread(
                embedding_service.embed_query, intent.original_query
            )
            embedding_str = "[" + ",".join(map(str, query_embedding)) + "]"

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
                lambda: session.execute(sql, {
                    "user_id": user_id,
                    "limit": intent.target_chunk_count,
                }).fetchall()
            )

            for row in result:
                chunk_id, doc_id, content, metadata, similarity = row

                if similarity < 0.15:
                    continue

                metadata = metadata or {}

                results.append({
                    "id": str(chunk_id),
                    "document_id": str(doc_id),
                    "content": content or "",
                    "source": metadata.get("source", "Unknown"),
                    "score": float(similarity),
                    "metadata": metadata,
                    "search_type": "narrow_scope",
                })

        finally:
            session.close()

    except Exception as e:
        logger.error(f"Narrow scope retrieval error: {e}", exc_info=True)

    return results


# =============================================================================
# LANGGRAPH NODE
# =============================================================================

async def intent_aware_retrieval_node(state: dict[str, Any]) -> dict[str, Any]:
    """
    LangGraph node for intent-aware retrieval.

    Replaces standard retrieval when intent understanding is needed.
    """
    from app.rag.langgraph.state import track_node, add_documents, add_error
    import time

    start_time = time.time()
    logger.info("Starting intent-aware retrieval node")

    query = state.get("original_query", "")
    user_id = state.get("user_id")
    document_ids = state.get("document_ids")

    updates = track_node(state, "intent_retrieval")

    try:
        # Run intent-aware retrieval
        documents, intent = await intent_aware_retrieval(query, user_id, document_ids)

        # Add intent info to state for downstream nodes
        updates["query_intent"] = {
            "scope": intent.scope.value,
            "type": intent.query_type.value,
            "domain": intent.domain,
            "expanded_queries": intent.expanded_queries,
            "confidence": intent.confidence,
        }

        if documents:
            updates.update(add_documents(state, documents))
            logger.info(f"Intent retrieval found {len(documents)} documents")
        else:
            logger.warning("Intent retrieval found no documents")
            updates.update(add_error(
                state, "intent_retrieval", "NO_RESULTS",
                "No documents found matching your query intent.",
                recoverable=True
            ))

    except Exception as e:
        logger.error(f"Intent retrieval error: {e}", exc_info=True)
        updates.update(add_error(
            state, "intent_retrieval", "RETRIEVAL_ERROR",
            str(e), recoverable=True
        ))

    duration_ms = (time.time() - start_time) * 1000
    logger.info(f"Intent-aware retrieval complete in {duration_ms:.1f}ms")

    return updates


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "classify_query_intent",
    "expand_query",
    "intent_aware_retrieval",
    "intent_aware_retrieval_node",
    "QueryIntent",
    "QueryScope",
    "QueryType",
    "CONCEPT_SYNONYMS",
]
