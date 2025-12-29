"""
RAG Data Models
===============
Data classes and enums for the RAG pipeline.

This module contains all configuration and response models
used throughout the RAG system.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict, Any


class RAGMode(Enum):
    """RAG operation modes."""

    STANDARD = "standard"
    CONVERSATIONAL = "conversational"
    STRICT = "strict"  # Only answer from context, never hallucinate
    CREATIVE = "creative"  # Allow some inference from context


@dataclass
class RAGConfig:
    """
    Configuration for RAG chain.

    Attributes:
        mode: RAG operation mode
        use_memory: Enable conversation memory
        use_query_rewriting: Rewrite queries with context
        use_hyde: Use Hypothetical Document Embeddings
        use_compression: Use contextual compression
        max_context_length: Maximum context length in characters
        min_confidence_score: Minimum confidence threshold
        max_retries: Maximum retry attempts
        retry_delay: Initial retry delay in seconds
        timeout: Operation timeout in seconds
        include_sources: Include source attribution
        stream_response: Enable streaming responses
    """

    mode: RAGMode = RAGMode.CONVERSATIONAL
    use_memory: bool = True
    use_query_rewriting: bool = True
    use_hyde: bool = False
    use_compression: bool = False
    max_context_length: int = 4000
    min_confidence_score: float = 0.4
    max_retries: int = 3
    retry_delay: float = 1.0
    timeout: float = 30.0
    include_sources: bool = True
    stream_response: bool = True


@dataclass
class RAGResponse:
    """
    RAG response with metadata.

    Attributes:
        answer: Generated answer text
        sources: List of source documents with metadata
        context_used: The context string used for generation
        retrieval_scores: Similarity scores for retrieved documents
        tokens_used: Number of tokens used (if available)
        processing_time: Total processing time in seconds
        confidence: Confidence score (0.0 to 1.0)
        query_rewritten: Rewritten query (if applicable)
        metadata: Additional metadata
        retrieved_chunks: Full retrieved chunks for debugging
    """

    answer: str
    sources: List[Dict[str, Any]]
    context_used: str
    retrieval_scores: List[float]
    tokens_used: Optional[int] = None
    processing_time: float = 0.0
    confidence: float = 0.0
    query_rewritten: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    retrieved_chunks: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class RAGMetrics:
    """
    Metrics for RAG pipeline monitoring.

    Attributes:
        total_queries: Total number of queries processed
        successful_queries: Number of successful queries
        failed_queries: Number of failed queries
        avg_retrieval_time: Average retrieval time in seconds
        avg_generation_time: Average generation time in seconds
        avg_confidence: Average confidence score
        cache_hit_rate: Cache hit rate (0.0 to 1.0)
    """

    total_queries: int = 0
    successful_queries: int = 0
    failed_queries: int = 0
    avg_retrieval_time: float = 0.0
    avg_generation_time: float = 0.0
    avg_confidence: float = 0.0
    cache_hit_rate: float = 0.0

    def update_averages(
        self,
        retrieval_time: float,
        generation_time: float,
        confidence: float,
        alpha: float = 0.1,
    ) -> None:
        """
        Update running averages using exponential moving average.

        Args:
            retrieval_time: Latest retrieval time
            generation_time: Latest generation time
            confidence: Latest confidence score
            alpha: Smoothing factor (0.0 to 1.0)
        """
        self.avg_retrieval_time = (
            alpha * retrieval_time + (1 - alpha) * self.avg_retrieval_time
        )
        self.avg_generation_time = (
            alpha * generation_time + (1 - alpha) * self.avg_generation_time
        )
        self.avg_confidence = alpha * confidence + (1 - alpha) * self.avg_confidence

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "total_queries": self.total_queries,
            "successful_queries": self.successful_queries,
            "failed_queries": self.failed_queries,
            "success_rate": (
                self.successful_queries / max(self.total_queries, 1)
            ),
            "avg_retrieval_time": self.avg_retrieval_time,
            "avg_generation_time": self.avg_generation_time,
            "avg_confidence": self.avg_confidence,
            "cache_hit_rate": self.cache_hit_rate,
        }
