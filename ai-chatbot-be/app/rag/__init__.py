"""
RAG Services
============

Document processing and LangGraph agent for document Q&A.

Usage:
    from app.rag import DocumentLoaderService, EmbeddingsService

    # Load documents
    loader = DocumentLoaderService()
    docs = await loader.load("document.pdf")

    # Use the LangGraph agent
    from app.rag.langgraph import Agent, create_agent

    agent = create_agent()
    result = await agent.invoke("What is the document about?", user_id="user123")
"""

# Document processing
from app.rag.documents.loader import DocumentLoaderService
from app.rag.documents.splitter import TextSplitterService

# Embeddings
from app.rag.embeddings.service import EmbeddingsService

# Models
from app.rag.models.schemas import (
    RAGConfig,
    RAGResponse,
    RAGMode,
)

# Utils
from app.rag.utils.text_utils import clean_text, extract_keywords
from app.rag.utils.resilience import with_retry, CircuitBreaker

__all__ = [
    # Documents
    "DocumentLoaderService",
    "TextSplitterService",
    # Embeddings
    "EmbeddingsService",
    # Models
    "RAGConfig",
    "RAGResponse",
    "RAGMode",
    # Utils
    "clean_text",
    "extract_keywords",
    "with_retry",
    "CircuitBreaker",
]
