"""
Embeddings Service
==================
Fixed embedding service using BAAI/bge-large-en-v1.5 (1024 dimensions).
Matches database expectations.
"""

import os
import logging
from typing import List, Optional, Dict, Any
import numpy as np

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)

# Try to import embedding providers
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
    except ImportError:
        HuggingFaceEmbeddings = None

try:
    from langchain_openai import OpenAIEmbeddings
except ImportError:
    OpenAIEmbeddings = None


class EmbeddingsService:
    """
    Embeddings service using BAAI/bge-large-en-v1.5 (1024 dimensions) by default.
    
    Fixes: Database expects 1024 dimensions, not 384.
    """

    # Embedding models - BAAI/bge-large-en-v1.5 (1024D) is default to match DB
    AVAILABLE_MODELS = {
        "huggingface": {
            "default": "BAAI/bge-large-en-v1.5",  # 1024 dimensions - MATCHES DATABASE
            "BAAI/bge-large-en-v1.5": 1024,       # Database expects 1024D
            "BAAI/bge-base-en-v1.5": 768,
            "BAAI/bge-small-en-v1.5": 384,
            "all-MiniLM-L6-v2": 384,
            "sentence-transformers/all-mpnet-base-v2": 768,
        },
        "openai": {
            "default": "text-embedding-3-small",
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
        }
    }

    def __init__(
        self,
        provider: str = "huggingface",
        model_name: Optional[str] = None,
        normalize: bool = True,
        batch_size: int = 32,
    ):
        """
        Initialize embeddings service.
        
        Args:
            provider: "huggingface" or "openai"
            model_name: Specific model name (defaults to BAAI/bge-large-en-v1.5 for huggingface)
            normalize: Normalize embeddings for cosine similarity
            batch_size: Batch size for processing
        """
        self.provider = provider
        self.normalize = normalize
        self.batch_size = batch_size
        
        # Force BAAI/bge-large-en-v1.5 for huggingface if not specified
        if provider == "huggingface" and model_name is None:
            model_name = self.AVAILABLE_MODELS["huggingface"]["default"]
            logger.info(f"Using default huggingface model: {model_name} (1024 dimensions)")
        
        # Initialize embedding model
        self.embeddings = self._init_embeddings(provider, model_name)
        self.model_name = model_name or self._get_default_model(provider)
        self.dimensions = self._get_dimensions()
        
        # Log dimensions to debug
        logger.info(
            f"EmbeddingsService: {provider}/{self.model_name}, "
            f"dimensions={self.dimensions} (database expects 1024), "
            f"normalize={normalize}"
        )
        
        # Warn if dimensions don't match expected 1024
        if self.dimensions != 1024:
            logger.warning(
                f"Using {self.dimensions} dimensions but database expects 1024. "
                f"Consider using BAAI/bge-large-en-v1.5 or updating database schema."
            )

    def _init_embeddings(self, provider: str, model_name: Optional[str]) -> Embeddings:
        """Initialize embedding model."""
        if provider == "huggingface":
            if HuggingFaceEmbeddings is None:
                raise ImportError(
                    "HuggingFace embeddings not available. "
                    "Install: pip install langchain-huggingface"
                )
            
            model = model_name or self.AVAILABLE_MODELS["huggingface"]["default"]
            return HuggingFaceEmbeddings(
                model_name=model,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": self.normalize},
            )
            
        elif provider == "openai":
            if OpenAIEmbeddings is None:
                raise ImportError(
                    "OpenAI embeddings not available. "
                    "Install: pip install langchain-openai"
                )
            
            model = model_name or self.AVAILABLE_MODELS["openai"]["default"]
            return OpenAIEmbeddings(model=model)
            
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    def _get_default_model(self, provider: str) -> str:
        """Get default model for provider."""
        return self.AVAILABLE_MODELS.get(provider, {}).get("default", "unknown")

    def _get_dimensions(self) -> int:
        """Get embedding dimensions."""
        provider_models = self.AVAILABLE_MODELS.get(self.provider, {})
        return provider_models.get(self.model_name, 1024)  # Default to 1024 to match DB

    def embed_query(self, query: str) -> List[float]:
        """
        Embed a single query.
        
        Args:
            query: Text to embed
            
        Returns:
            Embedding vector
        """
        return self.embeddings.embed_query(query)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed multiple documents.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        # Log dimensions being generated
        if texts and len(texts) > 0:
            sample_text = texts[0][:100] if len(texts[0]) > 100 else texts[0]
            logger.debug(f"Generating embeddings for {len(texts)} documents, sample: '{sample_text}...'")
        
        # Process in batches
        all_embeddings = []
        total_processed = 0
        
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            batch_embeddings = self.embeddings.embed_documents(batch)
            all_embeddings.extend(batch_embeddings)
            total_processed += len(batch)
            
            # Log progress for large batches
            if len(texts) > 100:
                logger.debug(f"Processed {total_processed}/{len(texts)} documents")
        
        # Validate dimensions
        if all_embeddings and len(all_embeddings) > 0:
            actual_dimensions = len(all_embeddings[0])
            if actual_dimensions != self.dimensions:
                logger.error(
                    f"Dimension mismatch! Expected {self.dimensions}, "
                    f"got {actual_dimensions}. Check model configuration."
                )
            else:
                logger.debug(f"Generated {len(all_embeddings)} embeddings with {actual_dimensions} dimensions")
        
        return all_embeddings

# Factory function that ensures 1024 dimensions
def create_embeddings_service(
    provider: str = "huggingface",
    model_name: Optional[str] = None,
    normalize: bool = True,
    batch_size: int = 32,
) -> EmbeddingsService:
    """
    Create embeddings service with proper dimension configuration.
    
    Ensures BAAI/bge-large-en-v1.5 (1024 dimensions) is used by default
    to match database expectations.
    
    Args:
        provider: "huggingface" or "openai"
        model_name: Model name (defaults to BAAI/bge-large-en-v1.5 for huggingface)
        normalize: Normalize embeddings
        batch_size: Batch size
        
    Returns:
        EmbeddingsService instance
    """
    # Force BAAI/bge-large-en-v1.5 for huggingface to match database
    if provider == "huggingface" and model_name is None:
        model_name = "BAAI/bge-large-en-v1.5"
        logger.info(f"Using default model to match database: {model_name} (1024 dimensions)")
    
    return EmbeddingsService(
        provider=provider,
        model_name=model_name,
        normalize=normalize,
        batch_size=batch_size,
    )


# Create default instance with 1024 dimensions
default_embeddings = create_embeddings_service()


def get_embeddings_service() -> EmbeddingsService:
    """
    Get embeddings service instance.
    
    Returns default BAAI/bge-large-en-v1.5 (1024 dimensions) instance
    to match database schema.
    """
    return default_embeddings
