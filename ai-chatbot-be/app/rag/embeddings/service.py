"""
Embeddings Service
==================
Production embeddings with caching and multiple provider support.
"""

import os
import logging
import hashlib
from typing import List, Optional, Dict, Any
from functools import lru_cache
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

try:
    from langchain_cohere import CohereEmbeddings
except ImportError:
    CohereEmbeddings = None

try:
    from langchain_mistralai import MistralAIEmbeddings
except ImportError:
    MistralAIEmbeddings = None


class EmbeddingsService:
    """
    Production embeddings service with caching.

    Features:
    - Multiple embedding providers (HuggingFace, OpenAI, Cohere)
    - In-memory caching to reduce API calls
    - Batch processing for efficiency
    - Embedding normalization for better similarity search
    """

    # Embedding model configurations
    MODELS = {
        "huggingface": {
            "default": "sentence-transformers/all-mpnet-base-v2",
            "production": "sentence-transformers/all-mpnet-base-v2",
            "multilingual": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            "dimensions": {
                "all-MiniLM-L6-v2": 384,
                "sentence-transformers/all-mpnet-base-v2": 768,
                "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2": 384,
            },
        },
        "openai": {
            "default": "text-embedding-3-small",
            "production": "text-embedding-3-large",
            "dimensions": {
                "text-embedding-3-small": 1536,
                "text-embedding-3-large": 3072,
            },
        },
        "mistral": {
            "default": "mistral-embed",
            "production": "mistral-embed",
            "dimensions": {
                "mistral-embed": 1024,
            },
        },
    }

    def __init__(
        self,
        provider: str = "huggingface",
        model_name: Optional[str] = None,
        use_cache: bool = True,
        normalize: bool = True,
        batch_size: int = 32,
    ):
        """
        Initialize embeddings service.

        Args:
            provider: Embedding provider ("huggingface", "openai", "cohere")
            model_name: Specific model to use
            use_cache: Enable embedding caching
            normalize: Normalize embeddings for cosine similarity
            batch_size: Batch size for processing
        """
        self.provider = provider
        self.use_cache = use_cache
        self.normalize = normalize
        self.batch_size = batch_size

        # Initialize the embedding model
        self.embeddings = self._init_embeddings(provider, model_name)
        self.model_name = model_name or self._get_default_model(provider)
        self.dimensions = self._get_dimensions()

        # In-memory cache
        self._cache: Dict[str, List[float]] = {}
        self._cache_hits = 0
        self._cache_misses = 0

        logger.info(
            f"EmbeddingsService initialized: provider={provider}, model={self.model_name}, dimensions={self.dimensions}"
        )

    def _init_embeddings(self, provider: str, model_name: Optional[str]) -> Embeddings:
        """Initialize embedding model based on provider."""
        if provider == "huggingface":
            if HuggingFaceEmbeddings is None:
                raise ImportError(
                    "HuggingFace embeddings not available. Install: pip install langchain-huggingface"
                )

            model = model_name or self.MODELS["huggingface"]["default"]
            return HuggingFaceEmbeddings(
                model_name=model,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": self.normalize},
            )

        elif provider == "openai":
            if OpenAIEmbeddings is None:
                raise ImportError(
                    "OpenAI embeddings not available. Install: pip install langchain-openai"
                )

            model = model_name or self.MODELS["openai"]["default"]
            return OpenAIEmbeddings(model=model)

        elif provider == "cohere":
            if CohereEmbeddings is None:
                raise ImportError(
                    "Cohere embeddings not available. Install: pip install langchain-cohere"
                )

            return CohereEmbeddings()

        elif provider == "mistral":
            if MistralAIEmbeddings is None:
                raise ImportError(
                    "Mistral embeddings not available. Install: pip install langchain-mistralai"
                )

            # Get API key from environment - MUST be passed explicitly
            api_key = os.getenv("MISTRAL_API_KEY")

            if not api_key or api_key.strip() == "":
                raise ValueError(
                    "MISTRAL_API_KEY environment variable is not set or empty. "
                    "Get your API key from https://console.mistral.ai/"
                )

            model = model_name or self.MODELS["mistral"]["default"]
            logger.info(f"Initializing Mistral embeddings with model: {model}")

            return MistralAIEmbeddings(
                model=model,
                api_key=api_key.strip()  # Pass API key explicitly
            )

        else:
            raise ValueError(f"Unsupported provider: {provider}")

    def _get_default_model(self, provider: str) -> str:
        """Get default model for provider."""
        return self.MODELS.get(provider, {}).get("default", "unknown")

    def _get_dimensions(self) -> int:
        """Get embedding dimensions for current model."""
        provider_models = self.MODELS.get(self.provider, {}).get("dimensions", {})
        return provider_models.get(self.model_name, 384)  # Default to 384

    def embed_query(self, query: str) -> List[float]:
        """
        Embed a single query string.

        Args:
            query: Query text to embed

        Returns:
            Embedding vector
        """
        # Check cache
        if self.use_cache:
            cache_key = self._get_cache_key(query)
            if cache_key in self._cache:
                self._cache_hits += 1
                return self._cache[cache_key]
            self._cache_misses += 1

        # Generate embedding
        embedding = self.embeddings.embed_query(query)

        # Normalize if needed (and not already done by model)
        if self.normalize and self.provider == "openai":
            embedding = self._normalize_vector(embedding)

        # Cache result
        if self.use_cache:
            self._cache[cache_key] = embedding

        return embedding

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

        # Check cache for each text
        embeddings = []
        texts_to_embed = []
        text_indices = []

        if self.use_cache:
            for i, text in enumerate(texts):
                cache_key = self._get_cache_key(text)
                if cache_key in self._cache:
                    embeddings.append((i, self._cache[cache_key]))
                    self._cache_hits += 1
                else:
                    texts_to_embed.append(text)
                    text_indices.append(i)
                    self._cache_misses += 1
        else:
            texts_to_embed = texts
            text_indices = list(range(len(texts)))

        # Embed uncached texts in batches
        if texts_to_embed:
            for i in range(0, len(texts_to_embed), self.batch_size):
                batch = texts_to_embed[i : i + self.batch_size]
                batch_embeddings = self.embeddings.embed_documents(batch)

                # Normalize if needed
                if self.normalize and self.provider == "openai":
                    batch_embeddings = [self._normalize_vector(e) for e in batch_embeddings]

                # Cache and store results
                for j, embedding in enumerate(batch_embeddings):
                    idx = text_indices[i + j]
                    embeddings.append((idx, embedding))

                    if self.use_cache:
                        cache_key = self._get_cache_key(batch[j])
                        self._cache[cache_key] = embedding

        # Sort by original index and return just embeddings
        embeddings.sort(key=lambda x: x[0])
        return [e[1] for e in embeddings]

    def embed_document_objects(self, documents: List[Document]) -> List[Document]:
        """
        Embed documents and attach embeddings to metadata.

        Args:
            documents: List of Document objects

        Returns:
            Documents with embeddings in metadata
        """
        texts = [doc.page_content for doc in documents]
        embeddings = self.embed_documents(texts)

        for doc, embedding in zip(documents, embeddings):
            doc.metadata["embedding"] = embedding

        return documents

    def similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """
        Calculate cosine similarity between two embeddings.

        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector

        Returns:
            Similarity score between 0 and 1
        """
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)

        # If already normalized, dot product equals cosine similarity
        if self.normalize:
            return float(np.dot(vec1, vec2))

        # Calculate cosine similarity
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(np.dot(vec1, vec2) / (norm1 * norm2))

    def find_similar(
        self,
        query_embedding: List[float],
        document_embeddings: List[List[float]],
        top_k: int = 5,
        threshold: float = 0.0,
    ) -> List[tuple]:
        """
        Find most similar embeddings to query.

        Args:
            query_embedding: Query embedding vector
            document_embeddings: List of document embeddings
            top_k: Number of results to return
            threshold: Minimum similarity threshold

        Returns:
            List of (index, similarity_score) tuples
        """
        similarities = []

        for idx, doc_embedding in enumerate(document_embeddings):
            sim = self.similarity(query_embedding, doc_embedding)
            if sim >= threshold:
                similarities.append((idx, sim))

        # Sort by similarity descending
        similarities.sort(key=lambda x: x[1], reverse=True)

        return similarities[:top_k]

    def _normalize_vector(self, vector: List[float]) -> List[float]:
        """Normalize a vector to unit length."""
        vec = np.array(vector)
        norm = np.linalg.norm(vec)
        if norm == 0:
            return vector
        return (vec / norm).tolist()

    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text."""
        return hashlib.md5(text.encode()).hexdigest()

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total if total > 0 else 0

        return {
            "cache_size": len(self._cache),
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "hit_rate": hit_rate,
        }

    def clear_cache(self):
        """Clear the embedding cache."""
        self._cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0
        logger.info("Embedding cache cleared")


# Create instances for different use cases
def create_embeddings_service(environment: str = "development") -> EmbeddingsService:
    """
    Factory function to create embeddings service based on environment and config.

    Args:
        environment: "development" or "production"

    Returns:
        Configured EmbeddingsService
    """
    # Import settings here to avoid circular imports during startup
    from app.core.config import settings

    provider = settings.embedding_provider
    logger.info(f"Creating embeddings service using provider: {provider}")

    return EmbeddingsService(
        provider=provider,
        # Let the service pick the default model for the provider
        model_name=None,
        use_cache=True,
        normalize=True,
    )


# Global instance - lazy initialization
_embeddings_service: Optional[EmbeddingsService] = None


def get_embeddings_service() -> EmbeddingsService:
    """Get or create the global embeddings service."""
    global _embeddings_service
    if _embeddings_service is None:
        from app.core.config import settings

        _embeddings_service = create_embeddings_service(settings.environment)
    return _embeddings_service
