"""
Document Processor Service
==========================
Production document processing pipeline.
"""

import logging
from typing import List, Optional, Dict, Any

from langchain_core.documents import Document

from app.services.supabase_client import supabase_client
from app.rag.documents.loader import DocumentLoaderService, document_loader
from app.rag.documents.splitter import TextSplitterService, text_splitter, ChunkingStrategy
from app.rag.embeddings.service import EmbeddingsService, get_embeddings_service

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """
    Production document processor.

    Features:
    - Multi-format document loading
    - Intelligent text chunking
    - Embedding generation and storage via Supabase
    - Progress tracking
    """

    def __init__(self):
        self.document_loader = document_loader
        self.text_splitter = text_splitter
        self._embeddings_service: Optional[EmbeddingsService] = None

        logger.info("DocumentProcessor initialized")

    @property
    def embeddings_service(self) -> EmbeddingsService:
        """Lazy load embeddings service."""
        if self._embeddings_service is None:
            self._embeddings_service = get_embeddings_service()
        return self._embeddings_service

    async def process_document(
        self,
        document_id: str,
        file_content: bytes,
        content_type: str,
        filename: str = "document",
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process a document through the full pipeline.

        Args:
            document_id: Unique document ID
            file_content: Raw file bytes
            content_type: MIME type
            filename: Original filename
            metadata: Additional metadata

        Returns:
            Processing result with statistics
        """
        result = {
            "document_id": document_id,
            "success": False,
            "chunks_created": 0,
            "error": None,
        }

        try:
            logger.info(f"Processing document {document_id}: {filename} ({content_type})")

            # Step 1: Load document
            documents = self.document_loader.load_from_bytes(
                content=file_content,
                content_type=content_type,
                filename=filename,
                metadata={
                    "document_id": document_id,
                    **(metadata or {})
                }
            )

            if not documents:
                raise ValueError("No content extracted from document")

            logger.info(f"Loaded {len(documents)} document section(s)")

            # Step 2: Split into chunks
            chunks = self.text_splitter.split_documents(documents)

            if not chunks:
                raise ValueError("No chunks created from document")

            logger.info(f"Created {len(chunks)} chunks")

            # Step 3: Generate embeddings and store
            added_count = await self._store_chunks_with_embeddings(
                document_id=document_id,
                chunks=chunks,
                metadata=metadata,
            )

            # Step 4: Update document status
            supabase_client.table("documents").update({
                "processed": True,
                "error": None
            }).eq("id", document_id).execute()

            result["success"] = True
            result["chunks_created"] = added_count
            result["sections_loaded"] = len(documents)

            logger.info(f"Successfully processed document {document_id}: {added_count} chunks")

            return result

        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error processing document {document_id}: {error_msg}")

            # Update document with error
            supabase_client.table("documents").update({
                "processed": False,
                "error": error_msg
            }).eq("id", document_id).execute()

            result["error"] = error_msg
            return result

    async def _store_chunks_with_embeddings(
        self,
        document_id: str,
        chunks: List[Document],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """
        Generate embeddings and store chunks in Supabase.

        Args:
            document_id: Document ID
            chunks: List of document chunks
            metadata: Additional metadata

        Returns:
            Number of chunks stored
        """
        # Extract texts for embedding
        texts = [chunk.page_content for chunk in chunks]

        # Generate embeddings
        embeddings = self.embeddings_service.embed_documents(texts)

        # Prepare records for insertion
        records = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            chunk_metadata = {
                "document_id": document_id,
                "chunk_index": i,
                "source": chunk.metadata.get("source", ""),
                **(chunk.metadata or {}),
                **(metadata or {}),
            }

            records.append({
                "document_id": document_id,
                "content": chunk.page_content,
                "embedding": embedding,
                "metadata": chunk_metadata,
                "chunk_index": i,
            })

        # Insert into Supabase
        if records:
            supabase_client.table("document_chunks").insert(records).execute()

        return len(records)

    async def reprocess_document(
        self,
        document_id: str,
        file_content: bytes,
        content_type: str,
        filename: str = "document"
    ) -> Dict[str, Any]:
        """
        Reprocess an existing document (deletes old chunks first).

        Args:
            document_id: Document ID
            file_content: Raw file bytes
            content_type: MIME type
            filename: Original filename

        Returns:
            Processing result
        """
        # Delete existing chunks
        deleted = self.delete_document_chunks(document_id)
        logger.info(f"Deleted {deleted} existing chunks for document {document_id}")

        # Process again
        return await self.process_document(
            document_id=document_id,
            file_content=file_content,
            content_type=content_type,
            filename=filename,
        )

    def delete_document_chunks(self, document_id: str) -> int:
        """
        Delete all chunks for a document.

        Args:
            document_id: Document ID

        Returns:
            Number of chunks deleted
        """
        result = supabase_client.table("document_chunks").delete().eq(
            "document_id", document_id
        ).execute()

        return len(result.data) if result.data else 0

    def search_documents(
        self,
        query: str,
        user_id: str,
        document_ids: Optional[List[str]] = None,
        top_k: int = 5,
        match_threshold: float = 0.3
    ) -> List[Dict[str, Any]]:
        """
        Search for relevant document chunks using vector similarity.

        Uses the match_documents PostgreSQL function for efficient similarity search.
        Falls back to basic search if the function is not available.

        Args:
            query: Search query
            user_id: User ID for access control
            document_ids: Optional document IDs to search within
            top_k: Number of results to return
            match_threshold: Minimum similarity threshold (0.0-1.0)

        Returns:
            List of relevant chunks with content, score, and metadata
        """
        try:
            # Generate query embedding
            query_embedding = self.embeddings_service.embed_query(query)

            if not query_embedding:
                logger.warning("Failed to generate query embedding")
                return []

            # Try RPC-based similarity search first
            try:
                params = {
                    "query_embedding": query_embedding,
                    "match_threshold": match_threshold,
                    "match_count": top_k * 2,  # Get more to filter later
                }

                result = supabase_client.rpc("match_documents", params).execute()
                chunks = result.data if result.data else []

            except Exception as rpc_error:
                logger.warning(f"RPC search failed, using fallback: {rpc_error}")
                chunks = self._fallback_search(query_embedding, top_k * 2, match_threshold)

            if not chunks:
                logger.debug(f"No matching documents found for query: {query[:50]}...")
                return []

            # Filter by document_ids if provided
            if document_ids:
                chunks = [c for c in chunks if c.get("document_id") in document_ids]

            # Sort by similarity score and limit
            chunks = sorted(chunks, key=lambda x: x.get("similarity", 0), reverse=True)[:top_k]

            # Format results
            results = []
            for c in chunks:
                results.append({
                    "content": c.get("content", ""),
                    "score": round(c.get("similarity", 0.0), 4),
                    "document_id": c.get("document_id", ""),
                    "chunk_index": c.get("chunk_index", 0),
                    "metadata": c.get("metadata", {}),
                    "source": c.get("metadata", {}).get("source", "Unknown"),
                })

            logger.info(f"Found {len(results)} relevant chunks for query")
            return results

        except Exception as e:
            logger.error(f"Document search error: {e}", exc_info=True)
            return []

    def _fallback_search(
        self,
        query_embedding: List[float],
        limit: int,
        threshold: float
    ) -> List[Dict[str, Any]]:
        """
        Fallback search using direct SQL when RPC is not available.

        Args:
            query_embedding: Query embedding vector
            limit: Maximum results
            threshold: Minimum similarity

        Returns:
            List of matching chunks
        """
        try:
            from sqlalchemy import text
            from app.database.connection import SessionLocal

            session = SessionLocal()
            try:
                # Format embedding for PostgreSQL
                embedding_str = "[" + ",".join(map(str, query_embedding)) + "]"

                # Direct cosine similarity query
                sql = f"""
                    SELECT
                        dc.id,
                        dc.document_id,
                        dc.content,
                        dc.chunk_metadata as metadata,
                        1 - (dc.embedding <=> '{embedding_str}'::vector) as similarity
                    FROM document_chunks dc
                    WHERE dc.embedding IS NOT NULL
                      AND 1 - (dc.embedding <=> '{embedding_str}'::vector) > {threshold}
                    ORDER BY dc.embedding <=> '{embedding_str}'::vector
                    LIMIT {limit}
                """

                result = session.execute(text(sql)).fetchall()

                return [
                    {
                        "id": str(row[0]),
                        "document_id": str(row[1]),
                        "content": row[2],
                        "metadata": row[3] or {},
                        "similarity": float(row[4]) if row[4] else 0.0
                    }
                    for row in result
                ]

            finally:
                session.close()

        except Exception as e:
            logger.error(f"Fallback search error: {e}")
            return []

    def get_document_chunks(
        self,
        document_id: str,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get all chunks for a document.

        Args:
            document_id: Document ID
            limit: Maximum chunks to return

        Returns:
            List of chunks
        """
        result = supabase_client.table("document_chunks").select(
            "content, chunk_index, metadata"
        ).eq("document_id", document_id).limit(limit).execute()

        if not result.data:
            return []

        return [
            {
                "content": c.get("content", ""),
                "chunk_index": c.get("chunk_index", 0),
                "metadata": c.get("metadata", {}),
            }
            for c in result.data
        ]

    @staticmethod
    def is_supported_type(content_type: str) -> bool:
        """Check if content type is supported."""
        return DocumentLoaderService.is_supported(content_type)


# Global instance
doc_processor = DocumentProcessor()
