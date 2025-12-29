#!/usr/bin/env python3
"""
Document Re-indexing Script
============================
Re-processes all documents with new chunking and embedding settings.

This script will:
1. Get all existing documents from the database
2. Delete old embeddings/chunks
3. Re-split documents with new chunk sizes (300 chars)
4. Create new embeddings with the proper model

Usage:
    python scripts/reindex_documents.py

Options:
    --dry-run       Show what would be done without making changes
    --document-id   Re-index a specific document only
"""

import sys
import os
import argparse
import logging

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import text
from langchain_core.documents import Document as LCDocument
from app.database.connection import SessionLocal
from app.database.models import Document, DocumentChunk
from app.rag.embeddings.service import get_embeddings_service
from app.rag.documents.splitter import TextSplitterService, ChunkConfig, ChunkingStrategy
from app.services.supabase_client import supabase_client

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_all_documents():
    """Get all documents from database with their content from chunks."""
    session = SessionLocal()
    try:
        documents = session.query(Document).all()
        result = []
        for doc in documents:
            # Get content from existing chunks
            chunks = session.query(DocumentChunk).filter(
                DocumentChunk.document_id == doc.id
            ).order_by(DocumentChunk.chunk_index).all()

            # Combine chunk content to get full document
            full_content = "\n\n".join([chunk.content for chunk in chunks])

            result.append({
                'id': str(doc.id),
                'filename': doc.filename,
                'content': full_content,
                'user_id': str(doc.user_id),
                'chunk_count': len(chunks),
            })
        return result
    finally:
        session.close()


def delete_document_chunks(document_id: str, dry_run: bool = False):
    """Delete all chunks for a document."""
    session = SessionLocal()
    try:
        count = session.query(DocumentChunk).filter(
            DocumentChunk.document_id == document_id
        ).count()

        if dry_run:
            logger.info(f"[DRY RUN] Would delete {count} chunks for document {document_id}")
            return count

        session.query(DocumentChunk).filter(
            DocumentChunk.document_id == document_id
        ).delete()
        session.commit()

        logger.info(f"Deleted {count} old chunks for document {document_id}")
        return count
    except Exception as e:
        session.rollback()
        logger.error(f"Error deleting chunks: {e}")
        return 0
    finally:
        session.close()


def reindex_document(document: dict, dry_run: bool = False):
    """Re-index a single document with new chunk settings."""
    doc_id = document['id']
    filename = document['filename']
    content = document['content']
    old_chunk_count = document.get('chunk_count', 0)

    logger.info(f"\n{'='*60}")
    logger.info(f"Processing: {filename}")
    logger.info(f"Document ID: {doc_id}")
    logger.info(f"Content length: {len(content)} characters")
    logger.info(f"Current chunks: {old_chunk_count}")

    if not content or len(content.strip()) < 10:
        logger.warning(f"Document {filename} has no content, skipping")
        return False

    # Step 1: Delete old chunks
    old_count = delete_document_chunks(doc_id, dry_run)

    # Step 2: Split document with new settings
    splitter = TextSplitterService(
        config=ChunkConfig(
            chunk_size=300,  # Smaller chunks for precise matching
            chunk_overlap=30,
            strategy=ChunkingStrategy.SENTENCE,
        )
    )

    # Create a LangChain document
    lc_doc = LCDocument(
        page_content=content,
        metadata={"source": filename, "document_id": doc_id}
    )

    chunks = splitter.split_documents([lc_doc])

    logger.info(f"Split into {len(chunks)} chunks (was {old_count})")

    # Show sample chunks
    for i, chunk in enumerate(chunks[:3]):
        preview = chunk.page_content[:100].replace('\n', ' ')
        logger.info(f"  Chunk {i}: {preview}...")

    if dry_run:
        logger.info(f"[DRY RUN] Would create {len(chunks)} new embeddings")
        return True

    # Step 3: Generate embeddings and store
    try:
        embeddings_service = get_embeddings_service()
        texts = [chunk.page_content for chunk in chunks]
        embeddings = embeddings_service.embed_documents(texts)

        records = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            records.append({
                "document_id": doc_id,
                "content": chunk.page_content,
                "embedding": embedding,
                "metadata": chunk.metadata,
                "chunk_index": i,
            })

        if records:
            supabase_client.table("document_chunks").insert(records).execute()

        logger.info(f"Created {len(records)} new embeddings for {filename}")
        return True
    except Exception as e:
        logger.error(f"Error creating embeddings: {e}")
        return False


def verify_embeddings():
    """Verify embeddings are properly stored."""
    session = SessionLocal()
    try:
        # Check embedding count and dimensions
        result = session.execute(text("""
            SELECT
                COUNT(*) as total_chunks,
                COUNT(embedding) as with_embedding,
                COUNT(*) - COUNT(embedding) as without_embedding
            FROM document_chunks
        """)).fetchone()

        logger.info(f"\nEmbedding Verification:")
        logger.info(f"  Total chunks: {result.total_chunks}")
        logger.info(f"  With embeddings: {result.with_embedding}")
        logger.info(f"  Without embeddings: {result.without_embedding}")

        # Sample a chunk to check dimension
        sample = session.execute(text("""
            SELECT array_length(embedding::real[], 1) as dim
            FROM document_chunks
            WHERE embedding IS NOT NULL
            LIMIT 1
        """)).fetchone()

        if sample:
            logger.info(f"  Embedding dimension: {sample.dim}")

        return result.without_embedding == 0
    except Exception as e:
        logger.error(f"Error verifying embeddings: {e}")
        return False
    finally:
        session.close()


def main():
    parser = argparse.ArgumentParser(
        description='Re-index documents with new chunk settings'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be done without making changes'
    )
    parser.add_argument(
        '--document-id',
        type=str,
        help='Re-index only a specific document'
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Document Re-indexing Script")
    print("=" * 60)
    print("New settings:")
    print("  - Chunk size: 300 characters (smaller for precision)")
    print("  - Chunk overlap: 30 characters")
    print("  - Strategy: SENTENCE (semantic boundaries)")
    print("  - Model: sentence-transformers/all-mpnet-base-v2 (768 dim)")
    print("=" * 60)

    if args.dry_run:
        print("\n*** DRY RUN MODE - No changes will be made ***\n")

    # Get documents
    documents = get_all_documents()

    if args.document_id:
        documents = [d for d in documents if d['id'] == args.document_id]
        if not documents:
            logger.error(f"Document {args.document_id} not found")
            sys.exit(1)

    if not documents:
        logger.info("No documents found to re-index")
        sys.exit(0)

    logger.info(f"Found {len(documents)} documents to re-index")

    # Process each document
    success_count = 0
    error_count = 0

    for i, doc in enumerate(documents, 1):
        logger.info(f"\nProgress: {i}/{len(documents)}")

        if reindex_document(doc, args.dry_run):
            success_count += 1
        else:
            error_count += 1

    # Verify
    if not args.dry_run:
        print("\n" + "=" * 60)
        verify_embeddings()

    print("\n" + "=" * 60)
    print(f"Re-indexing complete!")
    print(f"  Success: {success_count}")
    print(f"  Errors: {error_count}")
    print("=" * 60)

    if args.dry_run:
        print("\nTo apply changes, run without --dry-run flag:")
        print("  python scripts/reindex_documents.py")


if __name__ == "__main__":
    main()
