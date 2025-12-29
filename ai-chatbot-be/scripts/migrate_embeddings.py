#!/usr/bin/env python3
"""
Embedding Migration Script
==========================
Migrates the database from 384-dimensional embeddings (all-MiniLM-L6-v2)
to 768-dimensional embeddings (sentence-transformers/all-mpnet-base-v2).

This script will:
1. Alter the embedding column to support 768 dimensions
2. Clear existing embeddings (they're incompatible with new model)
3. Re-embed all documents with the new model

Usage:
    python scripts/migrate_embeddings.py

Options:
    --dry-run       Show what would be done without making changes
    --skip-reembed  Only alter column, don't re-embed documents
"""

import sys
import os
import argparse
import logging
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import text
from app.database.connection import SessionLocal, engine
from app.database.models import Document, DocumentChunk
from app.rag.embeddings.service import get_embeddings_service
from app.rag.documents.splitter import TextSplitterService

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_current_dimension():
    """Check the current embedding column dimension."""
    session = SessionLocal()
    try:
        result = session.execute(text("""
            SELECT atttypmod
            FROM pg_attribute
            WHERE attrelid = 'document_chunks'::regclass
            AND attname = 'embedding';
        """)).fetchone()

        if result:
            # Vector dimension is stored in atttypmod
            dimension = result[0]
            return dimension
        return None
    except Exception as e:
        logger.error(f"Error checking dimension: {e}")
        return None
    finally:
        session.close()


def alter_embedding_column(dry_run: bool = False):
    """Alter the embedding column from 384 to 768 dimensions."""
    session = SessionLocal()

    try:
        logger.info("Step 1: Altering embedding column dimension from 384 to 768...")

        if dry_run:
            logger.info("[DRY RUN] Would execute: ALTER TABLE document_chunks ALTER COLUMN embedding TYPE vector(768)")
            return True

        # First, drop any existing index on the embedding column
        try:
            session.execute(text("""
                DROP INDEX IF EXISTS idx_document_chunks_embedding;
            """))
            logger.info("Dropped existing embedding index (if any)")
        except Exception as e:
            logger.warning(f"No index to drop or error: {e}")

        # Clear existing embeddings (they're incompatible)
        session.execute(text("""
            UPDATE document_chunks SET embedding = NULL;
        """))
        logger.info("Cleared existing embeddings (incompatible with new dimension)")

        # Alter the column type
        session.execute(text("""
            ALTER TABLE document_chunks
            ALTER COLUMN embedding TYPE vector(768);
        """))
        logger.info("Altered embedding column to vector(768)")

        # Recreate the index for faster similarity search
        session.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_document_chunks_embedding
            ON document_chunks
            USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = 100);
        """))
        logger.info("Recreated embedding index with ivfflat")

        session.commit()
        logger.info("Column alteration completed successfully!")
        return True

    except Exception as e:
        session.rollback()
        logger.error(f"Error altering column: {e}")
        return False
    finally:
        session.close()


def get_all_documents():
    """Get all documents that need re-embedding."""
    session = SessionLocal()
    try:
        documents = session.query(Document).all()
        return [doc.to_dict() for doc in documents]
    finally:
        session.close()


def reembed_document(document_id: str, dry_run: bool = False):
    """Re-embed a single document with the new model."""
    session = SessionLocal()

    try:
        # Get the document
        doc = session.query(Document).filter(Document.id == document_id).first()
        if not doc:
            logger.warning(f"Document {document_id} not found")
            return False

        # Get existing chunks
        chunks = session.query(DocumentChunk).filter(
            DocumentChunk.document_id == document_id
        ).order_by(DocumentChunk.chunk_index).all()

        if not chunks:
            logger.warning(f"No chunks found for document {document_id}")
            return False

        logger.info(f"Re-embedding document: {doc.filename} ({len(chunks)} chunks)")

        if dry_run:
            logger.info(f"[DRY RUN] Would re-embed {len(chunks)} chunks")
            return True

        # Get embeddings service
        embeddings_service = get_embeddings_service()

        # Re-embed each chunk
        chunk_contents = [chunk.content for chunk in chunks]
        new_embeddings = embeddings_service.embed_documents(chunk_contents)

        # Update embeddings in database
        for chunk, new_embedding in zip(chunks, new_embeddings):
            chunk.embedding = new_embedding

        session.commit()
        logger.info(f"Successfully re-embedded document: {doc.filename}")
        return True

    except Exception as e:
        session.rollback()
        logger.error(f"Error re-embedding document {document_id}: {e}")
        return False
    finally:
        session.close()


def reembed_all_documents(dry_run: bool = False):
    """Re-embed all documents in the database."""
    logger.info("Step 2: Re-embedding all documents with new model...")

    documents = get_all_documents()

    if not documents:
        logger.info("No documents found to re-embed")
        return True

    logger.info(f"Found {len(documents)} documents to re-embed")

    success_count = 0
    error_count = 0

    for i, doc in enumerate(documents, 1):
        logger.info(f"Processing document {i}/{len(documents)}: {doc['filename']}")

        if reembed_document(doc['id'], dry_run):
            success_count += 1
        else:
            error_count += 1

    logger.info(f"\nRe-embedding complete: {success_count} successful, {error_count} errors")
    return error_count == 0


def main():
    parser = argparse.ArgumentParser(
        description='Migrate embeddings from 384 to 768 dimensions'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be done without making changes'
    )
    parser.add_argument(
        '--skip-reembed',
        action='store_true',
        help='Only alter column, skip re-embedding documents'
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Embedding Migration Script")
    print("=" * 60)
    print(f"From: all-MiniLM-L6-v2 (384 dimensions)")
    print(f"To:   sentence-transformers/all-mpnet-base-v2 (768 dimensions)")
    print("=" * 60)

    if args.dry_run:
        print("\n*** DRY RUN MODE - No changes will be made ***\n")

    # Step 1: Alter the column
    if not alter_embedding_column(args.dry_run):
        logger.error("Failed to alter embedding column. Aborting.")
        sys.exit(1)

    # Step 2: Re-embed documents
    if not args.skip_reembed:
        if not reembed_all_documents(args.dry_run):
            logger.error("Some documents failed to re-embed.")
            sys.exit(1)
    else:
        logger.info("Skipping re-embedding (--skip-reembed flag set)")
        logger.warning("Remember to re-embed your documents manually!")

    print("\n" + "=" * 60)
    print("Migration completed successfully!")
    print("=" * 60)

    if args.dry_run:
        print("\nTo apply changes, run without --dry-run flag:")
        print("  python scripts/migrate_embeddings.py")


if __name__ == "__main__":
    main()
