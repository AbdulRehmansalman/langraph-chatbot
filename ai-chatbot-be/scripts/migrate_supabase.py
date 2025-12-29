#!/usr/bin/env python3
"""
Supabase Migration Script
=========================
Migrates the database schema to match the simplified LangGraph chatbot schema.

This script will:
1. Enable required extensions (uuid-ossp, vector)
2. Create all required tables if they don't exist
3. Add missing columns to existing tables
4. Create indexes for performance
5. Create vector similarity search functions

Usage:
    python scripts/migrate_supabase.py

Options:
    --dry-run       Show what would be done without making changes
    --drop-old      Drop old/deprecated tables and columns (use with caution)
"""

import sys
import os
import argparse
import logging

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import text
from app.database.connection import SessionLocal

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# SQL STATEMENTS
# =============================================================================

ENABLE_EXTENSIONS = """
-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "vector";
"""

CREATE_USERS_TABLE = """
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(255) UNIQUE NOT NULL,
    hashed_password VARCHAR(255) NOT NULL,
    full_name VARCHAR(255),
    timezone VARCHAR(50) DEFAULT 'UTC',
    is_active BOOLEAN DEFAULT TRUE,
    email_verified BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);
"""

CREATE_OTPS_TABLE = """
CREATE TABLE IF NOT EXISTS otps (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(255) NOT NULL,
    otp VARCHAR(10) NOT NULL,
    purpose VARCHAR(50) NOT NULL,
    expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
    used BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);
"""

CREATE_DOCUMENTS_TABLE = """
CREATE TABLE IF NOT EXISTS documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    filename VARCHAR(500) NOT NULL,
    content_type VARCHAR(100) NOT NULL,
    file_path VARCHAR(500),
    storage_url TEXT,
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    processed BOOLEAN DEFAULT FALSE,
    error TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);
"""

CREATE_DOCUMENT_CHUNKS_TABLE = """
CREATE TABLE IF NOT EXISTS document_chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    embedding VECTOR(768),
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);
"""

CREATE_CHAT_HISTORY_TABLE = """
CREATE TABLE IF NOT EXISTS chat_history (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    thread_id VARCHAR(255),
    user_message TEXT NOT NULL,
    bot_response TEXT NOT NULL,
    document_ids UUID[] DEFAULT '{}',
    response_time DECIMAL(8,3),
    has_documents BOOLEAN DEFAULT FALSE,
    sources_used INTEGER DEFAULT 0,
    provider VARCHAR(100),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);
"""

CREATE_CHECKPOINTS_TABLE = """
CREATE TABLE IF NOT EXISTS checkpoints (
    thread_id VARCHAR(255) NOT NULL,
    checkpoint_ns VARCHAR(255) NOT NULL DEFAULT '',
    checkpoint_id VARCHAR(255) NOT NULL,
    parent_checkpoint_id VARCHAR(255),
    type VARCHAR(255),
    checkpoint JSONB NOT NULL,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id)
);
"""

CREATE_CHECKPOINT_WRITES_TABLE = """
CREATE TABLE IF NOT EXISTS checkpoint_writes (
    thread_id VARCHAR(255) NOT NULL,
    checkpoint_ns VARCHAR(255) NOT NULL DEFAULT '',
    checkpoint_id VARCHAR(255) NOT NULL,
    task_id VARCHAR(255) NOT NULL,
    idx INTEGER NOT NULL,
    channel VARCHAR(255) NOT NULL,
    type VARCHAR(255),
    value JSONB,
    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id, task_id, idx)
);
"""

CREATE_USER_GOOGLE_AUTH_TABLE = """
CREATE TABLE IF NOT EXISTS user_google_auth (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE UNIQUE,
    access_token TEXT NOT NULL,
    refresh_token TEXT,
    token_expiry TIMESTAMP WITH TIME ZONE,
    scopes JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);
"""

CREATE_MEETINGS_TABLE = """
CREATE TABLE IF NOT EXISTS meetings (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    title VARCHAR(500) NOT NULL,
    description TEXT,
    scheduled_time TIMESTAMP WITH TIME ZONE NOT NULL,
    duration_minutes INTEGER DEFAULT 30,
    meeting_link TEXT,
    attendees JSONB DEFAULT '[]',
    status VARCHAR(50) DEFAULT 'scheduled',
    google_event_id VARCHAR(255),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);
"""

CREATE_INDEXES = """
-- Users
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);

-- Documents
CREATE INDEX IF NOT EXISTS idx_documents_user_id ON documents(user_id);

-- Document chunks
CREATE INDEX IF NOT EXISTS idx_document_chunks_document_id ON document_chunks(document_id);

-- Chat history
CREATE INDEX IF NOT EXISTS idx_chat_history_user_id ON chat_history(user_id);
CREATE INDEX IF NOT EXISTS idx_chat_history_thread_id ON chat_history(thread_id);
CREATE INDEX IF NOT EXISTS idx_chat_history_created_at ON chat_history(created_at DESC);

-- Checkpoints
CREATE INDEX IF NOT EXISTS idx_checkpoints_thread_id ON checkpoints(thread_id);

-- OTPs
CREATE INDEX IF NOT EXISTS idx_otps_email ON otps(email);

-- Meetings
CREATE INDEX IF NOT EXISTS idx_meetings_user_id ON meetings(user_id);
CREATE INDEX IF NOT EXISTS idx_meetings_scheduled_time ON meetings(scheduled_time);
"""

CREATE_VECTOR_INDEX = """
-- Vector similarity search index (IVFFlat)
CREATE INDEX IF NOT EXISTS idx_document_chunks_embedding ON document_chunks
    USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
"""

CREATE_MATCH_DOCUMENTS_FUNCTION = """
CREATE OR REPLACE FUNCTION match_documents(
    query_embedding VECTOR(768),
    match_threshold FLOAT DEFAULT 0.5,
    match_count INT DEFAULT 10
)
RETURNS TABLE (
    id UUID,
    document_id UUID,
    content TEXT,
    metadata JSONB,
    similarity FLOAT
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        dc.id,
        dc.document_id,
        dc.content,
        dc.metadata,
        1 - (dc.embedding <=> query_embedding) AS similarity
    FROM document_chunks dc
    WHERE dc.embedding IS NOT NULL
      AND 1 - (dc.embedding <=> query_embedding) > match_threshold
    ORDER BY dc.embedding <=> query_embedding
    LIMIT match_count;
END;
$$;
"""

CREATE_MATCH_USER_DOCUMENTS_FUNCTION = """
CREATE OR REPLACE FUNCTION match_user_documents(
    query_embedding VECTOR(768),
    user_id_param UUID,
    match_threshold FLOAT DEFAULT 0.5,
    match_count INT DEFAULT 10
)
RETURNS TABLE (
    id UUID,
    document_id UUID,
    content TEXT,
    metadata JSONB,
    similarity FLOAT
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        dc.id,
        dc.document_id,
        dc.content,
        dc.metadata,
        1 - (dc.embedding <=> query_embedding) AS similarity
    FROM document_chunks dc
    JOIN documents d ON dc.document_id = d.id
    WHERE d.user_id = user_id_param
      AND dc.embedding IS NOT NULL
      AND 1 - (dc.embedding <=> query_embedding) > match_threshold
    ORDER BY dc.embedding <=> query_embedding
    LIMIT match_count;
END;
$$;
"""

# Tables to drop if --drop-old is specified
DROP_OLD_TABLES = """
DROP TABLE IF EXISTS document_embeddings CASCADE;
DROP TABLE IF EXISTS conversation_threads CASCADE;
DROP TABLE IF EXISTS agent_metrics CASCADE;
"""


def execute_sql(session, sql: str, description: str, dry_run: bool = False):
    """Execute SQL statement with logging."""
    if dry_run:
        logger.info(f"[DRY RUN] Would execute: {description}")
        return True

    try:
        session.execute(text(sql))
        session.commit()
        logger.info(f"SUCCESS: {description}")
        return True
    except Exception as e:
        session.rollback()
        logger.error(f"ERROR executing {description}: {e}")
        return False


def check_table_exists(session, table_name: str) -> bool:
    """Check if a table exists in the database."""
    result = session.execute(text(f"""
        SELECT EXISTS (
            SELECT FROM information_schema.tables
            WHERE table_name = '{table_name}'
        );
    """)).fetchone()
    return result[0] if result else False


def check_column_exists(session, table_name: str, column_name: str) -> bool:
    """Check if a column exists in a table."""
    result = session.execute(text(f"""
        SELECT EXISTS (
            SELECT FROM information_schema.columns
            WHERE table_name = '{table_name}' AND column_name = '{column_name}'
        );
    """)).fetchone()
    return result[0] if result else False


def migrate_old_embeddings_table(session, dry_run: bool = False):
    """Migrate data from document_embeddings to document_chunks if needed."""
    if not check_table_exists(session, 'document_embeddings'):
        logger.info("No document_embeddings table found - skipping migration")
        return True

    if check_table_exists(session, 'document_chunks'):
        # Check if document_chunks already has data
        result = session.execute(text("SELECT COUNT(*) FROM document_chunks")).fetchone()
        if result and result[0] > 0:
            logger.info("document_chunks already has data - skipping migration")
            return True

    logger.info("Migrating data from document_embeddings to document_chunks...")

    if dry_run:
        logger.info("[DRY RUN] Would migrate data from document_embeddings to document_chunks")
        return True

    try:
        # Check if document_embeddings has data
        result = session.execute(text("SELECT COUNT(*) FROM document_embeddings")).fetchone()
        count = result[0] if result else 0

        if count > 0:
            # Migrate data
            session.execute(text("""
                INSERT INTO document_chunks (id, document_id, chunk_index, content, embedding, metadata, created_at)
                SELECT id, document_id, chunk_index, content, embedding,
                       COALESCE(chunk_metadata, '{}')::jsonb, created_at
                FROM document_embeddings
                ON CONFLICT (id) DO NOTHING;
            """))
            session.commit()
            logger.info(f"Migrated {count} records from document_embeddings to document_chunks")
        else:
            logger.info("No data to migrate from document_embeddings")

        return True
    except Exception as e:
        session.rollback()
        logger.error(f"Error migrating data: {e}")
        return False


def run_migration(dry_run: bool = False, drop_old: bool = False):
    """Run the full migration."""
    session = SessionLocal()
    success = True

    try:
        # Step 1: Enable extensions
        logger.info("\n=== Step 1: Enabling Extensions ===")
        if not execute_sql(session, ENABLE_EXTENSIONS, "Enable extensions", dry_run):
            logger.warning("Some extensions may not have been enabled")

        # Step 2: Create tables
        logger.info("\n=== Step 2: Creating Tables ===")
        tables = [
            (CREATE_USERS_TABLE, "Create users table"),
            (CREATE_OTPS_TABLE, "Create otps table"),
            (CREATE_DOCUMENTS_TABLE, "Create documents table"),
            (CREATE_DOCUMENT_CHUNKS_TABLE, "Create document_chunks table"),
            (CREATE_CHAT_HISTORY_TABLE, "Create chat_history table"),
            (CREATE_CHECKPOINTS_TABLE, "Create checkpoints table"),
            (CREATE_CHECKPOINT_WRITES_TABLE, "Create checkpoint_writes table"),
            (CREATE_USER_GOOGLE_AUTH_TABLE, "Create user_google_auth table"),
            (CREATE_MEETINGS_TABLE, "Create meetings table"),
        ]

        for sql, desc in tables:
            if not execute_sql(session, sql, desc, dry_run):
                success = False

        # Step 3: Migrate old data if needed
        logger.info("\n=== Step 3: Migrating Old Data ===")
        if not migrate_old_embeddings_table(session, dry_run):
            success = False

        # Step 4: Create indexes
        logger.info("\n=== Step 4: Creating Indexes ===")
        if not execute_sql(session, CREATE_INDEXES, "Create indexes", dry_run):
            success = False

        # Step 5: Create vector index (may fail if no data)
        logger.info("\n=== Step 5: Creating Vector Index ===")
        try:
            execute_sql(session, CREATE_VECTOR_INDEX, "Create vector index", dry_run)
        except Exception as e:
            logger.warning(f"Could not create vector index (may need data first): {e}")

        # Step 6: Create functions
        logger.info("\n=== Step 6: Creating Functions ===")
        if not execute_sql(session, CREATE_MATCH_DOCUMENTS_FUNCTION, "Create match_documents function", dry_run):
            success = False
        if not execute_sql(session, CREATE_MATCH_USER_DOCUMENTS_FUNCTION, "Create match_user_documents function", dry_run):
            success = False

        # Step 7: Drop old tables (optional)
        if drop_old:
            logger.info("\n=== Step 7: Dropping Old Tables ===")
            if not execute_sql(session, DROP_OLD_TABLES, "Drop old tables", dry_run):
                logger.warning("Some old tables may not have been dropped")

        return success

    except Exception as e:
        logger.error(f"Migration failed: {e}")
        return False
    finally:
        session.close()


def verify_migration(session):
    """Verify the migration was successful."""
    logger.info("\n=== Migration Verification ===")

    tables = [
        'users', 'otps', 'documents', 'document_chunks',
        'chat_history', 'checkpoints', 'checkpoint_writes',
        'user_google_auth', 'meetings'
    ]

    all_exist = True
    for table in tables:
        exists = check_table_exists(session, table)
        status = "OK" if exists else "MISSING"
        logger.info(f"  {table}: {status}")
        if not exists:
            all_exist = False

    # Check functions
    functions = ['match_documents', 'match_user_documents']
    for func in functions:
        result = session.execute(text(f"""
            SELECT EXISTS (
                SELECT FROM pg_proc WHERE proname = '{func}'
            );
        """)).fetchone()
        exists = result[0] if result else False
        status = "OK" if exists else "MISSING"
        logger.info(f"  {func}(): {status}")
        if not exists:
            all_exist = False

    return all_exist


def main():
    parser = argparse.ArgumentParser(
        description='Migrate Supabase database to simplified schema'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be done without making changes'
    )
    parser.add_argument(
        '--drop-old',
        action='store_true',
        help='Drop old/deprecated tables (document_embeddings, etc.)'
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Supabase Migration Script")
    print("=" * 60)
    print("Target schema: Simplified LangGraph Chatbot")
    print("Tables: users, otps, documents, document_chunks,")
    print("        chat_history, checkpoints, checkpoint_writes,")
    print("        user_google_auth, meetings")
    print("=" * 60)

    if args.dry_run:
        print("\n*** DRY RUN MODE - No changes will be made ***\n")

    if args.drop_old:
        print("\n*** WARNING: Old tables will be dropped ***\n")

    # Run migration
    success = run_migration(args.dry_run, args.drop_old)

    # Verify
    if not args.dry_run:
        session = SessionLocal()
        try:
            verify_migration(session)
        finally:
            session.close()

    print("\n" + "=" * 60)
    if success:
        print("Migration completed successfully!")
    else:
        print("Migration completed with some errors. Check logs above.")
    print("=" * 60)

    if args.dry_run:
        print("\nTo apply changes, run without --dry-run flag:")
        print("  python scripts/migrate_supabase.py")

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
