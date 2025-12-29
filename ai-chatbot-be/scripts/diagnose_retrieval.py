#!/usr/bin/env python3
"""
Retrieval Diagnostics Script
============================

Diagnose issues with the document retrieval pipeline.
Checks:
1. Database connection and document_chunks table
2. Embeddings generation
3. Vector search functionality
4. End-to-end retrieval

Usage:
    cd ai-chatbot-be
    python scripts/diagnose_retrieval.py

    # With a specific query:
    python scripts/diagnose_retrieval.py --query "What is the vacation policy?"

    # With a user_id:
    python scripts/diagnose_retrieval.py --user-id "your-user-uuid"
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def check_database_connection():
    """Check database connection and document_chunks table."""
    print("\n" + "=" * 60)
    print("1. DATABASE CONNECTION CHECK")
    print("=" * 60)

    try:
        from app.database.connection import SessionLocal
        from sqlalchemy import text

        session = SessionLocal()

        # Check connection
        result = session.execute(text("SELECT 1")).fetchone()
        print("✅ Database connection: OK")

        # Check document_chunks table
        result = session.execute(text("""
            SELECT COUNT(*) as total,
                   COUNT(embedding) as with_embeddings,
                   COUNT(*) - COUNT(embedding) as without_embeddings
            FROM document_chunks
        """)).fetchone()

        total, with_emb, without_emb = result
        print(f"✅ document_chunks table exists")
        print(f"   - Total chunks: {total}")
        print(f"   - With embeddings: {with_emb}")
        print(f"   - Without embeddings: {without_emb}")

        if total == 0:
            print("⚠️  WARNING: No document chunks found! Upload documents first.")
            return False

        if with_emb == 0:
            print("❌ ERROR: No chunks have embeddings! Run embedding generation.")
            return False

        # Check embedding dimensions
        result = session.execute(text("""
            SELECT vector_dims(embedding) as dims
            FROM document_chunks
            WHERE embedding IS NOT NULL
            LIMIT 1
        """)).fetchone()

        if result:
            dims = result[0]
            print(f"   - Embedding dimensions: {dims}")
        else:
            print("⚠️  WARNING: Could not determine embedding dimensions")

        # Check documents table
        result = session.execute(text("""
            SELECT COUNT(*) as total, COUNT(DISTINCT user_id) as users
            FROM documents
        """)).fetchone()

        print(f"   - Total documents: {result[0]}")
        print(f"   - Unique users: {result[1]}")

        # List existing user_ids for reference
        users_result = session.execute(text("""
            SELECT DISTINCT d.user_id, COUNT(*) as doc_count,
                   (SELECT COUNT(*) FROM document_chunks dc
                    JOIN documents dd ON dc.document_id = dd.id
                    WHERE dd.user_id = d.user_id AND dc.embedding IS NOT NULL) as embedded_count
            FROM documents d
            GROUP BY d.user_id
            LIMIT 5
        """)).fetchall()

        if users_result:
            print(f"   - Sample user_ids with documents:")
            for user_row in users_result:
                print(f"     • {user_row[0]}: {user_row[1]} docs, {user_row[2]} embedded chunks")

        session.close()
        return True

    except Exception as e:
        print(f"❌ Database error: {e}")
        return False


async def check_embeddings_service():
    """Check embeddings service."""
    print("\n" + "=" * 60)
    print("2. EMBEDDINGS SERVICE CHECK")
    print("=" * 60)

    try:
        from app.rag.embeddings.service import EmbeddingsService

        service = EmbeddingsService()
        print(f"✅ Embeddings service initialized")
        print(f"   - Provider: {service.provider}")
        print(f"   - Model: {service.model_name}")
        print(f"   - Dimensions: {service.dimensions}")

        # Test embedding generation
        test_text = "This is a test query for embedding generation."
        embedding = service.embed_query(test_text)

        print(f"✅ Test embedding generated")
        print(f"   - Dimensions: {len(embedding)}")
        print(f"   - First 5 values: {embedding[:5]}")

        return True

    except Exception as e:
        print(f"❌ Embeddings error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def check_vector_search(query: str, user_id: str = None):
    """Check vector search functionality."""
    print("\n" + "=" * 60)
    print("3. VECTOR SEARCH CHECK")
    print("=" * 60)

    try:
        from app.rag.embeddings.service import EmbeddingsService
        from app.database.connection import SessionLocal
        from sqlalchemy import text

        service = EmbeddingsService()
        session = SessionLocal()

        # Generate query embedding
        print(f"Query: {query}")
        embedding = service.embed_query(query)
        print(f"✅ Query embedding generated ({len(embedding)} dims)")

        # Convert to PostgreSQL vector format
        embedding_str = "[" + ",".join(map(str, embedding)) + "]"

        # Use direct SQL for vector search (same as retrieval.py)
        if user_id:
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
                LIMIT 10
            """)
            print(f"Using direct SQL with user_id: {user_id}")
            result = session.execute(sql, {"user_id": user_id}).fetchall()
        else:
            sql = text(f"""
                SELECT
                    dc.id,
                    dc.document_id,
                    dc.content,
                    dc.metadata,
                    1 - (dc.embedding <=> '{embedding_str}'::vector) AS similarity
                FROM document_chunks dc
                WHERE dc.embedding IS NOT NULL
                ORDER BY dc.embedding <=> '{embedding_str}'::vector
                LIMIT 10
            """)
            print("Using direct SQL (no user filter)")
            result = session.execute(sql).fetchall()

        if result:
            print(f"✅ Vector search returned {len(result)} results")
            print("\nTop 5 results:")
            print("-" * 60)
            for i, row in enumerate(result[:5]):
                chunk_id, doc_id, content, metadata, similarity = row
                source = metadata.get("source", "Unknown") if metadata else "Unknown"
                print(f"\n[{i+1}] Score: {similarity:.4f}")
                print(f"    Source: {source}")
                print(f"    Content: {content[:200] if content else 'empty'}...")
        else:
            print("⚠️  No results found!")
            print("   Possible causes:")
            print("   - No documents in document_chunks table")
            print("   - Embedding column is NULL for all chunks")

            # Debug: Check if we have any chunks
            count_result = session.execute(text("""
                SELECT COUNT(*) as total, COUNT(embedding) as with_emb
                FROM document_chunks
            """)).fetchone()
            print(f"\n   Debug: total_chunks={count_result[0]}, with_embeddings={count_result[1]}")

        session.close()
        return len(result) > 0 if result else False

    except Exception as e:
        print(f"❌ Vector search error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def check_retrieval_node(query: str, user_id: str = None):
    """Check the full retrieval node."""
    print("\n" + "=" * 60)
    print("4. RETRIEVAL NODE CHECK")
    print("=" * 60)

    try:
        from app.rag.langgraph.nodes.retrieval import document_retrieval_node
        from app.rag.langgraph.state import create_initial_state

        # Create initial state
        state = create_initial_state(
            query=query,
            user_id=user_id,
        )

        print(f"Query: {query}")
        print(f"User ID: {user_id or 'None'}")

        # Run retrieval
        result = await document_retrieval_node(state)

        documents = result.get("documents", [])
        context = result.get("context", "")
        errors = result.get("error_log", [])

        if errors:
            print(f"⚠️  Errors during retrieval:")
            for err in errors:
                print(f"   - {err}")

        if documents:
            print(f"✅ Retrieved {len(documents)} documents")
            print("\nDocuments:")
            print("-" * 60)
            for i, doc in enumerate(documents):
                print(f"\n[{i+1}] Score: {doc.get('score', 0):.4f}")
                print(f"    Source: {doc.get('source', 'Unknown')}")
                print(f"    Content: {doc.get('content', '')[:200]}...")

            print("\nContext string preview:")
            print("-" * 60)
            print(context[:500] + "..." if len(context) > 500 else context)
        else:
            print("❌ No documents retrieved!")
            print("   The retrieval node failed to find relevant documents.")

        return len(documents) > 0

    except Exception as e:
        print(f"❌ Retrieval node error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def run_diagnostics(query: str, user_id: str = None):
    """Run all diagnostic checks."""
    print("\n" + "=" * 60)
    print("RETRIEVAL DIAGNOSTICS")
    print("=" * 60)
    print(f"Query: {query}")
    print(f"User ID: {user_id or 'None (global search)'}")

    results = {
        "database": await check_database_connection(),
        "embeddings": await check_embeddings_service(),
        "vector_search": await check_vector_search(query, user_id),
        "retrieval_node": await check_retrieval_node(query, user_id),
    }

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    all_passed = True
    for check, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {check}: {status}")
        if not passed:
            all_passed = False

    if all_passed:
        print("\n✅ All checks passed! Retrieval should be working.")
    else:
        print("\n❌ Some checks failed. See details above for troubleshooting.")

    # Recommendations
    print("\n" + "=" * 60)
    print("RECOMMENDATIONS")
    print("=" * 60)

    if not results["database"]:
        print("• Check database connection string in .env")
        print("• Ensure PostgreSQL is running")
        print("• Run database migrations")

    if not results["embeddings"]:
        print("• Check EMBEDDING_PROVIDER in .env")
        print("• Install required packages (sentence-transformers, etc.)")

    if not results["vector_search"]:
        print("• Lower MATCH_THRESHOLD (try 0.1 or 0.2)")
        print("• Check that embeddings were generated for documents")
        print("• Verify embedding dimensions match between query and stored")

    if not results["retrieval_node"]:
        print("• Check logs for detailed error messages")
        print("• Ensure user_id exists in documents table (if using user filter)")


def main():
    parser = argparse.ArgumentParser(description="Diagnose retrieval pipeline")
    parser.add_argument(
        "--query",
        type=str,
        default="What is the company policy?",
        help="Test query for retrieval"
    )
    parser.add_argument(
        "--user-id",
        type=str,
        default=None,
        help="User ID for scoped search"
    )
    args = parser.parse_args()

    asyncio.run(run_diagnostics(args.query, args.user_id))


if __name__ == "__main__":
    main()
