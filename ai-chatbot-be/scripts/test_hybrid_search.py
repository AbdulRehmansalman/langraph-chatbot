#!/usr/bin/env python3
"""
Hybrid Search Test Script
=========================

Test the hybrid search functionality combining vector + full-text search.

Usage:
    cd ai-chatbot-be
    python scripts/test_hybrid_search.py

    # With a specific query:
    python scripts/test_hybrid_search.py --query "What is the vacation policy?"

    # With a user_id:
    python scripts/test_hybrid_search.py --user-id "your-user-uuid"
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


async def test_query_preprocessing():
    """Test the query preprocessing function."""
    print("\n" + "=" * 60)
    print("1. QUERY PREPROCESSING TEST")
    print("=" * 60)

    try:
        from app.rag.langgraph.nodes.retrieval import preprocess_query

        test_queries = [
            "What is the vacation policy?",
            "How do I submit a leave request form?",
            "Tell me about health benefits and insurance",
            "policy",
            "the",  # Should filter out stopword
        ]

        for query in test_queries:
            result = preprocess_query(query)
            print(f"\n  Query: '{query}'")
            print(f"  Keywords: {result['keywords']}")
            print(f"  TS Query: {result['ts_query']}")

        print("\n✅ Query preprocessing works!")
        return True

    except Exception as e:
        print(f"\n❌ Query preprocessing failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_vector_search(query: str, user_id: str = None):
    """Test vector search."""
    print("\n" + "=" * 60)
    print("2. VECTOR SEARCH TEST")
    print("=" * 60)

    try:
        from app.rag.langgraph.nodes.retrieval import _vector_search

        print(f"  Query: {query}")
        print(f"  User ID: {user_id or 'None'}")

        results = await _vector_search(query, user_id)

        if results:
            print(f"\n✅ Vector search returned {len(results)} results:")
            for i, doc in enumerate(results[:3]):
                print(f"\n  [{i+1}] Score: {doc['score']:.4f}")
                print(f"      Source: {doc.get('source', 'unknown')}")
                print(f"      Content: {doc['content'][:100]}...")
        else:
            print("\n⚠️  Vector search returned no results")

        return len(results) > 0

    except Exception as e:
        print(f"\n❌ Vector search failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_fulltext_search(query: str, user_id: str = None):
    """Test full-text search."""
    print("\n" + "=" * 60)
    print("3. FULL-TEXT SEARCH TEST")
    print("=" * 60)

    try:
        from app.rag.langgraph.nodes.retrieval import (
            preprocess_query,
            _fulltext_search,
        )

        query_info = preprocess_query(query)
        print(f"  Query: {query}")
        print(f"  Keywords: {query_info['keywords']}")
        print(f"  User ID: {user_id or 'None'}")

        results = await _fulltext_search(query_info, user_id)

        if results:
            print(f"\n✅ Full-text search returned {len(results)} results:")
            for i, doc in enumerate(results[:3]):
                print(f"\n  [{i+1}] Score: {doc['score']:.4f}")
                print(f"      Source: {doc.get('source', 'unknown')}")
                print(f"      Content: {doc['content'][:100]}...")
        else:
            print("\n⚠️  Full-text search returned no results")

        return len(results) > 0

    except Exception as e:
        print(f"\n❌ Full-text search failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_rrf_fusion():
    """Test Reciprocal Rank Fusion."""
    print("\n" + "=" * 60)
    print("4. RECIPROCAL RANK FUSION TEST")
    print("=" * 60)

    try:
        from app.rag.langgraph.nodes.retrieval import reciprocal_rank_fusion

        # Simulate two result lists
        vector_results = [
            {"id": "doc1", "content": "Doc 1", "score": 0.95, "rank": 1, "search_type": "vector"},
            {"id": "doc2", "content": "Doc 2", "score": 0.85, "rank": 2, "search_type": "vector"},
            {"id": "doc3", "content": "Doc 3", "score": 0.75, "rank": 3, "search_type": "vector"},
        ]

        fulltext_results = [
            {"id": "doc2", "content": "Doc 2", "score": 0.90, "rank": 1, "search_type": "fulltext"},
            {"id": "doc4", "content": "Doc 4", "score": 0.80, "rank": 2, "search_type": "fulltext"},
            {"id": "doc1", "content": "Doc 1", "score": 0.70, "rank": 3, "search_type": "fulltext"},
        ]

        fused = reciprocal_rank_fusion([vector_results, fulltext_results])

        print(f"\n  Vector results: {[d['id'] for d in vector_results]}")
        print(f"  Fulltext results: {[d['id'] for d in fulltext_results]}")
        print(f"\n  Fused results (by RRF score):")

        for i, doc in enumerate(fused):
            print(f"    [{i+1}] {doc['id']}: RRF={doc['rrf_score']:.4f}, found_by={doc['found_by']}")

        # Verify doc2 is ranked highest (found by both)
        if fused[0]["id"] == "doc2":
            print("\n✅ RRF correctly ranks doc2 (found by both) highest!")
            return True
        else:
            print("\n⚠️  RRF ranking may not be optimal")
            return False

    except Exception as e:
        print(f"\n❌ RRF fusion failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_hybrid_retrieval(query: str, user_id: str = None):
    """Test full hybrid retrieval pipeline."""
    print("\n" + "=" * 60)
    print("5. FULL HYBRID RETRIEVAL TEST")
    print("=" * 60)

    try:
        from app.rag.langgraph.nodes.retrieval import document_retrieval_node
        from app.rag.langgraph.state import create_initial_state

        state = create_initial_state(query=query, user_id=user_id)

        print(f"  Query: {query}")
        print(f"  User ID: {user_id or 'None'}")

        result = await document_retrieval_node(state)

        documents = result.get("documents", [])
        errors = result.get("error_log", [])

        if errors:
            print(f"\n⚠️  Errors during retrieval:")
            for err in errors:
                print(f"    - {err}")

        if documents:
            print(f"\n✅ Hybrid retrieval returned {len(documents)} documents:")
            for i, doc in enumerate(documents):
                found_by = doc.get("found_by", ["unknown"])
                print(f"\n  [{i+1}] Score: {doc.get('score', 0):.4f}")
                print(f"      Found by: {found_by}")
                print(f"      Source: {doc.get('source', 'unknown')}")
                print(f"      Content: {doc.get('content', '')[:150]}...")
            return True
        else:
            print("\n❌ No documents retrieved!")
            return False

    except Exception as e:
        print(f"\n❌ Hybrid retrieval failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def run_all_tests(query: str, user_id: str = None):
    """Run all hybrid search tests."""
    print("\n" + "=" * 60)
    print("HYBRID SEARCH TEST SUITE")
    print("=" * 60)
    print(f"Query: {query}")
    print(f"User ID: {user_id or 'None (global search)'}")

    results = {
        "preprocessing": await test_query_preprocessing(),
        "rrf_fusion": await test_rrf_fusion(),
        "vector_search": await test_vector_search(query, user_id),
        "fulltext_search": await test_fulltext_search(query, user_id),
        "hybrid_retrieval": await test_hybrid_retrieval(query, user_id),
    }

    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    all_passed = True
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {test_name}: {status}")
        if not passed:
            all_passed = False

    if all_passed:
        print("\n✅ All tests passed! Hybrid search is working.")
    else:
        print("\n⚠️  Some tests failed. Check the output above for details.")

    # Recommendations
    print("\n" + "=" * 60)
    print("CONFIGURATION")
    print("=" * 60)
    print("  Environment variables for tuning:")
    print("    HYBRID_SEARCH_ENABLED=true  (default: true)")
    print("    MATCH_THRESHOLD=0.1         (default: 0.1)")
    print("    RERANKER_ENABLED=true       (default: true)")
    print("    RERANKER_MODEL=BAAI/bge-reranker-base")


def main():
    parser = argparse.ArgumentParser(description="Test hybrid search")
    parser.add_argument(
        "--query",
        type=str,
        default="What is the vacation policy?",
        help="Test query for retrieval"
    )
    parser.add_argument(
        "--user-id",
        type=str,
        default=None,
        help="User ID for scoped search"
    )
    args = parser.parse_args()

    asyncio.run(run_all_tests(args.query, args.user_id))


if __name__ == "__main__":
    main()
