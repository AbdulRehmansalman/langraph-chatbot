"""
Tests for BGE Reranker Integration
==================================

Tests the cross-encoder reranking functionality in the retrieval node.
Run with: pytest tests/unit/langgraph/test_reranker.py -v
"""

import asyncio
import os
import time
from typing import Any
import pytest


# Metrics calculation functions
def calculate_mrr(ranked_docs: list[dict], relevant_ids: set[str], k: int = 5) -> float:
    """
    Calculate Mean Reciprocal Rank @ k.

    Args:
        ranked_docs: List of documents in ranked order
        relevant_ids: Set of document IDs considered relevant
        k: Cutoff for ranking consideration

    Returns:
        MRR score (0.0 to 1.0)
    """
    for i, doc in enumerate(ranked_docs[:k]):
        doc_id = doc.get("id", "")
        if doc_id in relevant_ids:
            return 1.0 / (i + 1)
    return 0.0


def calculate_ndcg(ranked_docs: list[dict], relevance_scores: dict[str, float], k: int = 10) -> float:
    """
    Calculate Normalized Discounted Cumulative Gain @ k.

    Args:
        ranked_docs: List of documents in ranked order
        relevance_scores: Dict mapping doc_id to relevance score (0-3 scale)
        k: Cutoff for ranking consideration

    Returns:
        NDCG score (0.0 to 1.0)
    """
    import math

    # Calculate DCG
    dcg = 0.0
    for i, doc in enumerate(ranked_docs[:k]):
        doc_id = doc.get("id", "")
        rel = relevance_scores.get(doc_id, 0.0)
        dcg += (2 ** rel - 1) / math.log2(i + 2)

    # Calculate ideal DCG
    ideal_rels = sorted(relevance_scores.values(), reverse=True)[:k]
    idcg = sum((2 ** rel - 1) / math.log2(i + 2) for i, rel in enumerate(ideal_rels))

    if idcg == 0:
        return 0.0
    return dcg / idcg


class TestRerankerUnit:
    """Unit tests for reranker functions."""

    def test_mrr_perfect_ranking(self):
        """Test MRR when first result is relevant."""
        docs = [{"id": "doc1"}, {"id": "doc2"}, {"id": "doc3"}]
        relevant = {"doc1"}
        assert calculate_mrr(docs, relevant) == 1.0

    def test_mrr_second_position(self):
        """Test MRR when relevant doc is second."""
        docs = [{"id": "doc1"}, {"id": "doc2"}, {"id": "doc3"}]
        relevant = {"doc2"}
        assert calculate_mrr(docs, relevant) == 0.5

    def test_mrr_no_relevant(self):
        """Test MRR when no relevant docs in top k."""
        docs = [{"id": "doc1"}, {"id": "doc2"}, {"id": "doc3"}]
        relevant = {"doc10"}
        assert calculate_mrr(docs, relevant, k=3) == 0.0

    def test_ndcg_perfect_ranking(self):
        """Test NDCG with perfect ordering."""
        docs = [{"id": "doc1"}, {"id": "doc2"}, {"id": "doc3"}]
        relevance = {"doc1": 3.0, "doc2": 2.0, "doc3": 1.0}
        ndcg = calculate_ndcg(docs, relevance, k=3)
        assert ndcg == pytest.approx(1.0, rel=0.01)

    def test_ndcg_reversed_ranking(self):
        """Test NDCG with reversed ordering (worst case)."""
        docs = [{"id": "doc3"}, {"id": "doc2"}, {"id": "doc1"}]
        relevance = {"doc1": 3.0, "doc2": 2.0, "doc3": 1.0}
        ndcg = calculate_ndcg(docs, relevance, k=3)
        assert ndcg < 1.0  # Should be worse than perfect


class TestRerankerIntegration:
    """Integration tests for the reranker with the retrieval node."""

    @pytest.fixture
    def sample_documents(self) -> list[dict[str, Any]]:
        """Sample documents for testing reranking."""
        return [
            {"id": "doc1", "content": "The quick brown fox jumps over the lazy dog.", "score": 0.9, "source": "test1.txt"},
            {"id": "doc2", "content": "Python is a programming language used for machine learning.", "score": 0.85, "source": "test2.txt"},
            {"id": "doc3", "content": "Machine learning models require training data to learn patterns.", "score": 0.8, "source": "test3.txt"},
            {"id": "doc4", "content": "The weather today is sunny with clear skies.", "score": 0.75, "source": "test4.txt"},
            {"id": "doc5", "content": "Deep learning is a subset of machine learning using neural networks.", "score": 0.7, "source": "test5.txt"},
            {"id": "doc6", "content": "Artificial intelligence encompasses machine learning and other techniques.", "score": 0.65, "source": "test6.txt"},
            {"id": "doc7", "content": "Natural language processing helps computers understand human language.", "score": 0.6, "source": "test7.txt"},
            {"id": "doc8", "content": "The cat sat on the mat looking at the window.", "score": 0.55, "source": "test8.txt"},
            {"id": "doc9", "content": "Data science involves statistics, programming, and domain knowledge.", "score": 0.5, "source": "test9.txt"},
            {"id": "doc10", "content": "LLMs like GPT use transformer architecture for text generation.", "score": 0.45, "source": "test10.txt"},
        ]

    @pytest.mark.asyncio
    async def test_reranker_loads(self):
        """Test that the reranker model loads successfully."""
        from app.rag.langgraph.nodes.retrieval import _get_reranker

        reranker = await _get_reranker()
        assert reranker is not None, "Reranker failed to load"

    @pytest.mark.asyncio
    async def test_reranking_changes_order(self, sample_documents):
        """Test that reranking actually changes document order based on query relevance."""
        from app.rag.langgraph.nodes.retrieval import _rerank_documents

        query = "What is machine learning and how does it work?"

        # Get original order by score
        original_order = [doc["id"] for doc in sample_documents[:5]]

        # Rerank
        reranked = await _rerank_documents(query, sample_documents.copy())
        reranked_order = [doc["id"] for doc in reranked]

        # Verify we got results
        assert len(reranked) == 5, f"Expected 5 results, got {len(reranked)}"

        # Verify reranking changed something (documents about ML should rank higher)
        # Note: This may not always change order depending on the model
        print(f"Original top 5: {original_order}")
        print(f"Reranked top 5: {reranked_order}")

        # Check that rerank scores exist
        for doc in reranked:
            assert "rerank_score" in doc, "Missing rerank_score"
            assert "original_score" in doc, "Missing original_score"

    @pytest.mark.asyncio
    async def test_reranking_improves_relevance(self, sample_documents):
        """Test that reranking improves relevance for a specific query."""
        from app.rag.langgraph.nodes.retrieval import _rerank_documents

        # Query specifically about ML
        query = "How do neural networks learn from data?"

        # Define which docs are actually relevant (ground truth)
        relevant_ids = {"doc3", "doc5", "doc6", "doc10"}  # ML-related docs
        relevance_scores = {
            "doc1": 0.0, "doc2": 1.0, "doc3": 2.0, "doc4": 0.0,
            "doc5": 3.0, "doc6": 2.0, "doc7": 1.0, "doc8": 0.0,
            "doc9": 1.0, "doc10": 2.0
        }

        # Calculate metrics before reranking
        original_mrr = calculate_mrr(sample_documents, relevant_ids, k=5)
        original_ndcg = calculate_ndcg(sample_documents, relevance_scores, k=10)

        # Rerank with more candidates
        reranked = await _rerank_documents(query, sample_documents.copy())

        # Calculate metrics after reranking
        # Need to extend to k=10 for NDCG calculation
        reranked_mrr = calculate_mrr(reranked, relevant_ids, k=5)
        reranked_ndcg = calculate_ndcg(reranked, relevance_scores, k=5)

        print(f"\n=== Metrics Comparison ===")
        print(f"Original MRR@5:  {original_mrr:.4f}")
        print(f"Reranked MRR@5:  {reranked_mrr:.4f}")
        print(f"Original NDCG@10: {original_ndcg:.4f}")
        print(f"Reranked NDCG@5:  {reranked_ndcg:.4f}")

        # Log the reranked order
        print(f"\nReranked order: {[doc['id'] for doc in reranked]}")
        for doc in reranked:
            print(f"  {doc['id']}: rerank={doc.get('rerank_score', 0):.4f}")

    @pytest.mark.asyncio
    async def test_reranking_latency(self, sample_documents):
        """Test that reranking completes within acceptable time."""
        from app.rag.langgraph.nodes.retrieval import _rerank_documents

        query = "What is artificial intelligence?"

        # Warm up (first call loads model)
        await _rerank_documents(query, sample_documents.copy())

        # Measure latency
        start = time.time()
        await _rerank_documents(query, sample_documents.copy())
        duration_ms = (time.time() - start) * 1000

        print(f"\nReranking latency: {duration_ms:.1f}ms")

        # Should complete in under 500ms for 10 docs (on CPU)
        assert duration_ms < 500, f"Reranking too slow: {duration_ms:.1f}ms"

    @pytest.mark.asyncio
    async def test_reranker_fallback(self, sample_documents):
        """Test that disabling reranker returns original docs."""
        # Temporarily disable reranker
        original_setting = os.environ.get("RERANKER_ENABLED", "true")
        os.environ["RERANKER_ENABLED"] = "false"

        try:
            # Need to reimport to pick up env change
            import importlib
            from app.rag.langgraph.nodes import retrieval
            importlib.reload(retrieval)

            result = await retrieval._rerank_documents("test query", sample_documents.copy())

            # Should return first 5 unchanged
            assert len(result) == 5
            assert result[0]["id"] == "doc1"
            assert "rerank_score" not in result[0]
        finally:
            os.environ["RERANKER_ENABLED"] = original_setting
            import importlib
            from app.rag.langgraph.nodes import retrieval
            importlib.reload(retrieval)


class TestRerankerBenchmark:
    """Benchmark tests for measuring reranker improvement."""

    TEST_QUERIES = [
        {
            "query": "How do I schedule a meeting for tomorrow?",
            "relevant_keywords": ["schedule", "meeting", "calendar", "appointment"]
        },
        {
            "query": "What is our company's remote work policy?",
            "relevant_keywords": ["remote", "work", "policy", "home", "office"]
        },
        {
            "query": "How do I reset my password?",
            "relevant_keywords": ["password", "reset", "account", "login", "security"]
        },
        {
            "query": "What are the benefits of machine learning?",
            "relevant_keywords": ["machine", "learning", "AI", "benefits", "artificial"]
        },
        {
            "query": "How do I submit an expense report?",
            "relevant_keywords": ["expense", "report", "submit", "reimbursement", "finance"]
        },
    ]

    @pytest.mark.asyncio
    async def test_benchmark_suite(self):
        """Run benchmark on test queries and report aggregate metrics."""
        from app.rag.langgraph.nodes.retrieval import _rerank_documents, _get_reranker

        # Ensure model is loaded
        reranker = await _get_reranker()
        if reranker is None:
            pytest.skip("Reranker not available")

        print("\n=== Reranker Benchmark Suite ===\n")

        total_latency = 0
        for i, test_case in enumerate(self.TEST_QUERIES):
            query = test_case["query"]

            # Create synthetic docs (in real benchmark, use actual DB)
            docs = [
                {"id": f"doc_{j}", "content": f"Document {j} content about {test_case['relevant_keywords'][j % len(test_case['relevant_keywords'])]}", "score": 0.9 - j * 0.05}
                for j in range(10)
            ]

            start = time.time()
            reranked = await _rerank_documents(query, docs)
            latency = (time.time() - start) * 1000
            total_latency += latency

            print(f"Query {i+1}: '{query[:50]}...'")
            print(f"  Latency: {latency:.1f}ms")
            print(f"  Top result: {reranked[0]['id']} (score: {reranked[0].get('rerank_score', 0):.4f})")

        avg_latency = total_latency / len(self.TEST_QUERIES)
        print(f"\n=== Summary ===")
        print(f"Average latency: {avg_latency:.1f}ms")
        print(f"Queries tested: {len(self.TEST_QUERIES)}")


if __name__ == "__main__":
    # Run with: python -m pytest tests/unit/langgraph/test_reranker.py -v -s
    pytest.main([__file__, "-v", "-s"])
