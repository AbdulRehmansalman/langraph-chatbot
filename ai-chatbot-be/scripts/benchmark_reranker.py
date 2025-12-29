#!/usr/bin/env python3
"""
Reranker Benchmark Script
=========================

Standalone script to benchmark the BGE reranker against your actual data.
Measures MRR@5, NDCG@10, and latency.

Usage:
    cd ai-chatbot-be
    python scripts/benchmark_reranker.py

    # With custom queries file:
    python scripts/benchmark_reranker.py --queries queries.json

    # Compare with/without reranking:
    python scripts/benchmark_reranker.py --compare
"""

import argparse
import asyncio
import json
import logging
import math
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Results from a single query benchmark."""
    query: str
    mrr_before: float
    mrr_after: float
    ndcg_before: float
    ndcg_after: float
    latency_ms: float
    top_docs_before: list[str]
    top_docs_after: list[str]


def calculate_mrr(docs: list[dict], relevant_ids: set[str], k: int = 5) -> float:
    """Calculate Mean Reciprocal Rank @ k."""
    for i, doc in enumerate(docs[:k]):
        if doc.get("id", "") in relevant_ids:
            return 1.0 / (i + 1)
    return 0.0


def calculate_ndcg(docs: list[dict], relevance: dict[str, float], k: int = 10) -> float:
    """Calculate Normalized Discounted Cumulative Gain @ k."""
    dcg = sum(
        (2 ** relevance.get(doc.get("id", ""), 0) - 1) / math.log2(i + 2)
        for i, doc in enumerate(docs[:k])
    )
    ideal_rels = sorted(relevance.values(), reverse=True)[:k]
    idcg = sum((2 ** rel - 1) / math.log2(i + 2) for i, rel in enumerate(ideal_rels))
    return dcg / idcg if idcg > 0 else 0.0


async def fetch_documents(query: str, user_id: str = None, count: int = 20) -> list[dict]:
    """Fetch documents from the vector store without reranking."""
    from app.rag.embeddings.service import EmbeddingsService
    from app.services.supabase_client import get_supabase_client

    try:
        supabase = await get_supabase_client()
        embedding_service = EmbeddingsService()

        query_embedding = await asyncio.to_thread(
            embedding_service.embed_query, query
        )

        query_builder = supabase.rpc(
            "match_documents",
            {
                "query_embedding": query_embedding,
                "match_threshold": 0.3,
                "match_count": count,
            }
        )

        if user_id:
            query_builder = query_builder.eq("user_id", user_id)

        result = await asyncio.to_thread(query_builder.execute)

        docs = []
        if result.data:
            for i, doc in enumerate(result.data):
                docs.append({
                    "id": doc.get("id", f"doc_{i}"),
                    "content": doc.get("content", ""),
                    "source": doc.get("metadata", {}).get("source", "Unknown"),
                    "score": doc.get("similarity", 0.0),
                    "metadata": doc.get("metadata", {}),
                })
        return docs

    except Exception as e:
        logger.error(f"Failed to fetch documents: {e}")
        return []


async def benchmark_query(
    query: str,
    relevant_ids: set[str],
    relevance_scores: dict[str, float],
    user_id: str = None,
) -> BenchmarkResult:
    """Benchmark a single query with and without reranking."""
    from app.rag.langgraph.nodes.retrieval import _rerank_documents

    # Fetch original documents
    docs = await fetch_documents(query, user_id, count=20)

    if not docs:
        return BenchmarkResult(
            query=query,
            mrr_before=0, mrr_after=0,
            ndcg_before=0, ndcg_after=0,
            latency_ms=0,
            top_docs_before=[], top_docs_after=[]
        )

    # Calculate metrics before reranking
    mrr_before = calculate_mrr(docs, relevant_ids, k=5)
    ndcg_before = calculate_ndcg(docs, relevance_scores, k=10)
    top_before = [d["id"] for d in docs[:5]]

    # Rerank and measure latency
    start = time.time()
    reranked = await _rerank_documents(query, docs.copy())
    latency_ms = (time.time() - start) * 1000

    # Calculate metrics after reranking
    mrr_after = calculate_mrr(reranked, relevant_ids, k=5)
    ndcg_after = calculate_ndcg(reranked, relevance_scores, k=5)
    top_after = [d["id"] for d in reranked[:5]]

    return BenchmarkResult(
        query=query,
        mrr_before=mrr_before,
        mrr_after=mrr_after,
        ndcg_before=ndcg_before,
        ndcg_after=ndcg_after,
        latency_ms=latency_ms,
        top_docs_before=top_before,
        top_docs_after=top_after,
    )


async def run_benchmark(queries_file: str = None, compare: bool = False):
    """Run the full benchmark suite."""
    from app.rag.langgraph.nodes.retrieval import _get_reranker

    print("\n" + "=" * 60)
    print("BGE RERANKER BENCHMARK")
    print("=" * 60 + "\n")

    # Load reranker first
    print("Loading reranker model...")
    reranker = await _get_reranker()
    if reranker is None:
        print("ERROR: Failed to load reranker!")
        return
    print("Reranker loaded successfully\n")

    # Default test queries if no file provided
    test_cases = [
        {
            "query": "What is our company's remote work policy?",
            "relevant_ids": [],  # Will be empty without ground truth
            "relevance_scores": {}
        },
        {
            "query": "How do I schedule a meeting with my manager?",
            "relevant_ids": [],
            "relevance_scores": {}
        },
        {
            "query": "What are the steps to submit an expense report?",
            "relevant_ids": [],
            "relevance_scores": {}
        },
        {
            "query": "How do I reset my password?",
            "relevant_ids": [],
            "relevance_scores": {}
        },
        {
            "query": "What holidays does the company observe?",
            "relevant_ids": [],
            "relevance_scores": {}
        },
    ]

    # Load custom queries if provided
    if queries_file and os.path.exists(queries_file):
        with open(queries_file) as f:
            test_cases = json.load(f)
        print(f"Loaded {len(test_cases)} queries from {queries_file}\n")

    results: list[BenchmarkResult] = []

    for i, tc in enumerate(test_cases):
        query = tc["query"]
        relevant_ids = set(tc.get("relevant_ids", []))
        relevance_scores = tc.get("relevance_scores", {})

        print(f"[{i+1}/{len(test_cases)}] {query[:60]}...")

        result = await benchmark_query(query, relevant_ids, relevance_scores)
        results.append(result)

        # Print per-query results
        if compare:
            print(f"  MRR@5:  {result.mrr_before:.4f} → {result.mrr_after:.4f} ({result.mrr_after - result.mrr_before:+.4f})")
            print(f"  NDCG:   {result.ndcg_before:.4f} → {result.ndcg_after:.4f} ({result.ndcg_after - result.ndcg_before:+.4f})")
        print(f"  Latency: {result.latency_ms:.1f}ms")
        print(f"  Top 5:   {result.top_docs_after}")
        print()

    # Aggregate results
    print("\n" + "=" * 60)
    print("AGGREGATE RESULTS")
    print("=" * 60)

    avg_mrr_before = sum(r.mrr_before for r in results) / len(results) if results else 0
    avg_mrr_after = sum(r.mrr_after for r in results) / len(results) if results else 0
    avg_ndcg_before = sum(r.ndcg_before for r in results) / len(results) if results else 0
    avg_ndcg_after = sum(r.ndcg_after for r in results) / len(results) if results else 0
    avg_latency = sum(r.latency_ms for r in results) / len(results) if results else 0
    p95_latency = sorted(r.latency_ms for r in results)[int(len(results) * 0.95)] if len(results) > 1 else avg_latency

    print(f"\nQueries tested: {len(results)}")
    print(f"\nMRR@5:")
    print(f"  Before reranking: {avg_mrr_before:.4f}")
    print(f"  After reranking:  {avg_mrr_after:.4f}")
    print(f"  Improvement:      {avg_mrr_after - avg_mrr_before:+.4f} ({((avg_mrr_after/avg_mrr_before - 1) * 100) if avg_mrr_before > 0 else 0:+.1f}%)")

    print(f"\nNDCG@10:")
    print(f"  Before reranking: {avg_ndcg_before:.4f}")
    print(f"  After reranking:  {avg_ndcg_after:.4f}")
    print(f"  Improvement:      {avg_ndcg_after - avg_ndcg_before:+.4f}")

    print(f"\nLatency:")
    print(f"  Average: {avg_latency:.1f}ms")
    print(f"  P95:     {p95_latency:.1f}ms")

    # Save results
    output_file = f"benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    output_data = {
        "timestamp": datetime.now().isoformat(),
        "queries_count": len(results),
        "aggregate": {
            "mrr_before": avg_mrr_before,
            "mrr_after": avg_mrr_after,
            "mrr_improvement": avg_mrr_after - avg_mrr_before,
            "ndcg_before": avg_ndcg_before,
            "ndcg_after": avg_ndcg_after,
            "ndcg_improvement": avg_ndcg_after - avg_ndcg_before,
            "avg_latency_ms": avg_latency,
            "p95_latency_ms": p95_latency,
        },
        "per_query": [
            {
                "query": r.query,
                "mrr_before": r.mrr_before,
                "mrr_after": r.mrr_after,
                "ndcg_before": r.ndcg_before,
                "ndcg_after": r.ndcg_after,
                "latency_ms": r.latency_ms,
                "top_docs_before": r.top_docs_before,
                "top_docs_after": r.top_docs_after,
            }
            for r in results
        ]
    }

    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nResults saved to: {output_file}")

    # Decision guidance
    print("\n" + "=" * 60)
    print("DECISION GUIDANCE")
    print("=" * 60)

    mrr_improvement = (avg_mrr_after - avg_mrr_before) / avg_mrr_before * 100 if avg_mrr_before > 0 else 0

    if mrr_improvement >= 15:
        print("\n✅ PROCEED TO PHASE 2: Reranking shows significant improvement (>15%)")
    elif mrr_improvement >= 10:
        print("\n⚠️  MARGINAL: Improvement is 10-15%. Consider more testing before Phase 2.")
    else:
        print("\n❌ INSUFFICIENT: Improvement <10%. Consider Phase 5 (Hybrid Search) before Phase 2.")
        print("   Debug steps:")
        print("   1. Check if your documents have enough content for semantic matching")
        print("   2. Try a larger reranker model (bge-reranker-large)")
        print("   3. Review top results manually for quality")


def main():
    parser = argparse.ArgumentParser(description="Benchmark BGE reranker")
    parser.add_argument("--queries", type=str, help="Path to JSON file with test queries")
    parser.add_argument("--compare", action="store_true", help="Show before/after comparison")
    args = parser.parse_args()

    asyncio.run(run_benchmark(queries_file=args.queries, compare=args.compare))


if __name__ == "__main__":
    main()
