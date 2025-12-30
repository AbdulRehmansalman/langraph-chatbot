"""
Tests for Query Router
======================

Tests the rule-based query classification and routing logic.
Run with: pytest tests/unit/langgraph/test_router.py -v
"""

import pytest
from app.rag.langgraph.graph import (
    classify_query_rules,
    get_routing_stats,
    _match_patterns,
    GREETING_PATTERNS,
    CALENDAR_PATTERNS,
    DOCUMENT_PATTERNS,
    DIRECT_ANSWER_PATTERNS,
    SENSITIVE_PATTERNS,
)


class TestPatternMatching:
    """Test regex pattern matching."""

    def test_greeting_patterns(self):
        """Test greeting pattern detection."""
        greetings = ["hi", "hello", "hey", "Hi!", "Hello.", "good morning", "Good Evening!"]
        for g in greetings:
            assert _match_patterns(g, GREETING_PATTERNS), f"Failed to match greeting: {g}"

        non_greetings = ["hi there, how are you doing today", "hello world program"]
        for ng in non_greetings:
            # These shouldn't match because they're too long or have extra content
            # The greeting patterns use ^ and $ anchors
            pass  # We check via classify_query_rules instead

    def test_calendar_patterns(self):
        """Test calendar pattern detection."""
        calendar_queries = [
            "schedule a meeting",
            "book an appointment",
            "what's on my calendar",
            "am I free tomorrow",
            "find a free slot",
            "reschedule my meeting",
            "cancel the appointment",
            "when can we meet",
        ]
        for q in calendar_queries:
            assert _match_patterns(q, CALENDAR_PATTERNS), f"Failed to match calendar: {q}"

    def test_document_patterns(self):
        """Test document pattern detection."""
        doc_queries = [
            "search in the document",
            "what does the policy say",
            "summarize the report",
            "from the file",
            "page 5",
            "our company policy",
            "the uploaded document",
        ]
        for q in doc_queries:
            assert _match_patterns(q, DOCUMENT_PATTERNS), f"Failed to match document: {q}"

    def test_direct_answer_patterns(self):
        """Test direct answer pattern detection."""
        direct_queries = ["thanks", "ok", "yes", "no", "bye", "help", "who are you"]
        for q in direct_queries:
            assert _match_patterns(q, DIRECT_ANSWER_PATTERNS), f"Failed to match direct: {q}"


class TestClassifyQueryRules:
    """Test the classify_query_rules function."""

    # Greeting tests
    @pytest.mark.parametrize("query", [
        "hi",
        "hello",
        "hey",
        "Hello!",
        "Hi there",
        "good morning",
        "Good evening!",
        "what's up",
    ])
    def test_greeting_classification(self, query):
        """Test greeting queries are classified correctly."""
        classification, reason = classify_query_rules(query)
        assert classification == "greeting", f"'{query}' should be greeting, got {classification}"

    # Calendar tests
    @pytest.mark.parametrize("query", [
        "schedule a meeting for tomorrow",
        "book an appointment with Dr. Smith",
        "what meetings do I have today",
        "am I free next Tuesday at 3pm",
        "find me a free slot this week",
        "reschedule my 2pm meeting",
        "cancel my appointment",
        "when can we meet next week",
        "check my calendar",
        "show my schedule",
    ])
    def test_calendar_classification(self, query):
        """Test calendar queries are classified correctly."""
        classification, reason = classify_query_rules(query)
        assert classification == "calendar", f"'{query}' should be calendar, got {classification}"

    # Document tests
    @pytest.mark.parametrize("query", [
        "search the document for project updates",
        "what does the policy say about remote work",
        "summarize the Q4 report",
        "find the section about benefits",
        "what's on page 5",
        "according to the handbook",
        "our company policy on vacation",
    ])
    def test_document_classification(self, query):
        """Test document queries are classified correctly."""
        classification, reason = classify_query_rules(query)
        assert classification == "document", f"'{query}' should be document, got {classification}"

    # Direct answer tests
    @pytest.mark.parametrize("query", [
        "thanks",
        "thank you",
        "ok",
        "okay",
        "yes",
        "no",
        "bye",
        "goodbye",
        "help",
        "who are you",
        "what can you do",
        "never mind",
    ])
    def test_direct_classification(self, query):
        """Test direct queries are classified correctly."""
        classification, reason = classify_query_rules(query)
        assert classification == "direct", f"'{query}' should be direct, got {classification}"

    # Sensitive topic tests
    @pytest.mark.parametrize("query", [
        "delete all my documents",
        "schedule a meeting with the CEO",
        "show salary information",
        "access medical records",
        "view confidential files",
    ])
    def test_sensitive_classification(self, query):
        """Test sensitive queries trigger human approval."""
        classification, reason = classify_query_rules(query)
        assert classification == "human_approval", f"'{query}' should be human_approval, got {classification}"

    # General tests (no specific pattern)
    @pytest.mark.parametrize("query", [
        "tell me a joke",
        "what's the weather like",
        "random question here",
    ])
    def test_general_classification(self, query):
        """Test general queries are classified correctly."""
        classification, reason = classify_query_rules(query)
        assert classification == "general", f"'{query}' should be general, got {classification}"

    def test_complex_question_with_documents(self):
        """Test that complex questions route to documents when user has docs."""
        query = "What are the main points discussed in the report?"

        # Without documents - should be general
        classification, _ = classify_query_rules(query, has_documents=False)
        assert classification == "general"

        # With documents - should be document (because it's a complex question)
        classification, _ = classify_query_rules(query, has_documents=True)
        assert classification == "document"


class TestRoutingStats:
    """Test routing statistics tracking."""

    def test_get_routing_stats(self):
        """Test that routing stats are returned correctly."""
        stats = get_routing_stats()

        assert "total" in stats
        assert "greeting" in stats
        assert "calendar" in stats
        assert "document" in stats
        assert "direct" in stats
        assert "general" in stats
        assert "skip_retrieval_rate" in stats

    def test_skip_retrieval_rate_calculation(self):
        """Test skip retrieval rate is calculated correctly."""
        # This tests the formula: (greeting + calendar + direct) / total * 100
        stats = get_routing_stats()

        if stats["total"] > 0:
            expected_rate = (
                (stats["greeting"] + stats["calendar"] + stats["direct"])
                / stats["total"]
            ) * 100
            assert abs(stats["skip_retrieval_rate"] - expected_rate) < 0.01


class TestRouterEdgeCases:
    """Test edge cases in routing."""

    def test_empty_query(self):
        """Test empty query handling."""
        classification, reason = classify_query_rules("")
        assert classification == "general"

    def test_whitespace_query(self):
        """Test whitespace-only query handling."""
        classification, reason = classify_query_rules("   ")
        assert classification == "general"

    def test_mixed_case(self):
        """Test case insensitivity."""
        classification1, _ = classify_query_rules("SCHEDULE A MEETING")
        classification2, _ = classify_query_rules("schedule a meeting")
        assert classification1 == classification2 == "calendar"

    def test_greeting_with_question(self):
        """Test greeting followed by a question is not just a greeting."""
        # Short greeting = greeting
        classification1, _ = classify_query_rules("Hi")
        assert classification1 == "greeting"

        # Greeting + question = depends on the rest
        classification2, _ = classify_query_rules("Hi, can you help me find a document?")
        # This should be document because of "find a document"
        assert classification2 == "document"

    def test_reason_provided(self):
        """Test that a reason is always provided."""
        queries = ["hi", "schedule meeting", "search document", "thanks", "random"]
        for q in queries:
            classification, reason = classify_query_rules(q)
            assert reason is not None
            assert len(reason) > 0


class TestRouterBenchmark:
    """Benchmark tests for router performance."""

    BENCHMARK_QUERIES = [
        ("hello", "greeting"),
        ("hi there!", "greeting"),
        ("schedule a meeting tomorrow at 2pm", "calendar"),
        ("what's on my calendar today", "calendar"),
        ("find information in the policy document", "document"),
        ("what does the handbook say about PTO", "document"),
        ("thanks for your help", "direct"),
        ("who are you", "direct"),
        ("delete all my files", "human_approval"),
        ("meeting with the CEO", "human_approval"),
        ("tell me something interesting", "general"),
    ]

    def test_classification_accuracy(self):
        """Test classification accuracy on benchmark queries."""
        correct = 0
        total = len(self.BENCHMARK_QUERIES)

        for query, expected in self.BENCHMARK_QUERIES:
            actual, _ = classify_query_rules(query)
            if actual == expected:
                correct += 1
            else:
                print(f"MISMATCH: '{query}' expected {expected}, got {actual}")

        accuracy = correct / total * 100
        print(f"\nClassification accuracy: {accuracy:.1f}% ({correct}/{total})")

        # Should have at least 90% accuracy on benchmark
        assert accuracy >= 90, f"Accuracy too low: {accuracy:.1f}%"

    def test_classification_speed(self):
        """Test that classification is fast enough."""
        import time

        queries = [q for q, _ in self.BENCHMARK_QUERIES] * 100  # 1100 queries

        start = time.time()
        for q in queries:
            classify_query_rules(q)
        duration_ms = (time.time() - start) * 1000

        avg_ms = duration_ms / len(queries)
        print(f"\nAverage classification time: {avg_ms:.3f}ms per query")

        # Should be under 1ms per query
        assert avg_ms < 1.0, f"Classification too slow: {avg_ms:.3f}ms"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
