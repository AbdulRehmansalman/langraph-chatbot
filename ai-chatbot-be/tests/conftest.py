"""
Pytest Configuration and Fixtures
=================================
Shared fixtures for all tests.
"""

import pytest
import asyncio
from datetime import datetime
from typing import Any
from unittest.mock import MagicMock, AsyncMock
from uuid import uuid4


# =============================================================================
# EVENT LOOP FIXTURE
# =============================================================================

@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# =============================================================================
# AGENT STATE FIXTURES
# =============================================================================

@pytest.fixture
def base_agent_state() -> dict[str, Any]:
    """Create a base agent state for testing."""
    return {
        # Schema
        "schema_version": 4,
        # Messages
        "messages": [],
        # Session
        "thread_id": str(uuid4()),
        "user_id": "test-user-456",
        "user_name": "Test User",
        "session_id": str(uuid4()),
        "timestamp": datetime.utcnow().isoformat(),
        # Query
        "original_query": "",
        "query_classification": "",
        # Documents
        "document_ids": None,
        "documents": [],
        "context": "",
        # Tools
        "tools_called": [],
        "tool_results": [],
        "current_tool": None,
        # Calendar
        "calendar_action": None,
        "calendar_events": [],
        "scheduled_meeting": None,
        # Response
        "response": "",
        "citations": [],
        # Human-in-the-loop
        "requires_approval": False,
        "approval_status": "not_required",
        "approval_request": None,
        "reviewer_id": None,
        # Execution
        "current_node": "",
        "next_node": None,
        "execution_path": [],
        "should_end": False,
        "iteration_count": 0,
        "max_iterations": 10,
        # Errors
        "error_log": [],
        "has_error": False,
        "last_error": None,
        # Metrics
        "metrics": {
            "total_duration_ms": 0.0,
            "llm_calls": 0,
            "tokens_input": 0,
            "tokens_output": 0,
            "estimated_cost_usd": 0.0,
            "tools_executed": [],
            "nodes_executed": [],
        },
    }


# Alias for backward compatibility
@pytest.fixture
def base_rag_state(base_agent_state) -> dict[str, Any]:
    """Alias for base_agent_state (backward compatibility)."""
    return base_agent_state


@pytest.fixture
def sample_query_state(base_agent_state) -> dict[str, Any]:
    """State with a sample query."""
    state = base_agent_state.copy()
    state["original_query"] = "What is the refund policy?"
    return state


@pytest.fixture
def sample_documents() -> list[dict]:
    """Sample retrieved documents for testing."""
    return [
        {
            "id": "doc-1",
            "content": "Our refund policy allows returns within 30 days of purchase.",
            "source": "policies.pdf",
            "score": 0.95,
            "metadata": {"page": 1, "chunk_id": "chunk-1"},
        },
        {
            "id": "doc-2",
            "content": "To request a refund, contact customer service with your order number.",
            "source": "faq.pdf",
            "score": 0.87,
            "metadata": {"page": 5, "chunk_id": "chunk-2"},
        },
        {
            "id": "doc-3",
            "content": "Digital products are non-refundable after download.",
            "source": "terms.pdf",
            "score": 0.72,
            "metadata": {"page": 12, "chunk_id": "chunk-3"},
        },
    ]


@pytest.fixture
def state_with_documents(sample_query_state, sample_documents) -> dict[str, Any]:
    """State with query and retrieved documents."""
    state = sample_query_state.copy()
    state["documents"] = sample_documents
    state["query_classification"] = "document"
    state["context"] = "\n\n---\n\n".join([
        f"[{i+1}] Source: {d['source']}\n{d['content']}"
        for i, d in enumerate(sample_documents)
    ])
    return state


@pytest.fixture
def state_with_response(state_with_documents) -> dict[str, Any]:
    """State with generated response."""
    state = state_with_documents.copy()
    state["response"] = (
        "Our refund policy allows returns within 30 days of purchase [1]. "
        "To request a refund, please contact customer service with your order number [2]. "
        "Note that digital products are non-refundable after download [3]."
    )
    state["citations"] = [
        {"index": 1, "document_id": "doc-1", "source": "policies.pdf", "snippet": "..."},
        {"index": 2, "document_id": "doc-2", "source": "faq.pdf", "snippet": "..."},
        {"index": 3, "document_id": "doc-3", "source": "terms.pdf", "snippet": "..."},
    ]
    return state


# =============================================================================
# MOCK SERVICE FIXTURES
# =============================================================================

@pytest.fixture
def mock_llm():
    """Mock LLM for testing."""
    mock = MagicMock()
    mock.invoke = MagicMock(return_value=MagicMock(content="Test response"))
    mock.ainvoke = AsyncMock(return_value=MagicMock(content="Test response"))
    mock.astream = AsyncMock(return_value=iter([
        MagicMock(content="Test"),
        MagicMock(content=" "),
        MagicMock(content="response"),
    ]))
    return mock


@pytest.fixture
def mock_vector_store():
    """Mock vector store for testing."""
    mock = MagicMock()
    mock.similarity_search = MagicMock(return_value=[
        MagicMock(
            page_content="Test content",
            metadata={"source": "test.pdf", "page": 1}
        )
    ])
    mock.asimilarity_search = AsyncMock(return_value=[
        MagicMock(
            page_content="Test content",
            metadata={"source": "test.pdf", "page": 1}
        )
    ])
    return mock


@pytest.fixture
def mock_embeddings():
    """Mock embeddings service for testing."""
    mock = MagicMock()
    mock.embed_query = MagicMock(return_value=[0.1] * 768)
    mock.embed_documents = MagicMock(return_value=[[0.1] * 768])
    return mock


@pytest.fixture
def mock_supabase():
    """Mock Supabase client for testing."""
    mock = MagicMock()
    mock.table = MagicMock(return_value=mock)
    mock.select = MagicMock(return_value=mock)
    mock.insert = MagicMock(return_value=mock)
    mock.update = MagicMock(return_value=mock)
    mock.delete = MagicMock(return_value=mock)
    mock.eq = MagicMock(return_value=mock)
    mock.rpc = MagicMock(return_value=mock)
    mock.execute = MagicMock(return_value=MagicMock(data=[]))
    return mock


# =============================================================================
# CALENDAR FIXTURES
# =============================================================================

@pytest.fixture
def sample_calendar_event() -> dict[str, Any]:
    """Sample calendar event for testing."""
    return {
        "id": "event-123",
        "title": "Team Meeting",
        "description": "Weekly sync",
        "start_time": "2024-01-15T10:00:00Z",
        "end_time": "2024-01-15T11:00:00Z",
        "duration_minutes": 60,
        "location": "Conference Room A",
        "attendees": ["user1@example.com", "user2@example.com"],
        "google_meet_link": "https://meet.google.com/abc-defg-hij",
        "status": "confirmed",
    }


@pytest.fixture
def mock_calendar_service():
    """Mock Google Calendar service for testing."""
    mock = MagicMock()
    mock.initialize = AsyncMock(return_value=True)
    mock.create_event = AsyncMock(return_value={
        "success": True,
        "event_id": "new-event-123",
    })
    mock.list_events = AsyncMock(return_value={
        "success": True,
        "events": [],
        "count": 0,
    })
    mock.check_availability = AsyncMock(return_value={
        "success": True,
        "free_slots": [
            {"start": "09:00", "end": "10:00"},
            {"start": "14:00", "end": "15:00"},
        ],
    })
    return mock


# =============================================================================
# UTILITY FIXTURES
# =============================================================================

@pytest.fixture
def capture_logs(caplog):
    """Capture logs at DEBUG level."""
    import logging
    caplog.set_level(logging.DEBUG)
    return caplog


# =============================================================================
# TEST MARKERS
# =============================================================================

def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )
