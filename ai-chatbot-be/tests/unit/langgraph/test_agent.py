"""
Unit Tests for LangGraph Agent
==============================

Tests for the agent graph, state, and nodes.
"""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from datetime import datetime


class TestAgentState:
    """Tests for AgentState creation and helpers."""

    @pytest.mark.unit
    def test_create_initial_state(self):
        """Should create valid initial state."""
        from app.rag.langgraph.state import create_initial_state

        state = create_initial_state(
            query="What is the refund policy?",
            user_id="user-123",
            user_name="Test User",
        )

        assert state["original_query"] == "What is the refund policy?"
        assert state["user_id"] == "user-123"
        assert state["user_name"] == "Test User"
        assert state["schema_version"] == 4
        assert len(state["messages"]) == 1
        assert state["thread_id"] is not None

    @pytest.mark.unit
    def test_create_initial_state_with_thread_id(self):
        """Should use provided thread_id."""
        from app.rag.langgraph.state import create_initial_state

        state = create_initial_state(
            query="Hello",
            thread_id="custom-thread-123",
        )

        assert state["thread_id"] == "custom-thread-123"

    @pytest.mark.unit
    def test_create_initial_state_validates_query_length(self):
        """Should reject overly long queries."""
        from app.rag.langgraph.state import create_initial_state

        long_query = "x" * 20000  # Exceeds max length

        with pytest.raises(ValueError):
            create_initial_state(query=long_query)

    @pytest.mark.unit
    def test_track_node(self, base_agent_state):
        """Should track node execution."""
        from app.rag.langgraph.state import track_node

        updates = track_node(base_agent_state, "router")

        assert updates["current_node"] == "router"
        assert "router" in updates["execution_path"]
        assert updates["iteration_count"] == 1

    @pytest.mark.unit
    def test_add_error(self, base_agent_state):
        """Should add error to state."""
        from app.rag.langgraph.state import add_error

        updates = add_error(
            base_agent_state,
            node="retrieval",
            error_type="TIMEOUT",
            message="Request timed out",
        )

        assert len(updates["error_log"]) == 1
        assert updates["last_error"] == "Request timed out"

    @pytest.mark.unit
    def test_add_documents(self, base_agent_state):
        """Should add documents and build context."""
        from app.rag.langgraph.state import add_documents

        docs = [
            {"id": "doc-1", "content": "Test content 1", "source": "file1.pdf"},
            {"id": "doc-2", "content": "Test content 2", "source": "file2.pdf"},
        ]

        updates = add_documents(base_agent_state, docs)

        assert len(updates["documents"]) == 2
        assert "Test content 1" in updates["context"]
        assert "Test content 2" in updates["context"]


class TestAgentRouting:
    """Tests for query routing logic."""

    @pytest.mark.unit
    def test_route_greeting(self):
        """Should classify greetings correctly."""
        from app.rag.langgraph.graph import route_after_router

        state = {"query_classification": "greeting", "has_error": False}
        assert route_after_router(state) == "greeting"

    @pytest.mark.unit
    def test_route_document(self):
        """Should route document queries correctly."""
        from app.rag.langgraph.graph import route_after_router

        state = {"query_classification": "document", "has_error": False}
        assert route_after_router(state) == "document"

    @pytest.mark.unit
    def test_route_calendar(self):
        """Should route calendar queries correctly."""
        from app.rag.langgraph.graph import route_after_router

        state = {"query_classification": "calendar", "has_error": False}
        assert route_after_router(state) == "calendar"

    @pytest.mark.unit
    def test_route_error(self):
        """Should route to error on error state."""
        from app.rag.langgraph.graph import route_after_router

        state = {"query_classification": "document", "has_error": True}
        assert route_after_router(state) == "error"


class TestAgentNodes:
    """Tests for individual agent nodes."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_entry_node(self, base_agent_state):
        """Entry node should initialize state."""
        from app.rag.langgraph.graph import entry_node

        base_agent_state["user_id"] = "test-user"
        base_agent_state["original_query"] = "Hello"

        result = await entry_node(base_agent_state)

        assert "execution_path" in result
        assert "entry" in result["execution_path"]

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_router_node_classifies_greeting(self, base_agent_state):
        """Router should classify greetings."""
        from app.rag.langgraph.graph import router_node

        base_agent_state["original_query"] = "Hello there!"

        result = await router_node(base_agent_state)

        assert result["query_classification"] == "greeting"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_router_node_classifies_calendar(self, base_agent_state):
        """Router should classify calendar queries."""
        from app.rag.langgraph.graph import router_node

        base_agent_state["original_query"] = "Schedule a meeting for tomorrow"

        result = await router_node(base_agent_state)

        assert result["query_classification"] == "calendar"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_router_node_classifies_document(self, base_agent_state):
        """Router should classify document queries."""
        from app.rag.langgraph.graph import router_node

        base_agent_state["original_query"] = "What does the policy document say?"

        result = await router_node(base_agent_state)

        assert result["query_classification"] == "document"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_greeting_node(self, base_agent_state):
        """Greeting node should generate response."""
        from app.rag.langgraph.graph import greeting_node

        base_agent_state["user_name"] = "John"

        result = await greeting_node(base_agent_state)

        assert "response" in result
        assert "John" in result["response"]
        assert result["should_end"] == True

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_error_node(self, base_agent_state):
        """Error node should generate error response."""
        from app.rag.langgraph.graph import error_node

        base_agent_state["last_error"] = "Connection failed"

        result = await error_node(base_agent_state)

        assert "response" in result
        assert "Connection failed" in result["response"]
        assert result["should_end"] == True


class TestAgentGraphBuilder:
    """Tests for AgentGraphBuilder."""

    @pytest.mark.unit
    def test_build_with_memory_checkpointer(self):
        """Should build graph with memory checkpointer."""
        from app.rag.langgraph.graph import AgentGraphBuilder

        builder = AgentGraphBuilder()
        builder.with_memory_checkpointer()
        graph = builder.build()

        assert graph is not None

    @pytest.mark.unit
    def test_build_with_human_review(self):
        """Should configure human review interrupt."""
        from app.rag.langgraph.graph import AgentGraphBuilder

        builder = AgentGraphBuilder()
        builder.with_human_review()

        assert "human_review" in builder._interrupt_before


class TestAgent:
    """Tests for the Agent class."""

    @pytest.mark.unit
    def test_agent_initialization(self):
        """Should initialize agent successfully."""
        from app.rag.langgraph.graph import Agent

        agent = Agent()
        assert agent._graph is not None

    @pytest.mark.unit
    def test_create_agent_factory(self):
        """Factory should create agent."""
        from app.rag.langgraph.graph import create_agent

        agent = create_agent()
        assert agent is not None
