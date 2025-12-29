"""
LangGraph Agent Implementation
==============================

Production-grade agent built with LangGraph featuring:
- Stateful conversation management
- Document retrieval
- Calendar integration
- Human-in-the-loop review gates
- Streaming support
- PostgreSQL checkpointing

Usage:
    from app.rag.langgraph import Agent, create_agent, AgentState

    # Create agent
    agent = create_agent(database_url="postgresql://...")

    # Invoke
    result = await agent.invoke("What is the refund policy?", user_id="user-123")

    # Stream
    async for event in agent.stream("Schedule a meeting"):
        print(event)
"""

from app.rag.langgraph.state import (
    AgentState,
    QueryClassification,
    HumanReviewStatus,
    ToolStatus,
    create_initial_state,
    track_node,
    add_error,
    add_documents,
    update_metrics,
)
from app.rag.langgraph.graph import (
    Agent,
    AgentGraphBuilder,
    create_agent,
)

__all__ = [
    # State
    "AgentState",
    "QueryClassification",
    "HumanReviewStatus",
    "ToolStatus",
    "create_initial_state",
    "track_node",
    "add_error",
    "add_documents",
    "update_metrics",
    # Agent
    "Agent",
    "AgentGraphBuilder",
    "create_agent",
]
