"""
LangGraph Agent
===============

Production-ready agent graph with:
- Query routing (document/calendar/general)
- Tool execution
- Human-in-the-loop approval
- Streaming support
- PostgreSQL checkpointing

Graph Structure:
    entry → router → [document | calendar | general] → response → END
                   ↘ greeting → END
                   ↘ human_review → response → END
"""

import logging
import random
from typing import Any, AsyncIterator, Literal, Optional

from langchain_core.messages import AIMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from app.rag.langgraph.state import (
    AgentState,
    create_initial_state,
    track_node,
    check_iteration_limit,
    add_error,
    update_metrics,
    get_response_context,
)

logger = logging.getLogger(__name__)


# =============================================================================
# ROUTING FUNCTIONS
# =============================================================================

def route_after_router(state: AgentState) -> Literal["document", "calendar", "general", "greeting", "human_review", "error"]:
    """Route based on query classification."""
    if state.get("has_error"):
        return "error"

    classification = state.get("query_classification", "general")

    if classification == "greeting":
        return "greeting"

    if classification == "human_approval":
        return "human_review"

    if classification == "document":
        return "document"

    if classification == "calendar":
        return "calendar"

    return "general"


def route_after_approval(state: AgentState) -> Literal["response", "end"]:
    """Route after human review."""
    status = state.get("approval_status", "not_required")

    if status == "approved":
        return "response"

    return "end"


def should_require_approval(state: AgentState) -> bool:
    """Check if action requires human approval."""
    action = state.get("calendar_action")
    if action in ["cancel", "reschedule"]:
        return True
    return state.get("requires_approval", False)


# =============================================================================
# NODE FUNCTIONS
# =============================================================================

async def entry_node(state: AgentState) -> dict:
    """Entry point - initialize and validate."""
    import time

    updates = track_node(state, "entry")
    updates["execution_start_time"] = time.time()

    # Log entry
    logger.info(f"Agent started for user: {state.get('user_id')}, query: {state.get('original_query', '')[:50]}...")

    return updates


async def router_node(state: AgentState) -> dict:
    """Classify query and determine routing."""
    query = state.get("original_query", "").lower()

    updates = track_node(state, "router")

    # Simple classification (in production, use LLM)
    if any(word in query for word in ["hello", "hi", "hey", "good morning", "good evening"]):
        updates["query_classification"] = "greeting"

    elif any(word in query for word in ["schedule", "meeting", "calendar", "appointment", "book", "available", "free slot"]):
        updates["query_classification"] = "calendar"

    elif any(word in query for word in ["document", "file", "search", "find", "what", "how", "policy", "procedure"]):
        updates["query_classification"] = "document"

    else:
        updates["query_classification"] = "general"

    logger.info(f"Query classified as: {updates['query_classification']}")

    return updates


async def greeting_node(state: AgentState) -> dict:
    """Handle greeting queries."""
    user_name = state.get("user_name", "there")

    greetings = [
        f"Hello {user_name}! How can I help you today?",
        f"Hi {user_name}! I'm here to assist with documents and calendar.",
        f"Hey {user_name}! What would you like to do?",
    ]

    return {
        **track_node(state, "greeting"),
        "response": random.choice(greetings),
        "should_end": True,
    }


async def document_node(state: AgentState) -> dict:
    """Handle document search queries."""
    from app.rag.langgraph.nodes.retrieval import document_retrieval_node
    return await document_retrieval_node(state)


async def calendar_node(state: AgentState) -> dict:
    """Handle calendar/appointment queries."""
    from app.rag.langgraph.tools.appointment_tools import (
        check_calendar,
        schedule_meeting,
        find_available_slots,
    )

    query = state.get("original_query", "").lower()
    user_id = state.get("user_id")

    updates = track_node(state, "calendar")

    try:
        if any(word in query for word in ["schedule", "book", "create", "set up"]):
            result = await schedule_meeting.ainvoke({
                "title": "Meeting",
                "datetime_str": "tomorrow at 10am",
                "user_id": user_id,
            })
            updates["calendar_action"] = "schedule"
            updates["scheduled_meeting"] = result

        elif any(word in query for word in ["available", "free", "slot"]):
            result = await find_available_slots.ainvoke({
                "date_range_start": "today",
                "user_id": user_id,
            })
            updates["calendar_action"] = "check"
            updates["calendar_events"] = result.get("available_slots", [])

        else:
            result = await check_calendar.ainvoke({
                "date": "today",
                "user_id": user_id,
            })
            updates["calendar_action"] = "check"
            updates["calendar_events"] = result.get("events", [])

    except Exception as e:
        logger.error(f"Calendar error: {e}")
        updates.update(add_error(state, "calendar", "CALENDAR_ERROR", str(e)))

    return updates


async def general_node(state: AgentState) -> dict:
    """Handle general queries."""
    return track_node(state, "general")


async def human_review_node(state: AgentState) -> dict:
    """Human-in-the-loop review gate."""
    from app.rag.langgraph.nodes.human_review import human_review_node as review
    return await review(state)


async def response_node(state: AgentState) -> dict:
    """Generate final response."""
    from app.rag.langgraph.nodes.generation import generation_node
    return await generation_node(state)


async def error_node(state: AgentState) -> dict:
    """Handle errors gracefully."""
    last_error = state.get("last_error", "An unexpected error occurred")

    return {
        **track_node(state, "error"),
        "response": f"I apologize, but I encountered an issue: {last_error}. Please try again.",
        "should_end": True,
    }


# =============================================================================
# GRAPH BUILDER
# =============================================================================

class AgentGraphBuilder:
    """
    Builder for the LangGraph agent.

    Usage:
        builder = AgentGraphBuilder()
        graph = builder.build()

        # With checkpointing
        graph = builder.with_postgres_checkpointer(db_url).build()
    """

    def __init__(self):
        self._checkpointer = None
        self._interrupt_before: list[str] = []

    def with_postgres_checkpointer(self, connection_string: str) -> "AgentGraphBuilder":
        """Add PostgreSQL checkpointing."""
        try:
            from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
            self._checkpointer = AsyncPostgresSaver.from_conn_string(connection_string)
            logger.info("PostgreSQL checkpointer configured")
        except ImportError:
            logger.warning("AsyncPostgresSaver not available, using memory")
            self._checkpointer = MemorySaver()
        except Exception as e:
            logger.error(f"Checkpointer error: {e}")
            self._checkpointer = MemorySaver()
        return self

    def with_memory_checkpointer(self) -> "AgentGraphBuilder":
        """Add in-memory checkpointing."""
        self._checkpointer = MemorySaver()
        return self

    def with_human_review(self) -> "AgentGraphBuilder":
        """Enable human-in-the-loop."""
        self._interrupt_before.append("human_review")
        return self

    def build(self) -> StateGraph:
        """Build and compile the graph."""
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("entry", entry_node)
        workflow.add_node("router", router_node)
        workflow.add_node("document", document_node)
        workflow.add_node("calendar", calendar_node)
        workflow.add_node("general", general_node)
        workflow.add_node("greeting", greeting_node)
        workflow.add_node("human_review", human_review_node)
        workflow.add_node("response", response_node)
        workflow.add_node("error", error_node)

        # Entry point
        workflow.set_entry_point("entry")

        # Edges
        workflow.add_edge("entry", "router")

        workflow.add_conditional_edges(
            "router",
            route_after_router,
            {
                "document": "document",
                "calendar": "calendar",
                "general": "general",
                "greeting": "greeting",
                "human_review": "human_review",
                "error": "error",
            }
        )

        workflow.add_edge("document", "response")

        workflow.add_conditional_edges(
            "calendar",
            lambda s: "human_review" if should_require_approval(s) else "response",
            {"human_review": "human_review", "response": "response"}
        )

        workflow.add_edge("general", "response")

        workflow.add_conditional_edges(
            "human_review",
            route_after_approval,
            {"response": "response", "end": END}
        )

        workflow.add_edge("greeting", END)
        workflow.add_edge("response", END)
        workflow.add_edge("error", END)

        # Compile
        compile_kwargs = {}
        if self._checkpointer:
            compile_kwargs["checkpointer"] = self._checkpointer
        if self._interrupt_before:
            compile_kwargs["interrupt_before"] = self._interrupt_before

        return workflow.compile(**compile_kwargs)


# =============================================================================
# AGENT CLASS
# =============================================================================

class Agent:
    """
    High-level agent interface.

    Usage:
        agent = Agent()
        result = await agent.invoke("What meetings do I have?", user_id="123")

        async for event in agent.stream("Schedule a meeting"):
            print(event)
    """

    def __init__(
        self,
        database_url: Optional[str] = None,
        enable_human_review: bool = False,
    ):
        builder = AgentGraphBuilder()

        if database_url:
            builder.with_postgres_checkpointer(database_url)
        else:
            builder.with_memory_checkpointer()

        if enable_human_review:
            builder.with_human_review()

        self._graph = builder.build()

    async def invoke(
        self,
        query: str,
        user_id: Optional[str] = None,
        user_name: Optional[str] = None,
        document_ids: Optional[list[str]] = None,
        thread_id: Optional[str] = None,
    ) -> dict:
        """Invoke agent with a query."""
        initial_state = create_initial_state(
            query=query,
            user_id=user_id,
            user_name=user_name,
            document_ids=document_ids,
            thread_id=thread_id,
        )

        config = {"configurable": {"thread_id": initial_state["thread_id"]}}
        result = await self._graph.ainvoke(initial_state, config)

        return {
            "response": result.get("response", ""),
            "thread_id": result.get("thread_id", ""),
            "scheduled_meeting": result.get("scheduled_meeting"),
            "documents": result.get("documents", []),
            "citations": result.get("citations", []),
            "metrics": result.get("metrics", {}),
        }

    async def stream(
        self,
        query: str,
        user_id: Optional[str] = None,
        user_name: Optional[str] = None,
        document_ids: Optional[list[str]] = None,
        thread_id: Optional[str] = None,
    ) -> AsyncIterator[dict]:
        """Stream agent execution."""
        initial_state = create_initial_state(
            query=query,
            user_id=user_id,
            user_name=user_name,
            document_ids=document_ids,
            thread_id=thread_id,
        )

        config = {"configurable": {"thread_id": initial_state["thread_id"]}}

        async for event in self._graph.astream_events(initial_state, config, version="v2"):
            event_type = event.get("event", "")

            if event_type == "on_chain_start":
                yield {"type": "node_start", "node": event.get("name", "")}

            elif event_type == "on_chain_end":
                yield {"type": "node_end", "node": event.get("name", "")}

            elif event_type == "on_llm_stream":
                chunk = event.get("data", {}).get("chunk", "")
                if hasattr(chunk, "content") and chunk.content:
                    yield {"type": "token", "content": chunk.content}

    def get_graph(self) -> StateGraph:
        """Get underlying graph."""
        return self._graph


# =============================================================================
# FACTORY
# =============================================================================

def create_agent(
    database_url: Optional[str] = None,
    enable_human_review: bool = False,
) -> Agent:
    """Create an agent instance."""
    return Agent(database_url=database_url, enable_human_review=enable_human_review)


__all__ = ["AgentGraphBuilder", "Agent", "create_agent"]
