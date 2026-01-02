"""
LangGraph Agent State Schema
============================

Simplified state structure for the LangGraph agent.
Focuses on agent patterns: routing, tool execution, memory, and human-in-the-loop.

Removed LangChain RAG-specific fields (vector_search_results, reranking, etc.)
in favor of a cleaner agent-focused state.
"""

from datetime import datetime
from enum import Enum
from typing import Annotated, Any, Optional, Sequence, TypedDict
from uuid import uuid4

from langchain_core.messages import AnyMessage, BaseMessage, HumanMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


# =============================================================================
# ENUMS
# =============================================================================

class QueryClassification(str, Enum):
    """Query classification for routing."""
    DOCUMENT = "document"
    CALENDAR = "calendar"
    GENERAL = "general"
    GREETING = "greeting"
    HUMAN_APPROVAL = "human_approval"


class HumanReviewStatus(str, Enum):
    """Human review gate status."""
    NOT_REQUIRED = "not_required"
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


class ToolStatus(str, Enum):
    """Tool execution status."""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"


# =============================================================================
# PYDANTIC MODELS
# =============================================================================

class ToolCall(BaseModel):
    """Tool execution record."""
    tool_name: str
    tool_input: dict[str, Any] = Field(default_factory=dict)
    tool_output: Optional[Any] = None
    status: ToolStatus = ToolStatus.PENDING
    error: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    duration_ms: float = 0.0


class CalendarEvent(BaseModel):
    """Calendar event data."""
    id: Optional[str] = None
    title: str = ""
    description: Optional[str] = None
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    duration_minutes: int = 60
    location: Optional[str] = None
    attendees: list[str] = Field(default_factory=list)
    google_meet_link: Optional[str] = None
    calendar_link: Optional[str] = None
    status: str = "pending"


class DocumentContext(BaseModel):
    """Retrieved document context for generation."""
    id: str = ""
    content: str = ""
    source: str = ""
    score: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)


class Citation(BaseModel):
    """Source citation."""
    index: int = 0
    document_id: str = ""
    source: str = ""
    snippet: str = ""


class ErrorEntry(BaseModel):
    """Error log entry."""
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    node: str = ""
    error_type: str = ""
    message: str = ""
    recoverable: bool = True


class ExecutionMetrics(BaseModel):
    """Execution metrics for observability."""
    total_duration_ms: float = 0.0
    llm_calls: int = 0
    tokens_input: int = 0
    tokens_output: int = 0
    estimated_cost_usd: float = 0.0
    tools_executed: list[str] = Field(default_factory=list)
    nodes_executed: list[str] = Field(default_factory=list)


# =============================================================================
# AGENT STATE
# =============================================================================

class AgentState(TypedDict, total=False):
    """
    LangGraph Agent State Schema.

    Clean state focused on agent patterns:
    - Message handling with automatic merging
    - Query routing and classification
    - Tool execution tracking
    - Calendar/appointment management
    - Human-in-the-loop review
    - Error handling and metrics

    Schema Version: 4 (Simplified for LangGraph)
    """

    # === Schema Version ===
    schema_version: int

    # === Messages (Auto-merged by LangGraph) ===
    messages: Annotated[Sequence[AnyMessage], add_messages]

    # === Session Context ===
    thread_id: str
    user_id: Optional[str]
    user_name: str
    session_id: Optional[str]
    timestamp: str

    # === Query & Routing ===
    original_query: str
    query_classification: str  # document, calendar, general, greeting, human_approval

    # === Document Context (Simplified) ===
    document_ids: Optional[list[str]]  # User-specified document scope
    documents: list[dict[str, Any]]  # Retrieved documents (DocumentContext as dicts)
    context: str  # Formatted context for generation

    # === Tool Execution ===
    tools_called: list[str]
    tool_results: list[dict[str, Any]]  # ToolCall as dicts
    current_tool: Optional[str]

    # === Calendar/Appointments ===
    calendar_action: Optional[str]  # check, schedule, reschedule, cancel
    calendar_events: list[dict[str, Any]]  # CalendarEvent as dicts
    scheduled_meeting: Optional[dict[str, Any]]

    # === Scheduling Flow ===
    scheduling_suggested: bool  # True if scheduling was suggested in last response
    awaiting_scheduling_confirmation: bool  # True if waiting for user to say "yes"
    scheduling_context: Optional[str]  # What service/test the user asked about
    pending_schedule: Optional[dict[str, Any]]  # Pending meeting details awaiting confirmation

    # === Response Generation ===
    response: str
    citations: list[dict[str, Any]]  # Citation as dicts

    # === Human-in-the-Loop ===
    requires_approval: bool
    approval_status: str  # not_required, pending, approved, rejected
    approval_request: Optional[dict[str, Any]]
    reviewer_id: Optional[str]

    # === Execution Control ===
    current_node: str
    next_node: Optional[str]
    execution_path: list[str]
    should_end: bool
    iteration_count: int
    max_iterations: int

    # === Error Handling ===
    error_log: list[dict[str, Any]]  # ErrorEntry as dicts
    has_error: bool
    last_error: Optional[str]

    # === Metrics ===
    metrics: dict[str, Any]  # ExecutionMetrics as dict


# =============================================================================
# CONSTANTS
# =============================================================================

CURRENT_SCHEMA_VERSION = 4
DEFAULT_MAX_ITERATIONS = 10
DEFAULT_MAX_QUERY_LENGTH = 10000


# =============================================================================
# STATE FACTORY
# =============================================================================

def create_initial_state(
    query: str,
    user_id: Optional[str] = None,
    user_name: Optional[str] = None,
    session_id: Optional[str] = None,
    document_ids: Optional[list[str]] = None,
    thread_id: Optional[str] = None,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
) -> AgentState:
    """
    Create initial state for a new agent query.

    Args:
        query: User's query
        user_id: User identifier
        user_name: User's display name
        session_id: Session identifier
        document_ids: Document IDs to search within
        thread_id: Thread ID for conversation continuity
        max_iterations: Maximum agent iterations

    Returns:
        Initialized AgentState
    """
    if len(query) > DEFAULT_MAX_QUERY_LENGTH:
        raise ValueError(f"Query exceeds maximum length of {DEFAULT_MAX_QUERY_LENGTH}")

    return AgentState(
        # Schema
        schema_version=CURRENT_SCHEMA_VERSION,

        # Messages
        messages=[HumanMessage(content=query)],

        # Session
        thread_id=thread_id or str(uuid4()),
        user_id=user_id,
        user_name=user_name or "User",
        session_id=session_id,
        timestamp=datetime.utcnow().isoformat(),

        # Query
        original_query=query,
        query_classification="",

        # Documents
        document_ids=document_ids,
        documents=[],
        context="",

        # Tools
        tools_called=[],
        tool_results=[],
        current_tool=None,

        # Calendar
        calendar_action=None,
        calendar_events=[],
        scheduled_meeting=None,

        # Scheduling Flow
        scheduling_suggested=False,
        awaiting_scheduling_confirmation=False,
        scheduling_context=None,
        pending_schedule=None,

        # Response
        response="",
        citations=[],

        # Human-in-the-loop
        requires_approval=False,
        approval_status="not_required",
        approval_request=None,
        reviewer_id=None,

        # Execution
        current_node="",
        next_node=None,
        execution_path=[],
        should_end=False,
        iteration_count=0,
        max_iterations=max_iterations,

        # Errors
        error_log=[],
        has_error=False,
        last_error=None,

        # Metrics
        metrics=ExecutionMetrics().model_dump(),
    )


# =============================================================================
# STATE HELPERS
# =============================================================================

def track_node(state: AgentState, node_name: str) -> dict[str, Any]:
    """Track node execution."""
    path = list(state.get("execution_path", []))
    path.append(node_name)
    return {
        "execution_path": path,
        "current_node": node_name,
        "iteration_count": state.get("iteration_count", 0) + 1,
    }


def add_error(
    state: AgentState,
    node: str,
    error_type: str,
    message: str,
    recoverable: bool = True,
) -> dict[str, Any]:
    """Add an error to the state."""
    error_entry = ErrorEntry(
        node=node,
        error_type=error_type,
        message=message,
        recoverable=recoverable,
    ).model_dump()

    error_log = list(state.get("error_log", []))
    error_log.append(error_entry)

    return {
        "error_log": error_log,
        "has_error": not recoverable,
        "last_error": message,
    }


def add_documents(
    state: AgentState,
    documents: list[dict[str, Any]],
) -> dict[str, Any]:
    """Add retrieved documents to state."""
    # Deduplicate by ID
    existing_ids = {d.get("id") for d in state.get("documents", [])}
    new_docs = [d for d in documents if d.get("id") not in existing_ids]

    all_docs = list(state.get("documents", [])) + new_docs

    # Build context string
    context_parts = []
    for i, doc in enumerate(all_docs[:10], 1):  # Limit to top 10
        source = doc.get("source", "Unknown")
        content = doc.get("content", "")[:2000]  # Truncate long content
        context_parts.append(f"[{i}] Source: {source}\n{content}")

    context = "\n\n---\n\n".join(context_parts)

    return {
        "documents": all_docs,
        "context": context,
    }


def check_iteration_limit(state: AgentState) -> bool:
    """Check if iteration limit has been reached."""
    count = state.get("iteration_count", 0)
    max_iter = state.get("max_iterations", DEFAULT_MAX_ITERATIONS)
    return count >= max_iter


def update_metrics(
    state: AgentState,
    llm_calls: int = 0,
    tokens_in: int = 0,
    tokens_out: int = 0,
    duration_ms: float = 0.0,
) -> dict[str, Any]:
    """Update execution metrics."""
    metrics = state.get("metrics", {}).copy()

    metrics["llm_calls"] = metrics.get("llm_calls", 0) + llm_calls
    metrics["tokens_input"] = metrics.get("tokens_input", 0) + tokens_in
    metrics["tokens_output"] = metrics.get("tokens_output", 0) + tokens_out
    metrics["total_duration_ms"] = metrics.get("total_duration_ms", 0.0) + duration_ms

    # Estimate cost (Claude pricing)
    cost_in = tokens_in / 1000 * 0.003  # $3/1M input tokens
    cost_out = tokens_out / 1000 * 0.015  # $15/1M output tokens
    metrics["estimated_cost_usd"] = metrics.get("estimated_cost_usd", 0.0) + cost_in + cost_out

    metrics["nodes_executed"] = list(state.get("execution_path", []))
    metrics["tools_executed"] = list(state.get("tools_called", []))

    return {"metrics": metrics}


def get_response_context(state: AgentState) -> str:
    """Get formatted context for response generation."""
    context = state.get("context", "")

    # Add tool results if any
    tool_results = state.get("tool_results", [])
    if tool_results:
        tool_context = "\n\nTool Results:\n"
        for result in tool_results[-3:]:  # Last 3 tool results
            tool_context += f"- {result.get('tool_name')}: {str(result.get('tool_output', ''))[:500]}\n"
        context += tool_context

    # Add calendar info if any
    meeting = state.get("scheduled_meeting")
    if meeting:
        context += f"\n\nScheduled Meeting:\n- Title: {meeting.get('title')}\n- Time: {meeting.get('start_time')}\n"
        if meeting.get("google_meet_link"):
            context += f"- Meet Link: {meeting.get('google_meet_link')}\n"

    return context


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Enums
    "QueryClassification",
    "HumanReviewStatus",
    "ToolStatus",
    # Models
    "ToolCall",
    "CalendarEvent",
    "DocumentContext",
    "Citation",
    "ErrorEntry",
    "ExecutionMetrics",
    # State
    "AgentState",
    # Constants
    "CURRENT_SCHEMA_VERSION",
    "DEFAULT_MAX_ITERATIONS",
    # Factory
    "create_initial_state",
    # Helpers
    "track_node",
    "add_error",
    "add_documents",
    "check_iteration_limit",
    "update_metrics",
    "get_response_context",
]
