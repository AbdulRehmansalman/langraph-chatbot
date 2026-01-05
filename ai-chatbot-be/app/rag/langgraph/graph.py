"""
LangGraph Agent
===============

Production-ready agent graph with:
- Intelligent query routing (skip retrieval when unnecessary)
- ReAct calendar agent with bound tools for real scheduling
- Tool execution with proper error handling
- Human-in-the-loop approval
- Streaming support
- PostgreSQL checkpointing

Graph Structure:
    entry → router → [document | calendar | direct] → response → END
                   ↘ greeting → END
                   ↘ execute_scheduling → END (user confirmed with "yes")
                   ↘ human_review → response → END

Scheduling Flow:
    1. User: "Schedule CBC test tomorrow at 3pm"
    2. calendar_node: Parses request, stores pending_schedule in state, asks for confirmation
    3. User: "yes"
    4. router: Detects confirmation, routes to execute_scheduling
    5. execute_scheduling: Calls schedule_meeting tool, confirms booking
"""

import logging
import os
import random
import re
from typing import Any, AsyncIterator, Literal, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

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
# ROUTING CONFIGURATION
# =============================================================================

ROUTING_USE_LLM = os.getenv("ROUTING_USE_LLM", "true").lower() == "true"  # Default to LLM routing


# =============================================================================
# ROUTING PATTERNS (Rule-Based - Simplified for fast path only)
# =============================================================================

# Greeting patterns - short greetings that don't need retrieval
GREETING_PATTERNS = [
    r"^(hi|hello|hey|howdy|yo|sup)[\s!.,]*$",
    r"^good\s+(morning|afternoon|evening|day)[\s!.,]*$",
    r"^(greetings|salutations)[\s!.,]*$",
    r"^what'?s?\s+up[\s!?.,]*$",
    r"^how\s+are\s+you[\s!?.,]*$",
]

# Simple/conversational patterns - can answer without retrieval
DIRECT_ANSWER_PATTERNS = [
    r"^(thanks|thank\s+you|thx)[\s!.,]*$",
    r"^(ok|okay|sure|got\s+it|understood)[\s!.,]*$",
    r"^(bye|goodbye|see\s+you|later|ciao)[\s!.,]*$",
    r"^what\s+can\s+you\s+(do|help\s+with)",
    r"^(who|what)\s+are\s+you",
    r"^help\s*$",
    r"^(never\s*mind|forget\s+it|cancel)[\s!.,]*$",
]

# Scheduling confirmation patterns - user saying "yes" to schedule
SCHEDULING_CONFIRMATION_PATTERNS = [
    r"^(yes|yeah|yep|yup|sure|ok|okay)[\s!.,]*$",
    r"^(yes\s*please|sure\s*thing|go\s*ahead)[\s!.,]*$",
    r"^(i\s*want\s*to|let'?s\s*do\s*it|book\s*it)[\s!.,]*$",
    r"^(schedule|book)\s*(it|that|one)?[\s!.,]*$",
]

# Sensitive topics requiring human approval
SENSITIVE_PATTERNS = [
    r"\b(delete|remove|destroy)\s+(all|every|my)",
    r"\b(executive|ceo|cfo|vp|director)\s+(meeting|schedule|calendar)",
    r"\b(salary|compensation|payroll|bonus)",
    r"\b(terminate|fire|layoff|dismiss)",
    r"\b(medical|health\s+record|hipaa)",
    r"\b(confidential|classified|secret)",
    r"\b(financial|audit|tax\s+return)",
]

# Routing metrics (in-memory for now)
_routing_stats = {
    "greeting": 0,
    "calendar": 0,
    "document": 0,
    "direct": 0,
    "general": 0,
    "human_approval": 0,
    "total": 0,
}


def get_routing_stats() -> dict:
    """Get current routing statistics."""
    stats = _routing_stats.copy()
    if stats["total"] > 0:
        stats["skip_retrieval_rate"] = (
            (stats["greeting"] + stats["calendar"] + stats["direct"]) / stats["total"]
        ) * 100
    else:
        stats["skip_retrieval_rate"] = 0.0
    return stats


def _match_patterns(text: str, patterns: list[str]) -> bool:
    """Check if text matches any of the regex patterns."""
    text_lower = text.lower().strip()
    return any(re.search(pattern, text_lower) for pattern in patterns)


def _is_scheduling_confirmation(query: str) -> bool:
    """Check if query is a confirmation to schedule an appointment."""
    return _match_patterns(query, SCHEDULING_CONFIRMATION_PATTERNS)


async def classify_query_rules(query: str, has_documents: bool = False) -> tuple[str, str]:
    """
    Simplified rule-based query classification for fast path only.

    Returns:
        Tuple of (classification, reason)
    """
    query_clean = query.strip()
    query_lower = query_clean.lower()
    word_count = len(query_clean.split())

    # 1. Check for greetings (ONLY short, simple greetings)
    if word_count <= 3 and _match_patterns(query_clean, GREETING_PATTERNS):
        return "greeting", "Short greeting detected"

    # 2. Check for direct answers (thanks, ok, bye - very short)
    if word_count <= 3 and _match_patterns(query_clean, DIRECT_ANSWER_PATTERNS):
        return "direct", "Conversational response (no retrieval)"

    # 4. Check for sensitive topics (need human approval)
    if _match_patterns(query_clean, SENSITIVE_PATTERNS):
        return "human_approval", "Sensitive topic detected"

    # Default to LLM for everything else
    return "llm_needed", "Fallback to LLM classification"


async def _classify_with_llm(query: str) -> str:
    """
    Use LLM for query classification.
    """
    from app.services.llm_factory import llm_factory

    try:
        llm = llm_factory.create_llm(temperature=0, max_tokens=50)

        # Compact prompt
        prompt = f"""Classify this query into exactly one category:
- greeting: Hello, hi, etc.
- calendar: Schedule, meeting, appointment, availability
- document: Search docs, policy, procedure questions
- direct: Thanks, ok, yes/no, help, who are you
- general: Other

Query: "{query[:100]}"
Category:"""

        response = await llm.ainvoke(prompt)
        result = response.content.strip().lower()

        # Validate response
        valid = {"greeting", "calendar", "document", "direct", "general"}
        if result in valid:
            return result
        return "document"  # Default safe

    except Exception as e:
        logger.error(f"LLM classification error: {e}")
        return "document"


# =============================================================================
# SERVICE RETRIEVAL VIA RAG
# =============================================================================

async def retrieve_service_from_documents(
    query: str,
    user_id: str = None,
    top_k: int = 5
) -> dict:
    """
    Retrieve and validate service/test from user's documents using RAG.

    Args:
        query: User's query containing service name
        user_id: User ID for document access
        top_k: Number of results to retrieve

    Returns:
        dict with:
        - found: bool - whether relevant documents were found
        - service_name: str - extracted service name from query
        - context: str - relevant document context
        - documents: list - retrieved documents
        - similar_services: list - similar services if exact match not found
    """
    from app.rag.langgraph.nodes.retrieval import _vector_search

    logger.info(f"SERVICE_RETRIEVAL: Searching for service in query '{query}' for user {user_id}")

    # Extract the service name from query first
    service_name = _extract_title_from_query(query)

    if not user_id:
        logger.info(f"SERVICE_RETRIEVAL: No user_id, using extracted name: {service_name}")
        return {
            "found": False,
            "service_name": service_name,
            "context": "",
            "documents": [],
            "similar_services": [],
        }

    try:
        # Search for the service/query in documents
        results = await _vector_search(
            query=query,
            user_id=user_id,
            limit=top_k
        )

        if not results:
            logger.info(f"SERVICE_RETRIEVAL: No documents found for '{query}'")
            return {
                "found": False,
                "service_name": service_name,
                "context": "",
                "documents": [],
                "similar_services": [],
            }

        # Build context from results
        context_parts = []
        for doc in results[:3]:  # Limit context to top 3
            content = doc.get("content", "")[:500]
            source = doc.get("source", "Unknown")
            context_parts.append(f"[{source}]: {content}")

        context = "\n\n".join(context_parts)

        # Simplified match check
        service_lower = service_name.lower()
        exact_match = any(service_lower in doc.get("content", "").lower() for doc in results)

        return {
            "found": exact_match or len(results) > 0,
            "exact_match": exact_match,
            "service_name": service_name,
            "context": context,
            "documents": results,
            "similar_services": [],
        }

    except Exception as e:
        logger.error(f"SERVICE_RETRIEVAL: Error - {e}", exc_info=True)
        return {
            "found": False,
            "service_name": service_name,
            "context": "",
            "documents": [],
            "similar_services": [],
        }


def _extract_title_from_query(query: str) -> str:
    """
    Extract appointment title from query.
    Simple approach: remove scheduling keywords and time expressions, use remainder as title.
    """
    import re

    text = query.strip()

    # Remove common scheduling phrases (simplified)
    scheduling_phrases = [
        r'\b(schedule|book|set up|create|make|reserve)\s+(a|an|my|the)?\s*',
        r'\b(appointment|meeting|session)\s+(for|at|on)?\s*',
        r'\b(at|around|by)\s+\d{1,2}(:\d{2})?\s*(am|pm|a\.m\.|p\.m\.)?\b',
    ]

    cleaned = text
    for pattern in scheduling_phrases:
        cleaned = re.sub(pattern, ' ', cleaned, flags=re.IGNORECASE)

    # Clean up whitespace
    cleaned = ' '.join(cleaned.split()).strip()

    # If we have something left, use it as title
    if cleaned and len(cleaned) > 2:
        return cleaned.title()

    # Fallback to "Appointment"
    return "Appointment"


# =============================================================================
# ROUTING FUNCTIONS
# =============================================================================

def route_after_router(state: AgentState) -> Literal["document", "calendar", "greeting", "direct", "human_review", "error"]:
    """
    Route based on query classification.

    Routes:
    - greeting: Simple greetings
    - direct: Thanks, bye, ok, help
    - calendar: Scheduling requests
    - document: All other queries (uses RAG retrieval)
    - human_review: Sensitive topics
    """
    if state.get("has_error"):
        return "error"

    classification = state.get("query_classification", "document")

    if classification == "greeting":
        return "greeting"

    if classification == "direct":
        return "direct"

    if classification == "human_approval":
        return "human_review"

    if classification == "calendar":
        return "calendar"

    # Default: document retrieval
    return "document"


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
    """
    Classify query and determine routing using hybrid approach.
    """
    query = state.get("original_query", "")
    has_documents = bool(state.get("document_ids"))

    logger.info(f"ROUTER: Query='{query}'")

    updates = track_node(state, "router")

    # Fast rule-based path for simple cases
    classification, reason = await classify_query_rules(query, has_documents)

    if classification == "llm_needed":
        classification = await _classify_with_llm(query)
        reason = "LLM classification"

    # Remap "general" to "document"
    if classification == "general":
        classification = "document"
        reason = f"Remapped: {reason}"

    updates["query_classification"] = classification
    updates["routing_reason"] = reason

    # Update routing metrics
    _routing_stats["total"] += 1
    _routing_stats[classification] = _routing_stats.get(classification, 0) + 1

    logger.info(f"ROUTER: Final classification = '{classification}' (reason: {reason})")

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


async def direct_node(state: AgentState) -> dict:
    """
    Handle simple conversational responses without document retrieval.
    Used for: thanks, ok, help, who are you, etc.
    """
    query = state.get("original_query", "").lower().strip()
    user_name = state.get("user_name", "there")

    updates = track_node(state, "direct")

    # Match specific patterns and respond directly (simplified)
    if "thanks" in query or "thank you" in query:
        updates["response"] = f"You're welcome, {user_name}! Let me know if you need anything else."
        updates["should_end"] = True
    elif "ok" in query or "sure" in query:
        updates["response"] = "Great! What would you like to do next?"
        updates["should_end"] = False
    elif "bye" in query or "goodbye" in query:
        updates["response"] = f"Goodbye, {user_name}! Have a great day!"
        updates["should_end"] = True
    else:
        updates["response"] = "How can I help you?"
        updates["should_end"] = False

    return updates


async def document_node(state: AgentState) -> dict:
    """
    Handle document search queries.
    """
    import time
    start = time.time()

    logger.info("RETRIEVAL NODE EXECUTING")

    from app.rag.langgraph.nodes.retrieval import document_retrieval_node
    result = await document_retrieval_node(state)

    duration = (time.time() - start) * 1000

    logger.info(f"RETRIEVAL NODE COMPLETED - Duration: {duration:.1f}ms")

    return result


def _get_calendar_tools():
    """Get all calendar tools for the agent."""
    from app.rag.langgraph.tools.appointment_tools import (
        check_calendar,
        schedule_meeting,
        reschedule_meeting,
        cancel_meeting,
        find_available_slots,
        send_invites,
        set_reminder,
    )
    return [
        check_calendar,
        schedule_meeting,
        reschedule_meeting,
        cancel_meeting,
        find_available_slots,
        send_invites,
        set_reminder,
    ]


def _get_calendar_system_prompt(current_date: str, chat_history: str) -> str:
    """Get the calendar agent system prompt with dynamic values."""
    return f"""You are a precise calendar assistant for scheduling lab tests appointments.

CONTEXT:
- User's documents mention services like CBC (Complete Blood Count test, price $25, report 2-4 hours).
- Today's date: {current_date}
- Parse dates relative to today (e.g., 'tomorrow' = next day, 'at 3pm' = 15:00).

CRITICAL RULES:
- ALWAYS use tools to check availability, schedule, or view calendar.
- NEVER assume or hallucinate success - you MUST actually call tools.
- Use 60 minutes default duration for tests like CBC unless specified.
- Title format: Use descriptive titles like "CBC - Complete Blood Count Test".
- If unclear, ask for clarification (e.g., date/time).
- For scheduling: ALWAYS ask for confirmation before calling schedule_meeting.
- Only call schedule_meeting after explicit user confirmation.

STEP-BY-STEP PROCESS:
1. Understand request: Is it scheduling, checking calendar, or finding availability?
2. Parse details: Extract title, datetime (ISO format), duration from the request.
3. Use tools: Call check_calendar first for conflicts.
4. Ask for confirmation if scheduling.
5. On confirmation, call schedule_meeting.
6. On tool error: Report honestly.
7. Respond concisely with the actual tool result.

CHAT HISTORY FOR CONTEXT:
{chat_history}
(Use this to resolve references like 'it' or 'the test' from previous messages)"""


# Cache the agent to avoid recreating it each time
_calendar_agent_cache = None


def get_calendar_agent():
    """Get or create the calendar agent (cached)."""
    global _calendar_agent_cache
    if _calendar_agent_cache is None:
        from app.services.llm_factory import llm_factory
        tools = _get_calendar_tools()
        llm = llm_factory.create_llm(temperature=0.0, max_tokens=2048)
        _calendar_agent_cache = create_react_agent(llm, tools)
    return _calendar_agent_cache


async def calendar_node(state: AgentState) -> dict:
    """
    Handle calendar/appointment queries using ReAct agent.

    The agent now handles confirmation internally.
    """
    query = state.get("original_query", "")
    user_id = state.get("user_id")
    user_timezone = state.get("user_timezone", "UTC")
    user_name = state.get("user_name", "User")

    updates = track_node(state, "calendar")

    try:
        agent = get_calendar_agent()

        from datetime import datetime
        current_date = datetime.now().strftime("%Y-%m-%d")

        state_messages = state.get("messages", [])
        chat_history = ""
        if state_messages:
            recent = state_messages[-5:]
            for msg in recent:
                role = "User" if isinstance(msg, HumanMessage) else "Assistant"
                chat_history += f"{role}: {msg.content[:200]}\n"

        system_prompt = _get_calendar_system_prompt(current_date, chat_history)

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"""User: {user_name}
User ID: {user_id}
Timezone: {user_timezone}

User request: {query}

Use the appropriate calendar tools to handle this request.""")
        ]

        # Invoke the agent
        result = await agent.ainvoke({"messages": messages})

        # Extract the final response
        agent_messages = result.get("messages", [])
        if agent_messages:
            for msg in reversed(agent_messages):
                if isinstance(msg, AIMessage) and msg.content:
                    updates["response"] = msg.content
                    break

        if not updates.get("response"):
            updates["response"] = "I processed your calendar request."

        updates["messages"] = [AIMessage(content=updates["response"])]

    except Exception as e:
        logger.error(f"Calendar agent error: {e}")
        updates.update(add_error(state, "calendar", "CALENDAR_ERROR", str(e)))
        updates["response"] = f"Sorry, error processing calendar request: {str(e)}"
        updates["messages"] = [AIMessage(content=updates["response"])]

    return updates


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
    """

    def __init__(self):
        self._checkpointer = None
        self._interrupt_before: list[str] = []

    def with_postgres_checkpointer(self, connection_string: str) -> "AgentGraphBuilder":
        try:
            from langgraph.checkpoint.postgres import PostgresSaver
            from psycopg_pool import ConnectionPool

            pool = ConnectionPool(connection_string, min_size=1, max_size=10)
            self._checkpointer = PostgresSaver(pool)
            self._checkpointer.setup()
            logger.info("PostgreSQL checkpointer configured")
        except Exception as e:
            logger.error(f"PostgreSQL checkpointer error: {e}, falling back to memory")
            self._checkpointer = MemorySaver()
        return self

    def with_memory_checkpointer(self) -> "AgentGraphBuilder":
        self._checkpointer = MemorySaver()
        return self

    def with_human_review(self) -> "AgentGraphBuilder":
        self._interrupt_before.append("human_review")
        return self

    def build(self) -> StateGraph:
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("entry", entry_node)
        workflow.add_node("router", router_node)
        workflow.add_node("document", document_node)
        workflow.add_node("calendar", calendar_node)
        workflow.add_node("greeting", greeting_node)
        workflow.add_node("direct", direct_node)
        workflow.add_node("human_review", human_review_node)
        workflow.add_node("response", response_node)
        workflow.add_node("error", error_node)

        # Entry point
        workflow.set_entry_point("entry")

        # Edges
        workflow.add_edge("entry", "router")

        # Route based on query classification
        workflow.add_conditional_edges(
            "router",
            route_after_router,
            {
                "document": "document",
                "calendar": "calendar",
                "greeting": "greeting",
                "direct": "direct",
                "human_review": "human_review",
                "error": "error",
            }
        )

        # Document retrieval to response
        workflow.add_edge("document", "response")

        # Calendar to END or human_review
        workflow.add_conditional_edges(
            "calendar",
            lambda s: "human_review" if should_require_approval(s) else END,
            {"human_review": "human_review", END: END}
        )

        # Direct conditional
        workflow.add_conditional_edges(
            "direct",
            lambda s: END if s.get("should_end", False) else "response",
            {END: END, "response": "response"}
        )

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

    # Words that should NEVER be cached (confirmations, short responses)
    UNCACHEABLE_QUERIES = {
        "yes", "yeah", "yep", "yup", "sure", "ok", "okay", "confirm",
        "no", "nope", "nah", "cancel", "nevermind", "never mind",
        "book it", "go ahead", "do it", "schedule it", "yes please",
        "sure thing", "absolutely", "definitely", "let's do it",
        "sounds good", "perfect", "great", "fine", "thanks", "thank you",
        "hi", "hello", "hey", "bye", "goodbye",
    }

    async def invoke(
        self,
        query: str,
        user_id: Optional[str] = None,
        user_name: Optional[str] = None,
        user_timezone: Optional[str] = None,
        document_ids: Optional[list[str]] = None,
        thread_id: Optional[str] = None,
        use_cache: bool = True,
    ) -> dict:
        """Invoke agent with a query, with optional caching."""
        from app.rag.cache import get_cached_response, set_cached_response

        query_lower = query.lower().strip()
        word_count = len(query_lower.split())

        # NEVER cache short queries or confirmation words
        is_uncacheable = (
            word_count <= 3 or
            query_lower in self.UNCACHEABLE_QUERIES or
            any(query_lower.startswith(w) for w in ["yes", "yeah", "sure", "ok", "no", "nope"])
        )

        # Try cache first ONLY for longer document queries
        if use_cache and not document_ids and not is_uncacheable:
            cached = await get_cached_response(query, user_id)
            if cached:
                logger.info(f"Cache HIT: returning cached response for '{query[:30]}...'")
                return {
                    "response": cached.get("response", ""),
                    "thread_id": thread_id or "",
                    "scheduled_meeting": None,
                    "documents": cached.get("documents", []),
                    "citations": [],
                    "metrics": {"cache_hit": True},
                    "from_cache": True,
                }
        elif is_uncacheable:
            logger.info(f"Cache SKIP: query='{query_lower}' is_uncacheable={is_uncacheable}")

        from langchain_core.messages import HumanMessage
        from uuid import uuid4

        effective_thread_id = thread_id or str(uuid4())

        input_state = {
            "messages": [HumanMessage(content=query)],
            "original_query": query,
            "user_id": user_id,
            "user_name": user_name or "User",
            "user_timezone": user_timezone or "UTC",
            "document_ids": document_ids,
            "thread_id": effective_thread_id,
        }

        config = {"configurable": {"thread_id": effective_thread_id}}
        result = await self._graph.ainvoke(input_state, config)

        response_data = {
            "response": result.get("response", ""),
            "thread_id": result.get("thread_id", ""),
            "scheduled_meeting": result.get("scheduled_meeting"),
            "documents": result.get("documents", []),
            "citations": result.get("citations", []),
            "metrics": result.get("metrics", {}),
        }

        # Cache document query responses
        classification = result.get("query_classification", "")
        if use_cache and classification == "document" and response_data["response"]:
            await set_cached_response(
                query=query,
                response=response_data["response"],
                documents=response_data["documents"],
                user_id=user_id,
            )

        return response_data

    async def stream(
        self,
        query: str,
        user_id: Optional[str] = None,
        user_name: Optional[str] = None,
        user_timezone: Optional[str] = None,
        document_ids: Optional[list[str]] = None,
        thread_id: Optional[str] = None,
    ) -> AsyncIterator[dict]:
        """Stream agent execution with real-time LLM token streaming."""
        initial_state = create_initial_state(
            query=query,
            user_id=user_id,
            user_name=user_name,
            document_ids=document_ids,
            thread_id=thread_id,
        )
        initial_state["user_timezone"] = user_timezone or "UTC"

        config = {"configurable": {"thread_id": initial_state["thread_id"]}}

        async for event in self._graph.astream_events(initial_state, config, version="v2"):
            event_type = event.get("event", "")

            if event_type == "on_chain_start":
                yield {"type": "node_start", "node": event.get("name", "")}

            elif event_type == "on_chain_end":
                yield {"type": "node_end", "node": event.get("name", "")}

            elif event_type == "on_llm_stream" or event_type == "on_chat_model_stream":
                chunk = event.get("data", {}).get("chunk", "")
                if hasattr(chunk, "content") and chunk.content:
                    yield {"type": "token", "content": chunk.content}


def create_agent(
    database_url: Optional[str] = None,
    enable_human_review: bool = False,
    use_postgres_checkpointer: bool = False,
) -> Agent:
    if use_postgres_checkpointer and database_url is None:
        from app.core.config import settings
        database_url = settings.database_url

    return Agent(
        database_url=database_url if use_postgres_checkpointer else None,
        enable_human_review=enable_human_review
    )


__all__ = [
    "AgentGraphBuilder",
    "Agent",
    "create_agent",
    "get_routing_stats",
    "classify_query_rules",
]