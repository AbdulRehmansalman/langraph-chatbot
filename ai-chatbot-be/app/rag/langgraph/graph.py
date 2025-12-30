"""
LangGraph Agent
===============

Production-ready agent graph with:
- Intelligent query routing (skip retrieval when unnecessary)
- Tool execution
- Human-in-the-loop approval
- Streaming support
- PostgreSQL checkpointing

Graph Structure:
    entry → router → [document | calendar | general | direct] → response → END
                   ↘ greeting → END
                   ↘ human_review → response → END
"""

import logging
import os
import random
import re
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

# Routing configuration
ROUTING_USE_LLM = os.getenv("ROUTING_USE_LLM", "false").lower() == "true"


# =============================================================================
# ROUTING PATTERNS (Rule-Based)
# =============================================================================

# Greeting patterns - short greetings that don't need retrieval
GREETING_PATTERNS = [
    r"^(hi|hello|hey|howdy|yo|sup)[\s!.,]*$",
    r"^good\s+(morning|afternoon|evening|day)[\s!.,]*$",
    r"^(greetings|salutations)[\s!.,]*$",
    r"^what'?s?\s+up[\s!?.,]*$",
    r"^how\s+are\s+you[\s!?.,]*$",
]

# Calendar patterns - need calendar tools, not document retrieval
CALENDAR_PATTERNS = [
    r"\b(schedule|book|create|set\s+up)\s+(a\s+)?(meeting|appointment|call|session)",
    r"\b(reschedule|cancel|postpone|move)\s+(my\s+|the\s+)?(meeting|appointment)",
    r"\b(what|show|check|view)\s+(is\s+|are\s+)?(on\s+)?(my\s+)?(appointments?|meetings?|calendar|schedule)",
    r"\b(am\s+i\s+)?(free|available|busy)\s+(on|at|tomorrow|today|next)",
    r"\b(find|show|check)\s+(me\s+)?(a\s+)?(free\s+)?(slot|time|availability)",
    r"\bwhen\s+(can|am)\s+(i|we)\s+(meet|schedule|book)",
    r"\b(tomorrow|today|next\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday|week))\s+at\s+\d",
    r"\bremind\s+(me|us)\s+(about|to|for)",
    r"\bmy\s+calendar\b",  # Any mention of "my calendar"
    r"\bmy\s+(meetings?|appointments?|schedule)\b",  # Any mention of personal schedule
]

# Document patterns - need RAG retrieval
DOCUMENT_PATTERNS = [
    r"\b(search|find|look\s+for)\s+(in\s+)?(the\s+)?(document|file|pdf|doc)",
    r"\b(what\s+does|according\s+to|based\s+on)\s+(the\s+)?(document|file|policy|report)",
    r"\b(summarize|analyze|extract)\s+(the\s+)?(document|file|report|page)",
    r"\b(in\s+the|from\s+the)\s+(document|file|pdf|report|policy)",
    r"\bpage\s+\d+",
    r"\b(our|the|company)\s+(policy|procedure|guideline|handbook)",
    r"\b(uploaded|attached)\s+(document|file)",
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


def classify_query_rules(query: str, has_documents: bool = False, awaiting_scheduling: bool = False) -> tuple[str, str]:
    """
    Rule-based query classification.

    IMPORTANT: Default to "document" for most questions to ensure retrieval runs.
    Only skip retrieval for greetings, direct responses, and calendar actions.

    Returns:
        Tuple of (classification, reason)
    """
    query_clean = query.strip()
    query_lower = query_clean.lower()
    word_count = len(query_clean.split())

    # 0. Check for scheduling confirmation (if we're awaiting one)
    if awaiting_scheduling and word_count <= 5:
        if _is_scheduling_confirmation(query_clean):
            return "scheduling_confirmation", "User confirmed scheduling"

    # 1. Check for greetings (ONLY short, simple greetings)
    if word_count <= 3 and _match_patterns(query_clean, GREETING_PATTERNS):
        return "greeting", "Short greeting detected"

    # 2. Check for direct answers (thanks, ok, bye - very short)
    if word_count <= 3 and _match_patterns(query_clean, DIRECT_ANSWER_PATTERNS):
        return "direct", "Conversational response (no retrieval)"

    # 3. Check for calendar-specific queries (scheduling actions)
    if _match_patterns(query_clean, CALENDAR_PATTERNS):
        return "calendar", "Calendar action detected"

    # 4. Check for sensitive topics (need human approval)
    if _match_patterns(query_clean, SENSITIVE_PATTERNS):
        return "human_approval", "Sensitive topic detected"

    # 5. ANY question should go to document retrieval
    # Questions start with: what, how, why, when, where, which, who, is, are, can, do, does, tell, explain
    question_starters = [
        "what", "how", "why", "when", "where", "which", "who",
        "is", "are", "can", "could", "do", "does", "did",
        "tell", "explain", "describe", "show", "find", "get",
        "give", "list", "summarize", "define"
    ]
    if any(query_lower.startswith(w) for w in question_starters):
        return "document", "Question detected - using document retrieval"

    # 6. Questions ending with ? should use document retrieval
    if query_clean.endswith("?"):
        return "document", "Question mark detected - using document retrieval"

    # 7. Explicit document patterns
    if _match_patterns(query_clean, DOCUMENT_PATTERNS):
        return "document", "Document query detected"

    # 8. If query has 4+ words, assume it needs document context
    if word_count >= 4:
        return "document", "Complex query - using document retrieval"

    # 9. Default to document retrieval (safer than hallucinating)
    return "document", "Default to document retrieval"


# =============================================================================
# ROUTING FUNCTIONS
# =============================================================================

def route_after_router(state: AgentState) -> Literal["document", "calendar", "greeting", "direct", "human_review", "scheduling_flow", "error"]:
    """
    Route based on query classification.

    IMPORTANT: "general" classification no longer exists - all substantive queries
    go through "document" retrieval to ensure RAG is always invoked.
    """
    if state.get("has_error"):
        return "error"

    classification = state.get("query_classification", "document")  # Default to document!

    if classification == "greeting":
        return "greeting"

    if classification == "direct":
        return "direct"

    if classification == "scheduling_confirmation":
        return "scheduling_flow"

    if classification == "human_approval":
        return "human_review"

    if classification == "calendar":
        return "calendar"

    # CRITICAL: All other queries (including "general") go through document retrieval
    # This ensures retrieval is ALWAYS invoked for substantive queries
    logger.info(f"Routing to document retrieval (classification={classification})")
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
    Classify query and determine routing.

    IMPORTANT: "general" classification is remapped to "document" to ensure
    retrieval is always invoked for substantive queries.
    """
    query = state.get("original_query", "")
    has_documents = bool(state.get("document_ids"))
    awaiting_scheduling = state.get("awaiting_scheduling_confirmation", False)

    updates = track_node(state, "router")

    # Use rule-based classification (fast, no LLM call)
    classification, reason = classify_query_rules(query, has_documents, awaiting_scheduling)

    # Optional: Use LLM for uncertain cases (when ROUTING_USE_LLM=true)
    if ROUTING_USE_LLM and classification == "general" and len(query.split()) > 3:
        try:
            llm_classification = await _classify_with_llm(query)
            if llm_classification:
                classification = llm_classification
                reason = "LLM classification"
        except Exception as e:
            logger.warning(f"LLM classification failed: {e}, using rule-based")

    # CRITICAL: Remap "general" to "document" to ensure retrieval runs
    if classification == "general":
        logger.info(f"Remapping 'general' to 'document' to ensure retrieval executes")
        classification = "document"
        reason = f"Remapped from general: {reason}"

    updates["query_classification"] = classification
    updates["routing_reason"] = reason

    # Update routing metrics
    _routing_stats["total"] += 1
    _routing_stats[classification] = _routing_stats.get(classification, 0) + 1

    logger.info(f"ROUTER: Query classified as '{classification}' ({reason})")
    logger.info(f"ROUTER: Will route to -> {classification}")
    logger.debug(f"Routing stats: {get_routing_stats()}")

    return updates


async def _classify_with_llm(query: str) -> Optional[str]:
    """
    Use LLM for query classification when rules are uncertain.
    Only called if ROUTING_USE_LLM=true and query is ambiguous.
    """
    from app.services.llm_factory import llm_factory

    try:
        llm = llm_factory.create_llm(temperature=0, max_tokens=50)

        # Compact prompt (under 200 tokens)
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
        return None

    except Exception as e:
        logger.error(f"LLM classification error: {e}")
        return None


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

    # Match specific patterns and respond directly
    if re.match(r"^(thanks|thank\s+you|thx)", query):
        updates["response"] = f"You're welcome, {user_name}! Let me know if you need anything else."
        updates["should_end"] = True

    elif re.match(r"^(ok|okay|sure|got\s+it|understood)", query):
        updates["response"] = "Great! What would you like to do next?"
        updates["should_end"] = False

    elif re.match(r"^(yes|yeah|yep)", query):
        updates["response"] = "Understood. How can I help with that?"
        updates["should_end"] = False

    elif re.match(r"^(no|nope)", query):
        updates["response"] = "No problem. Is there anything else I can help you with?"
        updates["should_end"] = False

    elif re.match(r"^(bye|goodbye|see\s+you|later)", query):
        updates["response"] = f"Goodbye, {user_name}! Have a great day!"
        updates["should_end"] = True

    elif re.match(r"^what\s+can\s+you\s+(do|help)", query):
        updates["response"] = (
            "I can help you with:\n"
            "• **Document search** - Find information in your uploaded documents\n"
            "• **Calendar management** - Schedule, check, or modify meetings\n"
            "• **Policy questions** - Answer questions about company policies\n\n"
            "What would you like to do?"
        )
        updates["should_end"] = False

    elif re.match(r"^(who|what)\s+are\s+you", query):
        updates["response"] = (
            "I'm DocScheduler AI, your document analysis and scheduling assistant. "
            "I can search through documents, schedule meetings, and answer questions about policies and procedures."
        )
        updates["should_end"] = False

    elif re.match(r"^help\s*$", query):
        updates["response"] = (
            "**How to use DocScheduler AI:**\n\n"
            "• Ask questions about your documents\n"
            "• Say 'schedule a meeting' to book appointments\n"
            "• Ask 'what's on my calendar?' to see your schedule\n"
            "• Ask about company policies\n\n"
            "Try: 'What is our remote work policy?' or 'Schedule a meeting tomorrow at 2pm'"
        )
        updates["should_end"] = False

    elif re.match(r"^(never\s*mind|forget\s+it|cancel)", query):
        updates["response"] = "No problem! Let me know if you need anything else."
        updates["should_end"] = False

    else:
        # Fallback - shouldn't happen if patterns are correct
        updates["response"] = "How can I help you?"
        updates["should_end"] = False

    return updates


async def scheduling_flow_node(state: AgentState) -> dict:
    """
    Handle scheduling conversation when user confirms they want to book.

    This node asks for appointment details (date, time, location).
    """
    user_name = state.get("user_name", "there")
    scheduling_context = state.get("scheduling_context", "your appointment")

    updates = track_node(state, "scheduling_flow")

    # Ask for scheduling details
    response = (
        f"Great, {user_name}! I'd be happy to help you schedule an appointment.\n\n"
        "📋 **Please provide the following details:**\n"
        "- **Preferred date** (e.g., tomorrow, next Monday, January 15th)\n"
        "- **Preferred time** (e.g., 10:00 AM, afternoon, 3pm)\n"
        "- **Location preference** (if applicable)\n\n"
        "For example: *'Tomorrow at 2pm at the main office'*"
    )

    updates["response"] = response
    updates["should_end"] = False
    updates["awaiting_scheduling_confirmation"] = False  # Reset - now awaiting details
    updates["messages"] = [AIMessage(content=response)]

    logger.info("Scheduling flow: Asked user for appointment details")

    return updates


async def document_node(state: AgentState) -> dict:
    """
    Handle document search queries.

    CRITICAL: This node MUST execute retrieval. If this log doesn't appear,
    routing is bypassing retrieval.
    """
    import time
    start = time.time()

    logger.info("=" * 50)
    logger.info("RETRIEVAL NODE EXECUTING")
    logger.info(f"  Query: {state.get('original_query', '')[:100]}")
    logger.info(f"  User ID: {state.get('user_id')}")
    logger.info("=" * 50)

    from app.rag.langgraph.nodes.retrieval import document_retrieval_node
    result = await document_retrieval_node(state)

    docs_found = len(result.get("documents", []))
    duration = (time.time() - start) * 1000

    logger.info("=" * 50)
    logger.info("RETRIEVAL NODE COMPLETED")
    logger.info(f"  Documents found: {docs_found}")
    logger.info(f"  Duration: {duration:.1f}ms")
    logger.info(f"  Context length: {len(result.get('context', ''))} chars")
    logger.info("=" * 50)

    return result


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
        # NOTE: "general" node removed - all substantive queries go through "document"
        workflow.add_node("entry", entry_node)
        workflow.add_node("router", router_node)
        workflow.add_node("document", document_node)  # ALWAYS runs retrieval
        workflow.add_node("calendar", calendar_node)
        workflow.add_node("greeting", greeting_node)
        workflow.add_node("direct", direct_node)  # Handles simple responses (thanks, bye, etc.)
        workflow.add_node("scheduling_flow", scheduling_flow_node)  # Handles scheduling confirmation
        workflow.add_node("human_review", human_review_node)
        workflow.add_node("response", response_node)
        workflow.add_node("error", error_node)

        # Entry point
        workflow.set_entry_point("entry")

        # Edges
        workflow.add_edge("entry", "router")

        # CRITICAL: No "general" route - all queries either go to specific handlers
        # or through document retrieval
        workflow.add_conditional_edges(
            "router",
            route_after_router,
            {
                "document": "document",  # ALL substantive queries go here
                "calendar": "calendar",
                "greeting": "greeting",
                "direct": "direct",  # Only for: thanks, bye, ok, help, who are you
                "scheduling_flow": "scheduling_flow",  # User confirmed scheduling
                "human_review": "human_review",
                "error": "error",
            }
        )

        # Document retrieval MUST complete before response
        workflow.add_edge("document", "response")

        workflow.add_conditional_edges(
            "calendar",
            lambda s: "human_review" if should_require_approval(s) else "response",
            {"human_review": "human_review", "response": "response"}
        )

        # Direct node can end or continue based on should_end flag
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
        workflow.add_edge("scheduling_flow", END)  # Scheduling flow ends after asking for details
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
        use_cache: bool = True,
    ) -> dict:
        """Invoke agent with a query, with optional caching."""
        from app.rag.cache import get_cached_response, set_cached_response

        # Try cache first for document queries (skip for calendar/stateful queries)
        if use_cache and not document_ids:  # Only cache general queries
            cached = await get_cached_response(query, user_id)
            if cached:
                logger.info(f"Cache HIT: returning cached response")
                return {
                    "response": cached.get("response", ""),
                    "thread_id": thread_id or "",
                    "scheduled_meeting": None,
                    "documents": cached.get("documents", []),
                    "citations": [],
                    "metrics": {"cache_hit": True},
                    "from_cache": True,
                }

        initial_state = create_initial_state(
            query=query,
            user_id=user_id,
            user_name=user_name,
            document_ids=document_ids,
            thread_id=thread_id,
        )

        config = {"configurable": {"thread_id": initial_state["thread_id"]}}
        result = await self._graph.ainvoke(initial_state, config)

        response_data = {
            "response": result.get("response", ""),
            "thread_id": result.get("thread_id", ""),
            "scheduled_meeting": result.get("scheduled_meeting"),
            "documents": result.get("documents", []),
            "citations": result.get("citations", []),
            "metrics": result.get("metrics", {}),
        }

        # Cache document query responses (skip calendar/greeting/direct)
        classification = result.get("query_classification", "")
        if use_cache and classification == "document" and response_data["response"]:
            await set_cached_response(
                query=query,
                response=response_data["response"],
                documents=response_data["documents"],
                user_id=user_id,
            )
            logger.debug("Response cached for future queries")

        return response_data

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


__all__ = ["AgentGraphBuilder", "Agent", "create_agent", "get_routing_stats", "classify_query_rules"]
