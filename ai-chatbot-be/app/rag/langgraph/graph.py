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
# PENDING SCHEDULE CACHE (User + Thread based fallback)
# =============================================================================
# Used when state checkpointing isn't preserving pending_schedule across turns
# Key format: "user_id:thread_id" or just "user_id" if no thread_id
_pending_schedule_cache: dict[str, dict] = {}


def _get_cache_key(user_id: str, thread_id: str = None) -> str:
    """Generate cache key from user_id and optional thread_id."""
    if thread_id:
        return f"{user_id}:{thread_id}"
    return user_id


def cache_pending_schedule(user_id: str, pending: dict, thread_id: str = None) -> None:
    """Cache pending schedule for a user (fallback for state persistence)."""
    if user_id and pending:
        # Store with thread_id if available
        cache_key = _get_cache_key(user_id, thread_id)
        _pending_schedule_cache[cache_key] = pending

        # Also store with just user_id as fallback (for cases where thread_id changes)
        if thread_id:
            _pending_schedule_cache[user_id] = pending

        logger.info(f"CACHE: ✅ Stored pending schedule for key '{cache_key}': {pending.get('title')} at {pending.get('datetime')}")
        logger.info(f"CACHE: Current cache keys = {list(_pending_schedule_cache.keys())}")
    else:
        logger.warning(f"CACHE: ⚠️ Cannot cache - user_id={user_id}, pending={pending is not None}")


def get_cached_pending_schedule(user_id: str, thread_id: str = None) -> dict | None:
    """Get cached pending schedule for a user."""
    if not user_id:
        logger.warning(f"CACHE: ⚠️ Cannot retrieve - user_id is None/empty")
        return None

    # Try with thread_id first
    if thread_id:
        cache_key = _get_cache_key(user_id, thread_id)
        result = _pending_schedule_cache.get(cache_key)
        if result:
            logger.info(f"CACHE: ✅ Found cached schedule for key '{cache_key}': {result.get('title')}")
            return result

    # Fall back to user_id only
    result = _pending_schedule_cache.get(user_id)
    if result:
        logger.info(f"CACHE: ✅ Found cached schedule for user '{user_id}': {result.get('title')}")
    else:
        logger.info(f"CACHE: ❌ No cached schedule for user '{user_id}' (cache keys: {list(_pending_schedule_cache.keys())})")
    return result


def clear_cached_pending_schedule(user_id: str, thread_id: str = None) -> None:
    """Clear cached pending schedule for a user."""
    keys_to_remove = []

    # Remove thread-specific key
    if thread_id:
        cache_key = _get_cache_key(user_id, thread_id)
        if cache_key in _pending_schedule_cache:
            keys_to_remove.append(cache_key)

    # Also remove user-only key
    if user_id in _pending_schedule_cache:
        keys_to_remove.append(user_id)

    for key in keys_to_remove:
        del _pending_schedule_cache[key]
        logger.info(f"Cleared pending schedule for key {key}")




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
    # Direct scheduling commands
    r"\b(schedule|book|create|set\s+up)\s+(a\s+)?(meeting|appointment|call|session|test|checkup|consultation)",
    r"\b(i\s+want|i\s+need|i'd\s+like|can\s+you|please|could\s+you)\s+(to\s+)?(schedule|book|set\s+up)",
    r"\b(schedule|book)\s+(me|my|a|an)\b",
    r"\b(book|schedule)\s+(it|this|that)\b",
    # Reschedule/cancel
    r"\b(reschedule|cancel|postpone|move)\s+(my\s+|the\s+)?(meeting|appointment|test)",
    # Check calendar
    r"\b(what|show|check|view)\s+(is\s+|are\s+)?(on\s+)?(my\s+)?(appointments?|meetings?|calendar|schedule)",
    r"\b(am\s+i\s+)?(free|available|busy)\s+(on|at|tomorrow|today|next)",
    r"\b(find|show|check)\s+(me\s+)?(a\s+)?(free\s+)?(slot|time|availability)",
    r"\bwhen\s+(can|am)\s+(i|we)\s+(meet|schedule|book)",
    # Date/time patterns with scheduling intent
    r"\b(tomorrow|today|next\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday|week))\s+at\s+\d",
    r"\bremind\s+(me|us)\s+(about|to|for)",
    r"\bmy\s+calendar\b",
    r"\bmy\s+(meetings?|appointments?|schedule)\b",
]

# Scheduling keywords that indicate intent to book an appointment
SCHEDULING_KEYWORDS = [
    "schedule", "book", "appointment", "reserve", "slot",
]


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

        # Check if service name appears in any result (exact/partial match)
        service_lower = service_name.lower()
        exact_match = False
        similar_services = []

        for doc in results:
            content = doc.get("content", "").lower()
            score = doc.get("score", 0)

            # Check for exact match
            if service_lower in content:
                exact_match = True

            # Extract potential service names from content for suggestions
            # Look for capitalized phrases or common patterns
            words = doc.get("content", "").split()
            for i, word in enumerate(words):
                # Find capitalized words that might be service names
                if word and len(word) > 3 and word[0].isupper():
                    # Check if it's part of a phrase (2-3 words)
                    phrase = word
                    # Safely check next word
                    if i + 1 < len(words):
                        next_word = words[i + 1]
                        if next_word and len(next_word) > 0 and next_word[0].isupper():
                            phrase = f"{word} {next_word}"
                    if phrase.lower() != service_lower and phrase not in similar_services:
                        similar_services.append(phrase)

        # Limit to top 3 similar services
        similar_services = similar_services[:3]

        # Build context from results
        context_parts = []
        for doc in results[:3]:  # Limit context to top 3
            content = doc.get("content", "")[:500]
            source = doc.get("source", "Unknown")
            context_parts.append(f"[{source}]: {content}")

        context = "\n\n".join(context_parts)

        logger.info(f"SERVICE_RETRIEVAL: Found {len(results)} documents, exact_match={exact_match}, similar={similar_services}")

        return {
            "found": exact_match or len(results) > 0,
            "exact_match": exact_match,
            "service_name": service_name,
            "context": context,
            "documents": results,
            "similar_services": similar_services if not exact_match else [],
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

# Time expressions that indicate scheduling intent
TIME_EXPRESSIONS = [
    r"\b(tomorrow|today|tonight)\b",
    r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
    r"\b(next|this)\s+(week|month|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
    r"\b(at|around|by)\s+\d{1,2}(:\d{2})?\s*(am|pm|a\.m\.|p\.m\.)?\b",
    r"\b\d{1,2}(:\d{2})?\s*(am|pm|a\.m\.|p\.m\.)\b",
    r"\b(morning|afternoon|evening|night)\b",
    r"\b(in\s+)?\d+\s+(hour|minute|day|week)s?\b",
]


def _has_scheduling_keyword(text: str) -> bool:
    """Check if text contains a scheduling keyword."""
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in SCHEDULING_KEYWORDS)


def _has_time_expression(text: str) -> bool:
    """Check if text contains a time expression."""
    text_lower = text.lower()
    return any(re.search(pattern, text_lower) for pattern in TIME_EXPRESSIONS)


def _is_scheduling_intent(text: str) -> bool:
    """
    Detect scheduling intent from text.
    Returns True if:
    1. Text matches calendar patterns, OR
    2. Text contains scheduling keywords + time expression
    """
    # Check calendar patterns
    if _match_patterns(text, CALENDAR_PATTERNS):
        return True

    # Scheduling keywords + time expression = scheduling intent
    if _has_scheduling_keyword(text) and _has_time_expression(text):
        return True

    # Just scheduling keywords alone
    if _has_scheduling_keyword(text):
        return True

    return False

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

    # 3. Check for calendar-specific queries using enhanced scheduling intent detection
    # This catches: "schedule X", "book Y", "CBC tomorrow at 3pm", "I want to book", etc.
    if _is_scheduling_intent(query_clean):
        return "calendar", "Scheduling intent detected"

    # 4. Check for sensitive topics (need human approval)
    if _match_patterns(query_clean, SENSITIVE_PATTERNS):
        return "human_approval", "Sensitive topic detected"

    # 5. ANY question should go to document retrieval
    question_starters = [
        "what", "how", "why", "where", "which", "who",
        "is", "are", "could", "do", "does", "did",
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

def route_after_router(state: AgentState) -> Literal["document", "calendar", "greeting", "direct", "human_review", "execute_scheduling", "error"]:
    """
    Route based on query classification.

    Routes:
    - greeting: Simple greetings
    - direct: Thanks, bye, ok, help
    - calendar: Scheduling requests (creates pending_schedule)
    - execute_scheduling: User confirmed with "yes" (executes booking)
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

    if classification == "scheduling_confirmation":
        return "execute_scheduling"

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
    Classify query and determine routing.

    Key routing logic:
    1. Check for scheduling confirmation ("yes" when pending schedule exists)
    2. Classify query using enhanced rule-based patterns
    3. Route to appropriate node
    """
    query = state.get("original_query", "")
    query_lower = query.lower().strip()
    has_documents = bool(state.get("document_ids"))
    user_id = state.get("user_id")
    thread_id = state.get("thread_id")

    logger.info(f"ROUTER: ============ ROUTING START ============")
    logger.info(f"ROUTER: Query='{query}' | user_id={user_id} | thread_id={thread_id}")
    logger.info(f"ROUTER: Intent signals - has_scheduling={_has_scheduling_keyword(query)}, has_time={_has_time_expression(query)}")

    updates = track_node(state, "router")

    # Check for pending schedule (state or cache fallback with thread_id)
    state_pending = state.get("pending_schedule")
    cache_pending = get_cached_pending_schedule(user_id, thread_id) if user_id else None
    pending = state_pending or cache_pending

    logger.info(f"ROUTER: State pending={state_pending is not None}, Cache pending={cache_pending is not None}")
    if pending:
        logger.info(f"ROUTER: Pending schedule found: {pending.get('title')} at {pending.get('datetime')}")

    # If user is confirming a pending schedule
    if pending:
        # Extended confirmation patterns - handle various ways users say "yes"
        confirmation_words = [
            "yes", "yeah", "yep", "yup", "sure", "ok", "okay", "confirm",
            "book it", "go ahead", "do it", "schedule it", "yes please",
            "sure thing", "absolutely", "definitely", "let's do it",
            "sounds good", "perfect", "great", "fine"
        ]
        # Check if query starts with or equals a confirmation word
        is_confirmation = (
            query_lower in confirmation_words or
            query_lower.startswith("yes") or
            query_lower.startswith("yeah") or
            query_lower.startswith("sure") or
            query_lower.startswith("ok") or
            any(query_lower.startswith(w + " ") for w in ["yes", "yeah", "yep", "sure", "ok", "okay"])
        )
        logger.info(f"ROUTER: Confirmation check - query_lower='{query_lower}', is_confirmation={is_confirmation}")
        if is_confirmation:
            logger.info(f"ROUTER: ✅ CONFIRMATION DETECTED - routing to execute_scheduling")
            logger.info(f"ROUTER: Pending schedule = {pending.get('title')} at {pending.get('datetime')}")
            updates["query_classification"] = "scheduling_confirmation"
            updates["routing_reason"] = "User confirmed pending schedule"
            updates["pending_schedule"] = pending  # Ensure pending is in state
            _routing_stats["total"] += 1
            _routing_stats["scheduling_confirmation"] = _routing_stats.get("scheduling_confirmation", 0) + 1
            logger.info(f"ROUTER: ============ ROUTING END (execute_scheduling) ============")
            return updates
        else:
            logger.info(f"ROUTER: Pending exists but query is not a confirmation")

    # Use rule-based classification with enhanced patterns
    awaiting_scheduling = bool(pending)
    classification, reason = classify_query_rules(query, has_documents, awaiting_scheduling)

    # Remap "general" to "document" to ensure retrieval runs
    if classification == "general":
        classification = "document"
        reason = f"Remapped: {reason}"

    updates["query_classification"] = classification
    updates["routing_reason"] = reason

    # Update routing metrics
    _routing_stats["total"] += 1
    _routing_stats[classification] = _routing_stats.get(classification, 0) + 1

    logger.info(f"ROUTER: Final classification = '{classification}' (reason: {reason})")
    logger.info(f"ROUTER: ============ ROUTING END ({classification}) ============")

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


async def execute_scheduling_node(state: AgentState) -> dict:
    """
    Execute the actual meeting booking when user confirms with 'yes'.

    Flow:
    1. Get pending_schedule from state (or cache fallback)
    2. Call schedule_meeting tool to create the meeting
    3. Return success/error response
    4. Clear pending schedule
    """
    from app.rag.langgraph.tools.appointment_tools import (
        schedule_meeting,
        format_meeting_success,
    )
    from dateutil import parser as date_parser

    updates = track_node(state, "execute_scheduling")
    user_id = state.get("user_id")
    thread_id = state.get("thread_id")

    logger.info(f"EXECUTE_SCHEDULING: Starting for user {user_id}, thread {thread_id}")

    # Get pending schedule from state or cache (with thread_id)
    pending = state.get("pending_schedule") or get_cached_pending_schedule(user_id, thread_id)

    if not pending:
        logger.warning("EXECUTE_SCHEDULING: No pending schedule found!")
        updates["response"] = "I don't have a pending appointment to book. What would you like to schedule?"
        updates["should_end"] = False
        updates["messages"] = [AIMessage(content=updates["response"])]
        return updates

    logger.info(f"EXECUTE_SCHEDULING: Booking - {pending.get('title')} at {pending.get('datetime')}")

    # Call the schedule_meeting tool
    try:
        result = await schedule_meeting.ainvoke({
            "title": pending["title"],
            "datetime_str": pending["datetime"],
            "duration_minutes": pending.get("duration", 60),
            "participants": pending.get("attendees"),
            "user_id": user_id or pending.get("user_id"),
        })

        logger.info(f"EXECUTE_SCHEDULING: Result = {result}")

        if result.get("success"):
            updates["scheduled_meeting"] = result
            updates["response"] = format_meeting_success(result)

            # Log complete calendar event details
            logger.info("=" * 60)
            logger.info("📅 CALENDAR EVENT CREATED SUCCESSFULLY")
            logger.info("=" * 60)
            logger.info(f"  Meeting ID:    {result.get('meeting_id')}")
            logger.info(f"  Title:         {result.get('title')}")
            logger.info(f"  Start Time:    {result.get('start_time')}")
            logger.info(f"  End Time:      {result.get('end_time')}")
            logger.info(f"  Duration:      {result.get('duration_minutes')} minutes")
            logger.info(f"  Participants:  {result.get('participants', [])}")
            logger.info(f"  Location:      {result.get('location', 'Not specified')}")
            logger.info(f"  Status:        {result.get('status')}")
            logger.info(f"  Storage:       {result.get('source', 'unknown')}")
            if result.get('calendar_link'):
                logger.info(f"  Calendar Link: {result.get('calendar_link')}")
            if result.get('google_meet_link'):
                logger.info(f"  Meet Link:     {result.get('google_meet_link')}")
            if result.get('note'):
                logger.info(f"  Note:          {result.get('note')}")
            logger.info("=" * 60)
        else:
            error_msg = result.get('message') or result.get('error') or 'Unknown error'
            updates["response"] = f"Sorry, I couldn't schedule the appointment: {error_msg}"

            # Log failed calendar event
            logger.warning("=" * 60)
            logger.warning("❌ CALENDAR EVENT CREATION FAILED")
            logger.warning("=" * 60)
            logger.warning(f"  Error: {error_msg}")
            logger.warning(f"  Pending Title: {pending.get('title')}")
            logger.warning(f"  Pending Time: {pending.get('datetime')}")
            logger.warning("=" * 60)

            # Show alternative slots if available
            if result.get("alternative_slots"):
                updates["response"] += "\n\n**Alternative times available:**\n"
                for slot in result["alternative_slots"][:3]:
                    try:
                        slot_dt = date_parser.parse(slot['start'])
                        updates["response"] += f"- {slot_dt.strftime('%A, %B %d at %I:%M %p')}\n"
                    except Exception:
                        updates["response"] += f"- {slot['start']}\n"

    except Exception as e:
        logger.error(f"EXECUTE_SCHEDULING: Error - {e}", exc_info=True)
        updates["response"] = f"I encountered an error while scheduling: {str(e)}. Please try again."

    # Clear pending schedule from state and cache
    updates["pending_schedule"] = None
    updates["awaiting_scheduling_confirmation"] = False
    updates["should_end"] = True
    updates["messages"] = [AIMessage(content=updates["response"])]

    if user_id:
        clear_cached_pending_schedule(user_id, thread_id)

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


def _get_calendar_agent():
    """Create a ReAct agent for calendar operations with bound tools."""
    from app.services.llm_factory import llm_factory

    # Get calendar tools
    tools = _get_calendar_tools()

    # Create LLM with zero temperature for reliability (important for Ollama/llama3.1)
    llm = llm_factory.create_llm(temperature=0.0, max_tokens=2048)

    # Create ReAct agent with tools
    # Note: Dynamic context (date, history) is injected in calendar_node before invoking
    agent = create_react_agent(llm, tools)

    return agent


def _get_calendar_system_prompt(current_date: str, chat_history: str) -> str:
    """Get the calendar agent system prompt with dynamic values."""
    return f"""You are a precise calendar assistant for scheduling lab tests and appointments.

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

STEP-BY-STEP PROCESS:
1. Understand request: Is it scheduling, checking calendar, or finding availability?
2. Parse details: Extract title, datetime (ISO format), duration from the request.
3. Use tools: Call check_calendar first for conflicts, then schedule_meeting.
4. On tool error (success=False): Report honestly (e.g., "Couldn't save - database issue").
5. Respond concisely with the actual tool result.

CHAT HISTORY FOR CONTEXT:
{chat_history}
(Use this to resolve references like 'it' or 'the test' from previous messages)"""


def _extract_title_from_query(query: str) -> str:
    """
    Extract appointment title from query.
    Simple approach: remove scheduling keywords and time expressions, use remainder as title.
    """
    import re

    text = query.strip()

    # Remove common scheduling phrases
    scheduling_phrases = [
        r'\b(schedule|book|set up|create|make|reserve)\s+(a|an|my|the)?\s*',
        r'\b(appointment|meeting|session)\s+(for|at|on)?\s*',
        r'\b(i\s+want|i\s+need|i\'d\s+like|can\s+you|please)\s+(to\s+)?',
        r'\b(tomorrow|today|tonight)\s+(at\s+)?',
        r'\b(next\s+)?(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\s+(at\s+)?',
        r'\b(at|around|by)\s+\d{1,2}(:\d{2})?\s*(am|pm|a\.m\.|p\.m\.)?\b',
        r'\b\d{1,2}(:\d{2})?\s*(am|pm|a\.m\.|p\.m\.)\b',
        r'\b(morning|afternoon|evening)\b',
        r'\b(for\s+)?\d+\s+(hour|minute|min)s?\b',
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


# Cache the agent to avoid recreating it each time
_calendar_agent_cache = None


def get_calendar_agent():
    """Get or create the calendar agent (cached)."""
    global _calendar_agent_cache
    if _calendar_agent_cache is None:
        _calendar_agent_cache = _get_calendar_agent()
    return _calendar_agent_cache


async def calendar_node(state: AgentState) -> dict:
    """
    Handle calendar/appointment queries.

    Flow:
    1. Retrieve service info from documents using RAG
    2. Parse the user's scheduling request
    3. Ask for confirmation
    4. On confirmation, execute_scheduling_node handles actual booking
    """
    from app.rag.langgraph.tools.appointment_tools import (
        parse_natural_datetime_enhanced,
        extract_duration,
        extract_attendees,
        format_confirmation_request,
    )

    query = state.get("original_query", "")
    user_id = state.get("user_id")
    timezone = state.get("timezone", "UTC")
    user_name = state.get("user_name", "User")

    updates = track_node(state, "calendar")

    try:
        query_lower = query.lower()

        # Check if this is a scheduling request vs calendar check
        is_availability_check = any(word in query_lower for word in ["available", "free", "check", "what's on", "show my", "view"])

        if not is_availability_check:
            # First, retrieve service info from documents
            service_info = await retrieve_service_from_documents(query, user_id)
            title = service_info.get("service_name", "Appointment")
            doc_context = service_info.get("context", "")
            similar_services = service_info.get("similar_services", [])
            exact_match = service_info.get("exact_match", False)

            logger.info(f"CALENDAR: Retrieved service '{title}' from documents, found={service_info.get('found')}, exact_match={exact_match}")

            # If no exact match found and we have similar services, suggest them
            if not exact_match and similar_services and not service_info.get("found"):
                suggestions = "\n".join([f"  - {svc}" for svc in similar_services])
                updates["response"] = (
                    f"I couldn't find '{title}' in your documents. Did you mean one of these?\n\n"
                    f"{suggestions}\n\n"
                    f"Please try again with the correct service name and time."
                )
                updates["messages"] = [AIMessage(content=updates["response"])]
                return updates

            # Parse datetime from user query
            parsed = parse_natural_datetime_enhanced(query, timezone)

            if not parsed["datetime"]:
                updates["response"] = (
                    "I couldn't understand the date/time. Please specify like:\n"
                    "- 'tomorrow at 2pm'\n"
                    "- 'next Monday at 10:30am'\n"
                    "- 'in 2 hours'"
                )
                updates["messages"] = [AIMessage(content=updates["response"])]
                return updates

            duration = extract_duration(query)
            attendees = extract_attendees(query)

            # Store pending schedule for confirmation (include document context)
            pending_schedule = {
                "datetime": parsed["datetime"].isoformat(),
                "duration": duration,
                "attendees": attendees,
                "title": title,
                "user_id": user_id,
                "document_context": doc_context[:500] if doc_context else "",
            }
            updates["pending_schedule"] = pending_schedule
            updates["awaiting_scheduling_confirmation"] = True
            updates["response"] = format_confirmation_request(
                parsed["datetime"], duration, title, attendees
            )
            updates["calendar_action"] = "pending_schedule"
            updates["messages"] = [AIMessage(content=updates["response"])]

            # Also store retrieved documents for reference
            if service_info.get("documents"):
                updates["documents"] = service_info["documents"]

            # Cache for state persistence
            thread_id = state.get("thread_id")
            cache_pending_schedule(user_id, pending_schedule, thread_id)

            return updates

        # For availability checks, use the ReAct agent with tools
        agent = get_calendar_agent()

        from datetime import datetime
        current_date = datetime.now().strftime("%Y-%m-%d")

        state_messages = state.get("messages", [])
        chat_history = ""
        if state_messages:
            recent = state_messages[-5:] if len(state_messages) > 5 else state_messages
            for msg in recent:
                role = "User" if hasattr(msg, "type") and msg.type == "human" else "Assistant"
                content = msg.content if hasattr(msg, "content") else str(msg)
                chat_history += f"{role}: {content[:200]}\n"
        if not chat_history:
            chat_history = "(No previous messages)"

        system_prompt = _get_calendar_system_prompt(current_date, chat_history)

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"""User: {user_name}
User ID: {user_id}
Timezone: {timezone}

User request: {query}

Use the appropriate calendar tools to handle this request.""")
        ]

        # Invoke the agent
        result = await agent.ainvoke({"messages": messages})

        # Extract the final response from agent output
        agent_messages = result.get("messages", [])
        if agent_messages:
            # Get the last AI message as the response
            for msg in reversed(agent_messages):
                if isinstance(msg, AIMessage) and msg.content:
                    updates["response"] = msg.content
                    break

        if not updates.get("response"):
            updates["response"] = "I processed your calendar request but couldn't generate a response."

        updates["calendar_action"] = "agent_handled"
        updates["messages"] = [AIMessage(content=updates["response"])]

    except Exception as e:
        logger.error(f"Calendar agent error: {e}")
        updates.update(add_error(state, "calendar", "CALENDAR_ERROR", str(e)))
        updates["response"] = f"Sorry, I encountered an error processing your calendar request: {str(e)}"
        updates["messages"] = [AIMessage(content=updates["response"])]

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
        """
        Add PostgreSQL checkpointing using connection pool.

        Note: PostgresSaver requires proper connection management.
        Falls back to MemorySaver if PostgreSQL setup fails.
        """
        try:
            from langgraph.checkpoint.postgres import PostgresSaver
            from psycopg_pool import ConnectionPool

            # Create connection pool for PostgresSaver
            pool = ConnectionPool(connection_string, min_size=1, max_size=10)
            self._checkpointer = PostgresSaver(pool)

            # Setup the checkpoint tables if they don't exist
            try:
                self._checkpointer.setup()
            except Exception as setup_err:
                logger.warning(f"Checkpoint table setup warning (may already exist): {setup_err}")

            logger.info("PostgreSQL checkpointer configured with connection pool")
        except ImportError as ie:
            logger.warning(f"PostgreSQL dependencies not available ({ie}), using memory checkpointer. "
                          "Install with: pip install langgraph-checkpoint-postgres psycopg-pool")
            self._checkpointer = MemorySaver()
        except Exception as e:
            logger.error(f"PostgreSQL checkpointer error: {e}, falling back to memory")
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
        workflow.add_node("greeting", greeting_node)
        workflow.add_node("direct", direct_node)
        workflow.add_node("execute_scheduling", execute_scheduling_node)  # Executes booking on "yes"
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
                "execute_scheduling": "execute_scheduling",  # User said "yes" to confirm
                "human_review": "human_review",
                "error": "error",
            }
        )

        # Document retrieval MUST complete before response
        workflow.add_edge("document", "response")

        # Calendar node handles its own response - route to END directly
        # Only go to human_review for cancel/reschedule operations
        workflow.add_conditional_edges(
            "calendar",
            lambda s: "human_review" if should_require_approval(s) else END,
            {"human_review": "human_review", END: END}
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
        workflow.add_edge("execute_scheduling", END)  # Booking confirmed, ends conversation
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
        document_ids: Optional[list[str]] = None,
        thread_id: Optional[str] = None,
        use_cache: bool = True,
    ) -> dict:
        """Invoke agent with a query, with optional caching."""
        from app.rag.cache import get_cached_response, set_cached_response

        query_lower = query.lower().strip()
        word_count = len(query_lower.split())

        # NEVER cache short queries or confirmation words - they are context-dependent
        is_uncacheable = (
            word_count <= 3 or  # Short queries are context-dependent
            query_lower in self.UNCACHEABLE_QUERIES or
            any(query_lower.startswith(w) for w in ["yes", "yeah", "sure", "ok", "no", "nope"])
        )

        # Check for pending schedule - if exists, skip cache entirely
        has_pending = bool(get_cached_pending_schedule(user_id, thread_id)) if user_id else False

        # Try cache first ONLY for longer document queries without pending state
        if use_cache and not document_ids and not is_uncacheable and not has_pending:
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
        elif is_uncacheable or has_pending:
            logger.info(f"Cache SKIP: query='{query_lower}' is_uncacheable={is_uncacheable} has_pending={has_pending}")

        # For continuing conversations, only pass minimal input to preserve state
        # The checkpointer will load previous state (pending_schedule, etc.)
        from langchain_core.messages import HumanMessage
        from uuid import uuid4

        effective_thread_id = thread_id or str(uuid4())

        # Log thread_id for debugging state persistence
        if thread_id:
            logger.info(f"Continuing conversation with thread_id: {thread_id}")
        else:
            logger.warning(f"No thread_id provided - creating new: {effective_thread_id}. State won't persist!")

        # Minimal input - only what changes between messages
        # This allows checkpointer to preserve pending_schedule, awaiting_confirmation, etc.
        input_state = {
            "messages": [HumanMessage(content=query)],
            "original_query": query,
            "user_id": user_id,
            "user_name": user_name or "User",
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
    use_postgres_checkpointer: bool = False,
) -> Agent:
    """
    Create an agent instance.

    Args:
        database_url: PostgreSQL connection string (optional)
        enable_human_review: Enable human-in-the-loop review
        use_postgres_checkpointer: If True, try to use PostgreSQL for checkpointing

    Note: By default uses MemorySaver which works reliably.
    PostgreSQL checkpointing requires psycopg-pool package.
    State persistence for scheduling is handled by _pending_schedule_cache.
    """
    # Only use PostgreSQL if explicitly requested and URL is available
    if use_postgres_checkpointer and database_url is None:
        try:
            from app.core.config import settings
            database_url = settings.database_url
            if database_url:
                logger.info("Using PostgreSQL checkpointer from settings.database_url")
        except Exception as e:
            logger.warning(f"Could not load database URL from settings: {e}")
            database_url = None

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
    "_is_scheduling_intent",
    "_has_scheduling_keyword",
    "_has_time_expression",
]
