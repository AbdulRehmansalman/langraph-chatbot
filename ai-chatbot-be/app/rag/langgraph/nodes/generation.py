"""
Response Generation Node
========================

Generates responses using LLM with:
- Context-grounded answers
- Citation formatting
- Streaming support
- Intelligent appointment scheduling suggestions
"""

import logging
import re
import time
from typing import Any, AsyncIterator, Optional

from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from app.rag.langgraph.state import (
    AgentState,
    Citation,
    track_node,
    get_response_context,
    update_metrics,
)

logger = logging.getLogger(__name__)

# Token limits
MAX_CONTEXT_TOKENS = 6000
TOKEN_ESTIMATION_RATIO = 4


def estimate_tokens(text: str) -> int:
    """Estimate token count from text length."""
    return len(text) // TOKEN_ESTIMATION_RATIO


# System prompts
SYSTEM_PROMPTS = {
    "document": """You are a helpful assistant that answers questions based on the provided context.

RULES:
1. Use ONLY information from the provided context
2. If the context doesn't contain the answer, say "I don't have enough information."
3. Cite sources using [1], [2], etc. notation
4. Be concise but thorough

Context:
{context}""",

    "calendar": """You are a helpful assistant that manages calendar and appointments.

Based on the calendar information provided, help the user with their request.

Calendar Context:
{context}""",

    "general": """You are a helpful AI assistant.
Be conversational, accurate, and helpful.

{context}""",
}

RESPONSE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "{system_prompt}"),
    ("human", "{question}"),
])

FALLBACK_RESPONSES = {
    "NO_DOCUMENTS": (
        "I don't have any documents to search through yet. "
        "Please upload some documents first, and then I'll be able to answer your questions."
    ),
    "NO_EMBEDDINGS": (
        "Your documents are still being processed. "
        "Please wait a moment and try again."
    ),
    "default": (
        "I couldn't find relevant information in your documents to answer that question. "
        "Try rephrasing your question or asking about different topics from your documents."
    ),
}

FALLBACK_RESPONSE = FALLBACK_RESPONSES["default"]

# Scheduling suggestion template
SCHEDULING_SUGGESTION = (
    "\n\n---\n"
    "📅 **Would you like to schedule an appointment for this?**\n"
    "Just reply **'yes'** and I'll help you book it."
)


def _should_suggest_scheduling(query: str, response: str, documents: list[dict]) -> bool:
    """
    Determine if we should suggest scheduling an appointment.

    This works with ANY type of documents - medical, business, training, etc.
    It detects if the user is asking about something that could be scheduled.

    Args:
        query: User's original query
        response: Generated response
        documents: Retrieved documents

    Returns:
        True if scheduling suggestion should be added
    """
    query_lower = query.lower()
    response_lower = response.lower()

    # Skip if query is too short or is a greeting
    if len(query.split()) < 3:
        return False

    # Skip if already asking to schedule
    scheduling_words = ["schedule", "book", "appointment", "reserve", "when can i"]
    if any(word in query_lower for word in scheduling_words):
        return False

    # Patterns that indicate user is INQUIRING about something (not just chatting)
    inquiry_patterns = [
        r"(what|which|tell me|explain|describe|how|can you|do you|is there|are there)",
        r"(about|offer|provide|available|options?|services?|tests?|packages?)",
        r"\?$",  # Questions ending with ?
    ]

    is_inquiry = any(re.search(pattern, query_lower) for pattern in inquiry_patterns)
    if not is_inquiry:
        return False

    # Check if response contains schedulable content indicators
    # These are GENERIC patterns that work across any domain
    schedulable_indicators = [
        # Services and offerings
        r"\b(service|test|package|plan|program|session|consultation)\b",
        r"\b(appointment|visit|meeting|booking)\b",
        r"\b(available|offered|provide|offer)\b",
        # Actions that can be scheduled
        r"\b(exam|examination|assessment|evaluation|check[-\s]?up)\b",
        r"\b(training|workshop|demo|demonstration|trial)\b",
        r"\b(treatment|procedure|therapy|course)\b",
        # Time-related (indicates something can be scheduled)
        r"\b(duration|takes|minutes|hours|weekly|daily|monthly)\b",
        r"\b(schedule|timing|slots?|availability)\b",
    ]

    # Check in both response and documents
    content_to_check = response_lower
    for doc in documents[:3]:
        content_to_check += " " + doc.get("content", "").lower()

    matches = sum(1 for pattern in schedulable_indicators
                  if re.search(pattern, content_to_check))

    # If we find at least 2 schedulable indicators, suggest scheduling
    if matches >= 2:
        logger.info(f"Scheduling suggestion triggered: {matches} indicators found")
        return True

    return False


def _extract_citations(response: str, documents: list[dict]) -> list[dict]:
    """Extract citations from response and match to documents."""
    citations = []
    citation_pattern = r"\[(\d+)\]"
    matches = re.findall(citation_pattern, response)

    seen_indices = set()
    for match in matches:
        index = int(match)
        if index not in seen_indices and 0 < index <= len(documents):
            seen_indices.add(index)
            doc = documents[index - 1]
            citations.append(Citation(
                index=index,
                document_id=doc.get("id", ""),
                source=doc.get("source", "Unknown"),
                snippet=doc.get("content", "")[:200],
            ).model_dump())

    return citations


async def generation_node(state: AgentState) -> dict[str, Any]:
    """
    Generate response using LLM.

    Args:
        state: Current agent state

    Returns:
        Updated state with generated response
    """
    start_time = time.time()

    query = state.get("original_query", "")
    classification = state.get("query_classification", "general")
    context = get_response_context(state)
    documents = state.get("documents", [])
    error_log = state.get("error_log", [])
    user_id = state.get("user_id")

    # DIAGNOSTIC LOGGING - trace what generation receives
    logger.info("=" * 50)
    logger.info("GENERATION NODE - STATE RECEIVED")
    logger.info(f"  Query: {query[:80]}...")
    logger.info(f"  Classification: {classification}")
    logger.info(f"  User ID: {user_id}")
    logger.info(f"  Documents received: {len(documents)}")
    logger.info(f"  Context length: {len(context)} chars")
    logger.info(f"  Error log entries: {len(error_log)}")
    if error_log:
        for err in error_log:
            logger.info(f"    - {err.get('error_type')}: {err.get('message', '')[:50]}")
    logger.info("=" * 50)

    updates = track_node(state, "generation")

    # Handle no context for document queries
    if classification == "document" and not documents:
        logger.warning("=" * 50)
        logger.warning("NO DOCUMENTS - RETURNING FALLBACK")
        logger.warning(f"  Reason: classification={classification}, documents={len(documents)}")
        logger.warning("=" * 50)

        # Check error log for specific error types to provide better responses
        fallback_response = FALLBACK_RESPONSE

        for error in error_log:
            error_type = error.get("error_type", "")
            if error_type in FALLBACK_RESPONSES:
                fallback_response = FALLBACK_RESPONSES[error_type]
                logger.info(f"  Using specific fallback for: {error_type}")
                break

        updates["response"] = fallback_response
        updates["citations"] = []
        updates["messages"] = [AIMessage(content=fallback_response)]
        return updates

    try:
        from app.services.llm_factory import llm_factory
        llm = llm_factory.create_llm()

        # Select prompt based on classification
        system_prompt = SYSTEM_PROMPTS.get(
            classification,
            SYSTEM_PROMPTS["general"]
        ).format(context=context)

        chain = RESPONSE_PROMPT | llm | StrOutputParser()
        response = await chain.ainvoke({
            "system_prompt": system_prompt,
            "question": query,
        })

        # Extract citations for document queries
        citations = []
        if classification == "document" and documents:
            citations = _extract_citations(response, documents)

        logger.info(f"Generated response with {len(citations)} citations")

        # Check if we should suggest scheduling an appointment
        scheduling_suggested = False
        if classification == "document" and documents:
            if _should_suggest_scheduling(query, response, documents):
                response += SCHEDULING_SUGGESTION
                scheduling_suggested = True
                logger.info("Added scheduling suggestion to response")

        updates["response"] = response
        updates["citations"] = citations
        updates["messages"] = [AIMessage(content=response)]
        updates["scheduling_suggested"] = scheduling_suggested
        updates["awaiting_scheduling_confirmation"] = scheduling_suggested  # Wait for "yes"
        if scheduling_suggested:
            # Store what the user was asking about for context
            updates["scheduling_context"] = query[:200]

        # Update metrics
        input_tokens = estimate_tokens(system_prompt + query)
        output_tokens = estimate_tokens(response)
        duration_ms = (time.time() - start_time) * 1000

        updates.update(update_metrics(
            state,
            llm_calls=1,
            tokens_in=input_tokens,
            tokens_out=output_tokens,
            duration_ms=duration_ms,
        ))

    except Exception as e:
        logger.error(f"Generation error: {e}")
        error_response = "I apologize, but I encountered an error. Please try again."
        updates["response"] = error_response
        updates["citations"] = []
        updates["messages"] = [AIMessage(content=error_response)]

    duration_ms = (time.time() - start_time) * 1000
    logger.info(f"Response generation complete in {duration_ms:.1f}ms")

    return updates


async def stream_generation(state: AgentState) -> AsyncIterator[dict[str, Any]]:
    """
    Stream response generation token by token.

    Args:
        state: Current agent state

    Yields:
        Token updates and final state
    """
    logger.info("Starting streaming generation")

    query = state.get("original_query", "")
    classification = state.get("query_classification", "general")
    context = get_response_context(state)
    documents = state.get("documents", [])

    # Handle no context
    if classification == "document" and not documents:
        yield {"token": FALLBACK_RESPONSE, "done": True}
        return

    try:
        from app.services.llm_factory import llm_factory
        llm = llm_factory.create_llm()

        system_prompt = SYSTEM_PROMPTS.get(
            classification,
            SYSTEM_PROMPTS["general"]
        ).format(context=context)

        chain = RESPONSE_PROMPT | llm

        full_response = ""
        async for chunk in chain.astream({
            "system_prompt": system_prompt,
            "question": query,
        }):
            token = chunk.content if hasattr(chunk, "content") else str(chunk)
            full_response += token
            yield {"token": token, "done": False}

        # Final yield with citations
        citations = []
        if classification == "document" and documents:
            citations = _extract_citations(full_response, documents)

        yield {
            "token": "",
            "done": True,
            "full_response": full_response,
            "citations": citations,
        }

    except Exception as e:
        logger.error(f"Streaming generation error: {e}")
        yield {
            "token": "An error occurred while generating the response.",
            "done": True,
            "error": str(e),
        }


__all__ = ["generation_node", "stream_generation"]
