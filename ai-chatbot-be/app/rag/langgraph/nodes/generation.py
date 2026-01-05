"""
Response Generation Node
========================

Generates responses using LLM with:
- Context-grounded answers
- Citation formatting
- Streaming support
- Intelligent appointment scheduling suggestions (now powered by LLM classifier)
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

RESPONSE_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", "{system_prompt}"),
        ("human", "{question}"),
    ]
)

FALLBACK_RESPONSES = {
    "NO_DOCUMENTS": (
        "I don't have any documents to search through yet. "
        "Please upload some documents first, and then I'll be able to answer your questions."
    ),
    "NO_EMBEDDINGS": (
        "Your documents are still being processed. " "Please wait a moment and try again."
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


# === NEW: LLM-BASED SCHEDULING INTENT CLASSIFIER ===
SCHEDULING_CLASSIFIER_PROMPT = ChatPromptTemplate.from_template(
    """You are an expert at detecting user interest in scheduling appointments, consultations, services, or bookings.

User query: {query}

Assistant response: {response}

Relevant document excerpts:
{snippets}

Does this conversation indicate the user is inquiring about something that can be scheduled (e.g., medical test, consultation, training session, service package, treatment, etc.)?

Answer with ONLY "yes" or "no". No explanation."""
)


from langchain_core.runnables import RunnableConfig

async def _should_suggest_scheduling_llm(
    query: str, 
    response: str, 
    documents: list[dict], 
    fallback: bool = True,
    config: Optional[RunnableConfig] = None
) -> bool:
    """
    Use a lightweight LLM to classify if scheduling should be suggested.
    Falls back to original regex method if needed.
    """
    try:
        from app.services.llm_factory import llm_factory

        # Use the default LLM for classification (Ollama in dev, Bedrock in prod)
        classifier_llm = llm_factory.create_llm(
            temperature=0.0,  # Low temp for deterministic classification
            max_tokens=50,
        )

        # Prepare concise snippets (top 4 docs, truncated)
        snippets = "\n---\n".join(
            [doc.get("content", "")[:600] for doc in documents[:4] if doc.get("content")]
        )
        if not snippets.strip():
            snippets = "No relevant document content."

        chain = SCHEDULING_CLASSIFIER_PROMPT | classifier_llm | StrOutputParser()

        result = await chain.ainvoke(
            {
                "query": query.strip(),
                "response": response.strip(),
                "snippets": snippets,
            },
            config=config,
        )

        decision = result.strip().lower()
        is_yes = "yes" in decision

        logger.info(f"LLM Scheduling Classifier result: '{result.strip()}' → Suggest: {is_yes}")

        return False  # is_yes (Temporarily disabled for performance)

    except Exception as e:
        logger.warning(f"LLM scheduling classifier failed: {e}. Falling back to regex method.")
        if fallback:
            return _should_suggest_scheduling_regex(query, response, documents)
        return False


def _should_suggest_scheduling_regex(query: str, response: str, documents: list[dict]) -> bool:
    """
    Original regex-based fallback method (kept as safety net).
    Improved with broader patterns and lower threshold.
    """
    query_lower = query.lower()
    response_lower = response.lower()

    if len(query.split()) < 3:
        return False

    scheduling_words = ["schedule", "book", "appointment", "reserve", "when can i", "slot"]
    if any(word in query_lower for word in scheduling_words):
        return False  # Already asking to schedule

    # Broad indicators of schedulable services
    schedulable_indicators = [
        r"\b(service|test|package|plan|program|session|consultation|procedure|therapy|course|treatment|check.?up|exam|scan|assessment|evaluation)\b",
        r"\b(appointment|visit|booking|reservation|slot|availability)\b",
        r"\b(available|offered|provide|offer|includes?|covers?)\b",
        r"\b(duration|time|minutes|hours|session.?length)\b",
        r"\b(price|cost|fee|charge)\b",
    ]

    content_to_check = response_lower
    for doc in documents[:6]:  # Check more docs
        content_to_check += " " + doc.get("content", "").lower()

    matches = sum(
        1
        for pattern in schedulable_indicators
        if re.search(pattern, content_to_check, re.IGNORECASE)
    )

    triggered = matches >= 1  # Lowered threshold
    if triggered:
        logger.info(f"Regex fallback triggered scheduling suggestion ({matches} matches)")

    return triggered


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
            citations.append(
                Citation(
                    index=index,
                    document_id=doc.get("id", ""),
                    source=doc.get("source", "Unknown"),
                    snippet=doc.get("content", "")[:200],
                ).model_dump()
            )

    return citations


from langchain_core.runnables import RunnableConfig

async def generation_node(state: AgentState, config: RunnableConfig = None) -> dict[str, Any]:
    """
    Generate response using LLM with improved scheduling suggestion logic.
    """
    start_time = time.time()

    query = state.get("original_query", "")
    classification = state.get("query_classification", "general")
    context = get_response_context(state)
    documents = state.get("documents", [])
    error_log = state.get("error_log", [])
    user_id = state.get("user_id")

    logger.info("=" * 50)
    logger.info("GENERATION NODE - STATE RECEIVED")
    logger.info(f"  Query: {query[:80]}...")
    logger.info(f"  Classification: {classification}")
    logger.info(f"  Documents received: {len(documents)}")
    logger.info(f"  Context length: {len(context)} chars")
    logger.info("=" * 50)

    updates = track_node(state, "generation")

    # Handle no documents
    if classification == "document" and not documents:
        logger.warning("NO DOCUMENTS - RETURNING FALLBACK")
        fallback_response = FALLBACK_RESPONSE
        for error in error_log:
            error_type = error.get("error_type", "")
            if error_type in FALLBACK_RESPONSES:
                fallback_response = FALLBACK_RESPONSES[error_type]
                break

        updates["response"] = fallback_response
        updates["citations"] = []
        updates["messages"] = [AIMessage(content=fallback_response)]
        return updates

    try:
        from app.services.llm_factory import llm_factory

        llm = llm_factory.create_llm()

        system_prompt = SYSTEM_PROMPTS.get(classification, SYSTEM_PROMPTS["general"]).format(
            context=context
        )

        chain = RESPONSE_PROMPT | llm | StrOutputParser()
        response = await chain.ainvoke(
            {
                "system_prompt": system_prompt,
                "question": query,
            },
            config=config,  # Pass config for callback propagation
        )

        # Extract citations
        citations = []
        if classification == "document" and documents:
            citations = _extract_citations(response, documents)

        logger.info(f"Generated response with {len(citations)} citations")

        # === IMPROVED SCHEDULING SUGGESTION LOGIC ===
        scheduling_suggested = False
        if classification == "document" and documents and len(query.strip()) > 0:
            try:
                scheduling_suggested = await _should_suggest_scheduling_llm(
                    query=query,
                    response=response,
                    documents=documents,
                    fallback=True,  # Use regex if LLM fails
                    config=config,   # Pass config for callback propagation
                )

                if scheduling_suggested:
                    response += SCHEDULING_SUGGESTION
                    logger.info("Added scheduling suggestion (LLM or regex triggered)")
            except Exception as e:
                logger.error(f"Scheduling suggestion decision failed: {e}")
                scheduling_suggested = False

        updates["response"] = response
        updates["citations"] = citations
        updates["messages"] = [AIMessage(content=response)]
        updates["scheduling_suggested"] = scheduling_suggested
        updates["awaiting_scheduling_confirmation"] = scheduling_suggested
        if scheduling_suggested:
            updates["scheduling_context"] = query[:200]

        # Metrics
        input_tokens = estimate_tokens(system_prompt + query)
        output_tokens = estimate_tokens(response)
        duration_ms = (time.time() - start_time) * 1000

        updates.update(
            update_metrics(
                state,
                llm_calls=1 + (1 if scheduling_suggested else 0),  # +1 if classifier ran
                tokens_in=input_tokens,
                tokens_out=output_tokens,
                duration_ms=duration_ms,
            )
        )

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
    Scheduling suggestion added at the end.
    """
    logger.info("Starting streaming generation")

    query = state.get("original_query", "")
    classification = state.get("query_classification", "general")
    context = get_response_context(state)
    documents = state.get("documents", [])

    if classification == "document" and not documents:
        yield {"token": FALLBACK_RESPONSE, "done": True}
        return

    try:
        from app.services.llm_factory import llm_factory

        llm = llm_factory.create_llm()

        system_prompt = SYSTEM_PROMPTS.get(classification, SYSTEM_PROMPTS["general"]).format(
            context=context
        )

        chain = RESPONSE_PROMPT | llm

        full_response = ""
        async for chunk in chain.astream(
            {
                "system_prompt": system_prompt,
                "question": query,
            }
        ):
            token = chunk.content if hasattr(chunk, "content") else str(chunk)
            full_response += token
            yield {"token": token, "done": False}

        # Add scheduling suggestion if needed
        scheduling_suggested = False
        if classification == "document" and documents:
            try:
                scheduling_suggested = await _should_suggest_scheduling_llm(
                    query=query, response=full_response, documents=documents, fallback=True
                )
                if scheduling_suggested:
                    full_response += SCHEDULING_SUGGESTION
                    # Stream the suggestion
                    for token in SCHEDULING_SUGGESTION:
                        yield {"token": token, "done": False}
            except Exception as e:
                logger.error(f"Streaming scheduling suggestion failed: {e}")

        # Final yield
        citations = (
            _extract_citations(full_response, documents)
            if classification == "document" and documents
            else []
        )

        yield {
            "token": "",
            "done": True,
            "full_response": full_response,
            "citations": citations,
            "scheduling_suggested": scheduling_suggested,
        }

    except Exception as e:
        logger.error(f"Streaming generation error: {e}")
        yield {
            "token": "An error occurred while generating the response.",
            "done": True,
            "error": str(e),
        }


__all__ = ["generation_node", "stream_generation"]
