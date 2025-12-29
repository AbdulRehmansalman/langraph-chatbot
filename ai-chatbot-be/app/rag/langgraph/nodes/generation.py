"""
Response Generation Node
========================

Generates responses using LLM with:
- Context-grounded answers
- Citation formatting
- Streaming support
"""

import logging
import re
import time
from typing import Any, AsyncIterator

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

FALLBACK_RESPONSE = (
    "I apologize, but I couldn't find relevant information to answer your question. "
    "Could you please rephrase or ask something else?"
)


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
    logger.info("Starting response generation")

    query = state.get("original_query", "")
    classification = state.get("query_classification", "general")
    context = get_response_context(state)
    documents = state.get("documents", [])

    updates = track_node(state, "generation")

    # Handle no context for document queries
    if classification == "document" and not documents:
        logger.warning("No documents available for document query")
        updates["response"] = FALLBACK_RESPONSE
        updates["citations"] = []
        updates["messages"] = [AIMessage(content=FALLBACK_RESPONSE)]
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

        updates["response"] = response
        updates["citations"] = citations
        updates["messages"] = [AIMessage(content=response)]

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
