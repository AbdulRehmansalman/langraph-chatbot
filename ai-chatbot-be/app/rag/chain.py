"""
RAG Chain Adapter
=================

Compatibility layer that wraps LangGraph agent with RAG chain interface.
Provides backward compatibility for existing endpoints.
"""

import asyncio
import logging
from typing import Any, Iterator, Optional

from app.services.llm_factory import llm_factory

logger = logging.getLogger(__name__)

# =============================================================================
# SHARED AGENT SINGLETON
# =============================================================================
# The agent must be shared across requests to preserve conversation state
# via the MemorySaver checkpointer. Each request uses the same thread_id
# to continue a conversation.

_shared_agent = None
_shared_agent_lock = asyncio.Lock() if hasattr(asyncio, 'Lock') else None


def _get_shared_agent():
    """Get or create the shared agent singleton."""
    global _shared_agent
    if _shared_agent is None:
        from app.rag.langgraph import create_agent
        _shared_agent = create_agent()
        logger.info("Created shared agent singleton for state persistence")
    return _shared_agent


class RAGChainAdapter:
    """
    Adapter that provides RAG chain-like interface using LangGraph agent.

    Usage:
        chain = create_rag_chain(user_id="123", thread_id="conversation-123")
        response = await chain.invoke("What is the policy?")

        # Or streaming
        for token in chain.stream("What is the policy?"):
            print(token)
    """

    def __init__(
        self,
        user_id: str,
        document_ids: Optional[list[str]] = None,
        thread_id: Optional[str] = None,
    ):
        self.user_id = user_id
        self.document_ids = document_ids
        self.thread_id = thread_id
        self._user_timezone: Optional[str] = None

    def _get_user_timezone(self) -> str:
        """Fetch user's timezone from database. Cached for the session."""
        if self._user_timezone is not None:
            return self._user_timezone

        try:
            from app.services.supabase_client import supabase_client
            result = supabase_client.table("users").select("timezone").eq("id", self.user_id).execute()
            if result.data and result.data[0].get("timezone"):
                self._user_timezone = result.data[0]["timezone"]
                logger.info(f"User timezone loaded: {self._user_timezone}")
            else:
                self._user_timezone = "UTC"
                logger.info(f"No timezone for user {self.user_id}, defaulting to UTC")
        except Exception as e:
            logger.warning(f"Failed to fetch user timezone: {e}, defaulting to UTC")
            self._user_timezone = "UTC"

        return self._user_timezone

    def _get_agent(self):
        """Get the shared agent singleton for state persistence."""
        return _get_shared_agent()

    async def invoke(self, query: str) -> "RAGResponse":
        """
        Invoke the RAG chain with a query.

        Args:
            query: User's query

        Returns:
            RAGResponse with answer and metadata
        """
        agent = self._get_agent()
        user_timezone = self._get_user_timezone()

        result = await agent.invoke(
            query=query,
            user_id=self.user_id,
            user_timezone=user_timezone,
            document_ids=self.document_ids,
            thread_id=self.thread_id,  # Pass thread_id for state persistence
        )

        return RAGResponse(
            answer=result.get("response", ""),
            sources=[
                {"source": doc.get("source", ""), "content": doc.get("content", "")[:200]}
                for doc in result.get("documents", [])
            ],
            context_used="\n".join([d.get("content", "") for d in result.get("documents", [])]),
            retrieval_scores=[doc.get("score", 0.0) for doc in result.get("documents", [])],
            citations=result.get("citations", []),
        )

    def stream(self, query: str) -> Iterator[str]:
        """
        Stream response tokens.

        NOTE: This method now uses the same retrieval path as invoke() to ensure
        retrieval is ALWAYS executed. The sync wrapper runs the async agent.

        Args:
            query: User's query

        Yields:
            Response tokens
        """
        import asyncio

        logger.info(f"STREAM: Starting streaming for query: {query[:50]}...")

        try:
            # Run the full agent to ensure retrieval executes
            # This guarantees retrieval is not bypassed
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            try:
                agent = self._get_agent()
                user_timezone = self._get_user_timezone()
                result = loop.run_until_complete(
                    agent.invoke(
                        query=query,
                        user_id=self.user_id,
                        user_timezone=user_timezone,
                        document_ids=self.document_ids,
                        thread_id=self.thread_id,  # Pass thread_id for state persistence
                    )
                )

                # Get the full response from agent
                response = result.get("response", "")
                documents = result.get("documents", [])

                logger.info(f"STREAM: Agent returned {len(documents)} documents")

                # Yield response in chunks to simulate streaming
                # This maintains streaming UX while ensuring retrieval runs
                chunk_size = 20
                for i in range(0, len(response), chunk_size):
                    yield response[i:i + chunk_size]

            finally:
                loop.close()

        except Exception as e:
            logger.error(f"Streaming error: {e}")
            yield f"Error: {str(e)}"


class RAGResponse:
    """RAG response with answer and metadata."""

    def __init__(
        self,
        answer: str,
        sources: list[dict] = None,
        context_used: str = "",
        retrieval_scores: list[float] = None,
        citations: list[dict] = None,
    ):
        self.answer = answer
        self.sources = sources or []
        self.context_used = context_used
        self.retrieval_scores = retrieval_scores or []
        self.citations = citations or []

    def to_dict(self) -> dict:
        return {
            "answer": self.answer,
            "sources": self.sources,
            "context_used": self.context_used,
            "retrieval_scores": self.retrieval_scores,
            "citations": self.citations,
        }


def create_rag_chain(
    user_id: str,
    document_ids: Optional[list[str]] = None,
    thread_id: Optional[str] = None,
) -> RAGChainAdapter:
    """
    Create a RAG chain for the given user.

    Args:
        user_id: User ID
        document_ids: Optional document IDs to search
        thread_id: Thread ID for conversation continuity

    Returns:
        RAGChainAdapter instance
    """
    return RAGChainAdapter(user_id=user_id, document_ids=document_ids, thread_id=thread_id)


def get_llm_provider() -> str:
    """Get current LLM provider info."""
    return llm_factory.get_provider_info()


__all__ = ["create_rag_chain", "get_llm_provider", "RAGChainAdapter", "RAGResponse"]
