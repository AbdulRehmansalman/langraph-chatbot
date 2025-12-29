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


class RAGChainAdapter:
    """
    Adapter that provides RAG chain-like interface using LangGraph agent.

    Usage:
        chain = create_rag_chain(user_id="123")
        response = await chain.invoke("What is the policy?")

        # Or streaming
        for token in chain.stream("What is the policy?"):
            print(token)
    """

    def __init__(
        self,
        user_id: str,
        document_ids: Optional[list[str]] = None,
    ):
        self.user_id = user_id
        self.document_ids = document_ids
        self._agent = None

    def _get_agent(self):
        """Lazy load the agent."""
        if self._agent is None:
            from app.rag.langgraph import create_agent
            self._agent = create_agent()
        return self._agent

    async def invoke(self, query: str) -> "RAGResponse":
        """
        Invoke the RAG chain with a query.

        Args:
            query: User's query

        Returns:
            RAGResponse with answer and metadata
        """
        agent = self._get_agent()

        result = await agent.invoke(
            query=query,
            user_id=self.user_id,
            document_ids=self.document_ids,
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

        Args:
            query: User's query

        Yields:
            Response tokens
        """
        # For sync streaming, use the LLM directly for simplicity
        # Full agent streaming would require async context
        try:
            from app.services.document_processor import doc_processor

            # Get relevant documents
            documents = doc_processor.search_documents(
                query=query,
                user_id=self.user_id,
                document_ids=self.document_ids,
                top_k=5,
            )

            # Build context
            context = "\n\n".join([
                f"[{i+1}] {doc['content']}"
                for i, doc in enumerate(documents)
            ])

            # Create prompt
            system_prompt = f"""You are a helpful assistant that answers questions based on the provided context.

RULES:
1. Use ONLY information from the provided context
2. If the context doesn't contain the answer, say "I don't have enough information."
3. Cite sources using [1], [2], etc. notation
4. Be concise but thorough

Context:
{context}"""

            # Get LLM and stream
            llm = llm_factory.create_llm(streaming=True)

            from langchain_core.prompts import ChatPromptTemplate
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", "{question}"),
            ])

            chain = prompt | llm

            for chunk in chain.stream({"question": query}):
                if hasattr(chunk, "content") and chunk.content:
                    yield chunk.content

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
) -> RAGChainAdapter:
    """
    Create a RAG chain for the given user.

    Args:
        user_id: User ID
        document_ids: Optional document IDs to search

    Returns:
        RAGChainAdapter instance
    """
    return RAGChainAdapter(user_id=user_id, document_ids=document_ids)


def get_llm_provider() -> str:
    """Get current LLM provider info."""
    return llm_factory.get_provider_info()


__all__ = ["create_rag_chain", "get_llm_provider", "RAGChainAdapter", "RAGResponse"]
