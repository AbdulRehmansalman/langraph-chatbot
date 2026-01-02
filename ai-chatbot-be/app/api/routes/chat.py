"""
Chat API Routes
===============
Production-ready chat endpoints with enterprise validation and guardrails.

Enterprise Features:
- Strong Pydantic validation at API boundary
- Prompt injection detection
- Input sanitization before RAG
- Request size limits
- Clear error messages
"""

from fastapi import APIRouter, Depends, HTTPException
import time
import logging

from app.models.schemas import ChatMessage, ChatResponse, ChatHistory
from app.api.dependencies.auth import get_current_user_id
from app.core.exceptions import ValidationException
from app.services.supabase_client import supabase_client
from app.validation.input import validate_and_sanitize
from app.repositories import chat_history_repository
from app.rag.chain import create_rag_chain, get_llm_provider, RAGResponse


router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/message", response_model=ChatResponse)
async def send_message(
    message: ChatMessage,
    user_id: str = Depends(get_current_user_id)
):
    """
    Send a chat message and get a response.

    Enterprise Guardrails:
    - Pydantic validation (1-10000 chars, max 50 document IDs)
    - Prompt injection detection (15+ patterns)
    - Input sanitization (control chars, zero-width chars)
    - Token count validation (max 2000 tokens)
    """
    start_time = time.time()

    try:
        # GUARDRAIL: Validate and sanitize input
        sanitized_message, validated_doc_ids = validate_and_sanitize(
            message.message,
            message.document_ids
        )

        logger.info(
            f"Chat request - User: {user_id}, "
            f"Message length: {len(sanitized_message)}, "
            f"Documents: {len(validated_doc_ids) if validated_doc_ids else 0}"
        )

        # Create RAG chain with thread_id for conversation continuity
        rag_chain = create_rag_chain(
            user_id=user_id,
            document_ids=validated_doc_ids,
            thread_id=message.thread_id,  # Required for multi-turn scheduling
        )

        # Generate response
        rag_response: RAGResponse = await rag_chain.invoke(sanitized_message)
        response = rag_response.answer
        response_time = round(time.time() - start_time, 3)

        # Save chat history using repository pattern
        # Only include fields that exist in the ChatHistory model
        chat_data = {
            "user_id": user_id,
            "user_message": sanitized_message,
            "bot_response": response,
            "document_ids": validated_doc_ids if validated_doc_ids else [],
            "response_time": response_time,
            "has_documents": bool(validated_doc_ids),
            "sources_used": len(validated_doc_ids) if validated_doc_ids else 0,
            "provider": get_llm_provider(),
        }

        # Use repository instead of direct database access
        saved_chat = await chat_history_repository.create(chat_data)

        return ChatResponse(
            id=saved_chat["id"],
            user_message=sanitized_message,
            bot_response=response,
            document_ids=validated_doc_ids,
            created_at=saved_chat["created_at"]
        )

    except ValidationException as e:
        # Return validation errors with 400 status
        logger.warning(f"Validation error for user {user_id}: {e.message}")
        raise HTTPException(status_code=400, detail=e.message)

    except HTTPException:
        raise

    except Exception as e:
        logger.error(f"Chat error for user {user_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="An error occurred processing your request")


@router.get("/history", response_model=ChatHistory)
async def get_chat_history(
    limit: int = 50,
    user_id: str = Depends(get_current_user_id)
):
    """Get user's chat history."""
    if limit <= 0 or limit > 1000:
        raise HTTPException(status_code=400, detail="Limit must be between 1 and 1000")

    try:
        result = supabase_client.table("chat_history") \
            .select("*") \
            .eq("user_id", user_id) \
            .order("created_at", desc=True) \
            .limit(limit) \
            .execute()

        messages = [
            ChatResponse(
                id=msg["id"],
                user_message=msg["user_message"],
                bot_response=msg["bot_response"],
                document_ids=msg.get("document_ids"),
                created_at=msg["created_at"]
            )
            for msg in result.data
        ]

        return ChatHistory(messages=messages)

    except Exception as e:
        logger.error(f"Error fetching chat history for user {user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to fetch chat history")
