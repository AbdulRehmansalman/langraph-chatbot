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

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
import time
import logging

import asyncio
from typing import Set, Optional

from app.models.schemas import ChatHistory, StreamingChatRequest
from app.api.dependencies.auth import get_current_user_id
from app.core.exceptions import ValidationException
from app.validation.input import validate_and_sanitize
from app.repositories import chat_history_repository
from app.rag.chain import create_rag_chain, get_llm_provider
from app.streaming.sse import (
    StreamingManager,
    StatusEvent,
    StreamStatus,
    CompleteEvent,
    ErrorEvent,
    create_error_stream
)

router = APIRouter()
logger = logging.getLogger(__name__)


# Background task tracking to prevent garbage collection of pending saves
_background_tasks: Set[asyncio.Task] = set()
_MAX_BACKGROUND_TASKS = 1000


def _track_background_task(task: asyncio.Task) -> None:
    """Track background task and clean up when done."""
    if len(_background_tasks) > _MAX_BACKGROUND_TASKS:
        completed = {t for t in _background_tasks if t.done()}
        _background_tasks.difference_update(completed)

    _background_tasks.add(task)
# Runs when tASK fINISHES
    def _on_done(t: asyncio.Task) -> None:
        _background_tasks.discard(t)
        if not t.cancelled():
            exc = t.exception()
            if exc:
                logger.error(f"Background task failed: {exc}", exc_info=exc)
# Registers _on_done to automatically run when the task completes.
    task.add_done_callback(_on_done)


@router.post("/stream")
async def stream_chat_response(
    message: StreamingChatRequest,
    user_id: str = Depends(get_current_user_id)
):
    """Stream chat response using SSE."""
    try:
        sanitized_message, validated_doc_ids = validate_and_sanitize(
            message.message,
            message.document_ids
        )
        # as when user sen dtoo ong message so it gives streaming eror not crahs
    except ValidationException as e:
        return StreamingResponse(
            iter([create_error_stream(e.message, "VALIDATION_ERROR")]),
            media_type="text/event-stream"
        )

    logger.info(f"Streaming request - User: {user_id}, Message: {sanitized_message[:50]}...")
    llm_provider = get_llm_provider()

    async def event_generator():
        start_time = time.time()
        full_response = ""
        stream_completed = False

        try:
            yield StatusEvent(data={"status": StreamStatus.STARTING, "message": "Initializing..."}).to_sse()
            await asyncio.sleep(0.05)
            yield StatusEvent(data={"status": StreamStatus.RETRIEVING, "message": "Searching..."}).to_sse()

            rag_chain = create_rag_chain(
                user_id=user_id,
                document_ids=validated_doc_ids,
                thread_id=message.thread_id,
            )

            streaming_manager = StreamingManager(
                timeout=float(message.stream_timeout),
                heartbeat_interval=2.0,
                buffer_size=1   
            )

            yield StatusEvent(data={"status": StreamStatus.GENERATING, "message": "Generating..."}).to_sse()

            async_stream = rag_chain.stream(sanitized_message)

            async for event in streaming_manager.stream_with_timeout(async_stream, send_heartbeat=True):
                yield event.to_sse()
                if event.type == "token":
                    full_response += event.data
                if event.type == "error":
                    stream_completed = True
                    break

            if not stream_completed:
                total_time = time.time() - start_time
                yield CompleteEvent(data={
                    "total_time": round(total_time, 3),
                    "total_tokens": streaming_manager._tokens_sent,
                    "provider": llm_provider,
                    "status": "success"
                }).to_sse()
                stream_completed = True

                save_task = asyncio.create_task(_save_streaming_history(
                    user_id=user_id,
                    message=sanitized_message,
                    document_ids=validated_doc_ids,
                    response=full_response,
                    response_time=total_time,
                    provider=llm_provider,
                    thread_id=message.thread_id
                ))
                _track_background_task(save_task)

        except Exception as e:
            logger.error(f"Streaming error: {e}", exc_info=True)
            if not stream_completed:
                yield ErrorEvent(data={"message": str(e), "code": "STREAMING_ERROR"}).to_sse()

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )


async def _save_streaming_history(user_id, message, document_ids, response, response_time, provider, thread_id=None):
    """Save history in background."""
    try:
        chat_data = {
            "user_id": user_id,
            "user_message": message,
            "bot_response": response,
            "document_ids": document_ids or [],
            "response_time": round(response_time, 3),
            "has_documents": bool(document_ids),
            "sources_used": len(document_ids) if document_ids else 0,
            "provider": provider,
            "thread_id": thread_id
        }
        await chat_history_repository.create(chat_data)
    except Exception as e:
        logger.error(f"Failed to save history: {e}")


@router.get("/history", response_model=ChatHistory)
async def get_chat_history(
    limit: int = 50,
    user_id: str = Depends(get_current_user_id)
):
    """Get chat history."""
    try:
        messages = await chat_history_repository.get_by_user(user_id, limit)
        return ChatHistory(messages=messages)
    except Exception as e:
        logger.error(f"Error fetching history: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch history")
