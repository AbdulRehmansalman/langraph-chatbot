"""
Chat API Routes
===============
Production-ready chat endpoints with real-time streaming.
"""

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
import asyncio
import time
import logging
from typing import Set

from app.models.schemas import ChatHistory, StreamingChatRequest
from app.api.dependencies.auth import get_current_user_id
from app.core.exceptions import ValidationException
from app.validation.input import validate_and_sanitize
from app.repositories import chat_history_repository
from app.rag.langgraph import create_agent
from app.services.llm_factory import llm_factory
from app.streaming.sse import (
    StreamingManager,
    StreamEventType,
    StatusEvent,
    StreamStatus,
    CompleteEvent,
    ErrorEvent,
    create_error_stream
)

router = APIRouter()
logger = logging.getLogger(__name__)

# Shared agent singleton
_agent = None


def _get_agent():
    """Get or create shared agent with Human-in-the-Loop enabled."""
    global _agent
    if _agent is None:
        # Enable human review gate and postgres persistence
        _agent = create_agent(
            enable_human_review=True,
            use_postgres_checkpointer=True
        )
        logger.info("Created shared agent with HITL and Postgres Checkpointer")
    return _agent


def _get_user_timezone(user_id: str) -> str:
    """Fetch user timezone from database."""
    try:
        from app.services.supabase_client import supabase_client
        result = supabase_client.table("users").select("timezone").eq("id", user_id).execute()
        if result.data and result.data[0].get("timezone"):
            return result.data[0]["timezone"]
    except Exception as e:
        logger.warning(f"Failed to fetch timezone: {e}")
    return "UTC"


# Background task tracking
_background_tasks: Set[asyncio.Task] = set()
_MAX_BACKGROUND_TASKS = 1000

def _track_background_task(task: asyncio.Task) -> None:
    """Track background task and clean up when done."""
    if len(_background_tasks) > _MAX_BACKGROUND_TASKS:
        _background_tasks.difference_update({t for t in _background_tasks if t.done()})
    _background_tasks.add(task)
    task.add_done_callback(lambda t: _background_tasks.discard(t))


@router.post("/stream")
async def stream_chat_response(
    message: StreamingChatRequest,
    user_id: str = Depends(get_current_user_id)
):
    """Stream chat response using SSE with real-time LLM streaming."""
    try:
        sanitized_message, validated_doc_ids = validate_and_sanitize(
            message.message,
            message.document_ids
        )
    except ValidationException as e:
        return StreamingResponse(
            iter([create_error_stream(e.message, "VALIDATION_ERROR")]),
            media_type="text/event-stream"
        )

    logger.info(f"Stream request - User: {user_id}, Message: {sanitized_message[:50]}...")

    async def event_generator():
        start_time = time.time()
        full_response = ""
        stream_completed = False

        try:
            logger.info("STREAM: Starting event generator")
            yield StatusEvent(data={"status": StreamStatus.STARTING, "message": "Initializing..."}).to_sse()
            await asyncio.sleep(0)  # Force flush

            agent = _get_agent()
            user_timezone = _get_user_timezone(user_id)

            logger.info("STREAM: Agent retrieved")
            yield StatusEvent(data={"status": StreamStatus.GENERATING, "message": "Generating..."}).to_sse()
            await asyncio.sleep(0)  # Force flush
# manages how tokens are streamed
            streaming_manager = StreamingManager(
                timeout=float(message.stream_timeout),
                heartbeat_interval=2.0,
                buffer_size=1
            )

            async def token_stream():
                logger.debug(f"STREAM: Starting token_stream for {sanitized_message[:20]}...")
                async for event in agent.stream(
                    query=sanitized_message,
                    user_id=user_id,
                    user_timezone=user_timezone,
                    document_ids=validated_doc_ids,
                    thread_id=message.thread_id,
                ):
                    event_type = event.get("type")
                    if event_type == "token" and event.get("content"):
                        yield event["content"]
                    elif event_type == "node_start":
                        node = event.get("node")
                        logger.debug(f"STREAM: Node started - {node}")
                        # We can't yield status events here easily as streaming_manager expects tokens
                        # But logging confirms progress
                    elif event_type == "node_end":
                         logger.debug(f"STREAM: Node ended - {event.get('node')}")

            async for event in streaming_manager.stream_with_timeout(token_stream(), send_heartbeat=True):
                yield event.to_sse()
                if event.type == StreamEventType.TOKEN:
                    full_response += event.data
                if event.type == StreamEventType.ERROR:
                    stream_completed = True
                    break

            if not stream_completed:
                total_time = time.time() - start_time
                yield CompleteEvent(data={
                    "total_time": round(total_time, 3),
                    "total_tokens": streaming_manager._tokens_sent,
                    "provider": llm_factory.get_provider_info(),
                    "status": "success"
                }).to_sse()
                stream_completed = True

                save_task = asyncio.create_task(_save_history(
                    user_id=user_id,
                    message=sanitized_message,
                    document_ids=validated_doc_ids,
                    response=full_response,
                    response_time=total_time,
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


async def _save_history(user_id, message, document_ids, response, response_time, thread_id=None):
    """Save chat history in background."""
    try:
        await chat_history_repository.create({
            "user_id": user_id,
            "user_message": message,
            "bot_response": response,
            "document_ids": document_ids or [],
            "response_time": round(response_time, 3),
            "has_documents": bool(document_ids),
            "sources_used": len(document_ids) if document_ids else 0,
            "provider": llm_factory.get_provider_info(),
            "thread_id": thread_id
        })
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
