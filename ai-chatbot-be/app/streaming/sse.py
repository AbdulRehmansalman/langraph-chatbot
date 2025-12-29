"""
SSE Streaming Infrastructure
============================
Production Server-Sent Events with heartbeat, timeout, and error handling.

Event Types:
- STATUS: Pipeline state changes (starting, retrieving, generating)
- TOKEN: Generated text chunks
- PROGRESS: Periodic heartbeat with token count and timing
- COMPLETE: Success completion with metrics
- ERROR: Failure with error code and recovery info

Streaming Contract:
1. START event (status: starting) - Always first
2. RETRIEVAL event (status: retrieving) - Document search
3. GENERATING event (status: generating) - LLM generation
4. TOKEN events - Text chunks as they arrive
5. PROGRESS events - Every 2s heartbeat
6. COMPLETE or ERROR event - Always sent at end

Usage:
    streaming_manager = StreamingManager(timeout=3600, heartbeat_interval=2.0)

    async for event in streaming_manager.stream_with_timeout(token_generator):
        yield event.to_sse()
"""

import asyncio
import time
import logging
from typing import AsyncIterator, Dict, Any, Optional, Literal
from enum import Enum
from datetime import datetime
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class StreamEventType(str, Enum):
    """Types of streaming events."""
    STATUS = "status"
    TOKEN = "token"
    SOURCES = "sources"
    PROGRESS = "progress"
    COMPLETE = "complete"
    ERROR = "error"
    METADATA = "metadata"


class StreamStatus(str, Enum):
    """Status values for pipeline states."""
    STARTING = "starting"
    RETRIEVING = "retrieving"
    GENERATING = "generating"
    COMPLETE = "complete"
    ERROR = "error"


class StreamEvent(BaseModel):
    """
    Base streaming event model.

    All streaming events follow this structure for consistency.
    Frontend can parse the type field to handle each event appropriately.
    """
    type: StreamEventType
    data: Dict[str, Any] | str | None = None
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")

    def to_sse(self) -> str:
        """
        Convert to Server-Sent Events wire format.

        Returns:
            SSE-formatted string: "data: {json}\n\n"
        """
        return f"data: {self.model_dump_json()}\n\n"


class TokenEvent(StreamEvent):
    """Token streaming event - contains generated text chunk."""
    type: Literal[StreamEventType.TOKEN] = StreamEventType.TOKEN
    data: str  # The actual token/text chunk


class StatusEvent(StreamEvent):
    """Status update event - pipeline state changes."""
    type: Literal[StreamEventType.STATUS] = StreamEventType.STATUS
    data: Dict[str, Any]  # {"status": StreamStatus, "message": str}


class SourcesEvent(StreamEvent):
    """Sources information event - retrieved documents info."""
    type: Literal[StreamEventType.SOURCES] = StreamEventType.SOURCES
    data: Dict[str, Any]  # {"count": int, "preview": list, "retrieval_time": float}


class ProgressEvent(StreamEvent):
    """Progress/heartbeat event - periodic updates during generation."""
    type: Literal[StreamEventType.PROGRESS] = StreamEventType.PROGRESS
    data: Dict[str, Any]  # {"tokens": int, "time": float, "estimated_remaining": float}


class CompleteEvent(StreamEvent):
    """Completion event - success with metrics."""
    type: Literal[StreamEventType.COMPLETE] = StreamEventType.COMPLETE
    data: Dict[str, Any]  # {"total_time": float, "total_tokens": int, "provider": str}


class ErrorEvent(StreamEvent):
    """Error event - failure with details."""
    type: Literal[StreamEventType.ERROR] = StreamEventType.ERROR
    data: Dict[str, Any]  # {"message": str, "code": str, "recoverable": bool}


class MetadataEvent(StreamEvent):
    """Metadata event - additional information."""
    type: Literal[StreamEventType.METADATA] = StreamEventType.METADATA
    data: Dict[str, Any]  # Any additional metadata


class StreamingManager:
    """
    Manages streaming operations with timeout, heartbeat, and error handling.

    Features:
    - Per-token timeout to prevent hanging between tokens
    - Periodic heartbeat events for connection keepalive
    - Token buffering for efficiency (configurable)
    - Graceful cancellation support
    - Client disconnect detection
    - Automatic progress estimation

    Usage:
        manager = StreamingManager(timeout=3600, heartbeat_interval=2.0)

        async for event in manager.stream_with_timeout(token_generator):
            yield event.to_sse()
    """

    def __init__(
        self,
        timeout: float = 3600.0,  # Max stream duration (1 hour default)
        heartbeat_interval: float = 2.0,  # Heartbeat frequency
        buffer_size: int = 1,  # Tokens to buffer (1 = immediate, no buffering)
        disconnect_check_interval: float = 5.0  # Client connection check interval
    ):
        """
        Initialize streaming manager.

        Args:
            timeout: Maximum time for streaming operation (seconds)
            heartbeat_interval: Interval for sending heartbeat/progress events (seconds)
            buffer_size: Number of tokens to buffer before sending (1 = immediate)
            disconnect_check_interval: How often to check if client is still connected
        """
        self.timeout = timeout
        self.heartbeat_interval = heartbeat_interval
        self.buffer_size = buffer_size
        self.disconnect_check_interval = disconnect_check_interval

        # Runtime state
        self._start_time: Optional[float] = None
        self._last_heartbeat: Optional[float] = None
        self._last_disconnect_check: Optional[float] = None
        self._tokens_sent = 0
        self._cancelled = False

    def cancel(self) -> None:
        """Signal the stream to stop gracefully."""
        self._cancelled = True

    async def stream_with_timeout(
        self,
        stream_generator: AsyncIterator[str],
        send_heartbeat: bool = True
    ) -> AsyncIterator[StreamEvent]:
        """
        Wrap a stream generator with timeout and heartbeat support.

        Args:
            stream_generator: Async generator yielding tokens
            send_heartbeat: Whether to send periodic heartbeat/progress events

        Yields:
            StreamEvent objects for SSE encoding

        Contract:
        - Yields TOKEN events for text chunks
        - Yields PROGRESS events at heartbeat_interval
        - Yields ERROR event on timeout or error (terminates stream)
        - Never raises exceptions - errors become ERROR events
        """
        self._start_time = time.time()
        self._last_heartbeat = self._start_time
        self._last_disconnect_check = self._start_time
        self._tokens_sent = 0
        self._cancelled = False

        buffer = ""

        # Per-token timeout to prevent hanging (max 60s between tokens)
        per_token_timeout = min(60.0, self.timeout / 10)

        try:
            async for token in stream_generator:
                # Check if cancelled (client disconnected)
                if self._cancelled:
                    yield ErrorEvent(data={
                        "message": "Stream cancelled by client",
                        "code": "CLIENT_CANCELLED",
                        "recoverable": False
                    })
                    break

                # Check total timeout
                elapsed = time.time() - self._start_time
                if elapsed > self.timeout:
                    yield ErrorEvent(data={
                        "message": f"Streaming timeout after {self.timeout}s",
                        "code": "STREAMING_TIMEOUT",
                        "recoverable": False
                    })
                    break

                # Buffer tokens
                buffer += token
                self._tokens_sent += 1

                # Send buffered tokens when buffer is full or at sentence boundaries
                if len(buffer) >= self.buffer_size or token in [".", "!", "?", "\n"]:
                    yield TokenEvent(data=buffer)
                    buffer = ""

                # Send heartbeat/progress if needed
                if send_heartbeat:
                    current_time = time.time()
                    if current_time - self._last_heartbeat >= self.heartbeat_interval:
                        yield ProgressEvent(data={
                            "tokens": self._tokens_sent,
                            "time": round(current_time - self._start_time, 2),
                            "estimated_remaining": self._estimate_remaining_time()
                        })
                        self._last_heartbeat = current_time

            # Send any remaining buffered content
            if buffer:
                yield TokenEvent(data=buffer)

        except asyncio.TimeoutError:
            yield ErrorEvent(data={
                "message": "Stream interrupted due to timeout",
                "code": "TIMEOUT",
                "recoverable": False,
                "tokens_sent": self._tokens_sent
            })

        except asyncio.CancelledError:
            yield ErrorEvent(data={
                "message": "Stream cancelled",
                "code": "CANCELLED",
                "recoverable": False,
                "tokens_sent": self._tokens_sent
            })

        except Exception as e:
            logger.error(f"Stream error: {e}")
            yield ErrorEvent(data={
                "message": f"Stream error: {str(e)}",
                "code": "STREAMING_ERROR",
                "recoverable": False,
                "tokens_sent": self._tokens_sent
            })

    def _estimate_remaining_time(self) -> float:
        """
        Estimate remaining time based on token generation rate.

        Returns:
            Estimated seconds remaining (0.0 if cannot estimate)
        """
        if self._tokens_sent == 0:
            return 0.0

        elapsed = time.time() - self._start_time
        avg_time_per_token = elapsed / self._tokens_sent

        # Conservative estimate of total tokens (assume 2x current, max 500)
        estimated_total = min(self._tokens_sent * 2, 500)
        remaining_tokens = max(estimated_total - self._tokens_sent, 0)

        return round(remaining_tokens * avg_time_per_token, 2)


async def create_sse_stream(
    events: AsyncIterator[StreamEvent],
    include_heartbeat: bool = True
) -> AsyncIterator[str]:
    """
    Convert stream events to SSE format with optional keepalive.

    Args:
        events: Async iterator of StreamEvent objects
        include_heartbeat: Whether to include SSE comment heartbeats

    Yields:
        SSE-formatted strings
    """
    last_heartbeat = time.time()
    heartbeat_interval = 15.0  # SSE comment heartbeat every 15s

    try:
        async for event in events:
            # Send the event
            yield event.to_sse()

            # Send SSE comment heartbeat to keep connection alive
            if include_heartbeat:
                current_time = time.time()
                if current_time - last_heartbeat >= heartbeat_interval:
                    yield ": heartbeat\n\n"
                    last_heartbeat = current_time

    except asyncio.CancelledError:
        # Client disconnected
        yield ErrorEvent(data={
            "message": "Client disconnected",
            "code": "CLIENT_DISCONNECTED",
            "recoverable": False
        }).to_sse()

    except Exception as e:
        # Unexpected error
        logger.error(f"SSE stream error: {e}")
        yield ErrorEvent(data={
            "message": f"Unexpected streaming error: {str(e)}",
            "code": "UNEXPECTED_ERROR",
            "recoverable": False
        }).to_sse()


def create_error_stream(error_message: str, error_code: str = "ERROR") -> str:
    """
    Create an SSE error event string.

    Args:
        error_message: Error message to include
        error_code: Error code for categorization

    Returns:
        SSE-formatted error event string
    """
    event = ErrorEvent(data={
        "message": error_message,
        "code": error_code,
        "recoverable": False
    })
    return event.to_sse()
