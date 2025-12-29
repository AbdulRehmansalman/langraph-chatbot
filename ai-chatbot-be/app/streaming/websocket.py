"""
WebSocket Streaming Infrastructure
==================================
Production WebSocket implementation for real-time bidirectional chat streaming.

Features:
- Bidirectional communication
- Connection state management
- Automatic reconnection support
- Heartbeat/ping-pong for connection health
- Message queuing and buffering
- Rate limiting
- Graceful shutdown

Message Types:
- chat_message: User sends a message
- chat_response: Token streaming response
- node_update: LangGraph node execution updates
- tool_update: Tool execution notifications
- meeting_update: Calendar meeting scheduling
- error: Error messages
- ping/pong: Connection keepalive
- status: Connection status updates
"""

import asyncio
import json
import time
import logging
from typing import Any, Optional, Callable, Awaitable
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
from contextlib import asynccontextmanager

from fastapi import WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# =============================================================================
# MESSAGE TYPES
# =============================================================================

class WSMessageType(str, Enum):
    """WebSocket message types."""
    # Client -> Server
    CHAT_MESSAGE = "chat_message"
    CANCEL_STREAM = "cancel_stream"
    PING = "ping"

    # Server -> Client
    CHAT_RESPONSE = "chat_response"
    TOKEN = "token"
    NODE_UPDATE = "node_update"
    TOOL_UPDATE = "tool_update"
    MEETING_UPDATE = "meeting_update"
    SOURCES = "sources"
    COMPLETE = "complete"
    ERROR = "error"
    STATUS = "status"
    PONG = "pong"


class WSMessage(BaseModel):
    """WebSocket message format."""
    type: WSMessageType
    data: Any = None
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    message_id: Optional[str] = None


# =============================================================================
# CONNECTION MANAGER
# =============================================================================

@dataclass
class WebSocketConnection:
    """Represents an active WebSocket connection."""
    websocket: WebSocket
    user_id: str
    thread_id: Optional[str] = None
    connected_at: datetime = field(default_factory=datetime.utcnow)
    last_activity: datetime = field(default_factory=datetime.utcnow)
    is_streaming: bool = False
    message_count: int = 0


class ConnectionManager:
    """
    Manages WebSocket connections for multiple users and threads.

    Features:
    - User and thread-based connection tracking
    - Connection lifecycle management
    - Broadcast capabilities
    - Connection health monitoring
    """

    def __init__(self):
        # Active connections: {user_id: {thread_id: WebSocketConnection}}
        self._connections: dict[str, dict[str, WebSocketConnection]] = {}
        self._lock = asyncio.Lock()

    async def connect(
        self,
        websocket: WebSocket,
        user_id: str,
        thread_id: Optional[str] = None,
    ) -> WebSocketConnection:
        """
        Accept and register a new WebSocket connection.

        Args:
            websocket: FastAPI WebSocket instance
            user_id: User ID
            thread_id: Optional thread ID for conversation

        Returns:
            WebSocketConnection object
        """
        await websocket.accept()

        connection = WebSocketConnection(
            websocket=websocket,
            user_id=user_id,
            thread_id=thread_id or "default",
        )

        async with self._lock:
            if user_id not in self._connections:
                self._connections[user_id] = {}
            self._connections[user_id][connection.thread_id] = connection

        logger.info(
            f"WebSocket connected: user={user_id}, thread={thread_id}, "
            f"total_connections={self.connection_count}"
        )

        return connection

    async def disconnect(self, user_id: str, thread_id: Optional[str] = None):
        """
        Disconnect and remove a WebSocket connection.

        Args:
            user_id: User ID
            thread_id: Thread ID (default thread if not specified)
        """
        thread_id = thread_id or "default"

        async with self._lock:
            if user_id in self._connections:
                if thread_id in self._connections[user_id]:
                    del self._connections[user_id][thread_id]
                    logger.info(f"WebSocket disconnected: user={user_id}, thread={thread_id}")

                # Clean up empty user entries
                if not self._connections[user_id]:
                    del self._connections[user_id]

    def get_connection(
        self,
        user_id: str,
        thread_id: Optional[str] = None,
    ) -> Optional[WebSocketConnection]:
        """Get a specific connection."""
        thread_id = thread_id or "default"
        return self._connections.get(user_id, {}).get(thread_id)

    def get_user_connections(self, user_id: str) -> list[WebSocketConnection]:
        """Get all connections for a user."""
        return list(self._connections.get(user_id, {}).values())

    @property
    def connection_count(self) -> int:
        """Get total number of active connections."""
        return sum(len(threads) for threads in self._connections.values())

    async def send_message(
        self,
        user_id: str,
        message: WSMessage,
        thread_id: Optional[str] = None,
    ) -> bool:
        """
        Send a message to a specific connection.

        Args:
            user_id: User ID
            message: Message to send
            thread_id: Thread ID (sends to default if not specified)

        Returns:
            True if message sent successfully
        """
        connection = self.get_connection(user_id, thread_id)
        if not connection:
            return False

        try:
            await connection.websocket.send_json(message.model_dump())
            connection.last_activity = datetime.utcnow()
            connection.message_count += 1
            return True
        except Exception as e:
            logger.error(f"Failed to send message to {user_id}: {e}")
            await self.disconnect(user_id, thread_id)
            return False

    async def broadcast_to_user(self, user_id: str, message: WSMessage):
        """Broadcast a message to all connections for a user."""
        connections = self.get_user_connections(user_id)
        for conn in connections:
            await self.send_message(user_id, message, conn.thread_id)

    async def send_token(
        self,
        user_id: str,
        token: str,
        thread_id: Optional[str] = None,
    ):
        """Send a token event."""
        message = WSMessage(
            type=WSMessageType.TOKEN,
            data={"content": token},
        )
        await self.send_message(user_id, message, thread_id)

    async def send_node_update(
        self,
        user_id: str,
        node: str,
        status: str,
        message: Optional[str] = None,
        thread_id: Optional[str] = None,
    ):
        """Send a node execution update."""
        msg = WSMessage(
            type=WSMessageType.NODE_UPDATE,
            data={
                "node": node,
                "status": status,
                "message": message,
            },
        )
        await self.send_message(user_id, msg, thread_id)

    async def send_tool_update(
        self,
        user_id: str,
        tool: str,
        status: str,
        result: Optional[Any] = None,
        thread_id: Optional[str] = None,
    ):
        """Send a tool execution update."""
        message = WSMessage(
            type=WSMessageType.TOOL_UPDATE,
            data={
                "tool": tool,
                "status": status,
                "result": result,
            },
        )
        await self.send_message(user_id, message, thread_id)

    async def send_meeting_update(
        self,
        user_id: str,
        scheduled: bool,
        details: dict,
        thread_id: Optional[str] = None,
    ):
        """Send a meeting scheduling update."""
        message = WSMessage(
            type=WSMessageType.MEETING_UPDATE,
            data={
                "scheduled": scheduled,
                "details": details,
            },
        )
        await self.send_message(user_id, message, thread_id)

    async def send_error(
        self,
        user_id: str,
        error_message: str,
        error_code: str = "ERROR",
        recoverable: bool = False,
        thread_id: Optional[str] = None,
    ):
        """Send an error message."""
        message = WSMessage(
            type=WSMessageType.ERROR,
            data={
                "message": error_message,
                "code": error_code,
                "recoverable": recoverable,
            },
        )
        await self.send_message(user_id, message, thread_id)

    async def send_complete(
        self,
        user_id: str,
        data: dict,
        thread_id: Optional[str] = None,
    ):
        """Send completion message."""
        message = WSMessage(
            type=WSMessageType.COMPLETE,
            data=data,
        )
        await self.send_message(user_id, message, thread_id)


# Global connection manager
connection_manager = ConnectionManager()


# =============================================================================
# WEBSOCKET HANDLER
# =============================================================================

@dataclass
class WebSocketConfig:
    """WebSocket handler configuration."""
    ping_interval: float = 30.0
    ping_timeout: float = 10.0
    max_message_size: int = 1048576  # 1MB
    rate_limit_messages: int = 60  # per minute
    rate_limit_window: float = 60.0  # seconds


class WebSocketHandler:
    """
    Handles WebSocket communication for a single connection.

    Features:
    - Message parsing and routing
    - Ping/pong keepalive
    - Rate limiting
    - Graceful disconnection
    """

    def __init__(
        self,
        websocket: WebSocket,
        user_id: str,
        thread_id: Optional[str] = None,
        config: Optional[WebSocketConfig] = None,
        message_handler: Optional[Callable[[dict], Awaitable[None]]] = None,
    ):
        self.websocket = websocket
        self.user_id = user_id
        self.thread_id = thread_id or "default"
        self.config = config or WebSocketConfig()
        self.message_handler = message_handler

        self._connection: Optional[WebSocketConnection] = None
        self._running = False
        self._rate_limit_tokens: list[float] = []

    async def connect(self) -> bool:
        """Establish the WebSocket connection."""
        try:
            self._connection = await connection_manager.connect(
                self.websocket,
                self.user_id,
                self.thread_id,
            )
            self._running = True

            # Send initial status
            await self.send(WSMessage(
                type=WSMessageType.STATUS,
                data={"connected": True, "user_id": self.user_id, "thread_id": self.thread_id},
            ))

            return True

        except Exception as e:
            logger.error(f"WebSocket connection failed: {e}")
            return False

    async def disconnect(self):
        """Gracefully disconnect."""
        self._running = False
        await connection_manager.disconnect(self.user_id, self.thread_id)

    async def send(self, message: WSMessage) -> bool:
        """Send a message through this connection."""
        return await connection_manager.send_message(
            self.user_id,
            message,
            self.thread_id,
        )

    def _check_rate_limit(self) -> bool:
        """Check if rate limit is exceeded."""
        now = time.time()

        # Remove old tokens
        self._rate_limit_tokens = [
            t for t in self._rate_limit_tokens
            if now - t < self.config.rate_limit_window
        ]

        if len(self._rate_limit_tokens) >= self.config.rate_limit_messages:
            return False

        self._rate_limit_tokens.append(now)
        return True

    async def handle_message(self, raw_message: str) -> Optional[WSMessage]:
        """
        Parse and handle an incoming message.

        Args:
            raw_message: Raw JSON message string

        Returns:
            Response message or None
        """
        try:
            data = json.loads(raw_message)
            message_type = WSMessageType(data.get("type", ""))

            # Handle ping
            if message_type == WSMessageType.PING:
                return WSMessage(type=WSMessageType.PONG)

            # Check rate limit for chat messages
            if message_type == WSMessageType.CHAT_MESSAGE:
                if not self._check_rate_limit():
                    return WSMessage(
                        type=WSMessageType.ERROR,
                        data={
                            "message": "Rate limit exceeded",
                            "code": "RATE_LIMIT",
                            "recoverable": True,
                        },
                    )

            # Handle cancel
            if message_type == WSMessageType.CANCEL_STREAM:
                if self._connection:
                    self._connection.is_streaming = False
                return WSMessage(
                    type=WSMessageType.STATUS,
                    data={"streaming": False, "cancelled": True},
                )

            # Delegate to message handler for chat messages
            if message_type == WSMessageType.CHAT_MESSAGE and self.message_handler:
                await self.message_handler(data)
                return None

            return None

        except json.JSONDecodeError:
            return WSMessage(
                type=WSMessageType.ERROR,
                data={"message": "Invalid JSON", "code": "INVALID_JSON"},
            )
        except ValueError:
            return WSMessage(
                type=WSMessageType.ERROR,
                data={"message": "Unknown message type", "code": "UNKNOWN_TYPE"},
            )
        except Exception as e:
            logger.error(f"Message handling error: {e}")
            return WSMessage(
                type=WSMessageType.ERROR,
                data={"message": str(e), "code": "HANDLER_ERROR"},
            )

    async def run_receive_loop(self):
        """Main loop for receiving and handling messages."""
        try:
            while self._running:
                try:
                    raw_message = await asyncio.wait_for(
                        self.websocket.receive_text(),
                        timeout=self.config.ping_timeout * 2,
                    )

                    response = await self.handle_message(raw_message)
                    if response:
                        await self.send(response)

                except asyncio.TimeoutError:
                    # Send ping to check connection
                    await self.send(WSMessage(type=WSMessageType.PING))

        except WebSocketDisconnect:
            logger.info(f"WebSocket disconnected: user={self.user_id}")
        except Exception as e:
            logger.error(f"WebSocket receive error: {e}")
        finally:
            await self.disconnect()

    async def run_ping_loop(self):
        """Background loop for sending periodic pings."""
        try:
            while self._running:
                await asyncio.sleep(self.config.ping_interval)
                if self._running:
                    try:
                        await self.websocket.send_json(
                            WSMessage(type=WSMessageType.PING).model_dump()
                        )
                    except Exception:
                        self._running = False
                        break
        except Exception as e:
            logger.error(f"Ping loop error: {e}")


# =============================================================================
# STREAMING ADAPTER
# =============================================================================

class WebSocketStreamAdapter:
    """
    Adapter for streaming LangGraph events over WebSocket.

    Converts LangGraph streaming events to WebSocket messages.
    """

    def __init__(
        self,
        user_id: str,
        thread_id: Optional[str] = None,
    ):
        self.user_id = user_id
        self.thread_id = thread_id
        self._manager = connection_manager

    async def stream_tokens(self, token_generator):
        """Stream tokens from a generator to WebSocket."""
        connection = self._manager.get_connection(self.user_id, self.thread_id)
        if connection:
            connection.is_streaming = True

        try:
            async for token in token_generator:
                # Check if cancelled
                if connection and not connection.is_streaming:
                    break

                await self._manager.send_token(
                    self.user_id,
                    token,
                    self.thread_id,
                )

        finally:
            if connection:
                connection.is_streaming = False

    async def on_node_start(self, node: str, message: Optional[str] = None):
        """Handle node start event."""
        await self._manager.send_node_update(
            self.user_id,
            node,
            "start",
            message,
            self.thread_id,
        )

    async def on_node_end(self, node: str, message: Optional[str] = None):
        """Handle node end event."""
        await self._manager.send_node_update(
            self.user_id,
            node,
            "end",
            message,
            self.thread_id,
        )

    async def on_tool_start(self, tool: str):
        """Handle tool start event."""
        await self._manager.send_tool_update(
            self.user_id,
            tool,
            "start",
            thread_id=self.thread_id,
        )

    async def on_tool_end(self, tool: str, result: Any = None):
        """Handle tool end event."""
        await self._manager.send_tool_update(
            self.user_id,
            tool,
            "end",
            result,
            self.thread_id,
        )

    async def on_meeting_scheduled(self, details: dict):
        """Handle meeting scheduled event."""
        await self._manager.send_meeting_update(
            self.user_id,
            scheduled=True,
            details=details,
            thread_id=self.thread_id,
        )

    async def on_complete(
        self,
        total_time: float,
        total_tokens: int,
        provider: str = "unknown",
    ):
        """Handle completion event."""
        await self._manager.send_complete(
            self.user_id,
            {
                "total_time": total_time,
                "total_tokens": total_tokens,
                "provider": provider,
                "status": "complete",
            },
            self.thread_id,
        )

    async def on_error(
        self,
        message: str,
        code: str = "ERROR",
        recoverable: bool = False,
    ):
        """Handle error event."""
        await self._manager.send_error(
            self.user_id,
            message,
            code,
            recoverable,
            self.thread_id,
        )


# =============================================================================
# FASTAPI INTEGRATION
# =============================================================================

@asynccontextmanager
async def websocket_connection(
    websocket: WebSocket,
    user_id: str,
    thread_id: Optional[str] = None,
    message_handler: Optional[Callable[[dict], Awaitable[None]]] = None,
):
    """
    Context manager for WebSocket connections.

    Usage:
        @app.websocket("/ws/chat")
        async def websocket_endpoint(websocket: WebSocket, token: str):
            user_id = verify_token(token)

            async with websocket_connection(websocket, user_id) as handler:
                await handler.run_receive_loop()
    """
    handler = WebSocketHandler(
        websocket=websocket,
        user_id=user_id,
        thread_id=thread_id,
        message_handler=message_handler,
    )

    try:
        if await handler.connect():
            yield handler
        else:
            raise Exception("Failed to connect WebSocket")
    finally:
        await handler.disconnect()


# Export public API
__all__ = [
    "WSMessageType",
    "WSMessage",
    "WebSocketConnection",
    "ConnectionManager",
    "connection_manager",
    "WebSocketConfig",
    "WebSocketHandler",
    "WebSocketStreamAdapter",
    "websocket_connection",
]
