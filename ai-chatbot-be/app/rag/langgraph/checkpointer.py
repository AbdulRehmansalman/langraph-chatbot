"""
PostgreSQL Checkpointer Configuration
======================================

Provides thread-safe checkpointing setup for LangGraph state persistence.
Enables:
- Cross-session memory retrieval
- Time-travel debugging
- State recovery after failures
- Automatic checkpoint cleanup
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from contextvars import ContextVar
from datetime import datetime, timedelta
from typing import AsyncIterator, Optional

from langgraph.checkpoint.memory import MemorySaver

logger = logging.getLogger(__name__)

# Context variable for thread-safe checkpointer access
_checkpointer_context: ContextVar[Optional[any]] = ContextVar("checkpointer", default=None)

# Global async connection pool (initialized once)
_connection_pool = None
_pool_lock = asyncio.Lock()


class CheckpointerConfig:
    """Configuration for checkpointer behavior."""

    def __init__(
        self,
        connection_string: Optional[str] = None,
        pool_min_size: int = 2,
        pool_max_size: int = 10,
        checkpoint_retention_days: int = 30,
        enable_cleanup: bool = True,
    ):
        self.connection_string = connection_string
        self.pool_min_size = pool_min_size
        self.pool_max_size = pool_max_size
        self.checkpoint_retention_days = checkpoint_retention_days
        self.enable_cleanup = enable_cleanup


async def _get_connection_pool(config: CheckpointerConfig):
    """
    Get or create the async connection pool (singleton).

    Uses asyncio.Lock for thread-safe initialization.
    """
    global _connection_pool

    if _connection_pool is not None:
        return _connection_pool

    async with _pool_lock:
        # Double-check after acquiring lock
        if _connection_pool is not None:
            return _connection_pool

        try:
            from psycopg_pool import AsyncConnectionPool

            _connection_pool = AsyncConnectionPool(
                conninfo=config.connection_string,
                min_size=config.pool_min_size,
                max_size=config.pool_max_size,
                open=False,  # Open explicitly below
            )
            await _connection_pool.open()
            logger.info(
                f"PostgreSQL connection pool created: "
                f"min={config.pool_min_size}, max={config.pool_max_size}"
            )
            return _connection_pool

        except ImportError:
            logger.error("psycopg_pool not installed. Install with: pip install psycopg[pool]")
            return None
        except Exception as e:
            logger.error(f"Failed to create connection pool: {e}")
            return None


async def get_postgres_checkpointer_async(
    config: Optional[CheckpointerConfig] = None,
    connection_string: Optional[str] = None,
) -> any:
    """
    Get PostgreSQL checkpointer with connection pooling (async version).

    This is the recommended method for production use.

    Args:
        config: Checkpointer configuration
        connection_string: PostgreSQL connection URL (if config not provided)

    Returns:
        AsyncPostgresSaver or MemorySaver as fallback
    """
    # Check context variable first
    existing = _checkpointer_context.get()
    if existing is not None:
        return existing

    # Build config if needed
    if config is None:
        config = CheckpointerConfig(connection_string=connection_string)

    if not config.connection_string:
        logger.warning("No connection string provided, using MemorySaver")
        saver = MemorySaver()
        _checkpointer_context.set(saver)
        return saver

    try:
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

        pool = await _get_connection_pool(config)
        if pool is None:
            logger.warning("Connection pool unavailable, using MemorySaver")
            saver = MemorySaver()
            _checkpointer_context.set(saver)
            return saver

        saver = AsyncPostgresSaver(pool)
        await saver.setup()  # Create tables if needed
        _checkpointer_context.set(saver)
        logger.info("AsyncPostgresSaver initialized with connection pool")
        return saver

    except ImportError:
        logger.warning(
            "langgraph-checkpoint-postgres not installed. "
            "Install with: pip install langgraph-checkpoint-postgres"
        )
    except Exception as e:
        logger.error(f"Failed to initialize PostgreSQL checkpointer: {e}")

    # Fallback
    saver = MemorySaver()
    _checkpointer_context.set(saver)
    return saver


def get_postgres_checkpointer(connection_string: Optional[str] = None):
    """
    Get PostgreSQL checkpointer (sync version for backward compatibility).

    Note: Prefer get_postgres_checkpointer_async for production use.

    Args:
        connection_string: PostgreSQL connection URL

    Returns:
        PostgresSaver or MemorySaver as fallback
    """
    if not connection_string:
        logger.info("Using in-memory checkpointer (no persistence)")
        return MemorySaver()

    try:
        from langgraph.checkpoint.postgres import PostgresSaver

        saver = PostgresSaver.from_conn_string(connection_string)
        logger.info("PostgreSQL checkpointer initialized (sync)")
        return saver

    except ImportError:
        logger.warning(
            "langgraph-checkpoint-postgres not installed. "
            "Install with: pip install langgraph-checkpoint-postgres"
        )
    except Exception as e:
        logger.error(f"Failed to initialize PostgreSQL checkpointer: {e}")

    logger.info("Using in-memory checkpointer (no persistence)")
    return MemorySaver()


def get_memory_checkpointer():
    """Get in-memory checkpointer for development/testing."""
    return MemorySaver()


async def setup_postgres_schema(connection_string: str) -> bool:
    """
    Set up the required PostgreSQL schema for checkpointing.

    Args:
        connection_string: PostgreSQL connection URL

    Returns:
        True if successful, False otherwise
    """
    try:
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
        from psycopg_pool import AsyncConnectionPool

        async with AsyncConnectionPool(conninfo=connection_string) as pool:
            saver = AsyncPostgresSaver(pool)
            await saver.setup()
            logger.info("PostgreSQL checkpoint schema created successfully")
            return True

    except ImportError:
        logger.error("langgraph-checkpoint-postgres or psycopg not installed")
        return False
    except Exception as e:
        logger.error(f"Failed to set up PostgreSQL schema: {e}")
        return False


async def cleanup_old_checkpoints(
    connection_string: str,
    retention_days: int = 30,
    batch_size: int = 1000,
) -> int:
    """
    Clean up checkpoints older than retention period.

    This should be run periodically (e.g., daily cron job).

    Args:
        connection_string: PostgreSQL connection URL
        retention_days: Days to retain checkpoints
        batch_size: Number of checkpoints to delete per batch

    Returns:
        Number of checkpoints deleted
    """
    try:
        from psycopg_pool import AsyncConnectionPool

        cutoff_date = datetime.utcnow() - timedelta(days=retention_days)
        total_deleted = 0

        async with AsyncConnectionPool(conninfo=connection_string) as pool:
            async with pool.connection() as conn:
                while True:
                    # Delete in batches to avoid long locks
                    result = await conn.execute(
                        """
                        DELETE FROM checkpoints
                        WHERE checkpoint_id IN (
                            SELECT checkpoint_id FROM checkpoints
                            WHERE created_at < %s
                            LIMIT %s
                        )
                        RETURNING checkpoint_id
                        """,
                        (cutoff_date, batch_size),
                    )
                    deleted = result.rowcount
                    total_deleted += deleted
                    await conn.commit()

                    if deleted < batch_size:
                        break

                    # Yield control to allow other operations
                    await asyncio.sleep(0.1)

        logger.info(
            f"Checkpoint cleanup complete: {total_deleted} checkpoints deleted "
            f"(retention: {retention_days} days)"
        )
        return total_deleted

    except Exception as e:
        logger.error(f"Checkpoint cleanup failed: {e}")
        return 0


class CheckpointerManager:
    """
    Manager for handling checkpointer lifecycle and operations.

    Provides utilities for:
    - State snapshot management
    - Time-travel queries
    - State cleanup
    - Thread-safe access
    """

    def __init__(
        self,
        checkpointer,
        config: Optional[CheckpointerConfig] = None,
    ):
        """
        Initialize checkpointer manager.

        Args:
            checkpointer: LangGraph checkpointer instance
            config: Optional configuration
        """
        self.checkpointer = checkpointer
        self.config = config or CheckpointerConfig()

    async def get_state_history(
        self,
        thread_id: str,
        limit: int = 10,
    ) -> list[dict]:
        """
        Get state history for a thread (time-travel).

        Args:
            thread_id: Thread identifier
            limit: Maximum number of states to return

        Returns:
            List of historical states
        """
        try:
            config = {"configurable": {"thread_id": thread_id}}
            states = []

            # Check if checkpointer supports async listing
            if hasattr(self.checkpointer, "alist"):
                async for state in self.checkpointer.alist(config, limit=limit):
                    states.append({
                        "checkpoint_id": state.config.get("configurable", {}).get("checkpoint_id"),
                        "thread_id": thread_id,
                        "timestamp": state.metadata.get("created_at"),
                        "node": state.metadata.get("source"),
                        "step": state.metadata.get("step"),
                    })
            elif hasattr(self.checkpointer, "list"):
                for state in self.checkpointer.list(config, limit=limit):
                    states.append({
                        "checkpoint_id": state.config.get("configurable", {}).get("checkpoint_id"),
                        "thread_id": thread_id,
                        "timestamp": state.metadata.get("created_at"),
                        "node": state.metadata.get("source"),
                        "step": state.metadata.get("step"),
                    })

            return states

        except Exception as e:
            logger.error(f"Failed to get state history: {e}")
            return []

    async def get_state_at_checkpoint(
        self,
        thread_id: str,
        checkpoint_id: str,
    ) -> Optional[dict]:
        """
        Get state at a specific checkpoint (time-travel debugging).

        Args:
            thread_id: Thread identifier
            checkpoint_id: Specific checkpoint ID

        Returns:
            State at that checkpoint or None
        """
        try:
            config = {
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_id": checkpoint_id,
                }
            }

            if hasattr(self.checkpointer, "aget"):
                state = await self.checkpointer.aget(config)
            elif hasattr(self.checkpointer, "get"):
                state = self.checkpointer.get(config)
            else:
                return None

            if state:
                return state.values

            return None

        except Exception as e:
            logger.error(f"Failed to get state at checkpoint: {e}")
            return None

    async def delete_thread_history(self, thread_id: str) -> bool:
        """
        Delete all checkpoints for a thread (GDPR compliance).

        Args:
            thread_id: Thread identifier

        Returns:
            True if successful
        """
        try:
            config = {"configurable": {"thread_id": thread_id}}

            # Try async delete first
            if hasattr(self.checkpointer, "adelete"):
                await self.checkpointer.adelete(config)
                logger.info(f"Deleted history for thread {thread_id}")
                return True

            # Try sync delete
            if hasattr(self.checkpointer, "delete"):
                self.checkpointer.delete(config)
                logger.info(f"Deleted history for thread {thread_id}")
                return True

            logger.warning("Checkpointer does not support deletion")
            return False

        except Exception as e:
            logger.error(f"Failed to delete thread history: {e}")
            return False

    async def get_thread_stats(self, thread_id: str) -> dict:
        """
        Get statistics for a thread.

        Args:
            thread_id: Thread identifier

        Returns:
            Thread statistics
        """
        try:
            states = await self.get_state_history(thread_id, limit=1000)

            if not states:
                return {"exists": False}

            return {
                "exists": True,
                "checkpoint_count": len(states),
                "first_checkpoint": states[-1].get("timestamp") if states else None,
                "last_checkpoint": states[0].get("timestamp") if states else None,
                "nodes_executed": list({s.get("node") for s in states if s.get("node")}),
            }

        except Exception as e:
            logger.error(f"Failed to get thread stats: {e}")
            return {"exists": False, "error": str(e)}


@asynccontextmanager
async def checkpointer_context(
    connection_string: Optional[str] = None,
    config: Optional[CheckpointerConfig] = None,
) -> AsyncIterator[any]:
    """
    Context manager for checkpointer with automatic cleanup.

    Usage:
        async with checkpointer_context(conn_string) as saver:
            graph = create_graph(checkpointer=saver)
            await graph.ainvoke(...)
    """
    saver = await get_postgres_checkpointer_async(config, connection_string)
    try:
        yield saver
    finally:
        # Clear context variable
        _checkpointer_context.set(None)


async def close_connection_pool():
    """
    Close the global connection pool.

    Call this during application shutdown.
    """
    global _connection_pool

    if _connection_pool is not None:
        await _connection_pool.close()
        _connection_pool = None
        logger.info("PostgreSQL connection pool closed")
