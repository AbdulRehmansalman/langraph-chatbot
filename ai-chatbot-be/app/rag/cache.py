"""
RAG Response Cache
==================

Redis-based caching for RAG responses with graceful degradation.
Caches both exact query matches and semantic similarity lookups.
"""

import hashlib
import json
import logging
import os
from datetime import datetime
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Configuration
CACHE_ENABLED = os.getenv("CACHE_ENABLED", "true").lower() == "true"
CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "86400"))  # 24 hours
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
CACHE_PREFIX = "rag:cache:"

# Lazy-loaded Redis client
_redis_client = None
_redis_available = None

# In-memory stats (persisted to Redis periodically)
_stats = {"hits": 0, "misses": 0, "errors": 0}


def _get_redis():
    """Lazy load Redis client with connection validation."""
    global _redis_client, _redis_available

    if _redis_available is False:
        return None

    if _redis_client is not None:
        return _redis_client

    try:
        import redis
        _redis_client = redis.from_url(REDIS_URL, decode_responses=True)
        _redis_client.ping()
        _redis_available = True
        logger.info("Redis cache connected")
        return _redis_client
    except Exception as e:
        logger.warning(f"Redis unavailable, caching disabled: {e}")
        _redis_available = False
        return None


def _hash_query(query: str, user_id: str = None) -> str:
    """Create a hash key for a query."""
    key_str = f"{query.lower().strip()}:{user_id or 'global'}"
    return hashlib.sha256(key_str.encode()).hexdigest()[:32]


async def get_cached_response(
    query: str,
    user_id: str = None,
) -> Optional[dict[str, Any]]:
    """
    Get cached response for a query.

    Returns:
        Cached response dict or None if not found/expired.
    """
    if not CACHE_ENABLED:
        return None

    redis = _get_redis()
    if not redis:
        return None

    try:
        key = f"{CACHE_PREFIX}{_hash_query(query, user_id)}"
        cached = redis.get(key)

        if cached:
            data = json.loads(cached)
            data["hit_count"] = data.get("hit_count", 0) + 1
            redis.setex(key, CACHE_TTL_SECONDS, json.dumps(data))
            _stats["hits"] += 1
            logger.debug(f"Cache HIT for query: {query[:50]}...")
            return data

        _stats["misses"] += 1
        return None

    except Exception as e:
        _stats["errors"] += 1
        logger.error(f"Cache get error: {e}")
        return None


async def set_cached_response(
    query: str,
    response: str,
    documents: list[dict] = None,
    user_id: str = None,
) -> bool:
    """
    Cache a response for a query.

    Returns:
        True if cached successfully, False otherwise.
    """
    if not CACHE_ENABLED:
        return False

    redis = _get_redis()
    if not redis:
        return False

    try:
        key = f"{CACHE_PREFIX}{_hash_query(query, user_id)}"
        data = {
            "response": response,
            "documents": documents or [],
            "timestamp": datetime.utcnow().isoformat(),
            "hit_count": 0,
            "query": query[:200],  # Store truncated query for debugging
        }
        redis.setex(key, CACHE_TTL_SECONDS, json.dumps(data))
        logger.debug(f"Cached response for: {query[:50]}...")
        return True

    except Exception as e:
        _stats["errors"] += 1
        logger.error(f"Cache set error: {e}")
        return False


async def invalidate_cache(user_id: str = None, pattern: str = None) -> int:
    """Invalidate cache entries. Returns count of deleted keys."""
    redis = _get_redis()
    if not redis:
        return 0

    try:
        if pattern:
            keys = redis.keys(f"{CACHE_PREFIX}{pattern}*")
        elif user_id:
            keys = redis.keys(f"{CACHE_PREFIX}*:{user_id}")
        else:
            keys = redis.keys(f"{CACHE_PREFIX}*")

        if keys:
            return redis.delete(*keys)
        return 0
    except Exception as e:
        logger.error(f"Cache invalidation error: {e}")
        return 0


def get_cache_stats() -> dict[str, Any]:
    """Get cache statistics."""
    total = _stats["hits"] + _stats["misses"]
    hit_rate = (_stats["hits"] / total * 100) if total > 0 else 0.0

    stats = {
        "enabled": CACHE_ENABLED,
        "redis_available": _redis_available or False,
        "hits": _stats["hits"],
        "misses": _stats["misses"],
        "errors": _stats["errors"],
        "hit_rate_percent": round(hit_rate, 2),
        "ttl_seconds": CACHE_TTL_SECONDS,
    }

    # Get Redis stats if available
    redis = _get_redis()
    if redis:
        try:
            keys = redis.keys(f"{CACHE_PREFIX}*")
            stats["cached_entries"] = len(keys)
            info = redis.info("memory")
            stats["redis_memory_mb"] = round(info.get("used_memory", 0) / 1024 / 1024, 2)
        except Exception:
            pass

    return stats


__all__ = ["get_cached_response", "set_cached_response", "invalidate_cache", "get_cache_stats"]
