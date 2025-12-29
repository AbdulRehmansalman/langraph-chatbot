"""
Tests for RAG Response Cache
============================

Tests the Redis caching layer with graceful degradation.
Run with: pytest tests/unit/test_cache.py -v
"""

import os
import pytest
from unittest.mock import patch, MagicMock


class TestCacheModule:
    """Test cache module functions."""

    def test_hash_query_consistency(self):
        """Test that query hashing is consistent."""
        from app.rag.cache import _hash_query

        query = "What is the vacation policy?"
        hash1 = _hash_query(query)
        hash2 = _hash_query(query)

        assert hash1 == hash2
        assert len(hash1) == 32  # SHA256 truncated to 32 chars

    def test_hash_query_case_insensitive(self):
        """Test that hashing is case-insensitive."""
        from app.rag.cache import _hash_query

        hash1 = _hash_query("What is the policy?")
        hash2 = _hash_query("WHAT IS THE POLICY?")

        assert hash1 == hash2

    def test_hash_query_with_user_id(self):
        """Test that user_id affects the hash."""
        from app.rag.cache import _hash_query

        hash1 = _hash_query("query", user_id="user1")
        hash2 = _hash_query("query", user_id="user2")
        hash3 = _hash_query("query", user_id=None)

        assert hash1 != hash2
        assert hash1 != hash3

    def test_get_cache_stats(self):
        """Test that cache stats are returned correctly."""
        from app.rag.cache import get_cache_stats

        stats = get_cache_stats()

        assert "enabled" in stats
        assert "redis_available" in stats
        assert "hits" in stats
        assert "misses" in stats
        assert "errors" in stats
        assert "hit_rate_percent" in stats
        assert "ttl_seconds" in stats


class TestCacheDisabled:
    """Test cache behavior when disabled."""

    @pytest.fixture(autouse=True)
    def disable_cache(self):
        """Disable cache for these tests."""
        with patch.dict(os.environ, {"CACHE_ENABLED": "false"}):
            # Reload module to pick up new env var
            import importlib
            from app.rag import cache
            importlib.reload(cache)
            yield
            # Restore
            with patch.dict(os.environ, {"CACHE_ENABLED": "true"}):
                importlib.reload(cache)

    @pytest.mark.asyncio
    async def test_get_returns_none_when_disabled(self):
        """Test that get returns None when cache is disabled."""
        import importlib
        from app.rag import cache
        importlib.reload(cache)

        # Verify cache is disabled
        assert cache.CACHE_ENABLED == False

        result = await cache.get_cached_response("test query")
        assert result is None

    @pytest.mark.asyncio
    async def test_set_returns_false_when_disabled(self):
        """Test that set returns False when cache is disabled."""
        import importlib
        from app.rag import cache
        importlib.reload(cache)

        result = await cache.set_cached_response("query", "response")
        assert result == False


class TestCacheWithMockRedis:
    """Test cache with mocked Redis."""

    @pytest.fixture
    def mock_redis(self):
        """Create a mock Redis client."""
        mock = MagicMock()
        mock.ping.return_value = True
        mock.get.return_value = None
        mock.setex.return_value = True
        mock.keys.return_value = []
        mock.delete.return_value = 0
        mock.info.return_value = {"used_memory": 1024 * 1024}
        return mock

    @pytest.mark.asyncio
    async def test_cache_miss(self, mock_redis):
        """Test cache miss behavior."""
        import importlib
        from app.rag import cache

        with patch.dict(os.environ, {"CACHE_ENABLED": "true"}):
            importlib.reload(cache)
            cache._redis_client = mock_redis
            cache._redis_available = True

            result = await cache.get_cached_response("test query")

            assert result is None
            mock_redis.get.assert_called_once()

    @pytest.mark.asyncio
    async def test_cache_hit(self, mock_redis):
        """Test cache hit behavior."""
        import json
        import importlib
        from app.rag import cache

        cached_data = {
            "response": "Cached answer",
            "documents": [],
            "timestamp": "2024-01-01T00:00:00",
            "hit_count": 0,
        }
        mock_redis.get.return_value = json.dumps(cached_data)

        with patch.dict(os.environ, {"CACHE_ENABLED": "true"}):
            importlib.reload(cache)
            cache._redis_client = mock_redis
            cache._redis_available = True

            result = await cache.get_cached_response("test query")

            assert result is not None
            assert result["response"] == "Cached answer"
            assert result["hit_count"] == 1  # Incremented

    @pytest.mark.asyncio
    async def test_cache_set(self, mock_redis):
        """Test setting cache entries."""
        import importlib
        from app.rag import cache

        with patch.dict(os.environ, {"CACHE_ENABLED": "true"}):
            importlib.reload(cache)
            cache._redis_client = mock_redis
            cache._redis_available = True

            result = await cache.set_cached_response(
                query="test query",
                response="test response",
                documents=[{"id": "doc1", "content": "test"}],
            )

            assert result == True
            mock_redis.setex.assert_called_once()

    @pytest.mark.asyncio
    async def test_cache_invalidation(self, mock_redis):
        """Test cache invalidation."""
        import importlib
        from app.rag import cache

        mock_redis.keys.return_value = ["rag:cache:key1", "rag:cache:key2"]
        mock_redis.delete.return_value = 2

        with patch.dict(os.environ, {"CACHE_ENABLED": "true"}):
            importlib.reload(cache)
            cache._redis_client = mock_redis
            cache._redis_available = True

            deleted = await cache.invalidate_cache()

            assert deleted == 2


class TestCacheGracefulDegradation:
    """Test graceful degradation when Redis is unavailable."""

    @pytest.mark.asyncio
    async def test_get_handles_redis_failure(self):
        """Test that get handles Redis connection failure gracefully."""
        import importlib
        from app.rag import cache

        # Force Redis unavailable
        cache._redis_available = False
        cache._redis_client = None

        # Should return None without raising exception
        result = await cache.get_cached_response("test query")
        assert result is None

    @pytest.mark.asyncio
    async def test_set_handles_redis_failure(self):
        """Test that set handles Redis connection failure gracefully."""
        import importlib
        from app.rag import cache

        # Force Redis unavailable
        cache._redis_available = False
        cache._redis_client = None

        # Should return False without raising exception
        result = await cache.set_cached_response("query", "response")
        assert result == False

    def test_stats_when_redis_unavailable(self):
        """Test that stats work when Redis is unavailable."""
        import importlib
        from app.rag import cache

        cache._redis_available = False
        cache._redis_client = None

        stats = cache.get_cache_stats()

        assert stats["redis_available"] == False
        assert "cached_entries" not in stats  # Only present when Redis works


class TestCacheTTL:
    """Test cache TTL configuration."""

    def test_default_ttl(self):
        """Test default TTL is 24 hours."""
        import importlib
        from app.rag import cache

        with patch.dict(os.environ, {}, clear=False):
            importlib.reload(cache)
            assert cache.CACHE_TTL_SECONDS == 86400  # 24 hours

    def test_custom_ttl(self):
        """Test custom TTL from environment."""
        import importlib
        from app.rag import cache

        with patch.dict(os.environ, {"CACHE_TTL_SECONDS": "3600"}):
            importlib.reload(cache)
            assert cache.CACHE_TTL_SECONDS == 3600  # 1 hour


class TestCacheIntegration:
    """Integration tests (require running Redis)."""

    @pytest.fixture
    def redis_available(self):
        """Check if Redis is available."""
        try:
            import redis
            r = redis.from_url("redis://localhost:6379")
            r.ping()
            return True
        except Exception:
            return False

    @pytest.mark.asyncio
    async def test_full_cache_cycle(self, redis_available):
        """Test full cache cycle: miss -> set -> hit."""
        if not redis_available:
            pytest.skip("Redis not available")

        import importlib
        from app.rag import cache

        with patch.dict(os.environ, {"CACHE_ENABLED": "true"}):
            importlib.reload(cache)

            test_query = "integration_test_query_12345"
            test_response = "This is a cached response"

            # Clean up any existing entry
            await cache.invalidate_cache(pattern=cache._hash_query(test_query))

            # Should be a miss
            result = await cache.get_cached_response(test_query)
            assert result is None

            # Set the cache
            success = await cache.set_cached_response(
                query=test_query,
                response=test_response,
                documents=[],
            )
            assert success == True

            # Should be a hit
            result = await cache.get_cached_response(test_query)
            assert result is not None
            assert result["response"] == test_response

            # Clean up
            await cache.invalidate_cache(pattern=cache._hash_query(test_query))


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
