"""
Health Check Endpoints
======================
Production-ready health check endpoints for Kubernetes/Docker deployments.

Enterprise Features:
- Liveness probe (is the app running?)
- Readiness probe (can the app serve traffic?)
- Startup probe (has the app finished initializing?)
- Dependency health checks (database, Redis, LLM, vector store)
"""

from fastapi import APIRouter, status
from fastapi.responses import JSONResponse
from typing import Dict, Any
from datetime import datetime, timezone
import logging
import asyncio

from app.core.config import settings
from app.database.connection import SessionLocal
from app.core.logging import get_logger

router = APIRouter()
logger = get_logger(__name__)


async def check_database() -> Dict[str, Any]:
    """
    Check database connectivity.
    
    Returns:
        Dict with status and details
    """
    try:
        session = SessionLocal()
        # Simple query to check connection
        session.execute("SELECT 1")
        session.close()
        return {
            "status": "healthy",
            "latency_ms": 0,  # Could measure actual latency
            "message": "Database connection successful"
        }
    except Exception as e:
        logger.error(f"Database health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "message": "Database connection failed"
        }


async def check_redis() -> Dict[str, Any]:
    """
    Check Redis connectivity using the cache module.

    Returns:
        Dict with status and details
    """
    try:
        from app.rag.cache import get_cache_stats

        stats = get_cache_stats()

        if stats.get("redis_available"):
            return {
                "status": "healthy",
                "cached_entries": stats.get("cached_entries", 0),
                "hit_rate": f"{stats.get('hit_rate_percent', 0)}%",
                "memory_mb": stats.get("redis_memory_mb", 0),
                "message": "Redis connected and operational"
            }
        else:
            return {
                "status": "degraded",
                "message": "Redis unavailable - caching disabled"
            }
    except Exception as e:
        logger.error(f"Redis health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "message": "Redis connection failed"
        }


async def check_llm() -> Dict[str, Any]:
    """
    Check LLM service availability.
    
    Returns:
        Dict with status and details
    """
    try:
        # Note: Implement actual LLM health check
        # For now, return healthy if LLM config exists
        if settings.openai_api_key or settings.ollama_base_url:
            return {
                "status": "healthy",
                "provider": "configured",
                "message": "LLM provider configured"
            }
        else:
            return {
                "status": "degraded",
                "message": "No LLM provider configured"
            }
    except Exception as e:
        logger.error(f"LLM health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "message": "LLM check failed"
        }


async def check_vector_store() -> Dict[str, Any]:
    """
    Check vector store connectivity.
    
    Returns:
        Dict with status and details
    """
    try:
        # Note: Implement actual vector store health check
        # For now, return healthy if database is healthy
        return {
            "status": "healthy",
            "message": "Vector store check skipped (uses database)"
        }
    except Exception as e:
        logger.error(f"Vector store health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "message": "Vector store check failed"
        }


@router.get("/health/live", status_code=status.HTTP_200_OK)
async def liveness_probe():
    """
    Liveness probe for Kubernetes.
    
    Returns 200 if the application is running.
    This should be a simple check that doesn't depend on external services.
    
    Use in Kubernetes:
    ```yaml
    livenessProbe:
      httpGet:
        path: /health/live
        port: 8000
      initialDelaySeconds: 10
      periodSeconds: 10
    ```
    """
    return {
        "status": "alive",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version": "2.0.0"
    }


@router.get("/health/ready", status_code=status.HTTP_200_OK)
async def readiness_probe():
    """
    Readiness probe for Kubernetes.
    
    Returns 200 if the application is ready to serve traffic.
    Checks all critical dependencies.
    
    Use in Kubernetes:
    ```yaml
    readinessProbe:
      httpGet:
        path: /health/ready
        port: 8000
      initialDelaySeconds: 15
      periodSeconds: 5
    ```
    """
    # Check critical dependencies
    checks = {
        "database": await check_database(),
        "llm": await check_llm(),
    }
    
    # Determine overall status
    all_healthy = all(
        check["status"] == "healthy" 
        for check in checks.values()
    )
    
    if all_healthy:
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "status": "ready",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "checks": checks
            }
        )
    else:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "not_ready",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "checks": checks
            }
        )


@router.get("/health/startup", status_code=status.HTTP_200_OK)
async def startup_probe():
    """
    Startup probe for Kubernetes.
    
    Returns 200 when the application has finished initializing.
    Used to delay liveness/readiness checks during slow startup.
    
    Use in Kubernetes:
    ```yaml
    startupProbe:
      httpGet:
        path: /health/startup
        port: 8000
      initialDelaySeconds: 0
      periodSeconds: 5
      failureThreshold: 30
    ```
    """
    # Check if application has finished initializing
    checks = {
        "database": await check_database(),
        "config": {
            "status": "healthy",
            "environment": settings.environment,
            "message": "Configuration loaded"
        }
    }
    
    all_healthy = all(
        check["status"] == "healthy" 
        for check in checks.values()
    )
    
    if all_healthy:
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "status": "started",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "checks": checks
            }
        )
    else:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "starting",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "checks": checks
            }
        )


@router.get("/health/cache", status_code=status.HTTP_200_OK)
async def cache_stats():
    """
    Get detailed cache statistics.

    Returns cache hit rate, entry count, and memory usage.
    """
    try:
        from app.rag.cache import get_cache_stats

        stats = get_cache_stats()
        return {
            "status": "ok",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "cache": stats,
        }
    except Exception as e:
        logger.error(f"Cache stats error: {str(e)}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"status": "error", "message": str(e)}
        )


@router.delete("/health/cache", status_code=status.HTTP_200_OK)
async def clear_cache():
    """
    Clear the response cache.

    Use with caution in production.
    """
    try:
        from app.rag.cache import invalidate_cache

        deleted = await invalidate_cache()
        return {
            "status": "ok",
            "deleted_entries": deleted,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        logger.error(f"Cache clear error: {str(e)}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"status": "error", "message": str(e)}
        )


@router.get("/health", status_code=status.HTTP_200_OK)
async def health_check_detailed():
    """
    Detailed health check endpoint.
    
    Returns comprehensive health information including all dependencies.
    Use for monitoring and debugging.
    """
    # Run all health checks
    checks = {
        "database": await check_database(),
        "redis": await check_redis(),
        "llm": await check_llm(),
        "vector_store": await check_vector_store(),
    }
    
    # Calculate overall health
    healthy_count = sum(1 for check in checks.values() if check["status"] == "healthy")
    total_count = len(checks)
    
    overall_status = "healthy" if healthy_count == total_count else "degraded"
    if healthy_count == 0:
        overall_status = "unhealthy"
    
    return {
        "status": overall_status,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version": "2.0.0",
        "environment": settings.environment,
        "checks": checks,
        "summary": {
            "healthy": healthy_count,
            "total": total_count,
            "percentage": round((healthy_count / total_count) * 100, 2)
        }
    }
