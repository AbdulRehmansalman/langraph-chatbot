"""
Observability & Monitoring Module
=================================
Production-grade observability infrastructure including:
- LangSmith integration for LLM tracing
- Structured logging with context
- Performance monitoring and metrics
- Distributed tracing support
- Alert thresholds and health monitoring

Enterprise Features:
- Automatic trace propagation
- Cost tracking per request
- Latency percentile tracking (p50, p95, p99)
- Error rate monitoring with alerts
- Token usage analytics
"""

import os
import time
import logging
import asyncio
from typing import Any, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from functools import wraps
from contextlib import asynccontextmanager
from collections import defaultdict
import statistics

import structlog
from structlog.types import FilteringBoundLogger

# LangSmith imports
try:
    from langsmith import Client as LangSmithClient
    from langsmith.run_trees import RunTree
    from langchain_core.tracers.langchain import LangChainTracer
    LANGSMITH_AVAILABLE = True
except ImportError:
    LANGSMITH_AVAILABLE = False
    LangSmithClient = None
    RunTree = None
    LangChainTracer = None

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class ObservabilityConfig:
    """Observability configuration."""
    # LangSmith settings
    langsmith_enabled: bool = True
    langsmith_project: str = "langraph-chatbot-production"
    langsmith_tracing_level: str = "all"  # all, errors, samples
    langsmith_sample_rate: float = 0.1

    # Logging settings
    log_level: str = "INFO"
    log_format: str = "json"
    include_timestamps: bool = True
    include_trace_ids: bool = True

    # Metrics settings
    metrics_enabled: bool = True
    metrics_export_interval: int = 60

    # Performance thresholds
    latency_warning_ms: float = 5000.0
    latency_critical_ms: float = 10000.0
    error_rate_warning: float = 0.03  # 3%
    error_rate_critical: float = 0.05  # 5%
    cost_warning_usd: float = 0.50
    cost_critical_usd: float = 1.00


def load_observability_config() -> ObservabilityConfig:
    """Load observability configuration from environment and settings."""
    try:
        from app.core.config import settings
        return ObservabilityConfig(
            langsmith_enabled=bool(settings.langsmith_api_key),
            langsmith_project=settings.langsmith_project or "langraph-chatbot",
            log_level=settings.log_level,
            log_format=settings.log_format,
        )
    except Exception:
        return ObservabilityConfig()


# =============================================================================
# LANGSMITH INTEGRATION
# =============================================================================

class LangSmithManager:
    """
    LangSmith integration manager for LLM tracing.

    Features:
    - Automatic trace collection
    - Run feedback and evaluation
    - Cost tracking
    - Error categorization
    """

    _instance: Optional["LangSmithManager"] = None
    _client: Optional[LangSmithClient] = None
    _tracer: Optional[LangChainTracer] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, "_initialized"):
            return
        self._initialized = True
        self.config = load_observability_config()
        self._setup_langsmith()

    def _setup_langsmith(self):
        """Initialize LangSmith client and tracer."""
        if not LANGSMITH_AVAILABLE:
            logger.warning("LangSmith not available - install langsmith package")
            return

        if not self.config.langsmith_enabled:
            logger.info("LangSmith tracing disabled")
            return

        try:
            # Set environment variables for LangChain integration
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            os.environ["LANGCHAIN_PROJECT"] = self.config.langsmith_project

            from app.core.config import settings
            if settings.langsmith_api_key:
                os.environ["LANGCHAIN_API_KEY"] = settings.langsmith_api_key

            if settings.langsmith_endpoint:
                os.environ["LANGCHAIN_ENDPOINT"] = settings.langsmith_endpoint

            # Initialize client
            self._client = LangSmithClient()
            self._tracer = LangChainTracer(project_name=self.config.langsmith_project)

            logger.info(
                f"LangSmith initialized - project: {self.config.langsmith_project}"
            )

        except Exception as e:
            logger.error(f"Failed to initialize LangSmith: {e}")
            self._client = None
            self._tracer = None

    @property
    def client(self) -> Optional[LangSmithClient]:
        """Get LangSmith client."""
        return self._client

    @property
    def tracer(self) -> Optional[LangChainTracer]:
        """Get LangChain tracer for callbacks."""
        return self._tracer

    def get_callbacks(self) -> list:
        """Get LangChain callbacks for tracing."""
        if self._tracer:
            return [self._tracer]
        return []

    def should_trace(self, is_error: bool = False) -> bool:
        """Determine if this request should be traced based on config."""
        if not self._client:
            return False

        level = self.config.langsmith_tracing_level

        if level == "all":
            return True
        elif level == "errors":
            return is_error
        elif level == "samples":
            import random
            return random.random() < self.config.langsmith_sample_rate

        return False

    async def log_feedback(
        self,
        run_id: str,
        key: str,
        score: float,
        comment: Optional[str] = None,
    ):
        """Log feedback for a run."""
        if not self._client:
            return

        try:
            self._client.create_feedback(
                run_id=run_id,
                key=key,
                score=score,
                comment=comment,
            )
        except Exception as e:
            logger.error(f"Failed to log LangSmith feedback: {e}")


def get_langsmith_manager() -> LangSmithManager:
    """Get the LangSmith manager singleton."""
    return LangSmithManager()


# =============================================================================
# STRUCTURED LOGGING
# =============================================================================

def configure_structured_logging(config: Optional[ObservabilityConfig] = None):
    """Configure structlog for structured logging."""
    config = config or load_observability_config()

    processors = [
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    if config.log_format == "json":
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> FilteringBoundLogger:
    """Get a structured logger."""
    return structlog.get_logger(name)


# =============================================================================
# PERFORMANCE MONITORING
# =============================================================================

@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""
    request_count: int = 0
    error_count: int = 0
    total_latency_ms: float = 0.0
    latencies: list = field(default_factory=list)
    token_usage: int = 0
    cost_usd: float = 0.0
    node_times: dict = field(default_factory=dict)
    tool_calls: dict = field(default_factory=dict)
    cache_hits: int = 0
    cache_misses: int = 0

    def add_latency(self, latency_ms: float):
        """Add a latency measurement."""
        self.latencies.append(latency_ms)
        self.total_latency_ms += latency_ms
        # Keep only last 1000 measurements for percentile calculations
        if len(self.latencies) > 1000:
            self.latencies = self.latencies[-1000:]

    def get_percentile(self, p: float) -> float:
        """Get latency percentile."""
        if not self.latencies:
            return 0.0
        sorted_latencies = sorted(self.latencies)
        index = int(len(sorted_latencies) * p / 100)
        return sorted_latencies[min(index, len(sorted_latencies) - 1)]

    @property
    def p50(self) -> float:
        """Get 50th percentile latency."""
        return self.get_percentile(50)

    @property
    def p95(self) -> float:
        """Get 95th percentile latency."""
        return self.get_percentile(95)

    @property
    def p99(self) -> float:
        """Get 99th percentile latency."""
        return self.get_percentile(99)

    @property
    def average_latency_ms(self) -> float:
        """Get average latency."""
        if self.request_count == 0:
            return 0.0
        return self.total_latency_ms / self.request_count

    @property
    def error_rate(self) -> float:
        """Get error rate."""
        if self.request_count == 0:
            return 0.0
        return self.error_count / self.request_count

    @property
    def cache_hit_rate(self) -> float:
        """Get cache hit rate."""
        total = self.cache_hits + self.cache_misses
        if total == 0:
            return 0.0
        return self.cache_hits / total

    def to_dict(self) -> dict:
        """Export metrics as dictionary."""
        return {
            "request_count": self.request_count,
            "error_count": self.error_count,
            "error_rate": self.error_rate,
            "latency": {
                "average_ms": self.average_latency_ms,
                "p50_ms": self.p50,
                "p95_ms": self.p95,
                "p99_ms": self.p99,
            },
            "tokens": self.token_usage,
            "cost_usd": self.cost_usd,
            "cache_hit_rate": self.cache_hit_rate,
            "node_times": self.node_times,
            "tool_calls": self.tool_calls,
        }


class PerformanceMonitor:
    """
    Production performance monitoring system.

    Features:
    - Request/error counting
    - Latency percentile tracking
    - Token usage monitoring
    - Cost tracking
    - Node execution time analysis
    - Alert threshold checking
    """

    _instance: Optional["PerformanceMonitor"] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, "_initialized"):
            return
        self._initialized = True

        self.config = load_observability_config()
        self._metrics = PerformanceMetrics()
        self._hourly_metrics: dict[str, PerformanceMetrics] = defaultdict(PerformanceMetrics)
        self._lock = asyncio.Lock()
        self._alerts: list[dict] = []

    @property
    def metrics(self) -> PerformanceMetrics:
        """Get current metrics."""
        return self._metrics

    async def record_request(
        self,
        latency_ms: float,
        is_error: bool = False,
        tokens: int = 0,
        cost_usd: float = 0.0,
        node_times: Optional[dict] = None,
        tool_calls: Optional[list] = None,
        cache_hit: bool = False,
    ):
        """Record a request's metrics."""
        async with self._lock:
            self._metrics.request_count += 1
            self._metrics.add_latency(latency_ms)
            self._metrics.token_usage += tokens
            self._metrics.cost_usd += cost_usd

            if is_error:
                self._metrics.error_count += 1

            if cache_hit:
                self._metrics.cache_hits += 1
            else:
                self._metrics.cache_misses += 1

            # Track node times
            if node_times:
                for node, time_ms in node_times.items():
                    if node not in self._metrics.node_times:
                        self._metrics.node_times[node] = []
                    self._metrics.node_times[node].append(time_ms)

            # Track tool calls
            if tool_calls:
                for tool in tool_calls:
                    self._metrics.tool_calls[tool] = self._metrics.tool_calls.get(tool, 0) + 1

            # Record hourly metrics
            hour_key = datetime.utcnow().strftime("%Y-%m-%d-%H")
            hourly = self._hourly_metrics[hour_key]
            hourly.request_count += 1
            hourly.add_latency(latency_ms)
            if is_error:
                hourly.error_count += 1

            # Check alert thresholds
            await self._check_alerts(latency_ms, is_error, cost_usd)

    async def _check_alerts(
        self,
        latency_ms: float,
        is_error: bool,
        cost_usd: float,
    ):
        """Check and emit alerts based on thresholds."""
        alerts = []

        # Latency alerts
        if latency_ms > self.config.latency_critical_ms:
            alerts.append({
                "level": "critical",
                "type": "latency",
                "message": f"Critical latency: {latency_ms:.0f}ms",
                "threshold_ms": self.config.latency_critical_ms,
                "actual_ms": latency_ms,
            })
        elif latency_ms > self.config.latency_warning_ms:
            alerts.append({
                "level": "warning",
                "type": "latency",
                "message": f"High latency: {latency_ms:.0f}ms",
                "threshold_ms": self.config.latency_warning_ms,
                "actual_ms": latency_ms,
            })

        # Error rate alerts (check after every 100 requests)
        if self._metrics.request_count % 100 == 0:
            error_rate = self._metrics.error_rate
            if error_rate > self.config.error_rate_critical:
                alerts.append({
                    "level": "critical",
                    "type": "error_rate",
                    "message": f"Critical error rate: {error_rate:.1%}",
                    "threshold": self.config.error_rate_critical,
                    "actual": error_rate,
                })
            elif error_rate > self.config.error_rate_warning:
                alerts.append({
                    "level": "warning",
                    "type": "error_rate",
                    "message": f"High error rate: {error_rate:.1%}",
                    "threshold": self.config.error_rate_warning,
                    "actual": error_rate,
                })

        # Cost alerts
        if cost_usd > self.config.cost_critical_usd:
            alerts.append({
                "level": "critical",
                "type": "cost",
                "message": f"Critical cost per request: ${cost_usd:.2f}",
                "threshold_usd": self.config.cost_critical_usd,
                "actual_usd": cost_usd,
            })
        elif cost_usd > self.config.cost_warning_usd:
            alerts.append({
                "level": "warning",
                "type": "cost",
                "message": f"High cost per request: ${cost_usd:.2f}",
                "threshold_usd": self.config.cost_warning_usd,
                "actual_usd": cost_usd,
            })

        # Log and store alerts
        for alert in alerts:
            alert["timestamp"] = datetime.utcnow().isoformat()
            self._alerts.append(alert)

            if alert["level"] == "critical":
                logger.critical(f"ALERT: {alert['message']}")
            else:
                logger.warning(f"ALERT: {alert['message']}")

        # Keep only last 100 alerts
        if len(self._alerts) > 100:
            self._alerts = self._alerts[-100:]

    def get_recent_alerts(self, hours: int = 24) -> list[dict]:
        """Get alerts from the last N hours."""
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        return [
            alert for alert in self._alerts
            if datetime.fromisoformat(alert["timestamp"]) > cutoff
        ]

    def get_hourly_stats(self, hours: int = 24) -> dict:
        """Get hourly statistics."""
        result = {}
        now = datetime.utcnow()

        for i in range(hours):
            hour = now - timedelta(hours=i)
            key = hour.strftime("%Y-%m-%d-%H")
            if key in self._hourly_metrics:
                metrics = self._hourly_metrics[key]
                result[key] = {
                    "requests": metrics.request_count,
                    "errors": metrics.error_count,
                    "error_rate": metrics.error_rate,
                    "avg_latency_ms": metrics.average_latency_ms,
                    "p95_latency_ms": metrics.p95,
                }
            else:
                result[key] = {
                    "requests": 0,
                    "errors": 0,
                    "error_rate": 0,
                    "avg_latency_ms": 0,
                    "p95_latency_ms": 0,
                }

        return result

    def get_node_stats(self) -> dict:
        """Get average execution times per node."""
        result = {}
        for node, times in self._metrics.node_times.items():
            if times:
                result[node] = {
                    "count": len(times),
                    "avg_ms": statistics.mean(times),
                    "min_ms": min(times),
                    "max_ms": max(times),
                    "p95_ms": sorted(times)[int(len(times) * 0.95)] if len(times) > 20 else max(times),
                }
        return result

    def get_health_status(self) -> dict:
        """Get overall health status."""
        error_rate = self._metrics.error_rate
        p95 = self._metrics.p95

        # Determine health status
        if error_rate > self.config.error_rate_critical or p95 > self.config.latency_critical_ms:
            status = "unhealthy"
        elif error_rate > self.config.error_rate_warning or p95 > self.config.latency_warning_ms:
            status = "degraded"
        else:
            status = "healthy"

        return {
            "status": status,
            "timestamp": datetime.utcnow().isoformat(),
            "metrics": self._metrics.to_dict(),
            "recent_alerts_count": len(self.get_recent_alerts(hours=1)),
        }


def get_performance_monitor() -> PerformanceMonitor:
    """Get the performance monitor singleton."""
    return PerformanceMonitor()


# =============================================================================
# DECORATORS & CONTEXT MANAGERS
# =============================================================================

def trace_execution(
    name: Optional[str] = None,
    capture_args: bool = False,
    capture_result: bool = False,
):
    """
    Decorator to trace function execution with timing and logging.

    Args:
        name: Custom name for the trace (defaults to function name)
        capture_args: Whether to log function arguments
        capture_result: Whether to log function result

    Usage:
        @trace_execution(name="my_operation")
        async def my_function(arg1, arg2):
            ...
    """
    def decorator(func: Callable):
        trace_name = name or func.__name__

        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            log = get_logger(trace_name)

            log_context = {"function": trace_name}
            if capture_args:
                log_context["args"] = str(args)[:200]
                log_context["kwargs"] = str(kwargs)[:200]

            log.info("Starting execution", **log_context)

            try:
                result = await func(*args, **kwargs)

                duration_ms = (time.time() - start_time) * 1000
                log_context["duration_ms"] = duration_ms

                if capture_result:
                    log_context["result"] = str(result)[:200]

                log.info("Execution completed", **log_context)
                return result

            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                log.error(
                    "Execution failed",
                    error=str(e),
                    error_type=type(e).__name__,
                    duration_ms=duration_ms,
                    **log_context,
                )
                raise

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            log = get_logger(trace_name)

            log_context = {"function": trace_name}
            if capture_args:
                log_context["args"] = str(args)[:200]
                log_context["kwargs"] = str(kwargs)[:200]

            log.info("Starting execution", **log_context)

            try:
                result = func(*args, **kwargs)

                duration_ms = (time.time() - start_time) * 1000
                log_context["duration_ms"] = duration_ms

                if capture_result:
                    log_context["result"] = str(result)[:200]

                log.info("Execution completed", **log_context)
                return result

            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                log.error(
                    "Execution failed",
                    error=str(e),
                    error_type=type(e).__name__,
                    duration_ms=duration_ms,
                    **log_context,
                )
                raise

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


@asynccontextmanager
async def trace_context(
    name: str,
    metadata: Optional[dict] = None,
):
    """
    Async context manager for tracing a code block.

    Usage:
        async with trace_context("my_operation", {"user_id": "123"}):
            # ... code to trace ...
    """
    start_time = time.time()
    log = get_logger(name)

    log_context = metadata or {}
    log_context["trace_name"] = name

    log.info("Trace started", **log_context)

    try:
        yield
        duration_ms = (time.time() - start_time) * 1000
        log.info("Trace completed", duration_ms=duration_ms, **log_context)

    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        log.error(
            "Trace failed",
            error=str(e),
            error_type=type(e).__name__,
            duration_ms=duration_ms,
            **log_context,
        )
        raise


# =============================================================================
# HEALTH CHECK ENDPOINT DATA
# =============================================================================

def get_observability_health() -> dict:
    """Get health status of observability components."""
    langsmith = get_langsmith_manager()
    monitor = get_performance_monitor()

    return {
        "langsmith": {
            "enabled": langsmith.config.langsmith_enabled,
            "connected": langsmith.client is not None,
            "project": langsmith.config.langsmith_project,
        },
        "performance_monitor": {
            "status": monitor.get_health_status()["status"],
            "requests_tracked": monitor.metrics.request_count,
        },
        "logging": {
            "level": load_observability_config().log_level,
            "format": load_observability_config().log_format,
        },
    }


# =============================================================================
# INITIALIZATION
# =============================================================================

def initialize_observability():
    """Initialize all observability components."""
    config = load_observability_config()

    # Configure structured logging
    configure_structured_logging(config)

    # Initialize LangSmith
    get_langsmith_manager()

    # Initialize performance monitor
    get_performance_monitor()

    logger.info("Observability initialized")


# Export public API
__all__ = [
    "ObservabilityConfig",
    "LangSmithManager",
    "get_langsmith_manager",
    "PerformanceMonitor",
    "get_performance_monitor",
    "PerformanceMetrics",
    "configure_structured_logging",
    "get_logger",
    "trace_execution",
    "trace_context",
    "get_observability_health",
    "initialize_observability",
]
