from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
import time
import uuid
from typing import Callable

from app.core.logging import get_logger, app_logger

logger = get_logger(__name__)


class LoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for comprehensive request/response logging"""
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Generate unique request ID
        request_id = str(uuid.uuid4())
        
        # Add request ID to request state
        request.state.request_id = request_id
        
        # Start timing
        start_time = time.time()
        
        # Extract user info if available
        user_id = None
        if hasattr(request.state, 'user_id'):
            user_id = request.state.user_id
        
        # Log incoming request
        logger.info(
            f"Incoming {request.method} request to {request.url.path}",
            extra={
                "method": request.method,
                "endpoint": request.url.path,
                "query_params": str(request.query_params),
                "user_agent": request.headers.get("user-agent"),
                "client_host": request.client.host if request.client else None,
                "user_id": user_id,
                "request_id": request_id
            }
        )
        
        # Process request
        try:
            response = await call_next(request)
            
            # Calculate execution time
            execution_time = time.time() - start_time
            
            # Log response
            logger.info(
                f"Response {response.status_code} for {request.method} {request.url.path} in {execution_time:.3f}s",
                extra={
                    "method": request.method,
                    "endpoint": request.url.path,
                    "status_code": response.status_code,
                    "execution_time": execution_time,
                    "user_id": user_id,
                    "request_id": request_id
                }
            )
            
            # Add request ID to response headers
            response.headers["X-Request-ID"] = request_id
            
            return response
            
        except Exception as e:
            # Calculate execution time
            execution_time = time.time() - start_time
            
            # Log error
            app_logger.log_error(
                logger,
                e,
                context={
                    "method": request.method,
                    "endpoint": request.url.path,
                    "execution_time": execution_time
                },
                user_id=user_id,
                request_id=request_id
            )
            
            # Re-raise the exception
            raise


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Middleware to add request ID to all requests"""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Generate request ID if not already present
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        request.state.request_id = request_id
        
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        
        return response


class CORSLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware to log CORS requests"""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Log CORS preflight requests
        if request.method == "OPTIONS":
            logger.info(
                f"CORS preflight request to {request.url.path}",
                extra={
                    "method": request.method,
                    "endpoint": request.url.path,
                    "origin": request.headers.get("origin"),
                    "access_control_request_method": request.headers.get("access-control-request-method"),
                    "access_control_request_headers": request.headers.get("access-control-request-headers")
                }
            )
        
        response = await call_next(request)
        return response