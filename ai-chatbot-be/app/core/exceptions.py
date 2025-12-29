from fastapi import HTTPException, Request, Response, status
from fastapi.responses import JSONResponse
from fastapi.exception_handlers import http_exception_handler
from typing import Union, Dict, Any, Optional, List
from pydantic import BaseModel
import uuid
import traceback
from datetime import datetime, timezone

from app.core.logging import get_logger, app_logger

logger = get_logger(__name__)


class ErrorDetail(BaseModel):
    """Detailed error information"""
    code: str
    message: str
    field: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


class ErrorResponse(BaseModel):
    """Standardized error response format"""
    success: bool = False
    error: ErrorDetail
    request_id: str
    timestamp: str
    path: Optional[str] = None
    method: Optional[str] = None
    status_code: int
    trace_id: Optional[str] = None


class BaseCustomException(Exception):
    """Base exception for custom application errors"""
    
    def __init__(
        self,
        code: str,
        message: str,
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
        details: Optional[Dict[str, Any]] = None,
        field: Optional[str] = None
    ):
        self.code = code
        self.message = message
        self.status_code = status_code
        self.details = details
        self.field = field
        super().__init__(self.message)


class ValidationException(BaseCustomException):
    """Validation error exception"""
    
    def __init__(self, message: str, field: str = None, details: Dict[str, Any] = None):
        super().__init__(
            code="VALIDATION_ERROR",
            message=message,
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            field=field,
            details=details
        )


class AuthenticationException(BaseCustomException):
    """Authentication error exception"""
    
    def __init__(self, message: str = "Authentication failed", details: Dict[str, Any] = None):
        super().__init__(
            code="AUTHENTICATION_ERROR",
            message=message,
            status_code=status.HTTP_401_UNAUTHORIZED,
            details=details
        )


class AuthorizationException(BaseCustomException):
    """Authorization error exception"""
    
    def __init__(self, message: str = "Access denied", details: Dict[str, Any] = None):
        super().__init__(
            code="AUTHORIZATION_ERROR",
            message=message,
            status_code=status.HTTP_403_FORBIDDEN,
            details=details
        )


class ResourceNotFoundException(BaseCustomException):
    """Resource not found exception"""
    
    def __init__(self, resource: str, identifier: str = None, details: Dict[str, Any] = None):
        message = f"{resource} not found"
        if identifier:
            message = f"{resource} with identifier '{identifier}' not found"
        
        super().__init__(
            code="RESOURCE_NOT_FOUND",
            message=message,
            status_code=status.HTTP_404_NOT_FOUND,
            details=details
        )


class ConflictException(BaseCustomException):
    """Resource conflict exception"""
    
    def __init__(self, message: str, details: Dict[str, Any] = None):
        super().__init__(
            code="RESOURCE_CONFLICT",
            message=message,
            status_code=status.HTTP_409_CONFLICT,
            details=details
        )


class ExternalServiceException(BaseCustomException):
    """External service error exception"""
    
    def __init__(self, service: str, message: str = None, details: Dict[str, Any] = None):
        if not message:
            message = f"External service '{service}' is unavailable"
        
        super().__init__(
            code="EXTERNAL_SERVICE_ERROR",
            message=message,
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            details=details or {"service": service}
        )


class RateLimitException(BaseCustomException):
    """Rate limit exceeded exception"""
    
    def __init__(self, message: str = "Rate limit exceeded", details: Dict[str, Any] = None):
        super().__init__(
            code="RATE_LIMIT_EXCEEDED",
            message=message,
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            details=details
        )


class DatabaseException(BaseCustomException):
    """Database operation error exception"""
    
    def __init__(self, operation: str, table: str = None, details: Dict[str, Any] = None):
        message = f"Database operation '{operation}' failed"
        if table:
            message = f"Database operation '{operation}' failed for table '{table}'"
        
        super().__init__(
            code="DATABASE_ERROR",
            message=message,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            details=details or {"operation": operation, "table": table}
        )


async def custom_exception_handler(request: Request, exc: BaseCustomException) -> JSONResponse:
    """Handle custom exceptions and return standardized error responses"""
    
    request_id = str(uuid.uuid4())
    trace_id = str(uuid.uuid4()) if exc.status_code >= 500 else None
    
    # Log the error with context
    app_logger.log_error(
        logger,
        exc,
        context={
            "path": str(request.url.path),
            "method": request.method,
            "status_code": exc.status_code,
            "error_code": exc.code
        },
        request_id=request_id
    )
    
    error_response = ErrorResponse(
        error=ErrorDetail(
            code=exc.code,
            message=exc.message,
            field=exc.field,
            details=exc.details
        ),
        request_id=request_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
        path=str(request.url.path),
        method=request.method,
        status_code=exc.status_code,
        trace_id=trace_id
    )
    
    return JSONResponse(
        status_code=exc.status_code,
        content=error_response.model_dump()
    )


async def http_exception_middleware(request: Request, exc: HTTPException) -> JSONResponse:
    """Handle HTTPExceptions and return standardized error responses"""
    
    request_id = str(uuid.uuid4())
    trace_id = str(uuid.uuid4()) if exc.status_code >= 500 else None
    
    # Map status codes to error codes
    error_code_mapping = {
        400: "BAD_REQUEST",
        401: "AUTHENTICATION_ERROR",
        403: "AUTHORIZATION_ERROR",
        404: "RESOURCE_NOT_FOUND",
        405: "METHOD_NOT_ALLOWED",
        406: "NOT_ACCEPTABLE",
        409: "RESOURCE_CONFLICT",
        410: "RESOURCE_GONE",
        413: "PAYLOAD_TOO_LARGE",
        415: "UNSUPPORTED_MEDIA_TYPE",
        422: "VALIDATION_ERROR",
        429: "RATE_LIMIT_EXCEEDED",
        500: "INTERNAL_SERVER_ERROR",
        501: "NOT_IMPLEMENTED",
        502: "BAD_GATEWAY",
        503: "SERVICE_UNAVAILABLE",
        504: "GATEWAY_TIMEOUT"
    }
    
    error_code = error_code_mapping.get(exc.status_code, "UNKNOWN_ERROR")
    
    # Log the error
    app_logger.log_error(
        logger,
        Exception(exc.detail),
        context={
            "path": str(request.url.path),
            "method": request.method,
            "status_code": exc.status_code,
            "error_code": error_code
        },
        request_id=request_id
    )
    
    error_response = ErrorResponse(
        error=ErrorDetail(
            code=error_code,
            message=str(exc.detail)
        ),
        request_id=request_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
        path=str(request.url.path),
        method=request.method,
        status_code=exc.status_code,
        trace_id=trace_id
    )
    
    return JSONResponse(
        status_code=exc.status_code,
        content=error_response.model_dump()
    )


async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle general exceptions and return standardized error responses"""
    
    request_id = str(uuid.uuid4())
    trace_id = str(uuid.uuid4())
    
    # Log the unexpected error with full traceback
    app_logger.log_error(
        logger,
        exc,
        context={
            "path": str(request.url.path),
            "method": request.method,
            "error_type": "UNEXPECTED_ERROR"
        },
        request_id=request_id
    )
    
    # Don't expose internal error details in production
    error_response = ErrorResponse(
        error=ErrorDetail(
            code="INTERNAL_SERVER_ERROR",
            message="An unexpected error occurred. Please try again later.",
            details={"trace_id": trace_id}
        ),
        request_id=request_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
        path=str(request.url.path),
        method=request.method,
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        trace_id=trace_id
    )
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=error_response.model_dump()
    )


# Helper function for validation errors
def format_validation_errors(validation_errors: List[Dict[str, Any]]) -> List[ErrorDetail]:
    """Format Pydantic validation errors into standardized format"""
    errors = []
    for error in validation_errors:
        field_path = ".".join(str(loc) for loc in error["loc"])
        errors.append(ErrorDetail(
            code="VALIDATION_ERROR",
            message=error["msg"],
            field=field_path,
            details={"input": error.get("input"), "type": error["type"]}
        ))
    return errors