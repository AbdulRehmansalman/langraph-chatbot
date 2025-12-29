"""
Storage Exceptions
==================
Custom exceptions for Supabase storage operations.

Enterprise Rationale:
- Clear error messages for storage failures
- Structured error handling
- Easy debugging with operation context
"""

from typing import Dict, Any, Optional
from fastapi import status
from app.core.exceptions import BaseCustomException


class StorageException(BaseCustomException):
    """Base storage operation error exception"""
    
    def __init__(self, operation: str, message: str = None, details: Dict[str, Any] = None):
        if not message:
            message = f"Storage operation '{operation}' failed"
        
        super().__init__(
            code="STORAGE_ERROR",
            message=message,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            details=details or {"operation": operation}
        )


class StorageUploadException(StorageException):
    """Storage upload error exception"""
    
    def __init__(self, filename: str, message: str = None, details: Dict[str, Any] = None):
        if not message:
            message = f"Failed to upload file '{filename}'"
        
        super().__init__(
            operation="upload",
            message=message,
            details=details or {"filename": filename}
        )


class StorageDownloadException(StorageException):
    """Storage download error exception"""
    
    def __init__(self, filename: str, message: str = None, details: Dict[str, Any] = None):
        if not message:
            message = f"Failed to download file '{filename}'"
        
        super().__init__(
            operation="download",
            message=message,
            details=details or {"filename": filename}
        )


class StorageDeleteException(StorageException):
    """Storage delete error exception"""
    
    def __init__(self, filename: str, message: str = None, details: Dict[str, Any] = None):
        if not message:
            message = f"Failed to delete file '{filename}'"
        
        super().__init__(
            operation="delete",
            message=message,
            details=details or {"filename": filename}
        )


class StorageListException(StorageException):
    """Storage list error exception"""
    
    def __init__(self, path: str = None, message: str = None, details: Dict[str, Any] = None):
        if not message:
            message = f"Failed to list files{f' in {path}' if path else ''}"
        
        super().__init__(
            operation="list",
            message=message,
            details=details or {"path": path}
        )


class StorageConfigurationException(StorageException):
    """Storage configuration error exception"""
    
    def __init__(self, message: str = "Storage configuration is invalid", details: Dict[str, Any] = None):
        super().__init__(
            operation="configuration",
            message=message,
            details=details
        )
