"""
API Request/Response Schemas
=============================
Pydantic models for API contracts with strong validation.

Enterprise Features:
- Input validation at API boundary
- Prompt injection detection
- Size limits enforcement
- Clear error messages
"""

from pydantic import BaseModel, EmailStr, Field, field_validator
from typing import Optional, List
from datetime import datetime
import re


# ============================================================================
# AUTHENTICATION SCHEMAS
# ============================================================================

class UserCreate(BaseModel):
    email: EmailStr
    password: str
    full_name: Optional[str] = None
    timezone: Optional[str] = "UTC"


class UserLogin(BaseModel):
    email: EmailStr
    password: str
    timezone: Optional[str] = None


class EmailVerification(BaseModel):
    email: EmailStr
    otp: str


class ForgotPassword(BaseModel):
    email: EmailStr


class ResetPassword(BaseModel):
    email: EmailStr
    otp: str
    new_password: str


class ChangePassword(BaseModel):
    current_password: str
    new_password: str


class User(BaseModel):
    id: str
    email: str
    full_name: Optional[str] = None
    created_at: datetime
    is_active: bool = True


class Token(BaseModel):
    access_token: str
    token_type: str


# ============================================================================
# DOCUMENT SCHEMAS
# ============================================================================

class DocumentUpload(BaseModel):
    filename: str
    content_type: str


class Document(BaseModel):
    id: str
    filename: str
    content_type: str
    file_path: str
    storage_url: Optional[str] = None
    user_id: str
    created_at: datetime
    processed: bool = False


# ============================================================================
# CHAT SCHEMAS WITH VALIDATION
# ============================================================================

# Constants for validation
MAX_MESSAGE_LENGTH = 10000
MAX_DOCUMENT_IDS = 50


class ChatMessage(BaseModel):
    """
    Chat message request with enterprise validation.
    
    Validation:
    - Message: 1-10000 characters, no prompt injection, sanitized
    - Document IDs: Max 50, valid format, no duplicates
    """
    
    message: str = Field(
        ...,
        min_length=1,
        max_length=MAX_MESSAGE_LENGTH,
        description="User message (1-10000 characters)"
    )
    document_ids: Optional[List[str]] = Field(
        default=None,
        max_length=MAX_DOCUMENT_IDS,
        description="Optional document IDs for RAG (max 50)"
    )
    thread_id: Optional[str] = Field(
        default=None,
        max_length=255,
        description="Thread ID for conversation continuity (required for multi-turn scheduling)"
    )

    @field_validator('message')
    @classmethod
    def validate_message(cls, v: str) -> str:
        """
        Validate message content.

        Note: Detailed validation (prompt injection, token count)
        is performed in the validation module to keep schemas clean.
        This validator handles basic checks.
        """
        # Strip whitespace
        v = v.strip()
        
        if not v:
            raise ValueError("Message cannot be empty after trimming whitespace")
        
        # Basic length check (detailed check in validation module)
        if len(v) > MAX_MESSAGE_LENGTH:
            raise ValueError(
                f"Message too long: {len(v)} characters (max: {MAX_MESSAGE_LENGTH})"
            )
        
        return v
    
    @field_validator('document_ids')
    @classmethod
    def validate_document_ids(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        """
        Validate document IDs.
        
        Checks:
        - Not empty
        - Valid format (alphanumeric, hyphens, underscores)
        - No duplicates
        """
        if v is None:
            return None
        
        if len(v) > MAX_DOCUMENT_IDS:
            raise ValueError(
                f"Too many document IDs: {len(v)} (max: {MAX_DOCUMENT_IDS})"
            )
        
        # Remove duplicates while preserving order
        seen = set()
        unique_ids = []
        for doc_id in v:
            if doc_id not in seen:
                seen.add(doc_id)
                unique_ids.append(doc_id)
        
        # Validate each ID
        for doc_id in unique_ids:
            if not doc_id or not doc_id.strip():
                raise ValueError("Document ID cannot be empty")
            
            # Basic format validation (UUID-like or alphanumeric)
            if not re.match(r'^[a-zA-Z0-9_-]+$', doc_id):
                raise ValueError(
                    f"Invalid document ID format: '{doc_id}'. "
                    "Only alphanumeric characters, hyphens, and underscores allowed."
                )
        
        return unique_ids


class ChatResponse(BaseModel):
    """Chat response with metadata."""
    
    id: str
    user_message: str
    bot_response: str
    document_ids: Optional[List[str]] = None
    created_at: datetime
    
    class Config:
        from_attributes = True


class ChatHistory(BaseModel):
    """Chat history container."""
    
    messages: List[ChatResponse]


# ============================================================================
# STREAMING SCHEMAS
# ============================================================================

class StreamingChatRequest(BaseModel):
    """
    Streaming chat request with same validation as ChatMessage.

    This is a separate model to allow for future streaming-specific fields.
    """

    message: str = Field(
        ...,
        min_length=1,
        max_length=MAX_MESSAGE_LENGTH,
        description="User message (1-10000 characters)"
    )
    document_ids: Optional[List[str]] = Field(
        default=None,
        max_length=MAX_DOCUMENT_IDS,
        description="Optional document IDs for RAG (max 50)"
    )

    # Thread for conversation grouping
    thread_id: Optional[str] = Field(
        default=None,
        max_length=255,
        description="Optional thread ID for conversation grouping"
    )

    # Streaming-specific options
    stream_timeout: Optional[int] = Field(
        default=3600,
        ge=5,
        le=3600,
        description="Streaming timeout in seconds (5-3600)"
    )
    
    @field_validator('message')
    @classmethod
    def validate_message(cls, v: str) -> str:
        """Validate message content."""
        v = v.strip()
        
        if not v:
            raise ValueError("Message cannot be empty after trimming whitespace")
        
        if len(v) > MAX_MESSAGE_LENGTH:
            raise ValueError(
                f"Message too long: {len(v)} characters (max: {MAX_MESSAGE_LENGTH})"
            )
        
        return v
    
    @field_validator('document_ids')
    @classmethod
    def validate_document_ids(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        """Validate document IDs."""
        if v is None:
            return None
        
        if len(v) > MAX_DOCUMENT_IDS:
            raise ValueError(
                f"Too many document IDs: {len(v)} (max: {MAX_DOCUMENT_IDS})"
            )
        
        # Remove duplicates
        seen = set()
        unique_ids = []
        for doc_id in v:
            if doc_id not in seen:
                seen.add(doc_id)
                unique_ids.append(doc_id)
        
        # Validate format
        for doc_id in unique_ids:
            if not doc_id or not doc_id.strip():
                raise ValueError("Document ID cannot be empty")
            
            if not re.match(r'^[a-zA-Z0-9_-]+$', doc_id):
                raise ValueError(
                    f"Invalid document ID format: '{doc_id}'"
                )
        
        return unique_ids