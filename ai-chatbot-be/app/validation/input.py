"""
Input Validation and Sanitization
==================================
Enterprise-grade input validation for API endpoints.

Enterprise Rationale:
- Prevent prompt injection attacks
- Enforce request size limits
- Sanitize user input before RAG processing
- Provide clear validation errors
"""

import re
from typing import Optional, List, Tuple
from pydantic import BaseModel, Field, field_validator
import tiktoken

from app.core.exceptions import ValidationException


# ============================================================================
# CONSTANTS
# ============================================================================

# Request size limits
MAX_MESSAGE_LENGTH = 10000  # characters
MAX_MESSAGE_TOKENS = 2000   # tokens (for GPT models)
MAX_DOCUMENT_IDS = 50       # maximum number of document IDs

# Prompt injection patterns (deterministic detection)
PROMPT_INJECTION_PATTERNS = [
    # System role override attempts
    r"(?i)(system|assistant|user)\s*[:=]\s*[\"']",
    r"(?i)you\s+are\s+(now\s+)?(a|an)\s+",
    r"(?i)ignore\s+(previous|all|above|prior)\s+(instructions|prompts|rules)",
    r"(?i)disregard\s+(previous|all|above|prior)",
    
    # Instruction hijacking
    r"(?i)(new|updated|revised)\s+(instructions|rules|prompt)",
    r"(?i)from\s+now\s+on",
    r"(?i)instead\s+of\s+",
    r"(?i)forget\s+(everything|all|previous)",
    
    # Role confusion
    r"(?i)act\s+as\s+(if\s+)?(you\s+are|a|an)",
    r"(?i)pretend\s+(to\s+be|you\s+are)",
    r"(?i)simulate\s+(being|a|an)",
    
    # Direct system manipulation
    r"(?i)<\s*system\s*>",
    r"(?i)</\s*system\s*>",
    r"(?i)\[system\]",
    r"(?i)\[/system\]",
    r"(?i){{system}}",
    
    # Instruction termination attempts
    r"(?i)---\s*end\s+(of\s+)?(instructions|prompt)",
    r"(?i)###\s*",
    r"(?i)```\s*(system|instructions|prompt)",
]

# Suspicious character sequences
SUSPICIOUS_SEQUENCES = [
    r"[\x00-\x08\x0B\x0C\x0E-\x1F]",  # Control characters
    r"[\uFEFF\u200B-\u200D\uFFFE\uFFFF]",  # Zero-width and special Unicode
]


# ============================================================================
# VALIDATION MODELS
# ============================================================================

class ChatMessageRequest(BaseModel):
    """
    Validated chat message request.
    
    Enterprise Features:
    - Character and token limits
    - Prompt injection detection
    - Input sanitization
    - Document ID validation
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
    
    @field_validator('message')
    @classmethod
    def validate_message(cls, v: str) -> str:
        """
        Validate and sanitize message.
        
        Checks:
        1. Not empty after stripping
        2. Token count within limits
        3. No prompt injection patterns
        4. No suspicious character sequences
        """
        # Strip whitespace
        v = v.strip()
        
        if not v:
            raise ValueError("Message cannot be empty")
        
        # Check token count
        token_count = count_tokens(v)
        if token_count > MAX_MESSAGE_TOKENS:
            raise ValueError(
                f"Message too long: {token_count} tokens (max: {MAX_MESSAGE_TOKENS})"
            )
        
        # Check for prompt injection
        is_injection, pattern = detect_prompt_injection(v)
        if is_injection:
            raise ValueError(
                f"Potential prompt injection detected: {pattern}"
            )
        
        # Check for suspicious sequences
        if contains_suspicious_sequences(v):
            raise ValueError(
                "Message contains suspicious character sequences"
            )
        
        # Sanitize
        v = sanitize_input(v)
        
        return v
    
    @field_validator('document_ids')
    @classmethod
    def validate_document_ids(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        """Validate document IDs."""
        if v is None:
            return None
        
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
                    f"Invalid document ID format: {doc_id}"
                )
        
        return unique_ids


# ============================================================================
# VALIDATION FUNCTIONS
# ============================================================================

def count_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
    """
    Count tokens in text using tiktoken.
    
    Args:
        text: Input text
        model: Model name for tokenizer
    
    Returns:
        Token count
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except Exception:
        # Fallback: rough estimate (1 token ≈ 4 characters)
        return len(text) // 4


def detect_prompt_injection(text: str) -> Tuple[bool, Optional[str]]:
    """
    Detect potential prompt injection attempts.
    
    Uses deterministic pattern matching for common injection techniques.
    
    Args:
        text: Input text to check
    
    Returns:
        Tuple of (is_injection, matched_pattern)
    """
    for pattern in PROMPT_INJECTION_PATTERNS:
        if re.search(pattern, text):
            return True, pattern
    
    return False, None


def contains_suspicious_sequences(text: str) -> bool:
    """
    Check for suspicious character sequences.
    
    Args:
        text: Input text to check
    
    Returns:
        True if suspicious sequences found
    """
    for pattern in SUSPICIOUS_SEQUENCES:
        if re.search(pattern, text):
            return True
    
    return False


def sanitize_input(text: str) -> str:
    """
    Sanitize user input.
    
    Operations:
    1. Normalize whitespace
    2. Remove control characters
    3. Trim to reasonable length
    
    Args:
        text: Input text
    
    Returns:
        Sanitized text
    """
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove control characters (except newline and tab)
    text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F]', '', text)
    
    # Remove zero-width characters
    text = re.sub(r'[\uFEFF\u200B-\u200D]', '', text)
    
    # Trim
    text = text.strip()
    
    return text


def validate_message_size(message: str) -> None:
    """
    Validate message size (characters and tokens).
    
    Args:
        message: Message to validate
    
    Raises:
        ValidationException: If message exceeds limits
    """
    # Character limit
    if len(message) > MAX_MESSAGE_LENGTH:
        raise ValidationException(
            f"Message too long: {len(message)} characters (max: {MAX_MESSAGE_LENGTH})",
            field="message"
        )
    
    # Token limit
    token_count = count_tokens(message)
    if token_count > MAX_MESSAGE_TOKENS:
        raise ValidationException(
            f"Message too long: {token_count} tokens (max: {MAX_MESSAGE_TOKENS})",
            field="message"
        )


def validate_and_sanitize(message: str, document_ids: Optional[List[str]] = None) -> Tuple[str, Optional[List[str]]]:
    """
    Validate and sanitize chat request.
    
    This is the main entry point for request validation.
    
    Args:
        message: User message
        document_ids: Optional document IDs
    
    Returns:
        Tuple of (sanitized_message, validated_document_ids)
    
    Raises:
        ValidationException: If validation fails
    """
    try:
        # Create and validate request model
        request = ChatMessageRequest(
            message=message,
            document_ids=document_ids
        )
        
        return request.message, request.document_ids
    
    except ValueError as e:
        # Convert Pydantic validation errors to ValidationException
        raise ValidationException(
            str(e),
            field="message"
        )
