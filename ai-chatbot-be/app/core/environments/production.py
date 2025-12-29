"""
Production Environment Configuration
=====================================
Configuration for production deployment.

Enterprise Rationale:
- Maximum security settings
- Optimized performance
- Minimal logging overhead
- Strict validation
- Production-grade LLM (AWS Bedrock)
"""

from .base import BaseConfig
from pydantic import Field, field_validator
from typing import List


class ProductionConfig(BaseConfig):
    """
    Production environment configuration.
    
    Characteristics:
    - Debug mode disabled
    - Minimal logging (INFO level)
    - Strict CORS
    - Production LLM endpoints
    - Enhanced security
    """
    
    environment: str = "production"
    debug: bool = False
    log_level: str = "INFO"
    log_format: str = "json"  # Always JSON in production for log aggregation
    
    # Production CORS - must be explicitly configured
    allowed_origins: List[str] = Field(
        ...,  # Required - no defaults in production
        description="Allowed origins (must be explicitly set in production)"
    )
    
    # Production rate limiting (stricter)
    rate_limit_requests_per_minute: int = 30  # Lower than dev
    rate_limit_burst: int = 5
    
    # Production LLM - prefer AWS Bedrock
    aws_access_key_id: str = Field(
        ...,  # Required in production
        description="AWS access key for Bedrock"
    )
    aws_secret_access_key: str = Field(
        ...,  # Required in production
        description="AWS secret key for Bedrock"
    )
    
    # Production Redis - likely managed service
    redis_url: str = Field(
        ...,  # Required - must be explicitly configured
        description="Production Redis URL (managed service)"
    )
    redis_password: str = Field(
        ...,  # Required in production
        description="Redis password (required in production)"
    )
    
    # Longer TTLs for production (reduce load)
    session_ttl: int = 7 * 24 * 60 * 60  # 7 days
    cache_ttl: int = 60 * 60  # 1 hour
    
    # Production observability
    langsmith_api_key: str = Field(
        ...,  # Required for production monitoring
        description="LangSmith API key for production tracing"
    )
    langsmith_project: str = Field(
        ...,  # Required
        description="LangSmith project for production"
    )
    
    @field_validator('debug')
    @classmethod
    def validate_debug_disabled(cls, v: bool) -> bool:
        """Ensure debug is disabled in production."""
        if v is True:
            raise ValueError("Debug mode must be disabled in production")
        return v
    
    class Config:
        env_file = (".env", ".env.production")
        env_file_encoding = "utf-8"
