"""
Staging Environment Configuration
==================================
Configuration for staging/pre-production environment.

Enterprise Rationale:
- Production-like settings for realistic testing
- Slightly relaxed for testing convenience
- Separate credentials from production
- Full observability enabled
"""

from .base import BaseConfig
from pydantic import Field
from typing import List


class StagingConfig(BaseConfig):
    """
    Staging environment configuration.
    
    Characteristics:
    - Production-like settings
    - Debug mode disabled
    - Full logging for testing
    - Separate credentials
    """
    
    environment: str = "staging"
    debug: bool = False
    log_level: str = "DEBUG"  # More verbose than prod for testing
    log_format: str = "json"
    
    # Staging CORS - typically includes staging frontend URLs
    allowed_origins: List[str] = Field(
        default=[],  # Must be configured
        description="Allowed origins for staging environment"
    )
    
    # Staging rate limiting (same as production)
    rate_limit_requests_per_minute: int = 30
    rate_limit_burst: int = 5
    
    # Staging can use either Ollama or Bedrock
    # Credentials optional to allow testing with local models
    aws_access_key_id: str | None = None
    aws_secret_access_key: str | None = None
    
    # Staging Redis
    redis_url: str = Field(
        default="redis://localhost:6379",
        description="Staging Redis URL"
    )
    
    # Medium TTLs for staging
    session_ttl: int = 24 * 60 * 60  # 24 hours
    cache_ttl: int = 30 * 60  # 30 minutes
    
    # Staging observability (recommended but not required)
    langsmith_api_key: str | None = None
    langsmith_project: str | None = None
    
    class Config:
        env_file = (".env", ".env.staging")
        env_file_encoding = "utf-8"
