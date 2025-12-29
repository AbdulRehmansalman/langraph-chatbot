"""
Development Environment Configuration
======================================
Configuration overrides for local development.

Enterprise Rationale:
- Enables local testing without production credentials
- Relaxed security for developer convenience
- Verbose logging for debugging
- Local service endpoints
"""

from .base import BaseConfig
from pydantic import Field
from typing import List


class DevelopmentConfig(BaseConfig):
    """
    Development environment configuration.
    
    Characteristics:
    - Debug mode enabled
    - Verbose logging
    - Local service endpoints
    - Relaxed CORS for frontend development
    """
    
    environment: str = "development"
    debug: bool = True
    log_level: str = "DEBUG"
    
    # Development-specific CORS (allow all local ports)
    allowed_origins: List[str] = Field(
        default=[
            "http://localhost:3000",
            "http://localhost:5173",
            "http://localhost:8080",
            "http://127.0.0.1:3000",
            "http://127.0.0.1:5173",
            "http://127.0.0.1:8080",
        ],
        description="Allowed origins for development (all local ports)"
    )
    
    # Development LLM defaults (use local Ollama)
    ollama_base_url: str = "http://localhost:11434"
    
    # Development Redis (local)
    redis_url: str = "redis://localhost:6379"
    
    # Shorter TTLs for development (faster iteration)
    session_ttl: int = 60 * 60  # 1 hour
    cache_ttl: int = 5 * 60  # 5 minutes
    
    class Config:
        env_file = (".env", ".env.development")
        env_file_encoding = "utf-8"
