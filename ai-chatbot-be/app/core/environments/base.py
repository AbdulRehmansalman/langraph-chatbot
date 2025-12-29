"""
Base Configuration
==================
Shared configuration across all environments.
All sensitive values MUST come from environment variables.

Enterprise Rationale:
- No hardcoded secrets (security)
- Type-safe configuration (reliability)
- Fail-fast validation (operational excellence)
- Clear documentation (maintainability)
"""

from pydantic_settings import BaseSettings
from pydantic import Field, field_validator, ValidationError
from typing import Optional, List
import os


class BaseConfig(BaseSettings):
    """
    Base configuration shared across all environments.

    Security Notes:
    - All secrets MUST be loaded from environment variables
    - No default values for sensitive fields
    - Validation ensures required fields are present
    """

    # ============================================================================
    # APPLICATION SETTINGS
    # ============================================================================
    app_name: str = "Enterprise Document Chatbot API"
    app_version: str = "1.0.0"
    environment: str = Field(
        ...,  # Required field
        description="Environment name: development, staging, or production"
    )
    debug: bool = Field(
        default=False,
        description="Enable debug mode (should be False in production)"
    )

    # ============================================================================
    # LOGGING CONFIGURATION
    # ============================================================================
    log_level: str = Field(
        default="INFO",
        description="Logging level: DEBUG, INFO, WARNING, ERROR, CRITICAL"
    )
    log_format: str = Field(
        default="json",
        description="Log format: json or text"
    )
    log_file: str = Field(
        default="logs/app.log",
        description="Path to log file"
    )
    log_max_size: int = Field(
        default=100,
        description="Maximum log file size in MB"
    )
    log_backup_count: int = Field(
        default=5,
        description="Number of backup log files to keep"
    )

    # ============================================================================
    # CORS SETTINGS
    # ============================================================================
    allowed_origins: List[str] = Field(
        default=["http://localhost:5173", "http://localhost:3000"],
        description="Allowed CORS origins"
    )
    allowed_methods: List[str] = Field(
        default=["*"],
        description="Allowed HTTP methods"
    )
    allowed_headers: List[str] = Field(
        default=["*"],
        description="Allowed HTTP headers"
    )

    # ============================================================================
    # RATE LIMITING
    # ============================================================================
    rate_limit_requests_per_minute: int = Field(
        default=60,
        description="Maximum requests per minute per user"
    )
    rate_limit_burst: int = Field(
        default=10,
        description="Burst allowance for rate limiting"
    )

    # ============================================================================
    # SECURITY & AUTHENTICATION
    # ============================================================================
    secret_key: str = Field(
        ...,  # Required - no default
        description="Secret key for JWT token signing"
    )
    algorithm: str = Field(
        default="HS256",
        description="JWT algorithm"
    )
    access_token_expire_minutes: int = Field(
        default=30,
        description="JWT token expiration time in minutes"
    )

    # ============================================================================
    # DATABASE CONFIGURATION
    # ============================================================================
    database_url: str = Field(
        ...,  # Required - MUST come from environment
        description="PostgreSQL database connection URL"
    )

    # ============================================================================
    # SUPABASE CONFIGURATION
    # ============================================================================
    supabase_url: str = Field(
        ...,  # Required
        description="Supabase project URL"
    )
    supabase_key: str = Field(
        ...,  # Required
        description="Supabase anon/public key"
    )
    supabase_service_key: str = Field(
        ...,  # Required
        description="Supabase service role key (admin access)"
    )
    supabase_storage_bucket: str = Field(
        default="documents",
        description="Supabase storage bucket name for documents"
    )

    # ============================================================================
    # EMBEDDING CONFIGURATION
    # ============================================================================
    embedding_provider: str = Field(
        default="huggingface",
        description="Embedding provider: huggingface, openai, cohere, mistral"
    )

    # ============================================================================
    # LLM CONFIGURATION
    # ============================================================================
    llm_provider: str = Field(
        default="ollama",
        description="LLM provider: ollama (dev) or bedrock (prod)"
    )
    openai_api_key: Optional[str] = Field(
        default=None,
        description="OpenAI API key (required if using OpenAI)"
    )
    mistral_api_key: Optional[str] = Field(
        default=None,
        description="Mistral API key (required if using Mistral)"
    )
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        description="Ollama server base URL"
    )
    ollama_model: str = Field(
        default="llama3.1:8b",
        description="Ollama model name"
    )

    # ============================================================================
    # AWS BEDROCK CONFIGURATION (Production LLM)
    # ============================================================================
    aws_access_key_id: Optional[str] = Field(
        default=None,
        description="AWS access key ID for Bedrock"
    )
    aws_secret_access_key: Optional[str] = Field(
        default=None,
        description="AWS secret access key for Bedrock"
    )
    aws_bearer_token_bedrock: Optional[str] = Field(
        default=None,
        description="AWS bearer token for Bedrock"
    )
    aws_region: str = Field(
        default="us-east-1",
        description="AWS region for Bedrock"
    )
    bedrock_model_id: str = Field(
        default="us.anthropic.claude-3-5-sonnet-20241022-v2:0",
        description="Bedrock model identifier"
    )

    # ============================================================================
    # EMAIL CONFIGURATION
    # ============================================================================
    mail_username: str = Field(
        ...,  # Required
        description="SMTP username"
    )
    mail_password: str = Field(
        ...,  # Required
        description="SMTP password"
    )
    mail_from: str = Field(
        ...,  # Required
        description="From email address"
    )
    mail_port: int = Field(
        default=587,
        description="SMTP port"
    )
    mail_server: str = Field(
        default="smtp.gmail.com",
        description="SMTP server hostname"
    )
    mail_from_name: str = Field(
        default="AI Chatbot",
        description="From name for emails"
    )
    mail_starttls: bool = Field(
        default=True,
        description="Use STARTTLS for SMTP"
    )
    mail_ssl_tls: bool = Field(
        default=False,
        description="Use SSL/TLS for SMTP"
    )

    # ============================================================================
    # GOOGLE OAUTH CONFIGURATION
    # ============================================================================
    google_client_id: str = Field(
        ...,  # Required
        description="Google OAuth client ID"
    )
    google_client_secret: str = Field(
        ...,  # Required
        description="Google OAuth client secret"
    )
    google_redirect_uri: str = Field(
        default="http://localhost:8000/auth/google/callback",
        description="Google OAuth redirect URI"
    )

    # ============================================================================
    # LANGSMITH CONFIGURATION (Observability)
    # ============================================================================
    langsmith_api_key: Optional[str] = Field(
        default=None,
        description="LangSmith API key for tracing"
    )
    langsmith_project: Optional[str] = Field(
        default=None,
        description="LangSmith project name"
    )
    langsmith_endpoint: Optional[str] = Field(
        default=None,
        description="LangSmith API endpoint"
    )

    # ============================================================================
    # REDIS CONFIGURATION (Caching & Session)
    # ============================================================================
    redis_url: str = Field(
        default="redis://localhost:6379",
        description="Redis connection URL"
    )
    redis_password: Optional[str] = Field(
        default=None,
        description="Redis password (if required)"
    )
    redis_db: int = Field(
        default=0,
        description="Redis database number"
    )
    redis_max_connections: int = Field(
        default=10,
        description="Maximum Redis connection pool size"
    )

    # ============================================================================
    # MEMORY & CACHE TTL SETTINGS
    # ============================================================================
    session_ttl: int = Field(
        default=72 * 60 * 60,  # 72 hours
        description="Session TTL in seconds"
    )
    entity_ttl: int = Field(
        default=7 * 24 * 60 * 60,  # 7 days
        description="Entity memory TTL in seconds"
    )
    cache_ttl: int = Field(
        default=60 * 60,  # 1 hour
        description="Cache TTL in seconds"
    )

    # ============================================================================
    # RAG MEMORY SETTINGS
    # ============================================================================
    max_instant_messages: int = Field(
        default=10,
        description="Maximum messages in working memory"
    )
    max_context_tokens: int = Field(
        default=4000,
        description="Maximum tokens for context window"
    )
    summarization_threshold: int = Field(
        default=20,
        description="Messages before triggering summarization"
    )

    # ============================================================================
    # RAG PIPELINE CONFIGURATION
    # ============================================================================
    # Retrieval settings
    rag_retrieval_strategy: str = Field(
        default="hybrid",
        description="Retrieval strategy: similarity, mmr, hybrid, ensemble"
    )
    rag_top_k: int = Field(
        default=20,
        description="Number of documents to retrieve before reranking"
    )
    rag_score_threshold: float = Field(
        default=0.05,
        description="Minimum similarity score threshold"
    )
    rag_reranking_method: str = Field(
        default="cross_encoder",
        description="Reranking method: none, cross_encoder, llm, rrf"
    )
    rag_rerank_top_k: int = Field(
        default=10,
        description="Number of documents after reranking"
    )

    # Hybrid search weights
    rag_vector_weight: float = Field(
        default=0.4,
        description="Weight for vector similarity in hybrid search (0.0-1.0)"
    )
    rag_keyword_weight: float = Field(
        default=0.6,
        description="Weight for keyword matching in hybrid search (0.0-1.0)"
    )

    # Response generation settings
    rag_mode: str = Field(
        default="conversational",
        description="RAG mode: standard, conversational, strict, creative"
    )
    rag_use_memory: bool = Field(
        default=False,
        description="Enable conversation memory"
    )
    rag_use_query_rewriting: bool = Field(
        default=True,
        description="Enable query rewriting with context"
    )
    rag_use_hyde: bool = Field(
        default=False,
        description="Enable Hypothetical Document Embeddings"
    )
    rag_use_compression: bool = Field(
        default=False,
        description="Enable contextual compression"
    )
    rag_max_context_length: int = Field(
        default=4000,
        description="Maximum context length in characters"
    )
    rag_min_confidence: float = Field(
        default=0.3,
        description="Minimum confidence score for responses"
    )
    rag_include_sources: bool = Field(
        default=True,
        description="Include source attribution in responses"
    )

    # Resilience settings
    rag_max_retries: int = Field(
        default=3,
        description="Maximum retry attempts for RAG operations"
    )
    rag_retry_delay: float = Field(
        default=1.0,
        description="Initial retry delay in seconds"
    )
    rag_timeout: float = Field(
        default=30.0,
        description="Timeout for RAG operations in seconds"
    )

    # ============================================================================
    # VALIDATORS
    # ============================================================================

    @field_validator('environment')
    @classmethod
    def validate_environment(cls, v: str) -> str:
        """Ensure environment is one of the allowed values."""
        allowed = ['development', 'staging', 'production']
        if v not in allowed:
            raise ValueError(
                f"Invalid environment '{v}'. Must be one of: {', '.join(allowed)}"
            )
        return v

    @field_validator('log_level')
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Ensure log level is valid."""
        allowed = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        v_upper = v.upper()
        if v_upper not in allowed:
            raise ValueError(
                f"Invalid log level '{v}'. Must be one of: {', '.join(allowed)}"
            )
        return v_upper

    @field_validator('database_url')
    @classmethod
    def validate_database_url(cls, v: str) -> str:
        """Ensure database URL is properly formatted."""
        if not v.startswith(('postgresql://', 'postgres://')):
            raise ValueError(
                "database_url must start with 'postgresql://' or 'postgres://'"
            )
        return v

    @field_validator('secret_key')
    @classmethod
    def validate_secret_key(cls, v: str) -> str:
        """Ensure secret key is sufficiently strong."""
        if len(v) < 32:
            raise ValueError(
                "secret_key must be at least 32 characters for security"
            )
        return v

    @field_validator('rag_retrieval_strategy')
    @classmethod
    def validate_rag_strategy(cls, v: str) -> str:
        """Ensure RAG retrieval strategy is valid."""
        allowed = ['similarity', 'mmr', 'hybrid', 'ensemble', 'multi_query']
        v_lower = v.lower()
        if v_lower not in allowed:
            raise ValueError(
                f"Invalid rag_retrieval_strategy '{v}'. "
                f"Must be one of: {', '.join(allowed)}"
            )
        return v_lower

    @field_validator('rag_reranking_method')
    @classmethod
    def validate_rag_reranking(cls, v: str) -> str:
        """Ensure RAG reranking method is valid."""
        allowed = ['none', 'cross_encoder', 'llm', 'rrf', 'cohere']
        v_lower = v.lower()
        if v_lower not in allowed:
            raise ValueError(
                f"Invalid rag_reranking_method '{v}'. "
                f"Must be one of: {', '.join(allowed)}"
            )
        return v_lower

    @field_validator('rag_mode')
    @classmethod
    def validate_rag_mode(cls, v: str) -> str:
        """Ensure RAG mode is valid."""
        allowed = ['standard', 'conversational', 'strict', 'creative']
        v_lower = v.lower()
        if v_lower not in allowed:
            raise ValueError(
                f"Invalid rag_mode '{v}'. Must be one of: {', '.join(allowed)}"
            )
        return v_lower

    class Config:
        """Pydantic configuration."""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        # Allow extra fields for forward compatibility
        extra = "ignore"
