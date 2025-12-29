"""
Production LLM Factory
======================

Thread-safe singleton factory supporting multiple LLM providers:
- Ollama (development/local)
- AWS Bedrock Claude (production)

Environment Variables:
- LLM_PROVIDER: "ollama" or "bedrock" (default: "ollama" for dev)
- OLLAMA_BASE_URL: Ollama server URL (default: http://localhost:11434)
- OLLAMA_MODEL: Ollama model name (default: llama3.1:8b)
- AWS_ACCESS_KEY_ID: AWS credentials
- AWS_SECRET_ACCESS_KEY: AWS credentials
- AWS_REGION: AWS region (default: us-east-1)
- BEDROCK_MODEL_ID: Bedrock model ID
"""

import os
import threading
import logging
from typing import Optional, Union

from langchain_core.language_models import BaseChatModel

from app.core.config import settings

logger = logging.getLogger(__name__)


class LLMFactoryError(Exception):
    """LLM Factory error."""
    pass


class LLMFactory:
    """
    Thread-safe singleton factory for LLM instances.

    Supports:
    - Ollama (local development)
    - AWS Bedrock Claude (production)

    Usage:
        from app.services.llm_factory import llm_factory
        llm = llm_factory.create_llm()
    """

    _instance: Optional['LLMFactory'] = None
    _lock = threading.Lock()
    _cache_lock = threading.Lock()

    # Provider constants
    PROVIDER_OLLAMA = "ollama"
    PROVIDER_BEDROCK = "bedrock"

    # Default configurations
    DEFAULT_OLLAMA_URL = "http://localhost:11434"
    DEFAULT_OLLAMA_MODEL = "llama3.1:8b"
    DEFAULT_BEDROCK_REGION = "us-east-1"
    DEFAULT_BEDROCK_MODEL = "us.anthropic.claude-3-5-sonnet-20241022-v2:0"
    DEFAULT_TEMPERATURE = 0.7
    DEFAULT_MAX_TOKENS = 4096

    def __new__(cls) -> 'LLMFactory':
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialize()
        return cls._instance

    def _initialize(self) -> None:
        """Initialize factory state."""
        self._llm_cache: dict = {}
        self._credentials_configured = False
        self._provider = self._detect_provider()
        logger.info(f"LLMFactory initialized with provider: {self._provider}")

    def _detect_provider(self) -> str:
        """Detect which provider to use based on environment."""
        # Check explicit setting
        provider = getattr(settings, 'llm_provider', None) or os.getenv("LLM_PROVIDER")
        if provider:
            return provider.lower()

        # Check environment
        env = getattr(settings, 'environment', 'development')
        if env == "production":
            return self.PROVIDER_BEDROCK

        # Default to Ollama for development
        return self.PROVIDER_OLLAMA

    def create_llm(
        self,
        model_id: Optional[str] = None,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        streaming: bool = True,
        provider: Optional[str] = None,
    ) -> BaseChatModel:
        """
        Create or retrieve cached LLM instance.

        Args:
            model_id: Model identifier
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens in response
            streaming: Enable streaming responses
            provider: Override provider ("ollama" or "bedrock")

        Returns:
            Configured LLM instance

        Raises:
            LLMFactoryError: If LLM creation fails
        """
        provider = provider or self._provider

        if provider == self.PROVIDER_OLLAMA:
            return self._create_ollama_llm(model_id, temperature, max_tokens, streaming)
        elif provider == self.PROVIDER_BEDROCK:
            return self._create_bedrock_llm(model_id, temperature, max_tokens, streaming)
        else:
            raise LLMFactoryError(f"Unknown provider: {provider}")

    def _create_ollama_llm(
        self,
        model_id: Optional[str],
        temperature: float,
        max_tokens: int,
        streaming: bool,
    ) -> BaseChatModel:
        """Create Ollama LLM instance."""
        try:
            from langchain_ollama import ChatOllama
        except ImportError:
            raise LLMFactoryError(
                "langchain-ollama not installed. Install with: pip install langchain-ollama"
            )

        # Get configuration
        base_url = (
            getattr(settings, 'ollama_base_url', None) or
            os.getenv("OLLAMA_BASE_URL", self.DEFAULT_OLLAMA_URL)
        )
        model = (
            model_id or
            getattr(settings, 'ollama_model', None) or
            os.getenv("OLLAMA_MODEL", self.DEFAULT_OLLAMA_MODEL)
        )

        cache_key = f"ollama_{model}_{temperature}_{max_tokens}_{streaming}"

        # Return cached instance
        with self._cache_lock:
            if cache_key in self._llm_cache:
                logger.debug(f"Returning cached Ollama LLM: {model}")
                return self._llm_cache[cache_key]

        try:
            llm = ChatOllama(
                model=model,
                base_url=base_url,
                temperature=temperature,
                num_predict=max_tokens,
            )

            # Cache instance
            with self._cache_lock:
                self._llm_cache[cache_key] = llm

            logger.info(f"Ollama LLM created: {model} @ {base_url}")
            return llm

        except Exception as e:
            logger.error(f"Ollama LLM creation failed: {e}")
            raise LLMFactoryError(f"Failed to create Ollama LLM: {e}")

    def _create_bedrock_llm(
        self,
        model_id: Optional[str],
        temperature: float,
        max_tokens: int,
        streaming: bool,
    ) -> BaseChatModel:
        """Create AWS Bedrock Claude LLM instance."""
        try:
            import boto3
            from botocore.exceptions import ClientError, NoCredentialsError
            from langchain_aws import ChatBedrock
        except ImportError:
            raise LLMFactoryError(
                "AWS dependencies not installed. Install with: pip install boto3 langchain-aws"
            )

        # Get configuration
        model_id = (
            model_id or
            getattr(settings, 'bedrock_model_id', None) or
            os.getenv("BEDROCK_MODEL_ID", self.DEFAULT_BEDROCK_MODEL)
        )
        region = (
            getattr(settings, 'aws_region', None) or
            os.getenv("AWS_REGION", self.DEFAULT_BEDROCK_REGION)
        )

        cache_key = f"bedrock_{model_id}_{temperature}_{max_tokens}_{streaming}"

        # Return cached instance
        with self._cache_lock:
            if cache_key in self._llm_cache:
                logger.debug(f"Returning cached Bedrock LLM: {model_id}")
                return self._llm_cache[cache_key]

        # Configure AWS credentials
        self._configure_aws_credentials()

        try:
            # Create Bedrock client
            client = boto3.client("bedrock-runtime", region_name=region)

            # Create ChatBedrock instance
            llm = ChatBedrock(
                model_id=model_id,
                client=client,
                streaming=streaming,
                model_kwargs={
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                },
            )

            # Cache instance
            with self._cache_lock:
                self._llm_cache[cache_key] = llm

            logger.info(f"Bedrock LLM created: {model_id} in {region}")
            return llm

        except NoCredentialsError:
            logger.error("AWS credentials not found")
            raise LLMFactoryError(
                "AWS credentials missing. Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY"
            )
        except ClientError as e:
            logger.error(f"Bedrock client error: {e}")
            raise LLMFactoryError(f"Bedrock connection failed: {e}")
        except Exception as e:
            logger.error(f"Bedrock LLM creation failed: {e}")
            raise LLMFactoryError(f"Failed to create Bedrock LLM: {e}")

    def _configure_aws_credentials(self) -> None:
        """Configure AWS credentials from environment/settings."""
        if self._credentials_configured:
            return

        import boto3

        access_key = getattr(settings, 'aws_access_key_id', None) or os.getenv("AWS_ACCESS_KEY_ID")
        secret_key = getattr(settings, 'aws_secret_access_key', None) or os.getenv("AWS_SECRET_ACCESS_KEY")
        region = getattr(settings, 'aws_region', None) or os.getenv("AWS_REGION", self.DEFAULT_BEDROCK_REGION)

        if access_key:
            os.environ.setdefault("AWS_ACCESS_KEY_ID", access_key)
        if secret_key:
            os.environ.setdefault("AWS_SECRET_ACCESS_KEY", secret_key)
        os.environ.setdefault("AWS_DEFAULT_REGION", region)
        os.environ.setdefault("AWS_REGION", region)

        boto3.setup_default_session(
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            region_name=region
        )

        self._credentials_configured = True
        logger.info(f"AWS credentials configured for region: {region}")

    def get_provider_info(self) -> str:
        """Get current LLM provider information."""
        if self._provider == self.PROVIDER_OLLAMA:
            model = (
                getattr(settings, 'ollama_model', None) or
                os.getenv("OLLAMA_MODEL", self.DEFAULT_OLLAMA_MODEL)
            )
            return f"Ollama ({model})"
        else:
            model = (
                getattr(settings, 'bedrock_model_id', None) or
                os.getenv("BEDROCK_MODEL_ID", self.DEFAULT_BEDROCK_MODEL)
            )
            return f"AWS Bedrock ({model})"

    def get_current_provider(self) -> str:
        """Get current provider name."""
        return self._provider

    def set_provider(self, provider: str) -> None:
        """
        Set the LLM provider.

        Args:
            provider: "ollama" or "bedrock"
        """
        if provider not in [self.PROVIDER_OLLAMA, self.PROVIDER_BEDROCK]:
            raise ValueError(f"Invalid provider: {provider}")
        self._provider = provider
        self.clear_cache()
        logger.info(f"LLM provider set to: {provider}")

    def clear_cache(self) -> None:
        """Clear LLM instance cache."""
        with self._cache_lock:
            self._llm_cache.clear()
        logger.info("LLM cache cleared")


# Global singleton instance
llm_factory = LLMFactory()
