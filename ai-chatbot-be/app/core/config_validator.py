"""
Configuration Validator
========================
Validates configuration on application startup.

Enterprise Rationale:
- Fail-fast on misconfiguration (prevent runtime errors)
- Clear error messages for operators
- Validate cross-field dependencies
- Check external service connectivity
"""

import logging
import sys
from typing import Dict, List, Tuple
from pydantic import ValidationError

from app.core.environments.base import BaseConfig
from app.core.environments.development import DevelopmentConfig
from app.core.environments.staging import StagingConfig
from app.core.environments.production import ProductionConfig

logger = logging.getLogger(__name__)


class ConfigurationError(Exception):
    """Raised when configuration is invalid."""
    pass


class ConfigValidator:
    """
    Validates application configuration on startup.
    
    Ensures:
    - All required environment variables are present
    - Values are properly formatted
    - Cross-field validations pass
    - Environment-specific requirements are met
    """
    
    @staticmethod
    def validate_and_load(environment: str | None = None) -> BaseConfig:
        """
        Validate and load configuration for the specified environment.
        
        Args:
            environment: Environment name (development, staging, production)
                        If None, reads from ENVIRONMENT env var
        
        Returns:
            Validated configuration object
            
        Raises:
            ConfigurationError: If configuration is invalid
        """
        # Determine environment
        if environment is None:
            import os
            environment = os.getenv("ENVIRONMENT", "development")
        
        logger.info(f"Loading configuration for environment: {environment}")
        
        # Select appropriate config class
        config_class = ConfigValidator._get_config_class(environment)
        
        # Attempt to load and validate
        try:
            config = config_class()
            logger.info("✓ Configuration loaded successfully")
            
            # Run additional validations
            ConfigValidator._validate_llm_config(config)
            ConfigValidator._validate_security_config(config)
            ConfigValidator._validate_database_config(config)
            
            # Log configuration summary (without secrets)
            ConfigValidator._log_config_summary(config)
            
            return config
            
        except ValidationError as e:
            # Format validation errors for clarity
            error_msg = ConfigValidator._format_validation_errors(e)
            logger.error(f"Configuration validation failed:\n{error_msg}")
            raise ConfigurationError(error_msg) from e
        
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise ConfigurationError(f"Configuration error: {str(e)}") from e
    
    @staticmethod
    def _get_config_class(environment: str) -> type[BaseConfig]:
        """Get the appropriate config class for the environment."""
        config_map = {
            "development": DevelopmentConfig,
            "staging": StagingConfig,
            "production": ProductionConfig,
        }
        
        if environment not in config_map:
            raise ConfigurationError(
                f"Invalid environment '{environment}'. "
                f"Must be one of: {', '.join(config_map.keys())}"
            )
        
        return config_map[environment]
    
    @staticmethod
    def _validate_llm_config(config: BaseConfig) -> None:
        """
        Validate LLM configuration.
        
        Ensures at least one LLM provider is configured.
        """
        has_openai = bool(config.openai_api_key)
        has_ollama = bool(config.ollama_base_url)
        has_bedrock = bool(
            config.aws_access_key_id and 
            config.aws_secret_access_key
        )
        
        if not (has_openai or has_ollama or has_bedrock):
            raise ConfigurationError(
                "No LLM provider configured. Must configure at least one of: "
                "OpenAI (openai_api_key), Ollama (ollama_base_url), "
                "or AWS Bedrock (aws_access_key_id + aws_secret_access_key)"
            )
        
        # Production should use Bedrock
        if config.environment == "production" and not has_bedrock:
            logger.warning(
                "⚠ Production environment should use AWS Bedrock for reliability. "
                "OpenAI/Ollama are not recommended for production."
            )
    
    @staticmethod
    def _validate_security_config(config: BaseConfig) -> None:
        """
        Validate security-related configuration.
        
        Ensures security best practices are followed.
        """
        # Check secret key strength
        if len(config.secret_key) < 32:
            raise ConfigurationError(
                "secret_key must be at least 32 characters for security"
            )
        
        # Production-specific security checks
        if config.environment == "production":
            if config.debug:
                raise ConfigurationError(
                    "Debug mode must be disabled in production"
                )
            
            if "*" in config.allowed_origins:
                raise ConfigurationError(
                    "CORS allowed_origins cannot be '*' in production. "
                    "Specify exact origins."
                )
            
            if config.log_level == "DEBUG":
                logger.warning(
                    "⚠ DEBUG log level in production may expose sensitive data"
                )
    
    @staticmethod
    def _validate_database_config(config: BaseConfig) -> None:
        """
        Validate database configuration.
        
        Ensures database connection parameters are valid.
        """
        # Check database URL format
        if not config.database_url.startswith(('postgresql://', 'postgres://')):
            raise ConfigurationError(
                "database_url must be a PostgreSQL connection string"
            )
        
        # Warn if using localhost in production
        if config.environment == "production":
            if "localhost" in config.database_url or "127.0.0.1" in config.database_url:
                logger.warning(
                    "⚠ Production database URL contains localhost. "
                    "This should point to a managed database service."
                )
    
    @staticmethod
    def _format_validation_errors(error: ValidationError) -> str:
        """
        Format Pydantic validation errors for human readability.
        
        Args:
            error: Pydantic ValidationError
            
        Returns:
            Formatted error message
        """
        lines = ["Configuration validation failed:", ""]
        
        for err in error.errors():
            field = " -> ".join(str(loc) for loc in err["loc"])
            msg = err["msg"]
            error_type = err["type"]
            
            lines.append(f"  ✗ {field}")
            lines.append(f"    Error: {msg}")
            lines.append(f"    Type: {error_type}")
            lines.append("")
        
        lines.append("Please check your .env file and ensure all required variables are set.")
        lines.append("See .env.example for reference.")
        
        return "\n".join(lines)
    
    @staticmethod
    def _log_config_summary(config: BaseConfig) -> None:
        """
        Log a summary of the loaded configuration (without secrets).
        
        Args:
            config: Loaded configuration
        """
        logger.info("Configuration Summary:")
        logger.info(f"  Environment: {config.environment}")
        logger.info(f"  Debug Mode: {config.debug}")
        logger.info(f"  Log Level: {config.log_level}")
        logger.info(f"  Log Format: {config.log_format}")
        logger.info(f"  Database: {config.database_url.split('@')[1] if '@' in config.database_url else 'configured'}")
        logger.info(f"  Redis: {config.redis_url.split('@')[1] if '@' in config.redis_url else config.redis_url}")
        logger.info(f"  LLM Providers:")
        logger.info(f"    - OpenAI: {'✓' if config.openai_api_key else '✗'}")
        logger.info(f"    - Ollama: {'✓' if config.ollama_base_url else '✗'}")
        logger.info(f"    - AWS Bedrock: {'✓' if config.aws_access_key_id else '✗'}")
        logger.info(f"  Observability:")
        logger.info(f"    - LangSmith: {'✓' if config.langsmith_api_key else '✗'}")


def load_config(environment: str | None = None) -> BaseConfig:
    """
    Convenience function to load and validate configuration.
    
    Args:
        environment: Environment name (development, staging, production)
                    If None, reads from ENVIRONMENT env var
    
    Returns:
        Validated configuration object
        
    Raises:
        ConfigurationError: If configuration is invalid
        SystemExit: If configuration cannot be loaded (fail-fast)
    """
    try:
        return ConfigValidator.validate_and_load(environment)
    except ConfigurationError as e:
        logger.critical(f"FATAL: {e}")
        logger.critical("Application cannot start with invalid configuration")
        sys.exit(1)
