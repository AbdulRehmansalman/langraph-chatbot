"""
Application Configuration
==========================
Main configuration module with environment-based settings.

SECURITY NOTICE:
- NO hardcoded secrets allowed
- All sensitive values MUST come from environment variables
- Configuration is validated on startup
- Application fails fast on misconfiguration

Usage:
    from app.core.config import settings
    
    # Access configuration
    db_url = settings.database_url
    api_key = settings.openai_api_key

Enterprise Features:
- Environment-specific configuration (dev/staging/prod)
- Startup validation with clear error messages
- Type-safe configuration access
- Fail-fast on misconfiguration
"""

import os
import logging
from typing import TYPE_CHECKING

from app.core.config_validator import load_config, ConfigurationError

if TYPE_CHECKING:
    from app.core.environments.base import BaseConfig

# Configure basic logging for config loading
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# ============================================================================
# LOAD AND VALIDATE CONFIGURATION
# ============================================================================

# Get environment from env var (defaults to development)
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")

logger.info(f"Initializing application in {ENVIRONMENT} environment")

try:
    # Load and validate configuration
    # This will fail fast if configuration is invalid
    settings: "BaseConfig" = load_config(ENVIRONMENT)
    
    logger.info("✓ Configuration validated successfully")
    logger.info(f"✓ Application: {settings.app_name} v{settings.app_version}")
    logger.info(f"✓ Environment: {settings.environment}")
    logger.info(f"✓ Debug Mode: {settings.debug}")
    
except ConfigurationError as e:
    logger.critical("=" * 80)
    logger.critical("CONFIGURATION ERROR")
    logger.critical("=" * 80)
    logger.critical(str(e))
    logger.critical("=" * 80)
    logger.critical("Application cannot start. Please fix configuration and restart.")
    logger.critical("=" * 80)
    raise SystemExit(1) from e

except Exception as e:
    logger.critical(f"Unexpected error loading configuration: {e}")
    raise SystemExit(1) from e

# ============================================================================
# EXPORT
# ============================================================================

__all__ = ["settings"]
