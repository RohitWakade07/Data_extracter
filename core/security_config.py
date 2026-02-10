"""
Security and Configuration Management
======================================

Handles secrets, credentials, and secure configuration.
"""

import os
from typing import Optional, Dict, Any
from dataclasses import dataclass
from core.logging_config import Logger


logger = Logger(__name__)


@dataclass
class Credentials:
    """Secure credentials holder"""
    openai_api_key: Optional[str] = None
    nebula_password: Optional[str] = None
    weaviate_api_key: Optional[str] = None
    custom_secrets: Optional[Dict[str, str]] = None

    def __post_init__(self):
        if self.custom_secrets is None:
            self.custom_secrets = {}

    def get_secret(self, name: str, default: Optional[str] = None) -> Optional[str]:
        """Retrieve a secret safely"""
        # Check environment variable first
        env_key = name.upper()
        if env_key in os.environ:
            return os.environ[env_key]

        # Check stored secrets
        if self.custom_secrets is not None and name.lower() in self.custom_secrets:
            return self.custom_secrets[name.lower()]

        return default


class SecureConfig:
    """Secure configuration management with secret protection"""

    def __init__(self, env_file: Optional[str] = None):
        """
        Initialize configuration.

        Args:
            env_file: Optional path to .env file
        """
        self.env_file = env_file
        self.credentials = Credentials()
        self._load_configuration()

    def _load_configuration(self):
        """Load configuration from environment and .env file"""
        try:
            # Load from .env file if provided
            if self.env_file and os.path.exists(self.env_file):
                self._load_env_file(self.env_file)

            # Load from environment
            self.credentials.openai_api_key = os.getenv("OPENAI_API_KEY")
            self.credentials.nebula_password = os.getenv("NEBULA_PASSWORD", "nebula")
            self.credentials.weaviate_api_key = os.getenv("WEAVIATE_API_KEY")

            self._validate_critical_secrets()

        except Exception as e:
            logger.error(f"Configuration loading failed: {e}", exc_info=True)

    def _load_env_file(self, env_file: str):
        """Load environment variables from .env file"""
        try:
            from dotenv import load_dotenv
            load_dotenv(env_file)
            logger.info(f"Loaded configuration from {env_file}")
        except ImportError:
            logger.warning("python-dotenv not installed, skipping .env loading")
        except Exception as e:
            logger.error(f"Failed to load .env file: {e}")

    def _validate_critical_secrets(self):
        """Validate that critical secrets are available"""
        if not self.credentials.openai_api_key:
            logger.warning("OpenAI API key not configured. Vector embeddings will fail.")

    def get(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """Get a configuration value"""
        return self.credentials.get_secret(key, default)

    @property
    def openai_api_key(self) -> Optional[str]:
        """Get OpenAI API key (masked in logs)"""
        return self.credentials.openai_api_key

    @property
    def nebula_password(self) -> str:
        """Get Nebula password"""
        return self.credentials.nebula_password or "nebula"

    @property
    def weaviate_api_key(self) -> Optional[str]:
        """Get Weaviate API key"""
        return self.credentials.weaviate_api_key


class SecretsMask:
    """Utility for masking secrets in logs"""

    SENSITIVE_KEYS = [
        "api_key", "api_password", "password", "secret",
        "token", "authorization", "x-api-key"
    ]

    @classmethod
    def mask_dict(cls, data: Dict[str, Any], depth: int = 0) -> Dict[str, Any]:
        """Recursively mask sensitive values in a dictionary"""
        if depth > 5:  # Prevent infinite recursion
            return data

        masked = {}
        for key, value in data.items():
            if cls._is_sensitive_key(key):
                masked[key] = "***MASKED***"
            elif isinstance(value, dict):
                masked[key] = cls.mask_dict(value, depth + 1)
            elif isinstance(value, list) and value and isinstance(value[0], dict):
                masked[key] = [cls.mask_dict(item, depth + 1) for item in value]
            else:
                masked[key] = value

        return masked

    @classmethod
    def _is_sensitive_key(cls, key: str) -> bool:
        """Check if a key is sensitive"""
        key_lower = key.lower()
        return any(sensitive in key_lower for sensitive in cls.SENSITIVE_KEYS)

    @classmethod
    def mask_string(cls, text: str, patterns: Optional[list] = None) -> str:
        """Mask sensitive information in strings"""
        if not patterns:
            patterns = [
                (r"api.?key[=:\s]+['\"]?([^'\"\\s]+)['\"]?", "***MASKED***"),
                (r"password[=:\s]+['\"]?([^'\"\\s]+)['\"]?", "***MASKED***"),
                (r"Bearer\s+\S+", "Bearer ***MASKED***"),
            ]

        import re
        masked = text
        for pattern, replacement in patterns:
            masked = re.sub(pattern, replacement, masked, flags=re.IGNORECASE)

        return masked
