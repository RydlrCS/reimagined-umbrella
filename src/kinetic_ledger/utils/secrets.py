"""
Secret Manager integration with environment variable fallback.

Provides secure access to API keys and sensitive configuration using
Google Cloud Secret Manager in production, with local environment
variable fallback for development.
"""
import os
from typing import Optional

from .logging import get_logger
from .errors import ConfigMissingError

logger = get_logger(__name__)

# GCP Project ID for Secret Manager
_GCP_PROJECT_ID: str = "138150026675"

# Known secret paths (project-relative)
CIRCLE_TESTNET_API_KEY = "circle-testnet-api-key"
CIRCLE_TESTNET_CLIENT_KEY = "circle-testnet-client-key"
GEMINI_COMMS_KEY = "gemini-comms-key"

# Cache for fetched secrets (avoid repeated API calls)
_secret_cache: dict[str, str] = {}


def get_secret(
    secret_name: str,
    version: str = "latest",
    use_cache: bool = True,
) -> str:
    """
    Fetch a secret from Google Secret Manager with environment fallback.

    Order of precedence:
    1. In-memory cache (if use_cache=True)
    2. Environment variable (uppercase, underscores)
    3. Google Cloud Secret Manager

    Args:
        secret_name: Secret name (e.g., "circle-testnet-api-key")
        version: Secret version (default: "latest")
        use_cache: Whether to use cached values

    Returns:
        Secret value as string

    Raises:
        ConfigMissingError: If secret cannot be found

    Example:
        >>> api_key = get_secret("circle-testnet-api-key")
        >>> api_key = get_secret(CIRCLE_TESTNET_API_KEY)
    """
    logger.debug(f"[ENTRY] get_secret: secret_name={secret_name}, version={version}")

    # Check cache first
    cache_key = f"{secret_name}:{version}"
    if use_cache and cache_key in _secret_cache:
        logger.debug(f"[EXIT] get_secret: returning cached value for {secret_name}")
        return _secret_cache[cache_key]

    # Try environment variable (convert kebab-case to UPPER_SNAKE_CASE)
    env_name = secret_name.upper().replace("-", "_")
    env_value = os.getenv(env_name)
    if env_value:
        logger.info(f"Secret '{secret_name}' loaded from environment variable {env_name}")
        if use_cache:
            _secret_cache[cache_key] = env_value
        logger.debug(f"[EXIT] get_secret: returning env value for {secret_name}")
        return env_value

    # Try Google Cloud Secret Manager
    try:
        from google.cloud import secretmanager

        client = secretmanager.SecretManagerServiceClient()
        resource_name = f"projects/{_GCP_PROJECT_ID}/secrets/{secret_name}/versions/{version}"

        logger.info(f"Fetching secret from Secret Manager: {resource_name}")
        response = client.access_secret_version(request={"name": resource_name})
        secret_value = response.payload.data.decode("UTF-8")

        if use_cache:
            _secret_cache[cache_key] = secret_value

        logger.info(f"Secret '{secret_name}' loaded from Secret Manager")
        logger.debug(f"[EXIT] get_secret: returning Secret Manager value for {secret_name}")
        return secret_value

    except ImportError:
        logger.warning(
            "google-cloud-secret-manager not installed, "
            "falling back to environment variables only"
        )
        raise ConfigMissingError(
            f"Secret '{secret_name}' not found in environment and "
            "Secret Manager SDK not available"
        )
    except Exception as e:
        logger.error(f"Failed to fetch secret '{secret_name}' from Secret Manager: {e}")
        raise ConfigMissingError(
            f"Secret '{secret_name}' not found in environment or Secret Manager",
            details={"error": str(e)},
        )


def get_secret_optional(
    secret_name: str,
    version: str = "latest",
    default: Optional[str] = None,
) -> Optional[str]:
    """
    Fetch a secret, returning default if not found.

    Args:
        secret_name: Secret name
        version: Secret version
        default: Default value if secret not found

    Returns:
        Secret value or default
    """
    logger.debug(f"[ENTRY] get_secret_optional: secret_name={secret_name}")
    try:
        result = get_secret(secret_name, version)
        logger.debug(f"[EXIT] get_secret_optional: found secret {secret_name}")
        return result
    except ConfigMissingError:
        logger.debug(f"[EXIT] get_secret_optional: using default for {secret_name}")
        return default


def clear_secret_cache() -> None:
    """Clear the secret cache (useful for testing)."""
    logger.debug("[ENTRY] clear_secret_cache")
    _secret_cache.clear()
    logger.info("Secret cache cleared")
    logger.debug("[EXIT] clear_secret_cache")


def preload_secrets(*secret_names: str) -> dict[str, bool]:
    """
    Preload multiple secrets into cache.

    Args:
        *secret_names: Secret names to preload

    Returns:
        Dict mapping secret names to success status
    """
    logger.debug(f"[ENTRY] preload_secrets: {secret_names}")
    results: dict[str, bool] = {}

    for name in secret_names:
        try:
            get_secret(name)
            results[name] = True
            logger.info(f"Preloaded secret: {name}")
        except ConfigMissingError:
            results[name] = False
            logger.warning(f"Failed to preload secret: {name}")

    logger.debug(f"[EXIT] preload_secrets: {results}")
    return results
