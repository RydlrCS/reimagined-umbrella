"""
Idempotency key generation and nonce handling.
"""
import hashlib
import time
from typing import Dict, Any
from .canonicalize import canonicalize_json


def generate_idempotency_key(data: Dict[str, Any], prefix: str = "idem") -> str:
    """
    Generate deterministic idempotency key from request data.
    
    Args:
        data: Request payload
        prefix: Key prefix for categorization
    
    Returns:
        Idempotency key string
    """
    canonical = canonicalize_json(data)
    digest = hashlib.sha256(canonical).hexdigest()
    return f"{prefix}_{digest[:16]}"


class NonceManager:
    """
    Simple nonce manager for preventing replay attacks.
    
    In production, this should be backed by Redis or a database.
    """
    
    def __init__(self):
        self._nonces: Dict[str, int] = {}
        self._used_nonces: Dict[str, set] = {}
    
    def get_next_nonce(self, account: str) -> int:
        """Get next nonce for account."""
        current = self._nonces.get(account, 0)
        self._nonces[account] = current + 1
        return current + 1
    
    def is_nonce_valid(self, account: str, nonce: int, expiry: int) -> bool:
        """
        Check if nonce is valid and not expired.
        
        Args:
            account: Account address
            nonce: Nonce value
            expiry: Expiry timestamp
        
        Returns:
            True if valid and not used
        """
        # Check expiry
        if expiry < int(time.time()):
            return False
        
        # Check if nonce was used
        if account not in self._used_nonces:
            self._used_nonces[account] = set()
        
        if nonce in self._used_nonces[account]:
            return False
        
        return True
    
    def mark_nonce_used(self, account: str, nonce: int) -> None:
        """Mark nonce as used."""
        if account not in self._used_nonces:
            self._used_nonces[account] = set()
        self._used_nonces[account].add(nonce)
    
    def cleanup_expired(self, current_time: int) -> None:
        """Cleanup expired nonces (would run periodically in production)."""
        # This is a simplified version - production would track expiry per nonce
        pass


# Global nonce manager instance
nonce_manager = NonceManager()
