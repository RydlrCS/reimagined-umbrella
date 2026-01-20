"""
Typed domain errors for stable API error codes.
"""
from typing import Optional


class KineticLedgerError(Exception):
    """Base exception for all Kinetic Ledger errors."""
    
    code: str = "E_UNKNOWN"
    
    def __init__(self, message: str, details: Optional[dict] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


# Configuration errors
class ConfigError(KineticLedgerError):
    """Configuration missing or invalid."""
    code = "E_CFG_INVALID"


class ConfigMissingError(ConfigError):
    """Required configuration is missing."""
    code = "E_CFG_MISSING"


# Dependency errors
class DependencyError(KineticLedgerError):
    """External dependency unavailable."""
    code = "E_DEP_UNAVAILABLE"


class GeminiError(DependencyError):
    """Gemini API error."""
    code = "E_DEP_GEMINI"


class CircleError(DependencyError):
    """Circle API error."""
    code = "E_DEP_CIRCLE"


class ArcError(DependencyError):
    """Arc Network error."""
    code = "E_DEP_ARC"


class RpcError(DependencyError):
    """RPC endpoint error."""
    code = "E_DEP_RPC"


class VectorDbError(DependencyError):
    """Vector database error."""
    code = "E_DEP_VECTORDB"


# Validation errors
class ValidationError(KineticLedgerError):
    """Data validation failure."""
    code = "E_VAL_INVALID"


class SchemaValidationError(ValidationError):
    """Schema validation failed."""
    code = "E_VAL_SCHEMA"


class HashMismatchError(ValidationError):
    """Hash verification failed."""
    code = "E_VAL_HASH_MISMATCH"


# Policy errors
class PolicyError(KineticLedgerError):
    """Policy check failed."""
    code = "E_POLICY_VIOLATION"


class SafetyPolicyError(PolicyError):
    """Safety policy violation."""
    code = "E_POLICY_SAFETY"


class RateLimitError(PolicyError):
    """Rate limit exceeded."""
    code = "E_POLICY_RATE_LIMIT"


class PaymentRequiredError(PolicyError):
    """Payment required for operation."""
    code = "E_POLICY_PAYMENT_REQUIRED"


# Transaction errors
class TransactionError(KineticLedgerError):
    """Blockchain transaction error."""
    code = "E_TX_FAILED"


class NonceError(TransactionError):
    """Nonce error or replay attack."""
    code = "E_TX_NONCE"


class SignatureError(TransactionError):
    """Signature verification failed."""
    code = "E_TX_SIGNATURE"


class SettlementError(TransactionError):
    """Settlement transaction failed."""
    code = "E_TX_SETTLEMENT"


# Decision errors
class DecisionError(KineticLedgerError):
    """Attestation decision error."""
    code = "E_DECISION"


class NoveltyRejectionError(DecisionError):
    """Motion rejected due to low novelty."""
    code = "E_DECISION_NOVELTY_REJECT"


class ManualReviewRequiredError(DecisionError):
    """Manual review required."""
    code = "E_DECISION_REVIEW_REQUIRED"


# Error code constants for direct import
E_CFG_MISSING = "E_CFG_MISSING"
E_DEP_ARC = "E_DEP_ARC"
E_DEP_GEMINI = "E_DEP_GEMINI"
E_DEP_CIRCLE = "E_DEP_CIRCLE"
E_DEP_RPC = "E_DEP_RPC"
