"""Services package - all Phase-2 microservices."""
from .motion_ingest import MotionIngestService, MotionUploadRequest, TensorGenerationConfig
from .gemini_analyzer import GeminiAnalyzerService, GeminiAnalysisRequest
from .attestation_oracle import AttestationOracle, AttestationConfig, VectorStore
from .commerce_orchestrator import (
    CommerceOrchestrator,
    CircleWalletConfig,
    PaymentIntentRequest,
)
from .trustless_agent import TrustlessAgentLoop, TrustlessAgentConfig, AgentLoopResult

# V2 Services - Circle Wallets, x402, Royalty Chains, Gasless
try:
    from .circle_wallets import CircleWalletsService, CircleWalletsConfig
    from .x402_facilitator import X402FacilitatorService, X402FacilitatorConfig
    from .payment_automation import PaymentAutomationService, PaymentAutomationConfig
    from .gasless_executor import GaslessExecutor, GaslessExecutorConfig
    from .gemini_oracle_attestor import GeminiOracleAttestor, GeminiOracleConfig
    
    _V2_EXPORTS = [
        "CircleWalletsService",
        "CircleWalletsConfig",
        "X402FacilitatorService",
        "X402FacilitatorConfig",
        "PaymentAutomationService",
        "PaymentAutomationConfig",
        "GaslessExecutor",
        "GaslessExecutorConfig",
        "GeminiOracleAttestor",
        "GeminiOracleConfig",
    ]
except ImportError:
    _V2_EXPORTS = []

# SPADE Hierarchical Motion Blending (requires PyTorch)
try:
    from .spade_blend_service import (
        SPADEBlendService,
        get_spade_service,
        reset_spade_service,
        generate_hash_embedding,
        TORCH_AVAILABLE,
    )
    from .metrics import (
        compute_fid,
        compute_coverage,
        compute_diversity,
        compute_smoothness,
        compute_foot_sliding,
        compute_transition_quality,
        compute_all_metrics,
        MotionMetricsResult,
    )
    
    _SPADE_EXPORTS = [
        "SPADEBlendService",
        "get_spade_service",
        "reset_spade_service",
        "generate_hash_embedding",
        "TORCH_AVAILABLE",
        "compute_fid",
        "compute_coverage",
        "compute_diversity",
        "compute_smoothness",
        "compute_foot_sliding",
        "compute_transition_quality",
        "compute_all_metrics",
        "MotionMetricsResult",
    ]
except ImportError as e:
    import logging
    logging.getLogger(__name__).debug(f"SPADE services unavailable: {e}")
    _SPADE_EXPORTS = []

__all__ = [
    "MotionIngestService",
    "MotionUploadRequest",
    "TensorGenerationConfig",
    "GeminiAnalyzerService",
    "GeminiAnalysisRequest",
    "AttestationOracle",
    "AttestationConfig",
    "VectorStore",
    "CommerceOrchestrator",
    "CircleWalletConfig",
    "PaymentIntentRequest",
    "TrustlessAgentLoop",
    "TrustlessAgentConfig",
    "AgentLoopResult",
] + _V2_EXPORTS + _SPADE_EXPORTS
