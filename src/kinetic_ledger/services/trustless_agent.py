"""
Trustless Agent Loop - end-to-end orchestration of the Phase-2 workflow.

This orchestrator coordinates all services to implement the complete trustless agent loop:
1. Motion upload/ingest
2. Gemini multimodal analysis  
3. Attestation oracle validation (kNN + RkCNN)
4. Canonical pack creation and signing
5. Usage metering and USDC settlement
"""
import logging
import time
import uuid
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field

from ..schemas.models import (
    MotionBlendRequest,
    MotionAsset,
    GeminiAnalysis,
    SimilarityCheck,
    MotionCanonicalPack,
    MintAuthorization,
    UsageMeterEvent,
)
from ..utils.logging import setup_logging, set_correlation_id, get_correlation_id
from ..utils.errors import DecisionError, NoveltyRejectionError, ManualReviewRequiredError
from .motion_ingest import MotionIngestService, MotionUploadRequest, TensorGenerationConfig
from .gemini_analyzer import GeminiAnalyzerService, GeminiAnalysisRequest
from .attestation_oracle import AttestationOracle, AttestationConfig, VectorStore
from .commerce_orchestrator import (
    CommerceOrchestrator,
    CircleWalletConfig,
    PaymentIntentRequest,
)


logger = logging.getLogger(__name__)


class TrustlessAgentConfig(BaseModel):
    """Configuration for trustless agent loop."""
    # Service configs
    storage_url: str = "local://./data/storage"
    gemini_api_key: Optional[str] = None
    circle_api_key: str
    
    # Attestation
    novelty_threshold: float = Field(default=0.42, ge=0.0, le=1.0)
    chain_id: int = 1
    verifying_contract: str = Field(pattern=r"^0x[a-fA-F0-9]{40}$")
    
    # Commerce
    usdc_token_address: str = "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48"
    oracle_address: str = Field(pattern=r"^0x[a-fA-F0-9]{40}$")
    platform_address: str = Field(pattern=r"^0x[a-fA-F0-9]{40}$")
    ops_address: str = Field(pattern=r"^0x[a-fA-F0-9]{40}$")


class AgentLoopResult(BaseModel):
    """Result of trustless agent loop execution."""
    correlation_id: str
    motion_asset: MotionAsset
    gemini_analysis: GeminiAnalysis
    similarity_check: SimilarityCheck
    canonical_pack: MotionCanonicalPack
    mint_authorization: Optional[MintAuthorization] = None
    usage_event: Optional[UsageMeterEvent] = None
    decision: str  # "MINT", "REJECT", "REVIEW"
    pack_hash: str
    tx_hash: Optional[str] = None
    elapsed_seconds: float


class TrustlessAgentLoop:
    """
    Trustless Agent Loop Orchestrator.
    
    Coordinates the complete Phase-2 workflow:
    
    1. **Motion Ingest**: Upload BVH/FBX → generate tensors → create previews
    2. **Gemini Analysis**: Analyze previews → extract style labels/NPC tags
    3. **Attestation Oracle**: Run kNN + RkCNN → compute novelty → decide
    4. **Pack Creation**: Build MotionCanonicalPack v1 → compute pack_hash
    5. **Mint Authorization**: Sign EIP-712 payload (if MINT decision)
    6. **Usage Metering**: Verify x402 → settle USDC → route payouts
    
    All steps are idempotent and traceable via correlation IDs.
    """
    
    def __init__(self, config: TrustlessAgentConfig):
        self.config = config
        
        # Initialize services
        self.ingest_service = MotionIngestService(
            storage_url=config.storage_url,
        )
        
        self.gemini_service = GeminiAnalyzerService(
            gemini_api_key=config.gemini_api_key,
        )
        
        self.attestation_oracle = AttestationOracle(
            config=AttestationConfig(
                novelty_threshold=config.novelty_threshold,
                chain_id=config.chain_id,
                verifying_contract=config.verifying_contract,
            ),
            vector_store=VectorStore(),
        )
        
        self.commerce_orchestrator = CommerceOrchestrator(
            circle_config=CircleWalletConfig(
                api_key=config.circle_api_key,
            ),
            usdc_token_address=config.usdc_token_address,
        )
        
        setup_logging("trustless-agent-loop")
    
    def execute_blend_workflow(
        self,
        upload_request: MotionUploadRequest,
        blend_request: MotionBlendRequest,
        payment_proof: str,
        creator_address: str,
    ) -> AgentLoopResult:
        """
        Execute complete trustless agent loop workflow.
        
        Args:
            upload_request: Motion upload request
            blend_request: Blend request with NPC context and policy
            payment_proof: x402 payment proof
            creator_address: Creator wallet address for payout
        
        Returns:
            AgentLoopResult with all artifacts and decision
        """
        start_time = time.time()
        correlation_id = str(uuid.uuid4())
        set_correlation_id(correlation_id)
        
        logger.info(
            f"Starting trustless agent loop",
            extra={
                "correlation_id": correlation_id,
                "request_id": blend_request.request_id,
            },
        )
        
        # Step 1: Motion Ingest
        logger.info("Step 1/6: Motion Ingest")
        motion_asset = self.ingest_service.ingest_upload(
            request=upload_request,
            correlation_id=correlation_id,
        )
        
        # Step 2: Gemini Analysis
        logger.info("Step 2/6: Gemini Multimodal Analysis")
        gemini_request = GeminiAnalysisRequest(
            request_id=blend_request.request_id,
            preview_uri=motion_asset.preview.uri,
            blend_segments=[seg.model_dump() for seg in blend_request.blend_plan.segments],
            npc_context=blend_request.npc_context,
        )
        gemini_analysis = self.gemini_service.analyze(
            request=gemini_request,
            correlation_id=correlation_id,
        )
        
        # Step 3: Similarity Check (kNN + RkCNN)
        logger.info("Step 3/6: Attestation Oracle Validation")
        gemini_descriptors = self.gemini_service.build_query_descriptor(gemini_analysis)
        
        similarity_check = self.attestation_oracle.validate_similarity(
            analysis_id=gemini_analysis.analysis_id,
            tensor_hash=motion_asset.tensor.sha256,
            gemini_descriptors=gemini_descriptors,
            safety_flags=gemini_analysis.outputs.safety_flags,
            correlation_id=correlation_id,
        )
        
        decision = similarity_check.decision.result
        
        # Step 4: Create Canonical Pack
        logger.info("Step 4/6: Creating MotionCanonicalPack")
        canonical_pack = self.attestation_oracle.create_canonical_pack(
            motion_asset=motion_asset,
            blend_request=blend_request,
            gemini_analysis=gemini_analysis,
        )
        
        # Compute pack hash
        from ..utils.canonicalize import keccak256_json
        pack_hash = keccak256_json(canonical_pack.model_dump())
        
        # Step 5: Mint Authorization (if decision is MINT)
        mint_authorization = None
        if decision == "MINT":
            logger.info("Step 5/6: Signing Mint Authorization")
            mint_authorization = self.attestation_oracle.sign_mint_authorization(
                pack=canonical_pack,
                to_address=upload_request.owner_wallet,
            )
        elif decision == "REJECT":
            logger.warning("Motion rejected due to low novelty")
            raise NoveltyRejectionError(
                f"Motion rejected: {similarity_check.decision.reason}",
                details={
                    "separation_score": similarity_check.rkcnn.separation_score,
                    "threshold": similarity_check.decision.novelty_threshold,
                },
            )
        else:  # REVIEW
            logger.warning("Motion requires manual review")
            raise ManualReviewRequiredError(
                f"Manual review required: {similarity_check.decision.reason}",
                details={"safety_flags": gemini_analysis.outputs.safety_flags},
            )
        
        # Step 6: Usage Metering & Settlement (if MINT)
        usage_event = None
        tx_hash = None
        if decision == "MINT" and mint_authorization:
            logger.info("Step 6/6: Usage Metering & USDC Settlement")
            
            # Calculate usage from NPC generation request
            max_seconds = blend_request.policy.get("max_seconds", 10)
            
            payment_request = PaymentIntentRequest(
                user_wallet=blend_request.user_wallet,
                product="npc_generation",
                unit="seconds_generated",
                quantity=float(max_seconds),
                unit_price_usdc="0.50",  # $0.50 per second
                attestation_pack_hash=pack_hash,
            )
            
            usage_event = self.commerce_orchestrator.meter_usage(
                request=payment_request,
                payment_proof=payment_proof,
                creator_address=creator_address,
                oracle_address=self.config.oracle_address,
                platform_address=self.config.platform_address,
                ops_address=self.config.ops_address,
                correlation_id=correlation_id,
            )
            
            tx_hash = usage_event.settlement.tx_hash
        
        elapsed = time.time() - start_time
        
        result = AgentLoopResult(
            correlation_id=correlation_id,
            motion_asset=motion_asset,
            gemini_analysis=gemini_analysis,
            similarity_check=similarity_check,
            canonical_pack=canonical_pack,
            mint_authorization=mint_authorization,
            usage_event=usage_event,
            decision=decision,
            pack_hash=pack_hash,
            tx_hash=tx_hash,
            elapsed_seconds=elapsed,
        )
        
        logger.info(
            f"Trustless agent loop complete: {decision}",
            extra={
                "correlation_id": correlation_id,
                "decision": decision,
                "pack_hash": pack_hash,
                "tx_hash": tx_hash,
                "elapsed_seconds": elapsed,
            },
        )
        
        return result
    
    def execute_library_blend(
        self,
        blend_request: MotionBlendRequest,
        library_motion_ids: List[str],
        payment_proof: str,
        creator_address: str,
    ) -> AgentLoopResult:
        """
        Execute workflow for blending library motions.
        
        Args:
            blend_request: Blend request
            library_motion_ids: IDs of library motions to blend
            payment_proof: Payment proof
            creator_address: Creator address
        
        Returns:
            AgentLoopResult
        """
        # Similar to execute_blend_workflow but uses library motions
        # For demo: simplified version that reuses upload workflow
        logger.info("Library blend workflow not fully implemented - using upload workflow")
        
        # Would fetch library motions and blend them
        # For now, raise not implemented
        raise NotImplementedError("Library blend workflow coming soon")
