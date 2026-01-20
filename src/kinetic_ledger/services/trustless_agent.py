"""
Trustless Agent Loop - end-to-end orchestration of the Phase-2 workflow.

This orchestrator coordinates all services to implement the complete trustless agent loop:
1. Motion upload/ingest
2. Gemini multimodal analysis  
3. Attestation oracle validation (kNN + RkCNN)
4. Canonical pack creation and signing
5. On-chain vs off-chain transaction routing
6. Usage metering and USDC settlement
"""
import logging
import time
import uuid
from typing import Optional, Dict, Any, List, Literal
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
from .arc_network import ArcNetworkService, ArcNetworkConfig
from .commerce_orchestrator import (
    CommerceOrchestrator,
    CircleWalletConfig,
    PaymentIntentRequest,
)


logger = logging.getLogger(__name__)


class TransactionRouting(BaseModel):
    """Transaction routing decision for on-chain vs off-chain."""
    strategy: Literal["on-chain", "off-chain", "hybrid"]
    reason: str
    on_chain_operations: List[str] = Field(default_factory=list)
    off_chain_operations: List[str] = Field(default_factory=list)
    estimated_gas_usdc: Optional[str] = None
    use_arc_network: bool = False
    use_circle_wallets: bool = False


class TrustlessAgentConfig(BaseModel):
    """Configuration for trustless agent loop."""
    # Service configs
    storage_url: str = "local://./data/storage"
    gemini_api_key: Optional[str] = None
    circle_api_key: Optional[str] = None
    
    # Arc Network (on-chain)
    arc_rpc_url: Optional[str] = None
    arc_contract_address: Optional[str] = None
    arc_private_key: Optional[str] = None
    
    # Attestation
    novelty_threshold: float = Field(default=0.42, ge=0.0, le=1.0)
    chain_id: int = 1
    verifying_contract: str = Field(pattern=r"^0x[a-fA-F0-9]{40}$")
    
    # Commerce
    usdc_token_address: str = "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48"
    oracle_address: str = Field(pattern=r"^0x[a-fA-F0-9]{40}$")
    platform_address: str = Field(pattern=r"^0x[a-fA-F0-9]{40}$")
    ops_address: str = Field(pattern=r"^0x[a-fA-F0-9]{40}$")
    
    # Transaction routing preferences
    default_routing: Literal["on-chain", "off-chain", "hybrid"] = "hybrid"
    on_chain_threshold_usdc: float = 100.0  # Use on-chain for amounts >= $100
    force_on_chain_for_nfts: bool = True  # Always use on-chain for NFT minting


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
    routing: Optional[TransactionRouting] = None  # NEW: routing decision
    on_chain_tx_hash: Optional[str] = None  # NEW: on-chain transaction
    off_chain_tx_id: Optional[str] = None  # NEW: off-chain transaction
    tx_hash: Optional[str] = None  # Deprecated: use on_chain_tx_hash
    elapsed_seconds: float


class TrustlessAgentLoop:
    """
    Trustless Agent Loop Orchestrator.
    
    Coordinates the complete Phase-2 workflow with on-chain/off-chain routing:
    
    1. **Motion Ingest**: Upload BVH/FBX → generate tensors → create previews
    2. **Gemini Analysis**: Analyze previews → extract style labels/NPC tags
    3. **Attestation Oracle**: Run kNN + RkCNN → compute novelty → decide
    4. **Pack Creation**: Build MotionCanonicalPack v1 → compute pack_hash
    5. **Transaction Routing**: Decide on-chain (Arc) vs off-chain (Circle) based on:
       - Amount threshold ($100+ → on-chain)
       - NFT minting (always on-chain)
       - Payment complexity (multi-party → on-chain)
       - Gas costs (low-value → off-chain)
    6. **Mint Authorization**: Sign EIP-712 payload (if MINT decision)
    7. **Settlement Execution**:
       - **On-chain**: Submit to Arc Network smart contract (USDC gas token)
       - **Off-chain**: Execute via Circle Programmable Wallets API
       - **Hybrid**: Mint on-chain + payments off-chain
    8. **Usage Metering**: Verify x402 → settle USDC → route payouts
    
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
        
        # Arc Network service (on-chain) - optional
        self.arc_service = None
        if config.arc_rpc_url and config.arc_contract_address and config.arc_private_key:
            try:
                self.arc_service = ArcNetworkService(
                    config=ArcNetworkConfig(
                        rpc_url=config.arc_rpc_url,
                        contract_address=config.arc_contract_address,
                        private_key=config.arc_private_key,
                    )
                )
                logger.info("Arc Network service initialized (on-chain enabled)")
            except Exception as e:
                logger.warning(f"Arc Network service unavailable: {e}")
        
        # Commerce orchestrator (off-chain) - optional
        self.commerce_orchestrator = None
        if config.circle_api_key:
            try:
                self.commerce_orchestrator = CommerceOrchestrator(
                    circle_config=CircleWalletConfig(
                        api_key=config.circle_api_key,
                    ),
                    usdc_token_address=config.usdc_token_address,
                )
                logger.info("Circle Wallets service initialized (off-chain enabled)")
            except Exception as e:
                logger.warning(f"Circle Wallets service unavailable: {e}")
        
        setup_logging("trustless-agent-loop")
    
    def decide_transaction_routing(
        self,
        operation_type: Literal["nft_mint", "payment", "usage_metering"],
        amount_usdc: float,
        multi_party: bool = False,
    ) -> TransactionRouting:
        """
        Decide whether to use on-chain (Arc) vs off-chain (Circle) transactions.
        
        Routing Logic:
        1. **NFT Minting**: Always on-chain (immutability, ownership proof)
        2. **High-Value Payments** (≥$100): On-chain (transparency, auditability)
        3. **Multi-Party Settlements**: On-chain (atomic execution, trust)
        4. **Low-Value Micropayments** (<$100): Off-chain (gas efficiency)
        5. **Hybrid**: Mint on-chain + royalties off-chain
        
        Args:
            operation_type: Type of operation
            amount_usdc: Transaction amount in USDC
            multi_party: Whether multiple parties receive payouts
        
        Returns:
            TransactionRouting decision
        """
        # Check service availability
        has_arc = self.arc_service is not None
        has_circle = self.commerce_orchestrator is not None
        
        # Default routing
        strategy = self.config.default_routing
        reason = f"Default routing: {strategy}"
        on_chain_ops = []
        off_chain_ops = []
        
        # Rule 1: NFT minting always on-chain (if available)
        if operation_type == "nft_mint":
            if has_arc and self.config.force_on_chain_for_nfts:
                strategy = "on-chain"
                reason = "NFT minting requires on-chain immutability"
                on_chain_ops = ["mint_motion_pack", "record_canonical_hash"]
            elif has_circle:
                strategy = "off-chain"
                reason = "Arc Network unavailable - using off-chain fallback"
                off_chain_ops = ["mint_simulation"]
            else:
                raise DecisionError("No transaction infrastructure available")
        
        # Rule 2: High-value transactions on-chain
        elif amount_usdc >= self.config.on_chain_threshold_usdc:
            if has_arc:
                strategy = "on-chain"
                reason = f"Amount ${amount_usdc:.2f} ≥ threshold ${self.config.on_chain_threshold_usdc:.2f}"
                on_chain_ops = ["usdc_settlement", "royalty_distribution"]
            elif has_circle:
                strategy = "off-chain"
                reason = "Arc unavailable for high-value tx - using Circle"
                off_chain_ops = ["usdc_transfer", "royalty_transfers"]
        
        # Rule 3: Multi-party settlements prefer on-chain (atomic)
        elif multi_party:
            if has_arc:
                strategy = "on-chain"
                reason = "Multi-party settlement requires atomic on-chain execution"
                on_chain_ops = ["atomic_multi_payout"]
            elif has_circle:
                strategy = "off-chain"
                reason = "Multi-party via Circle sequential transfers"
                off_chain_ops = ["sequential_transfers"]
        
        # Rule 4: Low-value micropayments off-chain (gas efficient)
        else:
            if has_circle:
                strategy = "off-chain"
                reason = f"Low-value ${amount_usdc:.2f} - gas-efficient off-chain"
                off_chain_ops = ["usdc_transfer"]
            elif has_arc:
                strategy = "on-chain"
                reason = "Circle unavailable - using Arc (may have high gas)"
                on_chain_ops = ["usdc_settlement"]
        
        # Hybrid strategy: NFT on-chain + payments off-chain
        if operation_type == "nft_mint" and has_arc and has_circle:
            strategy = "hybrid"
            reason = "Hybrid: NFT minting on-chain + royalties off-chain"
            on_chain_ops = ["mint_motion_pack"]
            off_chain_ops = ["royalty_transfers"]
        
        # Estimate gas costs (Arc uses USDC as gas token)
        estimated_gas_usdc = None
        if strategy in ("on-chain", "hybrid") and has_arc:
            # Rough estimates: mint ~$0.10, transfer ~$0.01 on Arc L2
            gas_cost = 0.10 if "mint" in operation_type else 0.01
            estimated_gas_usdc = f"{gas_cost:.2f}"
        
        return TransactionRouting(
            strategy=strategy,
            reason=reason,
            on_chain_operations=on_chain_ops,
            off_chain_operations=off_chain_ops,
            estimated_gas_usdc=estimated_gas_usdc,
            use_arc_network=has_arc and strategy in ("on-chain", "hybrid"),
            use_circle_wallets=has_circle and strategy in ("off-chain", "hybrid"),
        )
    
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
        routing = None
        on_chain_tx_hash = None
        off_chain_tx_id = None
        
        if decision == "MINT":
            logger.info("Step 5/6: Transaction Routing Decision")
            
            # Calculate total payment amount
            max_seconds = blend_request.policy.get("max_seconds", 10)
            unit_price = 0.50  # $0.50 per second
            total_usdc = max_seconds * unit_price
            
            # Decide on-chain vs off-chain routing
            routing = self.decide_transaction_routing(
                operation_type="nft_mint",
                amount_usdc=total_usdc,
                multi_party=True,  # Creator + oracle + platform + ops
            )
            
            logger.info(
                f"Routing decision: {routing.strategy}",
                extra={
                    "strategy": routing.strategy,
                    "reason": routing.reason,
                    "amount_usdc": total_usdc,
                    "on_chain_ops": routing.on_chain_operations,
                    "off_chain_ops": routing.off_chain_operations,
                },
            )
            
            # Execute on-chain operations (Arc Network)
            if routing.use_arc_network and self.arc_service:
                logger.info("Executing on-chain operations via Arc Network")
                
                # Sign mint authorization
                mint_authorization = self.attestation_oracle.sign_mint_authorization(
                    pack=canonical_pack,
                    to_address=upload_request.owner_wallet,
                )
                
                # Submit to Arc Network smart contract
                try:
                    arc_result = self.arc_service.mint_motion_pack(
                        to_address=upload_request.owner_wallet,
                        canonical_hash=pack_hash,
                        motion_data_uri=motion_asset.tensor.uri,
                        creator_address=creator_address,
                        oracle_signature=mint_authorization.signature,
                    )
                    on_chain_tx_hash = arc_result["tx_hash"]
                    logger.info(
                        f"Arc Network mint successful: {on_chain_tx_hash}",
                        extra={"tx_hash": on_chain_tx_hash},
                    )
                except Exception as e:
                    logger.error(f"Arc Network mint failed: {e}")
                    # Fallback to off-chain if available
                    if routing.strategy == "hybrid" and routing.use_circle_wallets:
                        logger.warning("Falling back to off-chain for mint simulation")
                    else:
                        raise
            else:
                # Sign authorization for potential later on-chain submission
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
        
        # Step 6: Usage Metering & Settlement
        usage_event = None
        if decision == "MINT" and mint_authorization:
            logger.info("Step 6/6: Usage Metering & Payment Settlement")
            
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
            
            # Execute off-chain payment operations (Circle Wallets)
            if routing and routing.use_circle_wallets and self.commerce_orchestrator:
                logger.info("Executing off-chain payment operations via Circle Wallets")
                
                try:
                    usage_event = self.commerce_orchestrator.meter_usage(
                        request=payment_request,
                        payment_proof=payment_proof,
                        creator_address=creator_address,
                        oracle_address=self.config.oracle_address,
                        platform_address=self.config.platform_address,
                        ops_address=self.config.ops_address,
                        correlation_id=correlation_id,
                    )
                    
                    off_chain_tx_id = usage_event.settlement.tx_hash
                    logger.info(
                        f"Circle payment settlement successful",
                        extra={
                            "off_chain_tx_id": off_chain_tx_id,
                            "total_usdc": usage_event.metering.total_usdc,
                        },
                    )
                except Exception as e:
                    logger.error(f"Circle payment failed: {e}")
                    # If hybrid mode and Arc available, could retry on-chain
                    if routing and routing.strategy == "hybrid" and routing.use_arc_network:
                        logger.warning("Circle payment failed, but NFT already minted on-chain")
                    raise
            
            # Execute on-chain payment operations (Arc Network)
            elif routing and routing.use_arc_network and self.arc_service:
                logger.info("Executing on-chain payment via Arc Network smart contract")
                
                try:
                    # Record usage and execute atomic multi-party payout on-chain
                    total_usdc_wei = int(float(max_seconds) * 0.50 * 1_000_000)  # USDC has 6 decimals
                    
                    payout_result = self.arc_service.record_usage_and_pay(
                        motion_pack_id=0,  # Retrieved from mint result
                        usage_amount=total_usdc_wei,
                        creator_address=creator_address,
                    )
                    
                    on_chain_tx_hash = payout_result["tx_hash"]
                    logger.info(
                        f"Arc Network payment successful: {on_chain_tx_hash}",
                        extra={"tx_hash": on_chain_tx_hash},
                    )
                    
                    # Create usage event for logging
                    from ..schemas.models import Metering, X402, Settlement, PayoutItem
                    usage_event = UsageMeterEvent(
                        usage_id=str(uuid.uuid4()),
                        created_at=int(time.time()),
                        user_wallet=blend_request.user_wallet,
                        attestation_pack_hash=pack_hash,
                        product="npc_generation",
                        metering=Metering(
                            unit="seconds_generated",
                            quantity=float(max_seconds),
                            unit_price_usdc="0.50",
                            total_usdc=f"{max_seconds * 0.50:.6f}",
                        ),
                        x402=X402(
                            payment_proof=payment_proof,
                            facilitator_receipt_id="on-chain",
                            verified=True,
                        ),
                        settlement=Settlement(
                            chain="ARC",
                            token=self.config.usdc_token_address,
                            tx_hash=on_chain_tx_hash,
                        ),
                        payout_split=[
                            PayoutItem(to=creator_address, amount_usdc="70%", label="creator"),
                            PayoutItem(to=self.config.oracle_address, amount_usdc="10%", label="oracle"),
                            PayoutItem(to=self.config.platform_address, amount_usdc="15%", label="platform"),
                            PayoutItem(to=self.config.ops_address, amount_usdc="5%", label="ops"),
                        ],
                    )
                except Exception as e:
                    logger.error(f"Arc Network payment failed: {e}")
                    raise
            else:
                logger.warning("No payment infrastructure available - payment simulation only")
        
        elapsed = time.time() - start_time
        
        # Use on_chain_tx_hash if available, fallback to off_chain_tx_id for legacy tx_hash field
        tx_hash = on_chain_tx_hash or off_chain_tx_id
        
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
            routing=routing,
            on_chain_tx_hash=on_chain_tx_hash,
            off_chain_tx_id=off_chain_tx_id,
            tx_hash=tx_hash,  # Legacy field
            elapsed_seconds=elapsed,
        )
        
        logger.info(
            f"Trustless agent loop complete: {decision}",
            extra={
                "correlation_id": correlation_id,
                "decision": decision,
                "routing_strategy": routing.strategy if routing else "none",
                "pack_hash": pack_hash,
                "on_chain_tx": on_chain_tx_hash,
                "off_chain_tx": off_chain_tx_id,
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
