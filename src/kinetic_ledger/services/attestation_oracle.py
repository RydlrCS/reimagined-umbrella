"""
Attestation Oracle - validates motion novelty and signs mint authorizations.

Uses hybrid architecture:
- Gemini File Search: Natural language discovery and semantic search
- kNN + RkCNN: Precise embedding-based similarity for novelty detection
"""
import hashlib
import logging
import time
import uuid
import numpy as np
from typing import List, Optional, Tuple, Dict, Any
from pydantic import BaseModel, Field

from ..schemas.models import (
    SimilarityCheck,
    FeatureSpace,
    KNNNeighbor,
    KNNResult,
    RkCNNResult,
    Decision,
    MotionCanonicalPack,
    MintAuthorization,
    MintMessage,
    RoyaltyObligation,
    DerivativeDetectionResult,
    RoyaltyChain,
    RoyaltyNode,
    ROYALTY_DECAY_FACTOR,
    MAX_ROYALTY_CHAIN_DEPTH,
    DEFAULT_CREATOR_SHARE_BPS,
    DEFAULT_ORACLE_SHARE_BPS,
    DEFAULT_PLATFORM_SHARE_BPS,
    DEFAULT_OPS_SHARE_BPS,
)
from ..utils.logging import setup_logging, set_correlation_id
from ..utils.errors import DecisionError, NoveltyRejectionError, ManualReviewRequiredError
from ..utils.canonicalize import keccak256_json
from ..utils.idempotency import nonce_manager
from .similarity import knn, rkcnn
from ..connectors.file_search_connector import FileSearchConnector


logger = logging.getLogger(__name__)


class AttestationConfig(BaseModel):
    """Configuration for attestation oracle."""
    knn_k: int = Field(default=15, ge=1)
    rkcnn_k: int = Field(default=15, ge=1)
    rkcnn_ensembles: Optional[int] = None  # Auto-computed if None
    rkcnn_subspace_dim: Optional[int] = None  # Auto-computed if None
    novelty_threshold: float = Field(default=0.42, ge=0.0, le=1.0)
    vote_margin_threshold: float = Field(default=0.10, ge=0.0, le=1.0)
    distance_metric: str = Field(default="euclidean", pattern=r"^(euclidean|cosine)$")
    embedding_dim: int = Field(default=512, ge=128)
    embedding_model_id: str = "motion_encoder_v1"
    
    # Derivative detection
    derivative_threshold: float = Field(
        default=0.85, ge=0.0, le=1.0,
        description="Similarity threshold for derivative detection"
    )
    default_derivative_royalty: float = Field(
        default=0.10, ge=0.0, le=1.0,
        description="Default royalty percentage for derivatives"
    )
    
    # EIP-712 signing
    chain_id: int = Field(default=1, ge=1)
    verifying_contract: str = Field(pattern=r"^0x[a-fA-F0-9]{40}$")
    validator_private_key: Optional[str] = None  # Set via env in production
    attestation_expiry_seconds: int = Field(default=600, ge=60)


class VectorStore:
    """
    Simple in-memory vector store for demo.
    
    In production: integrate with Qdrant, pgvector, or Pinecone.
    """
    
    def __init__(self):
        self._vectors: Dict[str, np.ndarray] = {}
    
    def add(self, motion_id: str, embedding: np.ndarray) -> None:
        """Add embedding to store."""
        self._vectors[motion_id] = embedding
    
    def get_all(self) -> List[Tuple[str, np.ndarray]]:
        """Get all vectors as list of (id, vector) tuples."""
        return list(self._vectors.items())
    
    def size(self) -> int:
        """Get number of vectors."""
        return len(self._vectors)


class AttestationOracle:
    """
    Attestation Oracle Service.
    
    Hybrid architecture combining:
    1. Gemini File Search: Natural language discovery, semantic search, grounding
    2. kNN: Baseline similarity using cached embeddings
    3. RkCNN: High-dimensional robustness for novelty detection
    
    Responsibilities:
    1. Build query vectors from tensor features + Gemini descriptors
    2. Run kNN for baseline similarity (using embedding cache)
    3. Run RkCNN ensembles for high-dimensional robustness
    4. Compute separation score and vote margin
    5. Make novelty decision: MINT, REJECT, or REVIEW
    6. Create MotionCanonicalPack v1 and compute pack_hash
    7. Sign EIP-712 mint authorization
    """
    
    def __init__(
        self,
        config: AttestationConfig,
        vector_store: Optional[VectorStore] = None,
        file_search: Optional[FileSearchConnector] = None,
    ):
        self.config = config
        self.vector_store = vector_store or VectorStore()
        self.file_search = file_search or FileSearchConnector.get_instance()
        setup_logging("attestation-oracle")
    
    def _create_query_vector(
        self,
        tensor_hash: str,
        gemini_descriptors: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:
        """
        Create query vector from tensor hash and Gemini descriptors.
        
        In production:
        1. Load tensor features from storage
        2. Encode with neural network
        3. Combine with Gemini semantic embeddings
        
        For demo: create deterministic vector from hashes.
        
        Args:
            tensor_hash: SHA256 hash of tensor
            gemini_descriptors: Gemini analysis outputs
        
        Returns:
            Query embedding vector
        """
        # Deterministic seed from tensor hash
        seed_bytes = bytes.fromhex(tensor_hash[2:])  # Remove 0x prefix
        seed = int.from_bytes(seed_bytes[:8], "big")
        
        rng = np.random.default_rng(seed)
        embedding = rng.normal(size=(self.config.embedding_dim,)).astype(np.float32)
        
        # Normalize
        embedding = embedding / (np.linalg.norm(embedding) + 1e-9)
        
        # In production: blend with Gemini semantic features
        # For now: deterministic simulation
        
        return embedding
    
    def _decide_novelty(
        self,
        separation_score: float,
        vote_margin: float,
        safety_flags: List[str],
    ) -> Tuple[str, List[str]]:
        """
        Make novelty decision based on separation score and vote margin.
        
        Decision logic:
        - separation >= threshold AND no safety flags => MINT
        - separation < threshold => REJECT (too similar)
        - safety flags present => REVIEW (manual)
        
        Args:
            separation_score: Separation score from RkCNN
            vote_margin: Vote margin from RkCNN
            safety_flags: Safety flags from Gemini
        
        Returns:
            Tuple of (decision, reason_codes)
        """
        reason_codes = []
        
        # Check safety flags first
        if safety_flags:
            return "REVIEW", ["SAFETY_FLAGS_PRESENT"] + safety_flags
        
        # Check novelty
        if separation_score >= self.config.novelty_threshold:
            reason_codes.append("NOVELTY_PASS")
            
            # Check vote margin for consensus
            if vote_margin >= self.config.vote_margin_threshold:
                reason_codes.append("VOTE_CONSENSUS")
            else:
                reason_codes.append("VOTE_WEAK")
            
            return "MINT", reason_codes
        else:
            reason_codes.append("NOVELTY_FAIL")
            reason_codes.append(f"SEPARATION_{separation_score:.3f}_BELOW_{self.config.novelty_threshold}")
            return "REJECT", reason_codes
    
    def validate_similarity(
        self,
        analysis_id: str,
        tensor_hash: str,
        gemini_descriptors: Optional[Dict[str, Any]] = None,
        safety_flags: Optional[List[str]] = None,
        correlation_id: Optional[str] = None,
    ) -> SimilarityCheck:
        """
        Validate motion similarity using kNN + RkCNN with hybrid storage.
        
        Uses both File Search (for natural language discovery) and 
        embedding cache (for precise kNN/RkCNN similarity).
        
        Args:
            analysis_id: Gemini analysis ID
            tensor_hash: Tensor SHA256 hash
            gemini_descriptors: Gemini analysis outputs
            safety_flags: Safety flags from Gemini
            correlation_id: Correlation ID
        
        Returns:
            SimilarityCheck with decision
        """
        if correlation_id:
            set_correlation_id(correlation_id)
        
        similarity_id = str(uuid.uuid4())
        timestamp = int(time.time())
        
        logger.info(
            f"Starting similarity validation: {similarity_id}",
            extra={"analysis_id": analysis_id},
        )
        
        # Create query vector
        query = self._create_query_vector(tensor_hash, gemini_descriptors)
        
        # Get candidate vectors - try File Search cache first, fallback to VectorStore
        if self.file_search.is_available() and self.file_search.cache_size() > 0:
            items = self.file_search.get_all_embeddings()
            logger.info(
                f"Using File Search embedding cache: {len(items)} vectors",
                extra={"embedding_dim": query.shape[0]},
            )
        else:
            items = self.vector_store.get_all()
            logger.info(
                f"Using VectorStore: {len(items)} vectors",
                extra={"embedding_dim": query.shape[0]},
            )
        
        # Run kNN baseline
        knn_neighbors = knn(
            query=query,
            items=items,
            k=self.config.knn_k,
            distance_metric=self.config.distance_metric,
        )
        
        knn_result = KNNResult(
            k=self.config.knn_k,
            neighbors=[
                KNNNeighbor(motion_id=mid, dist=dist)
                for mid, dist in knn_neighbors
            ],
            min_dist=knn_neighbors[0][1] if knn_neighbors else 0.0,
        )
        
        # Run RkCNN ensemble
        separation, vote_margin, ensemble_results = rkcnn(
            query=query,
            items=items,
            k=self.config.rkcnn_k,
            ensembles=self.config.rkcnn_ensembles,
            subspace_dim=self.config.rkcnn_subspace_dim,
            distance_metric=self.config.distance_metric,
        )
        
        # Get actual ensemble/subspace params (may be auto-computed)
        actual_ensembles = len(ensemble_results)
        actual_subspace_dim = self.config.rkcnn_subspace_dim or query.shape[0]
        
        rkcnn_result = RkCNNResult(
            k=self.config.rkcnn_k,
            ensemble_size=actual_ensembles,
            subspace_dim=actual_subspace_dim,
            vote_margin=vote_margin,
            separation_score=separation,
        )
        
        # Make decision
        decision_result, reason_codes = self._decide_novelty(
            separation_score=separation,
            vote_margin=vote_margin,
            safety_flags=safety_flags or [],
        )
        
        decision = Decision(
            novelty_threshold=self.config.novelty_threshold,
            result=decision_result,
            reason=" | ".join(reason_codes) if reason_codes else None,
        )
        
        similarity_check = SimilarityCheck(
            similarity_id=similarity_id,
            analysis_id=analysis_id,
            created_at=timestamp,
            feature_space=FeatureSpace(
                embedding_dim=self.config.embedding_dim,
                embedding_model_id=self.config.embedding_model_id,
                distance=self.config.distance_metric,
            ),
            knn=knn_result,
            rkcnn=rkcnn_result,
            decision=decision,
        )
        
        logger.info(
            f"Similarity check complete: {decision_result}",
            extra={
                "similarity_id": similarity_id,
                "decision": decision_result,
                "separation": separation,
                "vote_margin": vote_margin,
                "reason_codes": reason_codes,
                "knn_neighbors": len(knn_neighbors),
                "rkcnn_ensembles": actual_ensembles,
            },
        )
        
        return similarity_check
    
    def create_canonical_pack(
        self,
        motion_asset: Any,  # MotionAsset
        blend_request: Any,  # MotionBlendRequest
        gemini_analysis: Any,  # GeminiAnalysis
    ) -> MotionCanonicalPack:
        """
        Create MotionCanonicalPack v1.
        
        Args:
            motion_asset: MotionAsset with all artifacts
            blend_request: Original blend request
            gemini_analysis: Gemini analysis results
        
        Returns:
            MotionCanonicalPack
        """
        pack = MotionCanonicalPack(
            pack_version="MotionCanonicalPack/v1",
            request_id=blend_request.request_id,
            owner_wallet=motion_asset.owner_wallet,
            raw_ref=motion_asset.source,
            tensor_ref=motion_asset.tensor,
            skeleton=motion_asset.skeleton,
            versions={"schema": "v1", "tensor": "v1"},
            gemini={
                "analysis_id": gemini_analysis.analysis_id,
                "model": gemini_analysis.model.model_dump(),
                "outputs": gemini_analysis.outputs.model_dump(),
            },
            policy=blend_request.policy,
        )
        
        return pack
    
    def sign_mint_authorization(
        self,
        pack: MotionCanonicalPack,
        to_address: str,
    ) -> MintAuthorization:
        """
        Sign EIP-712 mint authorization.
        
        Args:
            pack: MotionCanonicalPack to mint
            to_address: Recipient address
        
        Returns:
            MintAuthorization with signature
        """
        # Compute pack hash
        pack_dict = pack.model_dump()
        pack_hash = keccak256_json(pack_dict)
        
        # Get next nonce
        nonce = nonce_manager.get_next_nonce(to_address)
        
        # Compute expiry
        expiry = int(time.time()) + self.config.attestation_expiry_seconds
        
        # Policy digest (hash of policy object)
        policy_digest = keccak256_json(pack_dict["policy"])
        
        # Create message
        message = MintMessage(
            to=to_address,
            pack_hash=pack_hash,
            nonce=nonce,
            expiry=expiry,
            policy_digest=policy_digest,
        )
        
        # In production: sign with EIP-712 typed data
        # For demo: create placeholder signature
        message_dict = message.model_dump()
        message_hash = keccak256_json(message_dict)
        signature = f"0x{'0' * 128}_demo_signature"  # 65 bytes
        
        auth = MintAuthorization(
            chain_id=self.config.chain_id,
            verifying_contract=self.config.verifying_contract,
            message=message,
            signature=signature,
        )
        
        logger.info(
            f"Mint authorization signed",
            extra={
                "pack_hash": pack_hash,
                "to": to_address,
                "nonce": nonce,
                "expiry": expiry,
            },
        )
        
        return auth

    # -------------------------------------------------------------------------
    # Derivative Detection & Royalty Chain Management
    # -------------------------------------------------------------------------

    def detect_derivative(
        self,
        tensor_hash: str,
        knn_neighbors: List[KNNNeighbor],
        motion_metadata: Optional[Dict[str, Any]] = None,
    ) -> DerivativeDetectionResult:
        """
        Detect if motion is derivative of existing content.
        
        Uses KNN neighbor distances to determine if the new motion
        is too similar to existing content. If so, attaches royalty
        obligations to the original creators.
        
        Args:
            tensor_hash: Keccak256 hash of tensor data
            knn_neighbors: Nearest neighbors from similarity check
            motion_metadata: Optional metadata including creator info
        
        Returns:
            DerivativeDetectionResult with royalty obligations
        """
        logger.info(f"[ENTRY] detect_derivative: tensor_hash={tensor_hash[:16]}...")
        
        # No neighbors = definitely unique
        if not knn_neighbors:
            logger.debug("[EXIT] detect_derivative: no neighbors, unique")
            return DerivativeDetectionResult(
                is_derivative=False,
                similarity_score=0.0,
                detection_method="knn",
                reasoning="No existing motions to compare against",
            )
        
        # Get closest neighbor
        closest = knn_neighbors[0]
        distance = closest.dist
        
        # Convert distance to similarity (assuming normalized embeddings)
        # For euclidean distance on unit vectors: max distance is 2, so scale
        similarity_score = max(0.0, min(1.0, 1.0 - (distance / 2.0)))
        
        # Check against threshold
        is_derivative = similarity_score >= self.config.derivative_threshold
        
        royalty_obligations: List[RoyaltyObligation] = []
        source_motion_id: Optional[str] = None
        source_pack_hash: Optional[str] = None
        
        if is_derivative:
            source_motion_id = closest.motion_id
            # Generate deterministic pack hash from motion_id for demo
            source_pack_hash = f"0x{hashlib.sha256(source_motion_id.encode()).hexdigest()}"
            
            # Create royalty obligation
            original_creator = "0x0000000000000000000000000000000000000001"  # Default
            if motion_metadata and "creator_address" in motion_metadata:
                original_creator = motion_metadata["creator_address"]
            
            royalty_obligations.append(
                RoyaltyObligation(
                    original_pack_hash=source_pack_hash,
                    original_creator=original_creator,
                    original_motion_id=source_motion_id,
                    royalty_percentage=self.config.default_derivative_royalty,
                    derivation_score=similarity_score,
                )
            )
            
            logger.info(
                f"Derivative detected: source={source_motion_id}, "
                f"similarity={similarity_score:.3f}"
            )
        
        result = DerivativeDetectionResult(
            is_derivative=is_derivative,
            source_motion_id=source_motion_id,
            source_pack_hash=source_pack_hash,
            similarity_score=similarity_score,
            detection_method="knn",
            royalty_obligations=royalty_obligations,
            reasoning=f"Similarity {similarity_score:.3f} vs threshold {self.config.derivative_threshold}",
        )
        
        logger.debug(f"[EXIT] detect_derivative: is_derivative={is_derivative}")
        return result

    def build_royalty_chain(
        self,
        motion_id: str,
        creator_address: str,
        derivative_result: Optional[DerivativeDetectionResult] = None,
        oracle_address: str = "0x0000000000000000000000000000000000000001",
        platform_address: str = "0x0000000000000000000000000000000000000002",
        ops_address: str = "0x0000000000000000000000000000000000000003",
    ) -> RoyaltyChain:
        """
        Build royalty chain for a motion with optional parent derivation.
        
        Creates a RoyaltyChain with proper payout nodes. If the motion
        is derivative, includes parent chain reference for recursive payouts.
        
        Args:
            motion_id: New motion ID
            creator_address: Creator's wallet address
            derivative_result: Optional derivative detection result
            oracle_address: Oracle wallet for attestation fees
            platform_address: Platform wallet for service fees
            ops_address: Operations wallet for gas/maintenance
        
        Returns:
            RoyaltyChain with all payout nodes
        """
        logger.info(f"[ENTRY] build_royalty_chain: motion_id={motion_id}")
        
        # Build base nodes (direct recipients)
        nodes: List[RoyaltyNode] = [
            RoyaltyNode(
                wallet=creator_address,
                share_bps=DEFAULT_CREATOR_SHARE_BPS,
                role="creator",
                depth=0,
                motion_id=motion_id,
            ),
            RoyaltyNode(
                wallet=oracle_address,
                share_bps=DEFAULT_ORACLE_SHARE_BPS,
                role="oracle",
                depth=0,
            ),
            RoyaltyNode(
                wallet=platform_address,
                share_bps=DEFAULT_PLATFORM_SHARE_BPS,
                role="platform",
                depth=0,
            ),
            RoyaltyNode(
                wallet=ops_address,
                share_bps=DEFAULT_OPS_SHARE_BPS,
                role="ops",
                depth=0,
            ),
        ]
        
        parent_motion_id: Optional[str] = None
        parent_chain: Optional[RoyaltyChain] = None
        total_depth = 0
        
        # Add parent creator if derivative
        if derivative_result and derivative_result.is_derivative:
            for obligation in derivative_result.royalty_obligations:
                parent_motion_id = obligation.original_motion_id
                
                # Add parent creator as royalty recipient
                # Note: Parent's share comes from creator's portion via decay factor
                nodes.append(
                    RoyaltyNode(
                        wallet=obligation.original_creator,
                        share_bps=0,  # Calculated during payout via decay
                        role="parent_creator",
                        depth=1,
                        motion_id=obligation.original_motion_id,
                    )
                )
                
                total_depth = 1  # At least one parent
                
                logger.info(
                    f"Added parent creator: {obligation.original_creator}, "
                    f"motion={obligation.original_motion_id}"
                )
        
        chain = RoyaltyChain(
            motion_id=motion_id,
            nodes=nodes,
            parent_motion_id=parent_motion_id,
            parent_chain=parent_chain,  # Would be loaded from storage in production
            total_depth=total_depth,
        )
        
        logger.info(
            f"Royalty chain built: nodes={len(nodes)}, depth={total_depth}"
        )
        logger.debug(f"[EXIT] build_royalty_chain")
        return chain

    def validate_no_circular_reference(
        self,
        motion_id: str,
        parent_motion_ids: List[str],
    ) -> bool:
        """
        Validate that adding a motion doesn't create circular reference.
        
        Oracle-side validation at mint time to prevent circular derivation
        chains (A→B→C→A) which would cause infinite royalty loops.
        
        Args:
            motion_id: New motion being minted
            parent_motion_ids: All ancestor motion IDs
        
        Returns:
            True if valid (no circular reference)
        
        Raises:
            NoveltyRejectionError: If circular reference or depth exceeded
        """
        logger.debug(
            f"[ENTRY] validate_no_circular_reference: motion_id={motion_id}"
        )
        
        # Check if motion_id appears in its own ancestry
        if motion_id in parent_motion_ids:
            logger.error(f"Circular reference: {motion_id} is its own ancestor")
            raise NoveltyRejectionError(
                f"Circular derivation chain: {motion_id} appears in ancestry",
                details={"motion_id": motion_id, "ancestors": parent_motion_ids},
            )
        
        # Check chain depth against global constant
        if len(parent_motion_ids) > MAX_ROYALTY_CHAIN_DEPTH:
            logger.warning(
                f"Chain depth {len(parent_motion_ids)} exceeds max {MAX_ROYALTY_CHAIN_DEPTH}"
            )
            raise NoveltyRejectionError(
                f"Royalty chain depth {len(parent_motion_ids)} exceeds maximum {MAX_ROYALTY_CHAIN_DEPTH}",
                details={
                    "depth": len(parent_motion_ids),
                    "max_depth": MAX_ROYALTY_CHAIN_DEPTH,
                },
            )
        
        logger.info(f"Circular reference check passed: depth={len(parent_motion_ids)}")
        logger.debug(f"[EXIT] validate_no_circular_reference: valid=True")
        return True

    def validate_similarity_with_derivative_check(
        self,
        analysis_id: str,
        tensor_hash: str,
        creator_address: str,
        gemini_descriptors: Optional[Dict[str, Any]] = None,
        safety_flags: Optional[List[str]] = None,
        parent_motion_ids: Optional[List[str]] = None,
        correlation_id: Optional[str] = None,
    ) -> Tuple[SimilarityCheck, Optional[RoyaltyChain]]:
        """
        Extended similarity validation with derivative detection.
        
        Combines novelty check with derivative detection and royalty
        chain construction. This is the primary entry point for
        attestation with payment automation.
        
        Args:
            analysis_id: Gemini analysis ID
            tensor_hash: Tensor SHA256 hash
            creator_address: Creator's wallet address
            gemini_descriptors: Gemini analysis outputs
            safety_flags: Safety flags from Gemini
            parent_motion_ids: Known parent motion IDs (for circular check)
            correlation_id: Correlation ID for tracing
        
        Returns:
            Tuple of (SimilarityCheck, RoyaltyChain or None)
        """
        if correlation_id:
            set_correlation_id(correlation_id)
        
        logger.info(
            f"[ENTRY] validate_similarity_with_derivative_check: "
            f"analysis_id={analysis_id}"
        )
        
        # Validate no circular references if parent IDs provided
        if parent_motion_ids:
            # Generate motion ID from analysis for check
            motion_id = f"motion_{analysis_id[:16]}"
            self.validate_no_circular_reference(motion_id, parent_motion_ids)
        
        # Run standard similarity check
        similarity_check = self.validate_similarity(
            analysis_id=analysis_id,
            tensor_hash=tensor_hash,
            gemini_descriptors=gemini_descriptors,
            safety_flags=safety_flags,
            correlation_id=correlation_id,
        )
        
        royalty_chain: Optional[RoyaltyChain] = None
        
        # If decision is MINT, check for derivative and build royalty chain
        if similarity_check.decision.result == "MINT":
            # Detect derivative using kNN neighbors
            derivative_result = self.detect_derivative(
                tensor_hash=tensor_hash,
                knn_neighbors=similarity_check.knn.neighbors,
            )
            
            # Build royalty chain
            motion_id = f"motion_{analysis_id[:16]}"
            royalty_chain = self.build_royalty_chain(
                motion_id=motion_id,
                creator_address=creator_address,
                derivative_result=derivative_result,
            )
            
            logger.info(
                f"Royalty chain created: is_derivative={derivative_result.is_derivative}, "
                f"depth={royalty_chain.total_depth}"
            )
        
        logger.debug(f"[EXIT] validate_similarity_with_derivative_check")
        return similarity_check, royalty_chain

