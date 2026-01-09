"""
Attestation Oracle - validates motion novelty and signs mint authorizations.
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
)
from ..utils.logging import setup_logging, set_correlation_id
from ..utils.errors import DecisionError, NoveltyRejectionError, ManualReviewRequiredError
from ..utils.canonicalize import keccak256_json
from ..utils.idempotency import nonce_manager
from .similarity import knn, rkcnn


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
    
    Responsibilities:
    1. Build query vectors from tensor features + Gemini descriptors
    2. Run kNN for baseline similarity
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
    ):
        self.config = config
        self.vector_store = vector_store or VectorStore()
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
        Validate motion similarity using kNN + RkCNN.
        
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
        
        # Get candidate vectors from store
        items = self.vector_store.get_all()
        
        logger.info(
            f"Running similarity check against {len(items)} candidates",
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
