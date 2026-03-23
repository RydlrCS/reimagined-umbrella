from typing import List, Optional, Literal, Dict
from enum import Enum
from pydantic import BaseModel, Field, field_validator, model_validator

class SourceRef(BaseModel):
    type: Literal["upload", "uri", "library"]
    filename: Optional[str] = None
    content_type: Optional[str] = None
    uri: str
    sha256: str

class TensorRef(BaseModel):
    representation: Literal["quaternion", "rot6d"]
    fps: int = Field(ge=1, le=240)
    frame_count: int = Field(ge=1)
    joint_count: int = Field(ge=1)
    features: List[str]
    uri: str
    sha256: str

class PreviewRef(BaseModel):
    uri: str
    sha256: str

class SkeletonInfo(BaseModel):
    skeleton_id: str
    retarget_profile: Optional[str] = None
    joint_map_uri: Optional[str] = None

class MotionAsset(BaseModel):
    motion_id: str
    created_at: int
    owner_wallet: str
    source: SourceRef
    tensor: TensorRef
    preview: PreviewRef
    skeleton: SkeletonInfo

class BlendSegment(BaseModel):
    label: str
    start_frame: int = Field(ge=0)
    end_frame: int = Field(ge=0)

class BlendPlan(BaseModel):
    type: Literal["single_shot_temporal_conditioning", "two_stage", "custom"]
    segments: List[BlendSegment]

class MotionBlendRequest(BaseModel):
    request_id: str
    created_at: int
    user_wallet: str
    inputs: List[Dict[str, str]]
    blend_plan: BlendPlan
    npc_context: Dict[str, object]
    policy: Dict[str, object]

class ModelDescriptor(BaseModel):
    provider: str
    name: str
    version: str

class GeminiInputs(BaseModel):
    preview_uri: str
    metadata_ref: Optional[str] = None

class GeminiOutputs(BaseModel):
    style_labels: List[str]
    transition_window: Dict[str, int]
    npc_tags: List[str]
    safety_flags: List[str]
    summary: Optional[str] = None

class GeminiAnalysis(BaseModel):
    analysis_id: str
    request_id: str
    created_at: int
    model: ModelDescriptor
    inputs: GeminiInputs
    outputs: GeminiOutputs

class FeatureSpace(BaseModel):
    embedding_dim: int = Field(ge=1)
    embedding_model_id: str
    distance: Literal["euclidean", "cosine", "manhattan"]

class KNNNeighbor(BaseModel):
    motion_id: str
    dist: float = Field(ge=0)

class KNNResult(BaseModel):
    k: int = Field(ge=1)
    neighbors: List[KNNNeighbor]
    min_dist: float = Field(ge=0)

class RkCNNResult(BaseModel):
    k: int = Field(ge=1)
    ensemble_size: int = Field(ge=1)
    subspace_dim: int = Field(ge=1)
    vote_margin: float = Field(ge=0, le=1)
    separation_score: float = Field(ge=0, le=1)

class Decision(BaseModel):
    novelty_threshold: float = Field(ge=0, le=1)
    result: Literal["MINT", "REJECT", "REVIEW"]
    reason: Optional[str] = None

class SimilarityCheck(BaseModel):
    similarity_id: str
    analysis_id: str
    created_at: int
    feature_space: FeatureSpace
    knn: KNNResult
    rkcnn: RkCNNResult
    decision: Decision

class MotionCanonicalPack(BaseModel):
    pack_version: Literal["MotionCanonicalPack/v1"]
    request_id: str
    owner_wallet: str
    raw_ref: SourceRef
    tensor_ref: TensorRef
    skeleton: SkeletonInfo
    versions: Dict[str, str]
    gemini: Dict[str, object]
    policy: Dict[str, object]

class MintMessage(BaseModel):
    to: str
    pack_hash: str
    nonce: int
    expiry: int
    policy_digest: str

class MintAuthorization(BaseModel):
    chain_id: int
    verifying_contract: str
    message: MintMessage
    signature: str

class Metering(BaseModel):
    unit: Literal["seconds_generated", "frames_generated", "agent_steps"]
    quantity: float = Field(ge=0)
    unit_price_usdc: str
    total_usdc: str

class X402(BaseModel):
    payment_proof: str
    facilitator_receipt_id: str
    verified: bool

class Settlement(BaseModel):
    chain: str
    token: str
    tx_hash: str

class PayoutItem(BaseModel):
    to: str
    amount_usdc: str
    label: str

class UsageMeterEvent(BaseModel):
    usage_id: str
    created_at: int
    user_wallet: str
    attestation_pack_hash: str
    product: str
    metering: Metering
    x402: X402
    settlement: Settlement
    payout_split: List[PayoutItem]

class PayoutSplit(BaseModel):
    creator: float
    oracle: float
    platform: float
    ops: float

    @field_validator("creator", "oracle", "platform", "ops")
    @classmethod
    def _bounds(cls, v: float):
        if not (0.0 <= v <= 1.0):
            raise ValueError("payout percentages must be in [0,1]")
        return v

    @model_validator(mode="after")
    def _sum_to_one(self):
        total = self.creator + self.oracle + self.platform + self.ops
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"payout percentages must sum to 1.0, got {total}")
        return self

class EthicsModifiers(BaseModel):
    multipliers: Dict[str, float] = {}
    caps: Dict[str, float] = {}

    @field_validator("multipliers")
    @classmethod
    def _mult_bounds(cls, v: Dict[str, float]):
        for k, val in v.items():
            if not (0.0 <= val <= 2.0):
                raise ValueError(f"multiplier {k} out of bounds [0,2]")
        return v

    @field_validator("caps")
    @classmethod
    def _cap_bounds(cls, v: Dict[str, float]):
        for k, val in v.items():
            if not (0.0 <= val <= 1.0):
                raise ValueError(f"cap {k} out of bounds [0,1]")
        return v


# =============================================================================
# Global Payment Constants
# =============================================================================

# Royalty decay factor: each parent level receives this fraction of child's creator share
ROYALTY_DECAY_FACTOR: float = 0.5

# Maximum depth for royalty chain traversal (prevent infinite loops)
MAX_ROYALTY_CHAIN_DEPTH: int = 10

# Gas sponsorship budget replenish threshold (40% of treasury)
GAS_SPONSOR_REPLENISH_THRESHOLD: float = 0.40

# Minimum USDC balance to trigger auto-replenish
GAS_SPONSOR_MIN_BALANCE_USDC: str = "100.00"

# Default payout percentages (basis points for on-chain, decimals for Python)
DEFAULT_CREATOR_SHARE_BPS: int = 7000  # 70%
DEFAULT_ORACLE_SHARE_BPS: int = 1000   # 10%
DEFAULT_PLATFORM_SHARE_BPS: int = 1500 # 15%
DEFAULT_OPS_SHARE_BPS: int = 500       # 5%


# =============================================================================
# Circle Wallet Models
# =============================================================================


class CircleWallet(BaseModel):
    """Circle Programmable Wallet representation."""
    
    wallet_id: str
    address: str = Field(pattern=r"^0x[a-fA-F0-9]{40}$")
    blockchain: Literal["ARC", "ETH", "MATIC", "SOL"] = "ARC"
    status: Literal["ACTIVE", "PENDING", "FROZEN"] = "ACTIVE"
    created_at: int
    user_id: Optional[str] = None
    wallet_set_id: Optional[str] = None


class CircleTransfer(BaseModel):
    """USDC transfer record."""
    
    transfer_id: str
    source_wallet_id: str
    destination_address: str = Field(pattern=r"^0x[a-fA-F0-9]{40}$")
    amount_usdc: str  # Decimal string (e.g., "10.50")
    status: Literal["PENDING", "CONFIRMED", "FAILED"] = "PENDING"
    tx_hash: Optional[str] = None
    chain: str = "ARC"
    created_at: int
    confirmed_at: Optional[int] = None
    error_message: Optional[str] = None


class WalletBalanceResponse(BaseModel):
    """Wallet balance query response."""
    
    wallet_id: str
    address: str
    total_usdc: str
    native_balance: str = "0.00"  # Native token (USDC on Arc)
    updated_at: int


# =============================================================================
# Royalty Chain Models
# =============================================================================


class RoyaltyNode(BaseModel):
    """
    Node in royalty chain - represents a single payout recipient.
    
    For infinite royalty chains, each motion can have a parent,
    and payouts flow recursively through the chain.
    """
    
    wallet: str = Field(pattern=r"^0x[a-fA-F0-9]{40}$")
    share_bps: int = Field(ge=0, le=10000, description="Share in basis points")
    role: Literal["creator", "oracle", "platform", "ops", "parent_creator"]
    depth: int = Field(ge=0, default=0, description="0=direct creator, 1+=ancestors")
    motion_id: Optional[str] = None  # Source motion for parent creators


class RoyaltyObligation(BaseModel):
    """
    Royalty obligation for derivative motions.
    
    When a motion is detected as derivative of another,
    this records the royalty owed to the original creator.
    """
    
    original_pack_hash: str = Field(pattern=r"^0x[a-fA-F0-9]{64}$")
    original_creator: str = Field(pattern=r"^0x[a-fA-F0-9]{40}$")
    original_motion_id: str
    royalty_percentage: float = Field(ge=0, le=1, default=0.10)
    derivation_score: float = Field(ge=0, le=1, description="Similarity score")


class RoyaltyChain(BaseModel):
    """
    Infinite-depth royalty chain for derivative motions.
    
    Supports recursive parent references for unlimited derivation depth.
    Payout calculation uses ROYALTY_DECAY_FACTOR per level.
    """
    
    motion_id: str
    nodes: List[RoyaltyNode]
    parent_motion_id: Optional[str] = None
    parent_chain: Optional["RoyaltyChain"] = None  # Recursive reference
    total_depth: int = Field(ge=0, default=0)
    
    @model_validator(mode="after")
    def _validate_total_shares(self) -> "RoyaltyChain":
        """Validate that non-parent shares sum to 10000 bps."""
        # Only validate direct nodes (not recursive parent shares)
        direct_nodes = [n for n in self.nodes if n.depth == 0]
        total = sum(n.share_bps for n in direct_nodes)
        if direct_nodes and total != 10000:
            raise ValueError(f"Direct shares must sum to 10000 bps, got {total}")
        return self


# Enable forward reference for recursive model
RoyaltyChain.model_rebuild()


class DerivativeDetectionResult(BaseModel):
    """Result of derivative/reuse detection analysis."""
    
    is_derivative: bool
    source_motion_id: Optional[str] = None
    source_pack_hash: Optional[str] = None
    similarity_score: float = Field(ge=0, le=1)
    detection_method: Literal["knn", "rkcnn", "gemini_vision", "combined"]
    royalty_obligations: List[RoyaltyObligation] = []
    reasoning: Optional[str] = None


# =============================================================================
# Uniqueness Attestation Models
# =============================================================================


class UniquenessAttestation(BaseModel):
    """
    Structured output schema for Gemini uniqueness attestation.
    
    Used with Gemini's structured output feature to get verifiable
    uniqueness assessments for blend artifacts.
    """
    
    blend_hash: str = Field(pattern=r"^0x[a-fA-F0-9]{64}$")
    is_unique: bool
    uniqueness_score: float = Field(ge=0, le=1)
    similar_motions: List[str] = []  # Motion IDs of similar existing motions
    visual_novelty_score: float = Field(ge=0, le=1)
    style_originality: str
    reasoning: str
    timestamp: int
    oracle_model: str = "gemini-2.5-pro"
    signature: Optional[str] = None
    
    def with_signature(self, sig: str) -> "UniquenessAttestation":
        """Return copy with signature attached."""
        return self.model_copy(update={"signature": sig})


# =============================================================================
# Payment Automation Models
# =============================================================================


class PaymentTriggerEvent(BaseModel):
    """Event triggered when on-chain mint is confirmed."""
    
    event_id: str
    pack_hash: str = Field(pattern=r"^0x[a-fA-F0-9]{64}$")
    motion_id: str
    tx_hash: str = Field(pattern=r"^0x[a-fA-F0-9]{64}$")
    creator_address: str = Field(pattern=r"^0x[a-fA-F0-9]{40}$")
    payout_amount_usdc: str
    royalty_chain: Optional[RoyaltyChain] = None
    triggered_at: int
    status: Literal["PENDING", "PROCESSING", "COMPLETED", "FAILED"] = "PENDING"


class GaslessReceipt(BaseModel):
    """Receipt for gasless transaction execution."""
    
    receipt_id: str
    original_tx_hash: str
    sponsor_address: str = Field(pattern=r"^0x[a-fA-F0-9]{40}$")
    paymaster_tx_hash: Optional[str] = None
    gas_sponsored_usdc: str
    eip7702_used: bool = True
    created_at: int


class RecursivePayoutResult(BaseModel):
    """Result of recursive payout calculation."""
    
    motion_id: str
    total_usdc: str
    payouts: List[PayoutItem]
    chain_depth: int
    decay_factor_used: float = ROYALTY_DECAY_FACTOR
    gas_estimate_usdc: str = "0.00"


# =============================================================================
# Extended Motion Asset (V2)
# =============================================================================


class MotionAssetV2(BaseModel):
    """
    Extended motion asset with derivation tracking.
    
    Supports parent motion references for royalty chain calculation.
    """
    
    motion_id: str
    created_at: int
    owner_wallet: str = Field(pattern=r"^0x[a-fA-F0-9]{40}$")
    source: "SourceRef"
    tensor: "TensorRef"
    preview: "PreviewRef"
    skeleton: "SkeletonInfo"
    
    # Derivation tracking
    parent_motion_ids: List[str] = []
    derivation_depth: int = Field(ge=0, default=0)
    royalty_chain: Optional[RoyaltyChain] = None
    
    # Uniqueness attestation
    uniqueness_attestation: Optional[UniquenessAttestation] = None
    derivative_detection: Optional[DerivativeDetectionResult] = None


class PolicyConfig(BaseModel):
    allowed_use: Literal["npc_generation"]
    safety_level: Literal["low", "standard", "high"] = "standard"
    base_rate: float = Field(ge=0)
    unit: Literal["USD/s", "ETH/s", "USD/h", "ETH/h"]
    max_seconds: int = Field(ge=0)
    payout_split: PayoutSplit
    ethics_modifiers: Optional[EthicsModifiers] = None


# =============================================================================
# SPADE Hierarchical Motion Blending Models
# =============================================================================


class HierarchyLevel(str, Enum):
    """
    Motion hierarchy levels for SPADE conditioning.
    
    Based on academic research showing Level 1 (COARSE) conditioning
    yields optimal FID/coverage trade-off for motion blending.
    """
    COARSE = "coarse"    # Level 1: Root, pelvis - SPADE applied here (optimal)
    MID = "mid"          # Level 2: Spine, major limbs
    FINE = "fine"        # Level 3: Hands, feet, head
    DETAIL = "detail"    # Level 4: Fingers, facial


class JointHierarchyMapping(BaseModel):
    """
    Mapping of joint names to hierarchy levels.
    
    Default uses Mixamo 24-joint skeleton with research-based
    hierarchy assignments for optimal SPADE conditioning.
    """
    coarse_joints: List[str] = Field(
        default=["Hips", "Spine"],
        description="Level 1 joints - SPADE conditioning applied"
    )
    mid_joints: List[str] = Field(
        default=["Spine1", "Spine2", "LeftUpLeg", "RightUpLeg", 
                 "LeftArm", "RightArm"],
        description="Level 2 joints - standard temporal conditioning"
    )
    fine_joints: List[str] = Field(
        default=["Neck", "Head", "LeftLeg", "RightLeg", 
                 "LeftForeArm", "RightForeArm", "LeftFoot", "RightFoot"],
        description="Level 3 joints - standard temporal conditioning"
    )
    detail_joints: List[str] = Field(
        default=["LeftHand", "RightHand", "LeftToeBase", "RightToeBase",
                 "LeftShoulder", "RightShoulder"],
        description="Level 4 joints - standard temporal conditioning"
    )


class SPADEConfig(BaseModel):
    """
    Configuration for SPADE (Spatially-Adaptive Denormalization) layer.
    
    Research findings:
    - Level 1 (COARSE) SPADE yields best FID/coverage trade-off
    - Captures coarse motion features critical for overall structure
    - Trainable γ/β parameters for style-adaptive modulation
    """
    
    # Which hierarchy level to apply SPADE (1 = COARSE is optimal)
    spade_level: HierarchyLevel = Field(
        default=HierarchyLevel.COARSE,
        description="Hierarchy level for SPADE conditioning (COARSE recommended)"
    )
    
    # Neural network dimensions
    input_dim: int = Field(default=768, ge=1, description="Input embedding dimension")
    style_channels: int = Field(default=128, ge=1, description="Style conditioning channels")
    motion_channels: int = Field(default=256, ge=1, description="Motion feature channels")
    
    # Modulation parameters (trainable γ/β)
    gamma_init: float = Field(default=1.0, description="Scale parameter initialization")
    beta_init: float = Field(default=0.0, description="Bias parameter initialization")
    
    # Transition parameters
    transition_sharpness: float = Field(
        default=5.0, gt=0, 
        description="Controls blend boundary sharpness (sigmoid temperature)"
    )
    
    # Joint hierarchy
    joint_hierarchy: JointHierarchyMapping = Field(
        default_factory=JointHierarchyMapping,
        description="Joint-to-level mapping"
    )
    
    # Training config
    learning_rate: float = Field(default=1e-4, gt=0)
    weight_decay: float = Field(default=1e-5, ge=0)
    
    @model_validator(mode="after")
    def _validate_dims(self) -> "SPADEConfig":
        """Validate dimension relationships."""
        if self.style_channels > self.input_dim:
            raise ValueError(
                f"style_channels ({self.style_channels}) must be <= input_dim ({self.input_dim})"
            )
        return self


class SPADEBlendRequest(BaseModel):
    """
    Request for SPADE-enhanced motion blending.
    
    Uses hierarchical SPADE conditioning with trainable γ/β parameters
    for style-adaptive motion blending.
    """
    
    request_id: str
    motion_ids: List[str] = Field(
        min_length=2, max_length=8,
        description="Source motion IDs to blend (2-8 motions)"
    )
    motion_paths: Optional[List[str]] = Field(
        default=None,
        description="Optional FBX file paths (alternative to motion_ids)"
    )
    weights: List[float] = Field(
        description="Blend weights for duration allocation (must sum to 1.0)"
    )
    style_labels: List[List[str]] = Field(
        description="Style labels for each motion (e.g., [['capoeira'], ['breakdance']])"
    )
    
    # SPADE configuration
    hierarchy_level: int = Field(
        default=1, ge=1, le=4,
        description="Hierarchy level for SPADE conditioning (1=COARSE optimal)"
    )
    transition_frames: int = Field(
        default=30, ge=1, le=120,
        description="Frames for blend transition"
    )
    
    # Advanced options
    use_trainable_params: bool = Field(
        default=True,
        description="Use trainable γ/β parameters (requires PyTorch)"
    )
    checkpoint_path: Optional[str] = Field(
        default=None,
        description="Path to load pretrained SPADE weights"
    )
    
    @field_validator("weights")
    @classmethod
    def _validate_weights_sum(cls, v: List[float]) -> List[float]:
        """Validate weights sum to 1.0."""
        total = sum(v)
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"weights must sum to 1.0, got {total}")
        if any(w < 0 for w in v):
            raise ValueError("weights must be non-negative")
        return v
    
    @model_validator(mode="after")
    def _validate_lengths(self) -> "SPADEBlendRequest":
        """Validate array lengths match."""
        n_motions = len(self.motion_ids) if self.motion_ids else 0
        n_paths = len(self.motion_paths) if self.motion_paths else 0
        n_weights = len(self.weights)
        n_styles = len(self.style_labels)
        
        # Determine expected count
        expected = n_motions if n_motions > 0 else n_paths
        
        if expected == 0:
            raise ValueError("Either motion_ids or motion_paths must be provided")
        
        if n_weights != expected:
            raise ValueError(f"weights length ({n_weights}) must match motions ({expected})")
        
        if n_styles != expected:
            raise ValueError(f"style_labels length ({n_styles}) must match motions ({expected})")
        
        return self


class SPADEMetrics(BaseModel):
    """
    Evaluation metrics for SPADE blending quality.
    
    Includes FID (Fréchet Inception Distance), coverage, diversity,
    and motion-specific smoothness metrics.
    """
    
    # Distribution metrics
    fid_score: float = Field(
        ge=0,
        description="Fréchet Inception Distance (lower is better)"
    )
    coverage: float = Field(
        ge=0, le=1,
        description="Coverage of reference distribution (higher is better)"
    )
    diversity: float = Field(
        ge=0,
        description="Variance in generated motion features (higher is better)"
    )
    
    # Motion quality metrics
    smoothness: float = Field(
        ge=0, le=1,
        description="Motion smoothness (1.0 = perfectly smooth, acceleration-based)"
    )
    foot_sliding: float = Field(
        ge=0,
        description="Foot sliding distance in meters (lower is better)"
    )
    
    # SPADE-specific metrics
    spade_level_used: int = Field(ge=1, le=4)
    transition_quality: float = Field(
        ge=0, le=1,
        description="Blend transition quality (higher is better)"
    )
    
    # Timing
    blend_time_ms: float = Field(ge=0, description="Blend computation time")
    metrics_time_ms: float = Field(ge=0, description="Metrics computation time")


class SPADEBlendResponse(BaseModel):
    """
    Response from SPADE motion blending.
    
    Contains blended motion artifacts, quality metrics,
    and debug information for CI/CD validation.
    """
    
    status: Literal["success", "partial", "error"]
    blend_id: str
    
    # SPADE configuration used
    spade_config: SPADEConfig
    
    # Motion info
    source_motions: List[Dict[str, object]] = Field(
        description="Info about source motions (name, frames, style)"
    )
    total_frames: int = Field(ge=1)
    output_fps: int = Field(default=30, ge=1, le=120)
    
    # Artifacts
    blended_tensor_uri: Optional[str] = Field(
        default=None,
        description="URI to blended motion tensor (NPZ/JSON)"
    )
    preview_uri: Optional[str] = Field(
        default=None,
        description="URI to preview video/keyframes"
    )
    artifact_directory: Optional[str] = None
    
    # Quality metrics
    metrics: Optional[SPADEMetrics] = None
    
    # Debug info
    checkpoint_loaded: bool = False
    trainable_params_count: int = Field(default=0, ge=0)
    device_used: str = Field(default="cpu", description="cpu or cuda")
    
    # Error handling
    error_message: Optional[str] = None
    warnings: List[str] = []

