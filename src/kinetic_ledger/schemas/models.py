from typing import List, Optional, Literal, Dict
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

class PolicyConfig(BaseModel):
    allowed_use: Literal["npc_generation"]
    safety_level: Literal["low", "standard", "high"] = "standard"
    base_rate: float = Field(ge=0)
    unit: Literal["USD/s", "ETH/s", "USD/h", "ETH/h"]
    max_seconds: int = Field(ge=0)
    payout_split: PayoutSplit
    ethics_modifiers: Optional[EthicsModifiers] = None
