"""
Pydantic schemas for Gemini-powered structured motion analysis.

These schemas enable multimodal AI to analyze FBX animations visually
and generate intelligent blend parameters with explainable reasoning.
"""

from typing import List, Optional, Literal
from pydantic import BaseModel, Field


class JointVelocityProfile(BaseModel):
    """Velocity characteristics for a specific joint."""
    
    joint_name: str = Field(description="Name of the joint/bone")
    avg_velocity: float = Field(description="Average velocity magnitude in units/frame")
    peak_velocity: float = Field(description="Peak velocity magnitude")
    velocity_variance: float = Field(description="Variance in velocity over the motion")


class MotionCharacteristics(BaseModel):
    """Visual and kinematic characteristics of a motion clip."""
    
    motion_type: Literal["walk", "run", "jump", "idle", "combat", "dance", "other"] = Field(
        description="Primary category of the motion"
    )
    
    energy_level: Literal["low", "medium", "high"] = Field(
        description="Overall energy/intensity of the motion"
    )
    
    key_joints: List[str] = Field(
        description="List of joints that are most active/important in this motion",
        max_length=10
    )
    
    velocity_profiles: List[JointVelocityProfile] = Field(
        description="Velocity characteristics for key joints",
        max_length=10
    )
    
    has_cyclic_pattern: bool = Field(
        description="Whether the motion has a repeating cyclic pattern"
    )
    
    ground_contact: bool = Field(
        description="Whether the character maintains ground contact (vs aerial)"
    )
    
    motion_description: str = Field(
        description="Brief natural language description of the motion",
        max_length=200
    )


class BlendParameters(BaseModel):
    """Optimized parameters for blending two motions."""
    
    transition_frames: int = Field(
        description="Number of frames for the transition blend window",
        ge=10,
        le=60
    )
    
    crosshatch_offset: int = Field(
        description="Frame offset for crosshatch alignment",
        ge=0,
        le=30
    )
    
    omega_curve_type: Literal["smoothstep", "linear", "ease_in", "ease_out"] = Field(
        description="Type of temporal conditioning curve to use"
    )
    
    per_joint_weights: Optional[List[dict]] = Field(
        description="Optional per-joint blend weights for fine control",
        default=None
    )
    
    apply_velocity_matching: bool = Field(
        description="Whether to apply velocity matching at transition boundaries"
    )
    
    apply_root_motion_correction: bool = Field(
        description="Whether to correct root motion for spatial continuity"
    )


class CompatibilityScore(BaseModel):
    """Compatibility assessment between two motions."""
    
    overall_score: float = Field(
        description="Overall compatibility score (0-1)",
        ge=0.0,
        le=1.0
    )
    
    velocity_compatibility: float = Field(
        description="How well velocities match at potential transitions (0-1)",
        ge=0.0,
        le=1.0
    )
    
    pose_similarity: float = Field(
        description="Similarity of key poses between motions (0-1)",
        ge=0.0,
        le=1.0
    )
    
    energy_match: float = Field(
        description="How well energy levels match (0-1)",
        ge=0.0,
        le=1.0
    )
    
    reasoning: str = Field(
        description="Explanation of the compatibility assessment",
        max_length=300
    )


class QualityPrediction(BaseModel):
    """Predicted quality metrics for the blend result."""
    
    predicted_coverage: float = Field(
        description="Expected coverage metric (0-1)",
        ge=0.0,
        le=1.0
    )
    
    predicted_diversity: float = Field(
        description="Expected diversity metric (0-1)",
        ge=0.0,
        le=1.0
    )
    
    predicted_smoothness: float = Field(
        description="Expected smoothness at transition (0-1)",
        ge=0.0,
        le=1.0
    )
    
    confidence: float = Field(
        description="Confidence in the prediction (0-1)",
        ge=0.0,
        le=1.0
    )
    
    potential_issues: List[str] = Field(
        description="Potential quality issues to watch for",
        max_length=5
    )


class MotionBlendAnalysis(BaseModel):
    """Complete analysis and recommendations for blending two motions."""
    
    motion_a_characteristics: MotionCharacteristics = Field(
        description="Analysis of the first motion clip"
    )
    
    motion_b_characteristics: MotionCharacteristics = Field(
        description="Analysis of the second motion clip"
    )
    
    compatibility: CompatibilityScore = Field(
        description="Compatibility assessment between the two motions"
    )
    
    recommended_parameters: BlendParameters = Field(
        description="Recommended blend parameters for optimal quality"
    )
    
    quality_prediction: QualityPrediction = Field(
        description="Predicted quality metrics for the blend result"
    )
    
    alternative_parameters: Optional[BlendParameters] = Field(
        description="Alternative parameter set if different tradeoffs desired",
        default=None
    )
    
    overall_recommendation: str = Field(
        description="Summary recommendation and key insights",
        max_length=400
    )
