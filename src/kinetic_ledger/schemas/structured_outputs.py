"""
Pydantic schemas for Gemini structured outputs.

These schemas enable type-safe, validated responses from Gemini API
using the structured output feature documented at:
https://ai.google.dev/gemini-api/docs/structured-output
"""
from typing import List, Optional, Literal
from pydantic import BaseModel, Field


class SimilarityFeedback(BaseModel):
    """Structured feedback from Gemini on motion similarity."""
    
    sentiment: Literal["novel", "derivative", "ambiguous"] = Field(
        description="Overall novelty assessment of the motion"
    )
    
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence in the similarity assessment (0.0-1.0)"
    )
    
    similar_motions: List[str] = Field(
        default_factory=list,
        description="List of motion IDs that are semantically similar"
    )
    
    distinguishing_features: List[str] = Field(
        default_factory=list,
        description="Key features that make this motion unique or derivative"
    )
    
    summary: str = Field(
        description="Brief explanation of the similarity assessment"
    )


class MotionStyleClassification(BaseModel):
    """Structured classification of motion style."""
    
    primary_style: str = Field(
        description="Primary movement style (e.g., 'breakdance', 'capoeira', 'parkour')"
    )
    
    secondary_styles: List[str] = Field(
        default_factory=list,
        description="Secondary or blended movement styles present"
    )
    
    energy_level: Literal["low", "medium", "high", "explosive"] = Field(
        description="Overall energy and intensity of movement"
    )
    
    technical_difficulty: Literal["beginner", "intermediate", "advanced", "expert"] = Field(
        description="Technical skill level required"
    )
    
    characteristic_moves: List[str] = Field(
        default_factory=list,
        description="Signature moves or techniques identified"
    )


class NoveltyAssessment(BaseModel):
    """Comprehensive novelty assessment combining multiple signals."""
    
    is_novel: bool = Field(
        description="Whether the motion exhibits sufficient novelty for minting"
    )
    
    novelty_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Computed novelty score (0.0 = highly derivative, 1.0 = completely novel)"
    )
    
    knn_distance: float = Field(
        ge=0.0,
        description="Distance to nearest neighbor in embedding space"
    )
    
    separation_score: float = Field(
        ge=0.0,
        le=1.0,
        description="RkCNN separation score measuring boundary distance"
    )
    
    vote_consensus: float = Field(
        ge=0.0,
        le=1.0,
        description="RkCNN ensemble vote margin (higher = stronger consensus)"
    )
    
    semantic_similarity: Optional[SimilarityFeedback] = Field(
        default=None,
        description="Optional Gemini semantic similarity feedback"
    )
    
    decision: Literal["MINT", "REJECT", "REVIEW"] = Field(
        description="Final decision based on all signals"
    )
    
    reasoning: str = Field(
        description="Detailed explanation of the decision"
    )


class SearchResult(BaseModel):
    """Structured search result from File Search."""
    
    document_id: str = Field(
        description="Unique identifier of the matching document"
    )
    
    relevance_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Relevance score for the query (0.0-1.0)"
    )
    
    matched_content: str = Field(
        description="Excerpt of matching content from the document"
    )
    
    style_labels: List[str] = Field(
        default_factory=list,
        description="Style labels associated with the motion"
    )
    
    npc_tags: List[str] = Field(
        default_factory=list,
        description="NPC behavior tags"
    )
    
    summary: str = Field(
        description="Brief summary of why this result matches the query"
    )


class FileSearchResponse(BaseModel):
    """Complete File Search response with ranked results."""
    
    query: str = Field(
        description="Original search query"
    )
    
    total_results: int = Field(
        ge=0,
        description="Total number of matching results found"
    )
    
    results: List[SearchResult] = Field(
        default_factory=list,
        description="Ranked list of search results"
    )
    
    search_interpretation: str = Field(
        description="Gemini's interpretation and reformulation of the search query"
    )


class MotionBlendRecommendation(BaseModel):
    """Structured recommendation for motion blending."""
    
    source_motion_id: str = Field(
        description="ID of the source motion to blend from"
    )
    
    target_motion_id: str = Field(
        description="ID of the target motion to blend to"
    )
    
    blend_ratio: float = Field(
        ge=0.0,
        le=1.0,
        description="Recommended blend ratio (0.0 = all source, 1.0 = all target)"
    )
    
    transition_frame: int = Field(
        ge=0,
        description="Recommended frame number for blend transition"
    )
    
    blend_method: Literal["linear", "slerp", "cubic", "temporal_conditioning"] = Field(
        description="Recommended blending method"
    )
    
    compatibility_score: float = Field(
        ge=0.0,
        le=1.0,
        description="How well these motions blend together (0.0-1.0)"
    )
    
    reasoning: str = Field(
        description="Explanation of why this blend works well"
    )


class SafetyAssessment(BaseModel):
    """Structured safety and policy compliance assessment."""
    
    is_safe: bool = Field(
        description="Whether the motion passes safety checks"
    )
    
    safety_flags: List[str] = Field(
        default_factory=list,
        description="List of safety concerns identified"
    )
    
    violence_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Violence/aggression level (0.0 = none, 1.0 = extreme)"
    )
    
    injury_risk: Literal["none", "low", "moderate", "high"] = Field(
        description="Potential injury risk if performed"
    )
    
    policy_violations: List[str] = Field(
        default_factory=list,
        description="List of policy violations detected"
    )
    
    recommendation: Literal["APPROVE", "FLAG", "REJECT"] = Field(
        description="Safety recommendation"
    )
    
    notes: str = Field(
        description="Additional safety notes or context"
    )
