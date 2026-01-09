"""
Gemini Multimodal Analyzer - analyzes motion previews with Gemini.
"""
import logging
import time
import uuid
from typing import Optional, List, Dict, Any
from pydantic import BaseModel

from ..schemas.models import (
    GeminiAnalysis,
    ModelDescriptor,
    GeminiInputs,
    GeminiOutputs,
)
from ..utils.logging import setup_logging, set_correlation_id
from ..utils.errors import GeminiError, ValidationError
from ..utils.retry import retry_on_gemini_error


logger = logging.getLogger(__name__)


class GeminiAnalysisRequest(BaseModel):
    """Request for Gemini analysis."""
    request_id: str
    preview_uri: str
    metadata_ref: Optional[str] = None
    blend_segments: Optional[List[Dict[str, Any]]] = None
    npc_context: Optional[Dict[str, Any]] = None


class GeminiClient:
    """
    Gemini API client wrapper with retries and error handling.
    
    In production: integrate with Google Gemini SDK.
    For demo: simulate analysis with structured responses.
    """
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-latest"):
        self.api_key = api_key
        self.model = model
        self.provider = "google"
    
    @retry_on_gemini_error(max_attempts=5, min_wait=2, max_wait=30)
    def analyze_motion_preview(
        self,
        preview_uri: str,
        metadata: Optional[Dict[str, Any]] = None,
        segments: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Analyze motion preview with Gemini multimodal API.
        
        Args:
            preview_uri: URI to preview video/keyframes
            metadata: Additional metadata context
            segments: Blend segment information
        
        Returns:
            Analysis results
        """
        # In production: call Gemini API with multimodal prompt
        # For demo: simulate intelligent analysis based on inputs
        
        logger.info(f"Analyzing motion preview: {preview_uri}")
        
        # Simulate API call delay
        time.sleep(0.1)
        
        # Extract style labels from metadata or segments
        style_labels = []
        if segments:
            for seg in segments:
                label = seg.get("label", "")
                if label and label not in style_labels:
                    style_labels.append(label)
        
        if not style_labels:
            style_labels = ["dynamic_motion", "character_animation"]
        
        # Simulate transition window detection
        transition_window = {"start_frame": 110, "end_frame": 140}
        if segments and len(segments) > 1:
            # Find transition between first and second segment
            first_end = segments[0].get("end_frame", 100)
            second_start = segments[1].get("start_frame", 125)
            transition_window = {
                "start_frame": max(0, first_end - 15),
                "end_frame": min(second_start + 15, 250),
            }
        
        # Generate NPC tags based on style
        npc_tags = []
        if "capoeira" in " ".join(style_labels).lower():
            npc_tags.extend(["agile", "evasive", "high_energy"])
        if "breakdance" in " ".join(style_labels).lower():
            npc_tags.extend(["athletic", "dynamic", "acrobatic"])
        if not npc_tags:
            npc_tags = ["animated", "mobile"]
        
        # Safety flags
        safety_flags = []
        
        # Generate summary
        summary = f"Motion blend with {len(style_labels)} distinct styles"
        if len(style_labels) >= 2:
            summary = f"Blend transitions from {style_labels[0]} into {style_labels[1]}"
        
        return {
            "style_labels": style_labels[:5],  # Limit to 5
            "transition_window": transition_window,
            "npc_tags": list(set(npc_tags))[:10],  # Unique tags, limit 10
            "safety_flags": safety_flags,
            "summary": summary,
        }


class GeminiAnalyzerService:
    """
    Gemini Multimodal Analyzer Service.
    
    Responsibilities:
    1. Consume motion previews (video/keyframes) + metadata
    2. Call Gemini API for multimodal analysis
    3. Extract structured descriptors (style labels, NPC tags, safety flags)
    4. Detect transition windows
    5. Generate human-readable summaries
    """
    
    def __init__(
        self,
        gemini_api_key: Optional[str] = None,
        model_version: str = "gemini-latest",
    ):
        self.client = GeminiClient(api_key=gemini_api_key, model=model_version)
        self.model_version = model_version
        setup_logging("multimodal-gemini")
    
    def analyze(
        self,
        request: GeminiAnalysisRequest,
        correlation_id: Optional[str] = None,
    ) -> GeminiAnalysis:
        """
        Analyze motion preview with Gemini.
        
        Args:
            request: Analysis request
            correlation_id: Correlation ID for tracing
        
        Returns:
            GeminiAnalysis with structured outputs
        """
        if correlation_id:
            set_correlation_id(correlation_id)
        
        analysis_id = str(uuid.uuid4())
        timestamp = int(time.time())
        
        logger.info(
            f"Starting Gemini analysis for request: {request.request_id}",
            extra={"analysis_id": analysis_id},
        )
        
        try:
            # Call Gemini API
            results = self.client.analyze_motion_preview(
                preview_uri=request.preview_uri,
                metadata={"metadata_ref": request.metadata_ref} if request.metadata_ref else None,
                segments=request.blend_segments,
            )
            
            # Build GeminiAnalysis
            analysis = GeminiAnalysis(
                analysis_id=analysis_id,
                request_id=request.request_id,
                created_at=timestamp,
                model=ModelDescriptor(
                    provider=self.client.provider,
                    name=self.client.model,
                    version=self.model_version,
                ),
                inputs=GeminiInputs(
                    preview_uri=request.preview_uri,
                    metadata_ref=request.metadata_ref,
                ),
                outputs=GeminiOutputs(
                    style_labels=results["style_labels"],
                    transition_window=results["transition_window"],
                    npc_tags=results["npc_tags"],
                    safety_flags=results["safety_flags"],
                    summary=results.get("summary"),
                ),
            )
            
            logger.info(
                f"Gemini analysis complete: {analysis_id}",
                extra={
                    "analysis_id": analysis_id,
                    "style_labels": results["style_labels"],
                    "npc_tags": results["npc_tags"],
                    "has_safety_flags": len(results["safety_flags"]) > 0,
                },
            )
            
            return analysis
            
        except Exception as e:
            logger.error(f"Gemini analysis failed: {e}", exc_info=True)
            raise GeminiError(f"Failed to analyze motion preview: {e}")
    
    def build_query_descriptor(self, analysis: GeminiAnalysis) -> Dict[str, Any]:
        """
        Build query descriptor from Gemini analysis for similarity search.
        
        This combines style labels and NPC tags into a structured descriptor
        that can be embedded or used for vector search.
        
        Returns:
            Query descriptor dict
        """
        return {
            "style_labels": analysis.outputs.style_labels,
            "npc_tags": analysis.outputs.npc_tags,
            "transition_window": analysis.outputs.transition_window,
            "summary": analysis.outputs.summary or "",
        }
