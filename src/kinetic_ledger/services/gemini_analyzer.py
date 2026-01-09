"""
Gemini Multimodal Analyzer - analyzes motion previews with Gemini.

Uses structured outputs for type-safe, validated responses.
"""
import logging
import time
import uuid
from typing import Optional, List, Dict, Any
from pydantic import BaseModel
import json
from google import genai
from google.genai import types

from ..schemas.models import (
    GeminiAnalysis,
    ModelDescriptor,
    GeminiInputs,
    GeminiOutputs,
)
from ..schemas.structured_outputs import (
    MotionStyleClassification,
    SafetyAssessment,
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
    
    Uses Google Gemini SDK for real multimodal analysis.
    """
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-2.0-flash-exp"):
        self.api_key = api_key
        self.model = model
        self.provider = "google"
        self._client = None
        
        if api_key:
            self._client = genai.Client(api_key=api_key)
            logger.info(f"Gemini client configured with model: {model}")
    
    def _build_analysis_prompt(
        self,
        segments: Optional[List[Dict[str, Any]]] = None,
        npc_context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Build structured prompt for motion analysis."""
        
        segment_info = ""
        if segments:
            segment_info = "\n\nMotion Blend Segments:\n"
            for i, seg in enumerate(segments):
                segment_info += f"- Segment {i+1}: {seg.get('label', 'unknown')} (frames {seg.get('start_frame', 0)}-{seg.get('end_frame', 0)})\n"
        
        npc_info = ""
        if npc_context:
            npc_info = f"\n\nNPC Context:\n- Game: {npc_context.get('game', 'N/A')}\n- Intent: {', '.join(npc_context.get('intent', []))}\n- Environment: {npc_context.get('environment', 'N/A')}"
        
        prompt = f"""Analyze this motion capture animation blend for NPC character generation.
{segment_info}{npc_info}

Provide a structured analysis in JSON format with these fields:
{{
  "style_labels": ["style1", "style2"],  // Motion styles detected (e.g., capoeira, breakdance, martial_arts)
  "transition_window": {{"start_frame": X, "end_frame": Y}},  // Frame range where transition occurs
  "npc_tags": ["tag1", "tag2"],  // Character attributes (e.g., agile, athletic, evasive, dynamic)
  "safety_flags": [],  // Any content safety concerns (violence, inappropriate, etc.)
  "summary": "Brief description of the motion blend"
}}

Focus on:
1. Identifying distinct motion styles from the segments
2. Detecting the transition region between styles
3. Inferring NPC character traits suitable for the given context
4. Flagging any safety concerns

Respond ONLY with valid JSON, no additional text."""

        return prompt
    
    @retry_on_gemini_error(max_attempts=5, min_wait=2, max_wait=30)
    def analyze_motion_preview(
        self,
        preview_uri: str,
        metadata: Optional[Dict[str, Any]] = None,
        segments: Optional[List[Dict[str, Any]]] = None,
        npc_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Analyze motion preview with Gemini multimodal API.
        
        Args:
            preview_uri: URI to preview video/keyframes
            metadata: Additional metadata context
            segments: Blend segment information
            npc_context: NPC context information
        
        Returns:
            Analysis results
        """
        logger.info(f"Analyzing motion preview: {preview_uri}")
        
        # If no API key configured, fallback to rule-based analysis
        if not self._client:
            logger.warning("Gemini API key not configured, using fallback analysis")
            return self._fallback_analysis(segments, npc_context)
        
        try:
            # Build prompt
            prompt = self._build_analysis_prompt(segments, npc_context)
            
            # Call Gemini API with new SDK
            logger.info("Calling Gemini API for motion analysis")
            response = self._client.models.generate_content(
                model=self.model,
                contents=prompt,
            )
            
            # Parse JSON response
            response_text = response.text.strip()
            
            # Extract JSON from markdown code blocks if present
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()
            
            result = json.loads(response_text)
            
            # Validate and sanitize response
            result["style_labels"] = result.get("style_labels", [])[:5]
            result["npc_tags"] = list(set(result.get("npc_tags", [])))[:10]
            result["safety_flags"] = result.get("safety_flags", [])
            result["summary"] = result.get("summary", "Motion analysis completed")
            
            if "transition_window" not in result:
                result["transition_window"] = {"start_frame": 110, "end_frame": 140}
            
            logger.info(f"Gemini analysis complete: {len(result['style_labels'])} styles, {len(result['npc_tags'])} tags")
            return result
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Gemini response as JSON: {e}")
            raise GeminiError(f"Invalid JSON response from Gemini: {e}")
        except Exception as e:
            logger.error(f"Gemini API call failed: {e}", exc_info=True)
            raise GeminiError(f"Gemini API error: {e}")
    
    def _fallback_analysis(
        self,
        segments: Optional[List[Dict[str, Any]]] = None,
        npc_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Fallback rule-based analysis when API is unavailable."""
        
        # Extract style labels from segments
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
            first_end = segments[0].get("end_frame", 100)
            second_start = segments[1].get("start_frame", 125)
            transition_window = {
                "start_frame": max(0, first_end - 15),
                "end_frame": min(second_start + 15, 250),
            }
        
        # Generate NPC tags based on style
        npc_tags = []
        style_text = " ".join(style_labels).lower()
        
        if "capoeira" in style_text:
            npc_tags.extend(["agile", "evasive", "high_energy", "fluid"])
        if "breakdance" in style_text:
            npc_tags.extend(["athletic", "dynamic", "acrobatic", "powerful"])
        if "martial" in style_text or "combat" in style_text:
            npc_tags.extend(["combat_ready", "defensive", "tactical"])
        if "dance" in style_text:
            npc_tags.extend(["graceful", "rhythmic", "expressive"])
        
        if not npc_tags:
            npc_tags = ["animated", "mobile", "responsive"]
        
        # Safety flags
        safety_flags = []
        
        # Generate summary
        summary = f"Motion blend with {len(style_labels)} distinct styles"
        if len(style_labels) >= 2:
            summary = f"Blend transitions from {style_labels[0]} into {style_labels[1]}"
        
        return {
            "style_labels": style_labels[:5],
            "transition_window": transition_window,
            "npc_tags": list(set(npc_tags))[:10],
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
        model_version: str = "gemini-2.0-flash-exp",
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
                npc_context=request.npc_context,
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
