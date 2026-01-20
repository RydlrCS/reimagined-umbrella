"""
Gemini-powered multimodal motion analysis service.

Uses Gemini's video understanding and structured output capabilities
to intelligently analyze FBX animations and recommend blend parameters.
"""

import io
import base64
import logging
from typing import Optional, Tuple, List
from pathlib import Path

try:
    import google.generativeai as genai
except ImportError:
    genai = None
from PIL import Image

from ..schemas.gemini_motion_schemas import (
    MotionBlendAnalysis,
    MotionCharacteristics,
    BlendParameters,
    CompatibilityScore,
    QualityPrediction
)
from ..utils.logging import get_logger

logger = get_logger(__name__)


class GeminiMotionAnalyzer:
    """
    Analyzes motion clips using Gemini's multimodal capabilities.
    
    This service renders FBX animations to video frames and uses Gemini
    to perform visual analysis, generating structured recommendations
    for optimal blend parameters.
    """
    
    def __init__(
        self,
        api_key: str,
        model_name: str = "gemini-2.0-flash-exp",
        frame_sample_rate: int = 5
    ):
        """
        Initialize the Gemini Motion Analyzer.
        
        Args:
            api_key: Google AI API key
            model_name: Gemini model to use (2.0-flash-exp or gemini-exp-1206)
            frame_sample_rate: Sample every Nth frame for analysis
        """
        if genai is None:
            raise RuntimeError("google-generativeai package not installed")
            
        genai.configure(api_key=api_key)
        self.model_name = model_name
        self.frame_sample_rate = frame_sample_rate
        
        # Initialize model with JSON output (schema not compatible with multimodal)
        self.model = genai.GenerativeModel(
            model_name=model_name,
            generation_config=genai.types.GenerationConfig(
                response_mime_type="application/json"
            )
        )
        
        logger.info(f"Initialized GeminiMotionAnalyzer with model: {model_name}")
    
    async def analyze_motion_pair(
        self,
        motion_a_frames: List[Image.Image],
        motion_b_frames: List[Image.Image],
        motion_a_name: str = "Motion A",
        motion_b_name: str = "Motion B"
    ) -> MotionBlendAnalysis:
        """
        Analyze two motion clips and recommend blend parameters.
        
        Args:
            motion_a_frames: List of PIL Images from first motion
            motion_b_frames: List of PIL Images from second motion
            motion_a_name: Name/description of first motion
            motion_b_name: Name/description of second motion
            
        Returns:
            MotionBlendAnalysis with recommendations
        """
        logger.info(f"Analyzing motion pair: {motion_a_name} + {motion_b_name}")
        
        # Sample frames to reduce token usage
        sampled_a = self._sample_frames(motion_a_frames)
        sampled_b = self._sample_frames(motion_b_frames)
        
        logger.info(f"Sampled {len(sampled_a)} frames from {motion_a_name}, "
                   f"{len(sampled_b)} frames from {motion_b_name}")
        
        # Prepare prompt for Gemini
        prompt = self._create_analysis_prompt(
            motion_a_name, motion_b_name,
            len(motion_a_frames), len(motion_b_frames)
        )
        
        # Prepare content with frames - Gemini SDK accepts PIL Images directly
        content = [prompt]
        
        # Add motion A frames
        content.append(f"\n## {motion_a_name} Frames:")
        for frame in sampled_a:
            content.append(frame)  # PIL Image objects are accepted directly
        
        # Add motion B frames
        content.append(f"\n## {motion_b_name} Frames:")
        for frame in sampled_b:
            content.append(frame)  # PIL Image objects are accepted directly
        
        try:
            # Generate analysis with structured output
            logger.info(f"Sending {len(sampled_a) + len(sampled_b)} frames to Gemini...")
            response = self.model.generate_content(content)
            
            # Parse structured output
            analysis = MotionBlendAnalysis.model_validate_json(response.text)
            
            logger.info(f"Analysis complete. Compatibility: {analysis.compatibility.overall_score:.2f}")
            logger.info(f"Recommendation: {analysis.overall_recommendation}")
            
            return analysis
            
        except Exception as e:
            logger.error(f"Gemini analysis failed: {e}")
            raise
    
    def _sample_frames(self, frames: List[Image.Image]) -> List[Image.Image]:
        """Sample frames at the specified rate."""
        if not frames:
            return []
        
        sampled = []
        for i in range(0, len(frames), self.frame_sample_rate):
            sampled.append(frames[i])
        
        # Always include last frame
        if frames[-1] not in sampled:
            sampled.append(frames[-1])
        
        return sampled
    
    def _create_analysis_prompt(
        self,
        motion_a_name: str,
        motion_b_name: str,
        total_frames_a: int,
        total_frames_b: int
    ) -> str:
        """Create the analysis prompt for Gemini."""
        
        return f"""You are an expert animation technical director analyzing two motion capture clips for blending.

## Task
Analyze the visual characteristics of these two motions and provide structured recommendations for creating a high-quality blend using the BlendAnim algorithm.

## Motion Details
- **{motion_a_name}**: {total_frames_a} frames total
- **{motion_b_name}**: {total_frames_b} frames total

## Analysis Requirements

### 1. Motion Characteristics
For each motion, identify:
- **motion_type**: Primary category (walk, run, jump, idle, combat, dance, other)
- **energy_level**: Overall intensity (low, medium, high)
- **key_joints**: Most active/important joints (up to 10)
- **velocity_profiles**: Estimate velocity for key joints
- **has_cyclic_pattern**: Does the motion repeat cyclically?
- **ground_contact**: Is the character grounded or aerial?
- **motion_description**: Brief description

### 2. Compatibility Assessment
Evaluate how well these motions can blend:
- **overall_score**: 0-1 compatibility score
- **velocity_compatibility**: How well velocities match (0-1)
- **pose_similarity**: Similarity of key poses (0-1)
- **energy_match**: How well energy levels align (0-1)
- **reasoning**: Explain the compatibility assessment

### 3. Blend Parameters
Recommend optimal parameters:
- **transition_frames**: 10-60 frames for transition window
- **crosshatch_offset**: 0-30 frame offset for alignment
- **omega_curve_type**: smoothstep, linear, ease_in, or ease_out
- **apply_velocity_matching**: Should velocity be matched at boundaries?
- **apply_root_motion_correction**: Should root motion be corrected?

### 4. Quality Prediction
Predict blend result quality:
- **predicted_coverage**: Expected coverage metric (0-1)
- **predicted_diversity**: Expected diversity metric (0-1)
- **predicted_smoothness**: Expected smoothness (0-1)
- **confidence**: Prediction confidence (0-1)
- **potential_issues**: List potential quality issues

### 5. Overall Recommendation
Provide a summary recommendation (max 400 chars) with key insights for achieving the best blend quality.

## Key Considerations
- BlendAnim uses temporal conditioning: ω(t) = 3t² - 2t³ (smoothstep)
- Transition quality depends on pose similarity and velocity matching
- Per-joint weights can prioritize important joints (hands, feet, spine)
- Root motion correction prevents spatial discontinuities
- Coverage and diversity metrics measure motion space utilization

Analyze the provided frames and generate structured recommendations.

IMPORTANT: Return ONLY a valid JSON object matching the MotionBlendAnalysis schema. No additional text."""
    
    async def analyze_single_motion(
        self,
        frames: List[Image.Image],
        motion_name: str = "Motion"
    ) -> MotionCharacteristics:
        """
        Analyze a single motion clip characteristics.
        
        Args:
            frames: List of PIL Images
            motion_name: Name/description of motion
            
        Returns:
            MotionCharacteristics analysis
        """
        logger.info(f"Analyzing single motion: {motion_name}")
        
        sampled = self._sample_frames(frames)
        
        # Create model for single motion analysis
        model = genai.GenerativeModel(
            model_name=self.model_name,
            generation_config=genai.types.GenerationConfig(
                response_mime_type="application/json",
                response_schema=MotionCharacteristics
            )
        )
        
        prompt = f"""Analyze this motion capture animation and extract its characteristics.

Motion: {motion_name} ({len(frames)} frames)

Identify:
1. Primary motion type (walk, run, jump, idle, combat, dance, other)
2. Energy level (low, medium, high)
3. Key active joints (up to 10)
4. Velocity profiles for key joints
5. Whether motion has cyclic pattern
6. Whether character maintains ground contact
7. Brief motion description

Provide structured analysis based on the visual frames."""
        
        content = [prompt]
        for frame in sampled:
            img_bytes = io.BytesIO()
            frame.save(img_bytes, format='PNG')
            content.append({
                'mime_type': 'image/png',
                'data': img_bytes.getvalue()
            })
        
        try:
            response = model.generate_content(content)
            characteristics = MotionCharacteristics.model_validate_json(response.text)
            
            logger.info(f"Motion analysis complete: {characteristics.motion_type}, "
                       f"energy={characteristics.energy_level}")
            
            return characteristics
            
        except Exception as e:
            logger.error(f"Single motion analysis failed: {e}")
            raise
    
    async def predict_blend_quality(
        self,
        motion_a_frames: List[Image.Image],
        motion_b_frames: List[Image.Image],
        blend_params: BlendParameters
    ) -> QualityPrediction:
        """
        Predict quality of a blend with specific parameters.
        
        Args:
            motion_a_frames: First motion frames
            motion_b_frames: Second motion frames
            blend_params: Proposed blend parameters
            
        Returns:
            QualityPrediction assessment
        """
        logger.info("Predicting blend quality for given parameters")
        
        sampled_a = self._sample_frames(motion_a_frames)
        sampled_b = self._sample_frames(motion_b_frames)
        
        # Create model for quality prediction
        model = genai.GenerativeModel(
            model_name=self.model_name,
            generation_config=genai.types.GenerationConfig(
                response_mime_type="application/json",
                response_schema=QualityPrediction
            )
        )
        
        prompt = f"""Predict the quality of blending these two motions with the given parameters.

## Blend Parameters
- Transition frames: {blend_params.transition_frames}
- Crosshatch offset: {blend_params.crosshatch_offset}
- Omega curve: {blend_params.omega_curve_type}
- Velocity matching: {blend_params.apply_velocity_matching}
- Root motion correction: {blend_params.apply_root_motion_correction}

## Task
Predict:
1. Coverage metric (0-1): How well the blend covers the motion space
2. Diversity metric (0-1): Variety in the resulting motion
3. Smoothness (0-1): Smoothness at the transition
4. Confidence (0-1): Your confidence in this prediction
5. Potential issues: List up to 5 potential quality problems

Base your prediction on visual analysis of the motion frames."""
        
        content = [prompt, "\nMotion A:"] + sampled_a + ["\nMotion B:"] + sampled_b
        
        try:
            response = model.generate_content(content)
            prediction = QualityPrediction.model_validate_json(response.text)
            
            logger.info(f"Quality prediction: coverage={prediction.predicted_coverage:.2f}, "
                       f"smoothness={prediction.predicted_smoothness:.2f}")
            
            return prediction
            
        except Exception as e:
            logger.error(f"Quality prediction failed: {e}")
            raise
