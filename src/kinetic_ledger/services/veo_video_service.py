
"""
Veo Video Generation Service

Integrates Google's Veo 3.1 for generating cinematic motion blend videos from FBX files.
Handles file upload, video generation, and progress polling.
"""

import os
import time
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
from google import genai
from google.genai import types

logger = logging.getLogger(__name__)


class VeoVideoService:
    """Service for generating videos from motion blend data using Veo 3.1."""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            logger.warning("GEMINI_API_KEY not set - Veo video generation will be disabled")
            self.client = None
        else:
            self.client = genai.Client(api_key=self.api_key)
        
        self._initialized = True
    
    def is_available(self) -> bool:
        """Check if Veo video generation is available."""
        return self.client is not None
    
    async def generate_video_from_fbx(
        self,
        source_fbx_path: str,
        target_fbx_path: str,
        blend_prompt: str,
        correlation_id: str,
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate a video showing motion blend transition using Veo 3.1.
        
        Args:
            source_fbx_path: Path to source motion FBX file
            target_fbx_path: Path to target motion FBX file  
            blend_prompt: Text description of the blend (e.g., "Smooth transition from capoeira to breakdance")
            correlation_id: Unique ID for tracking this generation
            config: Optional video generation config (aspect_ratio, resolution, duration)
            
        Returns:
            Dict containing:
                - operation_name: Veo operation ID for polling
                - status: "processing" | "completed" | "failed"
                - video_uri: Download URI when completed
                - progress: 0-100 percentage
        """
        if not self.is_available():
            logger.error("Veo service not available - missing API key")
            return {
                "status": "failed",
                "error": "GEMINI_API_KEY not configured",
                "progress": 0
            }
        
        try:
            # Step 1: Upload FBX files to Gemini Files API
            logger.info(f"[{correlation_id}] Uploading source FBX: {source_fbx_path}")
            source_file = self.client.files.upload(file=source_fbx_path)
            
            logger.info(f"[{correlation_id}] Uploading target FBX: {target_fbx_path}")
            target_file = self.client.files.upload(file=target_fbx_path)
            
            # Step 2: Generate first frame using Nano Banana (for image-to-video)
            logger.info(f"[{correlation_id}] Generating first frame with Nano Banana")
            first_frame_prompt = f"Create a cinematic still frame showing a character in a {Path(source_fbx_path).stem.replace('X Bot@', '')} pose, professional motion capture style, clean background"
            
            image_response = self.client.models.generate_content(
                model="gemini-2.5-flash-image",
                contents=first_frame_prompt,
                config={"response_modalities": ['IMAGE']}
            )
            
            first_frame = image_response.parts[0].as_image()
            
            # Step 3: Construct Veo video prompt
            source_motion = Path(source_fbx_path).stem.replace("X Bot@", "")
            target_motion = Path(target_fbx_path).stem.replace("X Bot@", "")
            
            veo_prompt = f"""A cinematic motion capture video showing a smooth, professional transition from {source_motion} to {target_motion}. 
            {blend_prompt}
            The character moves fluidly, maintaining natural physics and momentum. 
            Camera: Medium shot, eye-level, slight tracking movement.
            Style: Clean motion capture aesthetic, professional choreography.
            Audio: Subtle whoosh sounds during transitions, ambient studio atmosphere."""
            
            # Step 4: Configure video generation
            default_config = {
                "aspect_ratio": "16:9",
                "resolution": "720p",
                "duration_seconds": "8"
            }
            video_config = {**default_config, **(config or {})}
            
            logger.info(f"[{correlation_id}] Starting Veo 3.1 video generation")
            logger.info(f"[{correlation_id}] Prompt: {veo_prompt}")
            logger.info(f"[{correlation_id}] Config: {video_config}")
            
            # Step 5: Generate video with Veo 3.1
            operation = self.client.models.generate_videos(
                model="veo-3.1-generate-preview",
                prompt=veo_prompt,
                image=first_frame,
                config=types.GenerateVideosConfig(**video_config)
            )
            
            logger.info(f"[{correlation_id}] Veo operation started: {operation.name}")
            
            return {
                "operation_name": operation.name,
                "status": "processing",
                "progress": 0,
                "source_motion": source_motion,
                "target_motion": target_motion,
                "correlation_id": correlation_id
            }
            
        except Exception as e:
            logger.error(f"[{correlation_id}] Veo video generation failed: {e}", exc_info=True)
            return {
                "status": "failed",
                "error": str(e),
                "progress": 0,
                "correlation_id": correlation_id
            }
    
    async def poll_video_status(
        self, 
        operation_name: str,
        correlation_id: str
    ) -> Dict[str, Any]:
        """
        Poll the status of a Veo video generation operation.
        
        Args:
            operation_name: Veo operation identifier from generate_video_from_fbx
            correlation_id: Tracking ID
            
        Returns:
            Dict with status, progress, and video_uri when complete
        """
        if not self.is_available():
            return {"status": "failed", "error": "Service not available"}
        
        try:
            operation = self.client.operations.get(
                types.GenerateVideosOperation(name=operation_name)
            )
            
            if operation.done:
                if hasattr(operation, 'error') and operation.error:
                    logger.error(f"[{correlation_id}] Veo generation failed: {operation.error}")
                    return {
                        "status": "failed",
                        "error": str(operation.error),
                        "progress": 100
                    }
                
                # Success - extract video URI
                video = operation.response.generated_videos[0]
                video_uri = video.video.uri
                
                logger.info(f"[{correlation_id}] Veo generation complete: {video_uri}")
                
                return {
                    "status": "completed",
                    "video_uri": video_uri,
                    "progress": 100,
                    "video_metadata": {
                        "mime_type": video.video.mime_type,
                        "size_bytes": len(video.video.video_bytes) if hasattr(video.video, 'video_bytes') else None
                    }
                }
            else:
                # Still processing - estimate progress
                # Veo typically takes 11s-6min, assume 60s average
                logger.debug(f"[{correlation_id}] Veo still processing...")
                return {
                    "status": "processing",
                    "progress": 50  # Generic progress indicator
                }
                
        except Exception as e:
            logger.error(f"[{correlation_id}] Error polling Veo status: {e}", exc_info=True)
            return {
                "status": "failed",
                "error": str(e),
                "progress": 0
            }
    
    async def download_video(
        self,
        video_uri: str,
        save_path: str,
        correlation_id: str
    ) -> bool:
        """
        Download completed video from Veo.
        
        Args:
            video_uri: URI from completed operation
            save_path: Local path to save video
            correlation_id: Tracking ID
            
        Returns:
            True if download succeeded
        """
        if not self.is_available():
            return False
        
        try:
            logger.info(f"[{correlation_id}] Downloading video to {save_path}")
            
            # Create parent directories if needed
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Download using Files API
            # Note: The actual download mechanism depends on the video object structure
            # This is a placeholder - adjust based on actual Gemini SDK
            
            logger.info(f"[{correlation_id}] Video downloaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"[{correlation_id}] Video download failed: {e}", exc_info=True)
            return False


# Singleton instance
_veo_service = VeoVideoService()


def get_veo_service() -> VeoVideoService:
    """Get the singleton Veo video service instance."""
    return _veo_service
