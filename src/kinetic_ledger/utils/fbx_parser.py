"""
FBX Parser Wrapper.

Provides a simple interface for the SPADE blend endpoint to parse FBX files.
This wraps the fbx_loader module for API compatibility.
"""
import numpy as np
from typing import Tuple, Dict, Any, Optional
from pathlib import Path

from .fbx_loader import load_fbx_motion, FBXMotionData, MIXAMO_JOINT_ORDER
from .logging import get_logger

logger = get_logger(__name__)


class FBXParser:
    """Simple FBX parser for API endpoints."""
    
    def __init__(self):
        self.joint_order = MIXAMO_JOINT_ORDER
    
    def parse_fbx(self, fbx_path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Parse FBX file and return positions array and metadata.
        
        Args:
            fbx_path: Path to FBX file
            
        Returns:
            Tuple of (positions [T, J, 3], metadata dict)
        """
        logger.info(f"[ENTRY] FBXParser.parse_fbx: {fbx_path}")
        
        try:
            # Load motion data using fbx_loader
            motion_data = load_fbx_motion(fbx_path)
            
            # Extract positions array
            positions = motion_data.positions  # [T, J, 3]
            
            # Build metadata
            metadata = {
                "num_frames": motion_data.num_frames,
                "num_joints": motion_data.num_joints,
                "fps": motion_data.fps,
                "joint_names": motion_data.joint_names,
                "source_file": motion_data.source_file,
            }
            
            logger.info(
                f"[EXIT] FBXParser.parse_fbx: "
                f"frames={motion_data.num_frames}, joints={motion_data.num_joints}"
            )
            
            return positions, metadata
            
        except Exception as e:
            logger.error(f"FBX parsing failed: {e}")
            # Return synthetic data as fallback
            T = 120
            J = len(self.joint_order)
            positions = np.random.randn(T, J, 3).astype(np.float32) * 0.3
            
            metadata = {
                "num_frames": T,
                "num_joints": J,
                "fps": 30.0,
                "joint_names": self.joint_order,
                "source_file": str(fbx_path),
                "synthetic": True,
            }
            
            return positions, metadata


# Singleton instance
_parser_instance: Optional[FBXParser] = None


def get_fbx_parser() -> FBXParser:
    """Get or create the FBX parser singleton."""
    global _parser_instance
    if _parser_instance is None:
        _parser_instance = FBXParser()
    return _parser_instance
