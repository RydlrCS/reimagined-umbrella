"""
BlendAnim Service Stub.

Provides a lightweight blending service for the API endpoints.
For full SPADE-based blending, use spade_blend_service.
"""
import logging
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass, field

import numpy as np

from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class MotionSequence:
    """Motion sequence data container."""
    
    positions: np.ndarray  # [T, J, D]
    joint_names: List[str] = field(default_factory=list)
    fps: float = 30.0
    style_labels: List[str] = field(default_factory=list)
    
    @property
    def num_frames(self) -> int:
        return self.positions.shape[0]
    
    @property
    def num_joints(self) -> int:
        return self.positions.shape[1]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "positions": self.positions.tolist(),
            "joint_names": self.joint_names,
            "fps": self.fps,
            "style_labels": self.style_labels,
            "num_frames": self.num_frames,
            "num_joints": self.num_joints,
        }


class BlendAnimService:
    """
    Simple motion blending service.
    
    For production, use SPADEBlendService which provides
    skeleton-aware conditioning with trainable γ/β parameters.
    """
    
    def __init__(self):
        logger.info("[ENTRY] BlendAnimService.__init__")
        self._initialized = True
        logger.info("[EXIT] BlendAnimService.__init__")
    
    def blend(
        self,
        motion_a: MotionSequence,
        motion_b: MotionSequence,
        weights: List[float] = None,
        transition_frames: int = 30,
    ) -> Tuple[MotionSequence, Dict[str, Any]]:
        """
        Blend two motion sequences.
        
        Args:
            motion_a: First motion
            motion_b: Second motion
            weights: Duration weights [w_a, w_b]
            transition_frames: Frames for blend transition
        
        Returns:
            Tuple of (blended_motion, timing_info)
        """
        logger.info(
            f"[ENTRY] blend: a={motion_a.num_frames}f, b={motion_b.num_frames}f"
        )
        
        if weights is None:
            weights = [0.5, 0.5]
        
        # Calculate output dimensions
        T_a = motion_a.num_frames
        T_b = motion_b.num_frames
        total_frames = int(T_a * weights[0] + T_b * weights[1])
        
        # Interpolate motions to match
        pos_a = self._interpolate(motion_a.positions, total_frames)
        pos_b = self._interpolate(motion_b.positions, total_frames)
        
        # Build blend curve (sigmoid transition)
        transition_start = int(total_frames * weights[0]) - transition_frames // 2
        transition_end = transition_start + transition_frames
        transition_start = max(0, transition_start)
        transition_end = min(total_frames, transition_end)
        
        omega = np.zeros(total_frames)
        omega[transition_end:] = 1.0
        
        if transition_end > transition_start:
            t = np.linspace(-5, 5, transition_end - transition_start)
            omega[transition_start:transition_end] = 1 / (1 + np.exp(-t))
        
        # Blend
        blended_pos = np.zeros_like(pos_a)
        for t in range(total_frames):
            w = omega[t]
            blended_pos[t] = (1 - w) * pos_a[t] + w * pos_b[t]
        
        # Create output sequence
        blended = MotionSequence(
            positions=blended_pos,
            joint_names=motion_a.joint_names or motion_b.joint_names,
            fps=motion_a.fps,
            style_labels=motion_a.style_labels + motion_b.style_labels,
        )
        
        timing = {
            "total_frames": total_frames,
            "transition_start": transition_start,
            "transition_end": transition_end,
        }
        
        logger.info(f"[EXIT] blend: output={blended.num_frames}f")
        
        return blended, timing
    
    def _interpolate(self, positions: np.ndarray, target_frames: int) -> np.ndarray:
        """Interpolate motion to target frame count."""
        T, J, D = positions.shape
        
        if T == target_frames:
            return positions.copy()
        
        result = np.zeros((target_frames, J, D), dtype=np.float32)
        
        for j in range(J):
            for d in range(D):
                result[:, j, d] = np.interp(
                    np.linspace(0, T - 1, target_frames),
                    np.arange(T),
                    positions[:, j, d],
                )
        
        return result


# Singleton instance
_service_instance: Optional[BlendAnimService] = None


def get_blendanim_service() -> BlendAnimService:
    """Get singleton BlendAnimService instance."""
    global _service_instance
    if _service_instance is None:
        _service_instance = BlendAnimService()
    return _service_instance


def reset_blendanim_service() -> None:
    """Reset singleton instance (for testing)."""
    global _service_instance
    _service_instance = None
