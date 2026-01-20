"""
Utilities for rendering FBX animations to video frames.

Provides headless rendering of character animations using PyOpenGL
or fallback to matplotlib-based visualization.
"""

import os
import logging
import tempfile
from typing import List, Optional, Tuple
from pathlib import Path

import numpy as np
from PIL import Image

try:
    import matplotlib
    matplotlib.use('Agg')  # Headless backend
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logging.warning("Matplotlib not available for frame rendering")

from ..utils.logging import get_logger

logger = get_logger(__name__)


class MotionFrameRenderer:
    """
    Renders FBX animation frames as images for Gemini analysis.
    
    Uses matplotlib for simple 3D skeleton visualization in headless mode.
    """
    
    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        dpi: int = 100,
        view_angle: Tuple[float, float] = (30, 45)
    ):
        """
        Initialize the frame renderer.
        
        Args:
            width: Image width in pixels
            height: Image height in pixels
            dpi: Dots per inch for rendering
            view_angle: (elevation, azimuth) viewing angles in degrees
        """
        if not MATPLOTLIB_AVAILABLE:
            raise RuntimeError("Matplotlib required for frame rendering")
        
        self.width = width
        self.height = height
        self.dpi = dpi
        self.view_angle = view_angle
        
        logger.info(f"Initialized MotionFrameRenderer: {width}x{height} @ {dpi} DPI")
    
    def render_frames_from_positions(
        self,
        positions: np.ndarray,
        skeleton_hierarchy: Optional[List[Tuple[int, int]]] = None,
        frame_indices: Optional[List[int]] = None
    ) -> List[Image.Image]:
        """
        Render animation frames from position data.
        
        Args:
            positions: Shape (num_frames, num_joints, 3) position array
            skeleton_hierarchy: List of (parent_idx, child_idx) bone connections
            frame_indices: Specific frames to render (None = all frames)
            
        Returns:
            List of PIL Images
        """
        if positions.ndim != 3 or positions.shape[2] != 3:
            raise ValueError(f"Expected (frames, joints, 3) array, got {positions.shape}")
        
        num_frames, num_joints, _ = positions.shape
        
        if frame_indices is None:
            frame_indices = list(range(num_frames))
        
        logger.info(f"Rendering {len(frame_indices)} frames from {num_frames} total")
        
        frames = []
        for frame_idx in frame_indices:
            if frame_idx < 0 or frame_idx >= num_frames:
                logger.warning(f"Frame index {frame_idx} out of range, skipping")
                continue
            
            frame_positions = positions[frame_idx]
            image = self._render_single_frame(frame_positions, skeleton_hierarchy)
            frames.append(image)
        
        logger.info(f"Successfully rendered {len(frames)} frames")
        return frames
    
    def _render_single_frame(
        self,
        positions: np.ndarray,
        skeleton_hierarchy: Optional[List[Tuple[int, int]]] = None
    ) -> Image.Image:
        """
        Render a single frame.
        
        Args:
            positions: Shape (num_joints, 3) positions for this frame
            skeleton_hierarchy: Bone connections
            
        Returns:
            PIL Image
        """
        fig = plt.figure(figsize=(self.width / self.dpi, self.height / self.dpi), dpi=self.dpi)
        ax = fig.add_subplot(111, projection='3d')
        
        # Extract XYZ coordinates
        x = positions[:, 0]
        y = positions[:, 1]
        z = positions[:, 2]
        
        # Plot joints as points
        ax.scatter(x, y, z, c='blue', marker='o', s=20, alpha=0.8)
        
        # Plot skeleton bones if hierarchy provided
        if skeleton_hierarchy:
            for parent_idx, child_idx in skeleton_hierarchy:
                if parent_idx < len(positions) and child_idx < len(positions):
                    ax.plot(
                        [positions[parent_idx, 0], positions[child_idx, 0]],
                        [positions[parent_idx, 1], positions[child_idx, 1]],
                        [positions[parent_idx, 2], positions[child_idx, 2]],
                        'r-', linewidth=1.5, alpha=0.6
                    )
        
        # Set equal aspect ratio and viewing angle
        max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2.0
        mid_x = (x.max() + x.min()) * 0.5
        mid_y = (y.max() + y.min()) * 0.5
        mid_z = (z.max() + z.min()) * 0.5
        
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        # Set viewing angle
        ax.view_init(elev=self.view_angle[0], azim=self.view_angle[1])
        
        # Remove grid for cleaner visualization
        ax.grid(False)
        ax.set_facecolor('white')
        
        # Render to image buffer
        fig.canvas.draw()
        
        # Convert to PIL Image (RGB for better compatibility with Gemini)
        buf = fig.canvas.buffer_rgba()
        image = Image.frombytes('RGBA', (self.width, self.height), buf)
        
        # Convert RGBA to RGB (Gemini prefers RGB)
        if image.mode == 'RGBA':
            rgb_image = Image.new('RGB', image.size, (255, 255, 255))
            rgb_image.paste(image, mask=image.split()[3])  # Use alpha as mask
            image = rgb_image
        
        plt.close(fig)
        
        return image
    
    def render_comparison_grid(
        self,
        positions_a: np.ndarray,
        positions_b: np.ndarray,
        frame_idx: int = 0,
        skeleton_hierarchy: Optional[List[Tuple[int, int]]] = None
    ) -> Image.Image:
        """
        Render side-by-side comparison of two motions at a specific frame.
        
        Args:
            positions_a: First motion positions (frames, joints, 3)
            positions_b: Second motion positions (frames, joints, 3)
            frame_idx: Frame index to visualize
            skeleton_hierarchy: Bone connections
            
        Returns:
            PIL Image with side-by-side comparison
        """
        fig = plt.figure(figsize=(self.width * 2 / self.dpi, self.height / self.dpi), dpi=self.dpi)
        
        # Motion A
        ax1 = fig.add_subplot(121, projection='3d')
        self._plot_skeleton(ax1, positions_a[frame_idx], skeleton_hierarchy, 'Motion A')
        
        # Motion B
        ax2 = fig.add_subplot(122, projection='3d')
        self._plot_skeleton(ax2, positions_b[frame_idx], skeleton_hierarchy, 'Motion B')
        
        plt.tight_layout()
        
        # Convert to image
        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        image = Image.frombytes('RGBA', (self.width * 2, self.height), buf)
        
        plt.close(fig)
        
        return image
    
    def _plot_skeleton(
        self,
        ax,
        positions: np.ndarray,
        skeleton_hierarchy: Optional[List[Tuple[int, int]]],
        title: str
    ):
        """Helper to plot skeleton on an axes."""
        x, y, z = positions[:, 0], positions[:, 1], positions[:, 2]
        
        ax.scatter(x, y, z, c='blue', marker='o', s=20, alpha=0.8)
        
        if skeleton_hierarchy:
            for parent_idx, child_idx in skeleton_hierarchy:
                if parent_idx < len(positions) and child_idx < len(positions):
                    ax.plot(
                        [positions[parent_idx, 0], positions[child_idx, 0]],
                        [positions[parent_idx, 1], positions[child_idx, 1]],
                        [positions[parent_idx, 2], positions[child_idx, 2]],
                        'r-', linewidth=1.5, alpha=0.6
                    )
        
        max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2.0
        mid_x, mid_y, mid_z = (x.max() + x.min()) * 0.5, (y.max() + y.min()) * 0.5, (z.max() + z.min()) * 0.5
        
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        ax.set_title(title)
        ax.view_init(elev=self.view_angle[0], azim=self.view_angle[1])
        ax.grid(False)


def create_standard_skeleton_hierarchy(num_joints: int = 52) -> List[Tuple[int, int]]:
    """
    Create a standard humanoid skeleton hierarchy.
    
    This is a simplified hierarchy for visualization purposes.
    For accurate bone connections, parse from FBX metadata.
    
    Args:
        num_joints: Number of joints in skeleton
        
    Returns:
        List of (parent, child) bone connections
    """
    # Simplified humanoid hierarchy (common Mixamo structure)
    # This is a basic approximation - actual hierarchy should come from FBX
    hierarchy = [
        # Spine
        (0, 1), (1, 2), (2, 3), (3, 4),  # Hips -> Spine -> Chest -> Neck -> Head
        
        # Left arm
        (3, 5), (5, 6), (6, 7), (7, 8),  # Chest -> LeftShoulder -> LeftArm -> LeftForearm -> LeftHand
        
        # Right arm
        (3, 9), (9, 10), (10, 11), (11, 12),  # Chest -> RightShoulder -> RightArm -> RightForearm -> RightHand
        
        # Left leg
        (0, 13), (13, 14), (14, 15), (15, 16),  # Hips -> LeftUpLeg -> LeftLeg -> LeftFoot -> LeftToe
        
        # Right leg
        (0, 17), (17, 18), (18, 19), (19, 20),  # Hips -> RightUpLeg -> RightLeg -> RightFoot -> RightToe
    ]
    
    return hierarchy


def render_fbx_to_frames(
    fbx_path: str,
    output_dir: Optional[str] = None,
    max_frames: int = 100,
    sample_rate: int = 5
) -> List[Image.Image]:
    """
    High-level function to render FBX file to frames.
    
    Args:
        fbx_path: Path to FBX file
        output_dir: Optional directory to save frames
        max_frames: Maximum frames to render
        sample_rate: Render every Nth frame
        
    Returns:
        List of PIL Images
    """
    from ..services.blendanim_service import BlendAnimService
    
    logger.info(f"Rendering frames from FBX: {fbx_path}")
    
    # Parse FBX to get positions
    blend_service = BlendAnimService()
    positions, metadata = blend_service._parse_fbx_positions(fbx_path)
    
    if positions is None:
        raise RuntimeError(f"Failed to parse FBX: {fbx_path}")
    
    # Sample frames
    num_frames = min(len(positions), max_frames)
    frame_indices = list(range(0, num_frames, sample_rate))
    
    # Create renderer
    renderer = MotionFrameRenderer()
    
    # Get skeleton hierarchy (simplified)
    num_joints = positions.shape[1]
    hierarchy = create_standard_skeleton_hierarchy(num_joints)
    
    # Render frames
    frames = renderer.render_frames_from_positions(
        positions,
        skeleton_hierarchy=hierarchy,
        frame_indices=frame_indices
    )
    
    # Optionally save frames
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for i, frame in enumerate(frames):
            frame_path = output_path / f"frame_{i:04d}.png"
            frame.save(frame_path)
            logger.info(f"Saved frame to {frame_path}")
    
    logger.info(f"Rendered {len(frames)} frames from {fbx_path}")
    return frames
