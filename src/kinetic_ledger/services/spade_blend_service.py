"""
SPADE Hierarchical Motion Blending Service.

Implements Spatially-Adaptive Denormalization (SPADE) layers for motion blending
with trainable γ/β parameters using PyTorch.

Research basis:
- Level 1 (COARSE) SPADE conditioning yields optimal FID/coverage trade-off
- Captures coarse motion features critical for overall blended motion structure
- γ(style) and β(style) are learned modulation parameters

References:
- SPADE: Semantic Image Synthesis with Spatially-Adaptive Normalization
- BlendAnim: Temporal conditioning for motion synthesis
"""
import os
import time
import hashlib
import logging
from typing import Optional, List, Tuple, Dict, Any
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Conditional PyTorch import
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    Tensor = Any  # Type alias fallback

from ..schemas.models import (
    HierarchyLevel,
    SPADEConfig,
    SPADEBlendRequest,
    SPADEBlendResponse,
    SPADEMetrics,
    JointHierarchyMapping,
)
from ..utils.logging import get_logger

# Setup logging with verbose entry/exit pattern
logger = get_logger(__name__)


# =============================================================================
# Constants
# =============================================================================

# Default Mixamo 24-joint skeleton
MIXAMO_24_JOINTS = [
    "Hips", "Spine", "Spine1", "Spine2", "Neck", "Head",
    "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand",
    "RightShoulder", "RightArm", "RightForeArm", "RightHand",
    "LeftUpLeg", "LeftLeg", "LeftFoot", "LeftToeBase",
    "RightUpLeg", "RightLeg", "RightFoot", "RightToeBase",
    "LeftEye", "RightEye",  # Optional eye joints
]

# Joint to hierarchy level mapping (research-based)
DEFAULT_JOINT_LEVELS: Dict[str, HierarchyLevel] = {
    # Level 1: COARSE - SPADE conditioning applied
    "Hips": HierarchyLevel.COARSE,
    "Spine": HierarchyLevel.COARSE,
    
    # Level 2: MID
    "Spine1": HierarchyLevel.MID,
    "Spine2": HierarchyLevel.MID,
    "LeftUpLeg": HierarchyLevel.MID,
    "RightUpLeg": HierarchyLevel.MID,
    "LeftArm": HierarchyLevel.MID,
    "RightArm": HierarchyLevel.MID,
    
    # Level 3: FINE
    "Neck": HierarchyLevel.FINE,
    "Head": HierarchyLevel.FINE,
    "LeftLeg": HierarchyLevel.FINE,
    "RightLeg": HierarchyLevel.FINE,
    "LeftForeArm": HierarchyLevel.FINE,
    "RightForeArm": HierarchyLevel.FINE,
    "LeftFoot": HierarchyLevel.FINE,
    "RightFoot": HierarchyLevel.FINE,
    
    # Level 4: DETAIL
    "LeftShoulder": HierarchyLevel.DETAIL,
    "RightShoulder": HierarchyLevel.DETAIL,
    "LeftHand": HierarchyLevel.DETAIL,
    "RightHand": HierarchyLevel.DETAIL,
    "LeftToeBase": HierarchyLevel.DETAIL,
    "RightToeBase": HierarchyLevel.DETAIL,
    "LeftEye": HierarchyLevel.DETAIL,
    "RightEye": HierarchyLevel.DETAIL,
}


# =============================================================================
# PyTorch SPADE Layer
# =============================================================================

# Mixamo skeleton adjacency (parent-child relationships)
MIXAMO_SKELETON_ADJACENCY = {
    "Hips": [],  # Root
    "Spine": ["Hips"],
    "Spine1": ["Spine"],
    "Spine2": ["Spine1"],
    "Neck": ["Spine2"],
    "Head": ["Neck"],
    "LeftShoulder": ["Spine2"],
    "LeftArm": ["LeftShoulder"],
    "LeftForeArm": ["LeftArm"],
    "LeftHand": ["LeftForeArm"],
    "RightShoulder": ["Spine2"],
    "RightArm": ["RightShoulder"],
    "RightForeArm": ["RightArm"],
    "RightHand": ["RightForeArm"],
    "LeftUpLeg": ["Hips"],
    "LeftLeg": ["LeftUpLeg"],
    "LeftFoot": ["LeftLeg"],
    "LeftToeBase": ["LeftFoot"],
    "RightUpLeg": ["Hips"],
    "RightLeg": ["RightUpLeg"],
    "RightFoot": ["RightLeg"],
    "RightToeBase": ["RightFoot"],
    "LeftEye": ["Head"],
    "RightEye": ["Head"],
}


def generate_skeleton_id_map(
    joint_names: List[str],
    hierarchy_level: HierarchyLevel,
    adjacency: Optional[Dict[str, List[str]]] = None,
) -> np.ndarray:
    """
    Generate skeleton ID map for temporal conditioning.
    
    The skeleton ID map is a binary matrix indicating which joints
    belong to a specific hierarchy level. This is used by SkeletonConv
    to produce γ and β modulation tensors.
    
    Based on Figure 2 from the paper:
        skeleton_id_maps → SkeletonConv → γ, β
    
    Args:
        joint_names: List of joint names in order
        hierarchy_level: Target hierarchy level to mark
        adjacency: Optional skeleton adjacency dict (default: Mixamo)
    
    Returns:
        Binary skeleton ID map [J, J] where 1 indicates joint in level
    """
    logger.debug(f"[ENTRY] generate_skeleton_id_map: level={hierarchy_level.value}")
    
    if adjacency is None:
        adjacency = MIXAMO_SKELETON_ADJACENCY
    
    J = len(joint_names)
    id_map = np.zeros((J, J), dtype=np.float32)
    
    # Get joints belonging to this level
    level_joints = set()
    for joint, level in DEFAULT_JOINT_LEVELS.items():
        if level == hierarchy_level and joint in joint_names:
            level_joints.add(joint)
    
    # Mark diagonal entries for level joints (self-connection)
    for i, joint in enumerate(joint_names):
        if joint in level_joints:
            id_map[i, i] = 1.0
            
            # Also mark parent-child connections within level
            if joint in adjacency:
                for parent in adjacency[joint]:
                    if parent in joint_names and parent in level_joints:
                        parent_idx = joint_names.index(parent)
                        id_map[i, parent_idx] = 1.0
                        id_map[parent_idx, i] = 1.0
    
    # Add border row/column for temporal conditioning (as shown in diagram)
    # The "11111" pattern at the top indicates temporal frame markers
    # We add this as a normalized weight sum in the first row/col
    id_map[0, :] = np.clip(id_map[0, :] + 0.5, 0, 1)
    id_map[:, 0] = np.clip(id_map[:, 0] + 0.5, 0, 1)
    
    logger.debug(
        f"[EXIT] generate_skeleton_id_map: shape={id_map.shape}, "
        f"level_joints={len(level_joints)}"
    )
    
    return id_map


if TORCH_AVAILABLE:
    
    class SkeletonConv(nn.Module):
        """
        Skeleton-aware convolution layer.
        
        Unlike standard convolutions, SkeletonConv respects the skeletal
        topology by using the skeleton ID map to weight connections.
        
        As shown in Figure 2:
            skeleton_id_maps → SkeletonConv → γ or β
        
        This enables the network to learn modulation parameters that
        are aware of joint hierarchy and connectivity.
        """
        
        def __init__(
            self,
            in_channels: int,
            out_channels: int,
            num_joints: int,
            kernel_size: int = 3,
        ):
            super().__init__()
            
            self.in_channels = in_channels
            self.out_channels = out_channels
            self.num_joints = num_joints
            
            logger.debug(
                f"[INIT] SkeletonConv: in={in_channels}, out={out_channels}, "
                f"joints={num_joints}, kernel={kernel_size}"
            )
            
            # Temporal convolution
            self.temporal_conv = nn.Conv1d(
                in_channels, out_channels, kernel_size, padding=kernel_size // 2
            )
            
            # Joint-wise mixing weights (learned)
            self.joint_weights = nn.Parameter(
                torch.ones(num_joints, num_joints) / num_joints
            )
            
            # Output projection
            self.proj = nn.Conv1d(out_channels, out_channels, 1)
            
            # Activation
            self.activation = nn.ReLU(inplace=True)
        
        def forward(
            self, 
            x: Tensor, 
            skeleton_id_map: Tensor,
        ) -> Tensor:
            """
            Forward pass with skeleton-aware convolution.
            
            Args:
                x: Input features [B, C, T] or [B*J, C, T]
                skeleton_id_map: Binary skeleton ID map [J, J]
            
            Returns:
                Output features [B, C_out, T]
            """
            # Temporal convolution
            out = self.temporal_conv(x)
            
            # Apply skeleton-aware weighting
            # The ID map modulates which joints influence which outputs
            if skeleton_id_map is not None:
                # Combine learned weights with skeleton structure
                effective_weights = self.joint_weights * skeleton_id_map
                effective_weights = F.softmax(effective_weights, dim=-1)
                
                # Apply as a form of attention over joint dimension
                # This is a simplified version - full impl would reshape properly
                weight_sum = effective_weights.sum()
                if weight_sum > 0:
                    scale = 1.0 + 0.1 * (weight_sum / skeleton_id_map.numel())
                    out = out * scale
            
            # Project and activate
            out = self.proj(out)
            out = self.activation(out)
            
            return out
    
    
    class SPADESkeletonBlock(nn.Module):
        """
        SPADE block using skeleton-aware convolutions.
        
        Implements the architecture from Figure 2:
            skeleton_id_maps → SkeletonConv (γ)
                            → SkeletonConv (β)
            upsampled_motions × γ + β → normalized_upsampled_motions
        """
        
        def __init__(
            self,
            motion_channels: int,
            num_joints: int,
            kernel_size: int = 3,
        ):
            super().__init__()
            
            logger.debug(
                f"[INIT] SPADESkeletonBlock: motion_ch={motion_channels}, "
                f"joints={num_joints}"
            )
            
            self.motion_channels = motion_channels
            self.num_joints = num_joints
            
            # Batch normalization (no affine - SPADE provides γ/β)
            self.bn = nn.BatchNorm1d(motion_channels, affine=False)
            
            # Input projection for skeleton ID map
            # The ID map is [J, J], we project it to [C, T] for conditioning
            self.id_map_proj = nn.Sequential(
                nn.Linear(num_joints, motion_channels),
                nn.ReLU(inplace=True),
            )
            
            # SkeletonConv for γ (scale)
            self.gamma_conv = SkeletonConv(
                motion_channels, motion_channels, num_joints, kernel_size
            )
            
            # SkeletonConv for β (bias)
            self.beta_conv = SkeletonConv(
                motion_channels, motion_channels, num_joints, kernel_size
            )
            
            # Initialize γ to produce ~1, β to produce ~0
            nn.init.xavier_uniform_(self.gamma_conv.temporal_conv.weight)
            nn.init.zeros_(self.beta_conv.temporal_conv.weight)
        
        def forward(
            self,
            x: Tensor,
            skeleton_id_map: Tensor,
            T: int,
        ) -> Tensor:
            """
            Forward pass with skeleton-aware SPADE modulation.
            
            Args:
                x: Input motion features [B, C, T]
                skeleton_id_map: Skeleton ID map [J, J]
                T: Temporal dimension for expansion
            
            Returns:
                Modulated motion features [B, C, T]
            """
            # Normalize input
            normalized = self.bn(x)
            
            # Project skeleton ID map to conditioning signal
            # [J, J] → [J, C] → expand to [B, C, T]
            J = skeleton_id_map.shape[0]
            id_cond = self.id_map_proj(skeleton_id_map)  # [J, C]
            id_cond = id_cond.mean(dim=0, keepdim=True)  # [1, C]
            id_cond = id_cond.unsqueeze(-1).expand(-1, -1, T)  # [1, C, T]
            
            B = x.shape[0]
            id_cond = id_cond.expand(B, -1, -1)  # [B, C, T]
            
            # Compute γ and β using SkeletonConv
            gamma = self.gamma_conv(id_cond, skeleton_id_map)
            beta = self.beta_conv(id_cond, skeleton_id_map)
            
            # Scale gamma to be centered around 1
            gamma = 1.0 + 0.1 * torch.tanh(gamma)
            
            # Apply SPADE modulation: γ × normalized + β
            output = gamma * normalized + beta
            
            return output


    class SPADENormLayer(nn.Module):
        """
        SPADE (Spatially-Adaptive Denormalization) Normalization Layer.
        
        Applies learned γ and β modulation conditioned on style embedding:
            output = γ(style) * BatchNorm(input) + β(style)
        
        This is the core trainable component for style-adaptive blending.
        
        Args:
            norm_channels: Number of channels to normalize
            style_channels: Dimension of style conditioning input
            kernel_size: Convolution kernel size for γ/β projection
        """
        
        def __init__(
            self, 
            norm_channels: int, 
            style_channels: int,
            kernel_size: int = 3,
        ):
            super().__init__()
            
            logger.debug(
                f"[INIT] SPADENormLayer: norm_ch={norm_channels}, "
                f"style_ch={style_channels}, kernel={kernel_size}"
            )
            
            self.norm_channels = norm_channels
            self.style_channels = style_channels
            
            # Batch normalization (no affine - SPADE provides γ/β)
            self.bn = nn.BatchNorm1d(norm_channels, affine=False)
            
            # Shared style encoder
            self.style_encoder = nn.Sequential(
                nn.Conv1d(style_channels, style_channels, kernel_size, padding=kernel_size // 2),
                nn.ReLU(inplace=True),
            )
            
            # Learned γ (scale) projection
            self.gamma_conv = nn.Conv1d(
                style_channels, norm_channels, kernel_size, padding=kernel_size // 2
            )
            
            # Learned β (bias) projection  
            self.beta_conv = nn.Conv1d(
                style_channels, norm_channels, kernel_size, padding=kernel_size // 2
            )
            
            # Initialize γ to 1, β to 0 (identity transform initially)
            nn.init.ones_(self.gamma_conv.weight.data[:, :, kernel_size // 2])
            nn.init.zeros_(self.gamma_conv.bias.data)
            nn.init.zeros_(self.beta_conv.weight.data)
            nn.init.zeros_(self.beta_conv.bias.data)
        
        def forward(self, x: Tensor, style: Tensor) -> Tensor:
            """
            Forward pass with SPADE modulation.
            
            Args:
                x: Input features [B, C, T] (batch, channels, time)
                style: Style conditioning [B, S, T] (batch, style_channels, time)
            
            Returns:
                Modulated features [B, C, T]
            """
            # Normalize input
            normalized = self.bn(x)
            
            # Encode style
            style_encoded = self.style_encoder(style)
            
            # Compute γ and β from style
            gamma = self.gamma_conv(style_encoded)
            beta = self.beta_conv(style_encoded)
            
            # Apply SPADE modulation: γ * normalized + β
            output = gamma * normalized + beta
            
            return output
    
    
    class SPADEBlendBlock(nn.Module):
        """
        SPADE blending block for a single hierarchy level.
        
        Combines motion features with style-conditioned normalization
        for smooth blending at a specific hierarchy level.
        """
        
        def __init__(
            self,
            motion_channels: int,
            style_channels: int,
            apply_spade: bool = True,
        ):
            super().__init__()
            
            self.apply_spade = apply_spade
            self.motion_channels = motion_channels
            
            if apply_spade:
                self.spade_norm = SPADENormLayer(motion_channels, style_channels)
                logger.debug(f"[INIT] SPADEBlendBlock: SPADE enabled, channels={motion_channels}")
            else:
                # Standard batch norm for non-SPADE levels
                self.bn = nn.BatchNorm1d(motion_channels)
                logger.debug(f"[INIT] SPADEBlendBlock: Standard BN, channels={motion_channels}")
            
            # Feature refinement
            self.refine = nn.Sequential(
                nn.Conv1d(motion_channels, motion_channels, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv1d(motion_channels, motion_channels, 3, padding=1),
            )
        
        def forward(
            self, 
            motion_a: Tensor, 
            motion_b: Tensor,
            omega: Tensor,
            style: Optional[Tensor] = None,
        ) -> Tensor:
            """
            Blend two motion sequences.
            
            Args:
                motion_a: First motion [B, C, T]
                motion_b: Second motion [B, C, T]
                omega: Blend weight [B, 1, T] in range [0, 1]
                style: Style conditioning [B, S, T] (required if apply_spade=True)
            
            Returns:
                Blended motion [B, C, T]
            """
            # Base blend: linear interpolation
            blended = (1 - omega) * motion_a + omega * motion_b
            
            # Apply normalization
            if self.apply_spade and style is not None:
                normalized = self.spade_norm(blended, style)
            else:
                normalized = self.bn(blended)
            
            # Refinement with residual connection
            refined = self.refine(normalized)
            output = normalized + 0.1 * refined  # Small residual weight
            
            return output
    
    
    class SPADEHierarchicalBlender(nn.Module):
        """
        Full hierarchical SPADE blender with 4-level architecture.
        
        Based on research showing that applying SPADE conditioning
        only at Level 1 (COARSE) yields optimal FID/coverage trade-off.
        
        Architecture:
            Level 1 (COARSE): SPADE conditioning ← optimal
            Level 2 (MID): Standard temporal conditioning
            Level 3 (FINE): Standard temporal conditioning  
            Level 4 (DETAIL): Standard temporal conditioning
        """
        
        def __init__(self, config: SPADEConfig):
            super().__init__()
            
            logger.info(
                f"[ENTRY] SPADEHierarchicalBlender.__init__: "
                f"spade_level={config.spade_level.value}, "
                f"motion_ch={config.motion_channels}, "
                f"style_ch={config.style_channels}"
            )
            
            self.config = config
            self.spade_level = config.spade_level
            
            # Map hierarchy level enum to integer
            self._level_to_int = {
                HierarchyLevel.COARSE: 1,
                HierarchyLevel.MID: 2,
                HierarchyLevel.FINE: 3,
                HierarchyLevel.DETAIL: 4,
            }
            self._spade_level_int = self._level_to_int[config.spade_level]
            
            # Create blend blocks for each level
            # Only the specified level gets SPADE, others get standard BN
            self.level_blocks = nn.ModuleDict({
                "coarse": SPADEBlendBlock(
                    config.motion_channels,
                    config.style_channels,
                    apply_spade=(config.spade_level == HierarchyLevel.COARSE),
                ),
                "mid": SPADEBlendBlock(
                    config.motion_channels,
                    config.style_channels,
                    apply_spade=(config.spade_level == HierarchyLevel.MID),
                ),
                "fine": SPADEBlendBlock(
                    config.motion_channels,
                    config.style_channels,
                    apply_spade=(config.spade_level == HierarchyLevel.FINE),
                ),
                "detail": SPADEBlendBlock(
                    config.motion_channels,
                    config.style_channels,
                    apply_spade=(config.spade_level == HierarchyLevel.DETAIL),
                ),
            })
            
            # Style encoder: input_dim → style_channels
            self.style_encoder = nn.Sequential(
                nn.Linear(config.input_dim, config.style_channels * 2),
                nn.ReLU(inplace=True),
                nn.Linear(config.style_channels * 2, config.style_channels),
            )
            
            # Motion encoder: raw_motion_dim (D) → motion_channels
            # This projects raw motion xyz (typically 3) to embedding space
            self.raw_motion_dim = 3  # xyz coordinates
            self.motion_encoder = nn.Conv1d(
                self.raw_motion_dim,
                config.motion_channels,
                kernel_size=1,
            )
            
            # Motion decoder: motion_channels → raw_motion_dim
            self.motion_decoder = nn.Conv1d(
                config.motion_channels,
                self.raw_motion_dim,
                kernel_size=1,
            )
            
            # Output projection per level
            self.output_proj = nn.Conv1d(
                config.motion_channels * 4, 
                config.motion_channels,
                1,
            )
            
            # Count trainable parameters
            self._param_count = sum(p.numel() for p in self.parameters() if p.requires_grad)
            
            logger.info(
                f"[EXIT] SPADEHierarchicalBlender.__init__: "
                f"trainable_params={self._param_count:,}"
            )
        
        def encode_style(self, style_embedding: Tensor, seq_len: int) -> Tensor:
            """
            Encode style embedding and expand to sequence length.
            
            Args:
                style_embedding: Style vector [B, D]
                seq_len: Target sequence length
            
            Returns:
                Style features [B, S, T]
            """
            # Encode style
            style_encoded = self.style_encoder(style_embedding)  # [B, S]
            
            # Expand to sequence
            style_seq = style_encoded.unsqueeze(-1).expand(-1, -1, seq_len)  # [B, S, T]
            
            return style_seq
        
        def build_omega(
            self,
            total_frames: int,
            transition_start: int,
            transition_end: int,
            device: torch.device,
        ) -> Tensor:
            """
            Build smooth blend weight omega(t) using sigmoid.
            
            Args:
                total_frames: Total output frames
                transition_start: Frame where transition begins
                transition_end: Frame where transition ends
                device: Target device
            
            Returns:
                Omega tensor [1, 1, T] with values in [0, 1]
            """
            omega = torch.zeros(1, 1, total_frames, device=device)
            
            # Before transition: 0 (motion A)
            # After transition: 1 (motion B)
            # During transition: smooth sigmoid
            
            transition_len = transition_end - transition_start
            if transition_len > 0:
                t = torch.linspace(
                    -self.config.transition_sharpness,
                    self.config.transition_sharpness,
                    transition_len,
                    device=device,
                )
                sigmoid = torch.sigmoid(t)
                omega[0, 0, transition_start:transition_end] = sigmoid
            
            omega[0, 0, transition_end:] = 1.0
            
            return omega
        
        def forward(
            self,
            motion_a: Tensor,
            motion_b: Tensor,
            style_a: Tensor,
            style_b: Tensor,
            omega: Tensor,
            joint_mask: Optional[Dict[str, Tensor]] = None,
        ) -> Tensor:
            """
            Hierarchical SPADE blending.
            
            Args:
                motion_a: First motion [B, C, T, J] (batch, channels, time, joints)
                         where C is raw motion dim (e.g., 3 for xyz)
                motion_b: Second motion [B, C, T, J]
                style_a: Style embedding for motion A [B, D]
                style_b: Style embedding for motion B [B, D]
                omega: Blend weight [B, 1, T]
                joint_mask: Optional dict mapping level names to joint indices
            
            Returns:
                Blended motion [B, C, T, J]
            """
            logger.debug(
                f"[ENTRY] forward: motion_a={motion_a.shape}, "
                f"omega_range=[{omega.min():.2f}, {omega.max():.2f}]"
            )
            
            B, C_raw, T, J = motion_a.shape
            device = motion_a.device
            
            # Encode raw motion to embedding space
            # Reshape [B, C_raw, T, J] → [B*J, C_raw, T] for Conv1d
            ma_flat_raw = motion_a.permute(0, 3, 1, 2).reshape(B * J, C_raw, T)
            mb_flat_raw = motion_b.permute(0, 3, 1, 2).reshape(B * J, C_raw, T)
            
            # Encode to embedding space
            ma_encoded = self.motion_encoder(ma_flat_raw)  # [B*J, motion_channels, T]
            mb_encoded = self.motion_encoder(mb_flat_raw)
            
            # Reshape back to [B, motion_channels, T, J]
            C_embed = self.config.motion_channels
            motion_a_emb = ma_encoded.reshape(B, J, C_embed, T).permute(0, 2, 3, 1)
            motion_b_emb = mb_encoded.reshape(B, J, C_embed, T).permute(0, 2, 3, 1)
            
            # Interpolate style embeddings based on omega
            # Use mean omega for global style
            omega_mean = omega.mean(dim=-1, keepdim=True)  # [B, 1, 1]
            style_blended = (1 - omega_mean.squeeze()) * style_a + omega_mean.squeeze() * style_b
            
            # Encode blended style
            style_seq = self.encode_style(style_blended, T)  # [B, S, T]
            
            # Process each level
            outputs = []
            level_names = ["coarse", "mid", "fine", "detail"]
            
            for level_name in level_names:
                block = self.level_blocks[level_name]
                
                # Get joints for this level (or use all if no mask)
                if joint_mask is not None and level_name in joint_mask:
                    joint_idx = joint_mask[level_name]
                    ma = motion_a_emb[:, :, :, joint_idx]  # [B, C_embed, T, J_level]
                    mb = motion_b_emb[:, :, :, joint_idx]
                else:
                    ma = motion_a_emb
                    mb = motion_b_emb
                
                # Reshape for 1D convolution: [B, C_embed, T, J] → [B*J, C_embed, T]
                _, C_embed_level, T_level, J_level = ma.shape
                ma_flat = ma.permute(0, 3, 1, 2).reshape(B * J_level, C_embed_level, T_level)
                mb_flat = mb.permute(0, 3, 1, 2).reshape(B * J_level, C_embed_level, T_level)
                omega_expanded = omega.expand(B * J_level, -1, -1)
                style_expanded = style_seq.unsqueeze(1).expand(-1, J_level, -1, -1)
                style_flat = style_expanded.reshape(B * J_level, -1, T_level)
                
                # Apply blend block
                blended_flat = block(ma_flat, mb_flat, omega_expanded, style_flat)
                
                # Reshape back: [B*J, C_embed, T] → [B, C_embed, T, J]
                blended = blended_flat.reshape(B, J_level, C_embed_level, T_level).permute(0, 2, 3, 1)
                outputs.append(blended)
            
            # Use coarse level output as primary (research shows Level 1 optimal)
            output_emb = outputs[0]  # [B, C_embed, T, J]
            
            # Decode back to raw motion space
            # Reshape [B, C_embed, T, J] → [B*J, C_embed, T]
            output_flat = output_emb.permute(0, 3, 1, 2).reshape(B * J, C_embed, T)
            output_decoded = self.motion_decoder(output_flat)  # [B*J, C_raw, T]
            
            # Reshape back to [B, C_raw, T, J]
            output = output_decoded.reshape(B, J, C_raw, T).permute(0, 2, 3, 1)
            
            logger.debug(f"[EXIT] forward: output={output.shape}")
            
            return output
        
        @property
        def trainable_params_count(self) -> int:
            """Get count of trainable parameters."""
            return self._param_count
        
        def save_checkpoint(self, path: str) -> None:
            """
            Save model checkpoint.
            
            Args:
                path: File path for checkpoint
            """
            logger.info(f"[ENTRY] save_checkpoint: path={path}")
            
            checkpoint = {
                "model_state_dict": self.state_dict(),
                "config": {
                    "spade_level": self.config.spade_level.value,
                    "input_dim": self.config.input_dim,
                    "style_channels": self.config.style_channels,
                    "motion_channels": self.config.motion_channels,
                    "transition_sharpness": self.config.transition_sharpness,
                },
                "param_count": self._param_count,
            }
            
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(checkpoint, path)
            
            logger.info(f"[EXIT] save_checkpoint: saved {self._param_count:,} params")
        
        def load_checkpoint(self, path: str) -> None:
            """
            Load model checkpoint.
            
            Args:
                path: File path to checkpoint
            """
            logger.info(f"[ENTRY] load_checkpoint: path={path}")
            
            checkpoint = torch.load(path, map_location="cpu")
            self.load_state_dict(checkpoint["model_state_dict"])
            
            logger.info(
                f"[EXIT] load_checkpoint: loaded {checkpoint.get('param_count', 'unknown')} params"
            )


# =============================================================================
# Style Embedding Generation (Hash-based MVP)
# =============================================================================

def generate_hash_embedding(
    style_labels: List[str],
    dim: int = 768,
    seed_prefix: str = "spade-mvp",
) -> np.ndarray:
    """
    Generate deterministic style embedding from labels using hashing.
    
    This is the MVP implementation using hash-based embeddings.
    Phase 2 will integrate real Gemini 768-dim embeddings.
    
    Args:
        style_labels: List of style labels (e.g., ["capoeira", "aggressive"])
        dim: Embedding dimension (default 768 to match Gemini)
        seed_prefix: Prefix for hash seed
    
    Returns:
        Normalized embedding vector [dim]
    """
    logger.debug(f"[ENTRY] generate_hash_embedding: labels={style_labels}, dim={dim}")
    
    # Create deterministic seed from labels
    label_str = ",".join(sorted(style_labels))
    seed_str = f"{seed_prefix}:{label_str}"
    seed = int(hashlib.sha256(seed_str.encode()).hexdigest()[:8], 16)
    
    # Generate embedding
    rng = np.random.default_rng(seed)
    embedding = rng.standard_normal(dim).astype(np.float32)
    
    # L2 normalize
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm
    
    logger.debug(f"[EXIT] generate_hash_embedding: norm={np.linalg.norm(embedding):.4f}")
    
    return embedding


# =============================================================================
# High-Level Blending Service
# =============================================================================

class SPADEBlendService:
    """
    High-level service for SPADE motion blending.
    
    Wraps PyTorch SPADE layers with motion loading, preprocessing,
    and metrics computation for end-to-end blending workflow.
    """
    
    def __init__(self, config: Optional[SPADEConfig] = None):
        """
        Initialize SPADE blend service.
        
        Args:
            config: SPADE configuration (uses defaults if None)
        """
        logger.info(f"[ENTRY] SPADEBlendService.__init__: torch_available={TORCH_AVAILABLE}")
        
        self.config = config or SPADEConfig()
        self._device = "cpu"
        self._model: Optional[Any] = None
        self._checkpoint_loaded = False
        
        if TORCH_AVAILABLE:
            # Check for CUDA
            if torch.cuda.is_available():
                self._device = "cuda"
                logger.info("CUDA available, using GPU")
            
            # Initialize model
            self._model = SPADEHierarchicalBlender(self.config)
            self._model.to(self._device)
            self._model.eval()  # Start in eval mode
            
            logger.info(
                f"[EXIT] SPADEBlendService.__init__: "
                f"device={self._device}, "
                f"params={self._model.trainable_params_count:,}"
            )
        else:
            logger.warning(
                "[EXIT] SPADEBlendService.__init__: "
                "PyTorch not available, using numpy fallback"
            )
    
    @property
    def model(self) -> Optional[Any]:
        """Get underlying PyTorch model."""
        return self._model
    
    @property
    def device(self) -> str:
        """Get device (cpu or cuda)."""
        return self._device
    
    @property
    def trainable_params_count(self) -> int:
        """Get trainable parameter count."""
        if self._model is not None:
            return self._model.trainable_params_count
        return 0
    
    def load_checkpoint(self, path: str) -> bool:
        """
        Load pretrained checkpoint.
        
        Args:
            path: Path to checkpoint file
        
        Returns:
            True if loaded successfully
        """
        logger.info(f"[ENTRY] load_checkpoint: path={path}")
        
        if self._model is None:
            logger.error("[EXIT] load_checkpoint: No model (PyTorch unavailable)")
            return False
        
        try:
            self._model.load_checkpoint(path)
            self._checkpoint_loaded = True
            logger.info("[EXIT] load_checkpoint: Success")
            return True
        except Exception as e:
            logger.error(f"[EXIT] load_checkpoint: Failed - {e}")
            return False
    
    def save_checkpoint(self, path: str) -> bool:
        """
        Save current model checkpoint.
        
        Args:
            path: Path to save checkpoint
        
        Returns:
            True if saved successfully
        """
        logger.info(f"[ENTRY] save_checkpoint: path={path}")
        
        if self._model is None:
            logger.error("[EXIT] save_checkpoint: No model")
            return False
        
        try:
            self._model.save_checkpoint(path)
            logger.info("[EXIT] save_checkpoint: Success")
            return True
        except Exception as e:
            logger.error(f"[EXIT] save_checkpoint: Failed - {e}")
            return False
    
    def blend(
        self,
        motion_a: np.ndarray,
        motion_b: np.ndarray,
        style_labels_a: List[str],
        style_labels_b: List[str],
        weights: List[float],
        transition_frames: int = 30,
        joint_names: Optional[List[str]] = None,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Blend two motions using SPADE conditioning.
        
        Args:
            motion_a: First motion [T_a, J, D]
            motion_b: Second motion [T_b, J, D]
            style_labels_a: Style labels for motion A
            style_labels_b: Style labels for motion B
            weights: Duration weights [w_a, w_b]
            transition_frames: Frames for blend transition
            joint_names: Optional joint names for hierarchy mapping
        
        Returns:
            Tuple of (blended_motion [T_out, J, D], timing_info dict)
        """
        logger.info(
            f"[ENTRY] blend: "
            f"motion_a={motion_a.shape}, motion_b={motion_b.shape}, "
            f"styles_a={style_labels_a}, styles_b={style_labels_b}, "
            f"weights={weights}, transition={transition_frames}"
        )
        
        start_time = time.perf_counter()
        
        # Generate style embeddings (hash-based MVP)
        style_embed_a = generate_hash_embedding(style_labels_a, self.config.input_dim)
        style_embed_b = generate_hash_embedding(style_labels_b, self.config.input_dim)
        
        # Calculate output dimensions
        T_a, J, D = motion_a.shape
        T_b = motion_b.shape[0]
        
        # Duration allocation based on weights
        total_frames = int(T_a * weights[0] + T_b * weights[1])
        transition_start = int(total_frames * weights[0]) - transition_frames // 2
        transition_end = transition_start + transition_frames
        
        # Clamp to valid range
        transition_start = max(0, transition_start)
        transition_end = min(total_frames, transition_end)
        
        logger.debug(
            f"blend: total_frames={total_frames}, "
            f"transition=[{transition_start}, {transition_end}]"
        )
        
        if TORCH_AVAILABLE and self._model is not None:
            blended = self._blend_pytorch(
                motion_a, motion_b,
                style_embed_a, style_embed_b,
                total_frames, transition_start, transition_end,
                joint_names,
            )
        else:
            blended = self._blend_numpy(
                motion_a, motion_b,
                style_embed_a, style_embed_b,
                total_frames, transition_start, transition_end,
            )
        
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        
        timing_info = {
            "blend_time_ms": elapsed_ms,
            "total_frames": total_frames,
            "transition_start": transition_start,
            "transition_end": transition_end,
            "used_pytorch": TORCH_AVAILABLE and self._model is not None,
        }
        
        logger.info(
            f"[EXIT] blend: output={blended.shape}, time={elapsed_ms:.2f}ms"
        )
        
        return blended, timing_info
    
    def _blend_pytorch(
        self,
        motion_a: np.ndarray,
        motion_b: np.ndarray,
        style_a: np.ndarray,
        style_b: np.ndarray,
        total_frames: int,
        transition_start: int,
        transition_end: int,
        joint_names: Optional[List[str]] = None,
    ) -> np.ndarray:
        """PyTorch-based SPADE blending."""
        logger.debug("[ENTRY] _blend_pytorch")
        
        T_a, J, D = motion_a.shape
        T_b = motion_b.shape[0]
        
        # Prepare tensors
        device = torch.device(self._device)
        
        # Pad/interpolate motions to match output length
        # For simplicity, use linear interpolation
        motion_a_interp = self._interpolate_motion(motion_a, total_frames)
        motion_b_interp = self._interpolate_motion(motion_b, total_frames)
        
        # Convert to tensors: [T, J, D] → [1, D, T, J]
        ma = torch.from_numpy(motion_a_interp).float().to(device)
        mb = torch.from_numpy(motion_b_interp).float().to(device)
        # Permute: [T, J, D] → [D, T, J] → [1, D, T, J]
        ma = ma.permute(2, 0, 1).unsqueeze(0)  # [1, D, T, J]
        mb = mb.permute(2, 0, 1).unsqueeze(0)
        
        # Style embeddings
        sa = torch.from_numpy(style_a).float().to(device).unsqueeze(0)  # [1, dim]
        sb = torch.from_numpy(style_b).float().to(device).unsqueeze(0)
        
        # Build omega
        omega = self._model.build_omega(total_frames, transition_start, transition_end, device)
        
        # Forward pass
        with torch.no_grad():
            blended = self._model(ma, mb, sa, sb, omega)
        
        # Convert back: [1, D, T, J] → [T, J, D]
        blended_np = blended.squeeze(0).permute(1, 2, 0).cpu().numpy()
        
        logger.debug(f"[EXIT] _blend_pytorch: shape={blended_np.shape}")
        
        return blended_np
    
    def _blend_numpy(
        self,
        motion_a: np.ndarray,
        motion_b: np.ndarray,
        style_a: np.ndarray,
        style_b: np.ndarray,
        total_frames: int,
        transition_start: int,
        transition_end: int,
    ) -> np.ndarray:
        """
        Numpy fallback for SPADE blending (when PyTorch unavailable).
        
        Uses simplified SPADE-like modulation with smoothstep blending.
        """
        logger.debug("[ENTRY] _blend_numpy (fallback)")
        
        T_a, J, D = motion_a.shape
        T_b = motion_b.shape[0]
        
        # Interpolate motions
        motion_a_interp = self._interpolate_motion(motion_a, total_frames)
        motion_b_interp = self._interpolate_motion(motion_b, total_frames)
        
        # Build omega (sigmoid transition)
        omega = np.zeros(total_frames)
        omega[transition_end:] = 1.0
        
        trans_len = transition_end - transition_start
        if trans_len > 0:
            t = np.linspace(-5, 5, trans_len)
            omega[transition_start:transition_end] = 1 / (1 + np.exp(-t))
        
        # Compute SPADE-like modulation
        # γ and β from style difference
        style_diff = style_b - style_a
        gamma = 1.0 + 0.1 * np.tanh(np.sum(style_diff[:self.config.style_channels]))
        beta = 0.05 * np.tanh(np.sum(style_diff[self.config.style_channels:]))
        
        # Blend with modulation
        blended = np.zeros((total_frames, J, D), dtype=np.float32)
        
        for t in range(total_frames):
            w = omega[t]
            base_blend = (1 - w) * motion_a_interp[t] + w * motion_b_interp[t]
            
            # Apply SPADE-like modulation to coarse joints (first 2)
            for j in range(J):
                if j < 2:  # Coarse level
                    # Normalize
                    mean = np.mean(base_blend[j])
                    std = np.std(base_blend[j]) + 1e-8
                    normalized = (base_blend[j] - mean) / std
                    # Modulate
                    modulated = gamma * normalized + beta
                    # Denormalize
                    blended[t, j] = modulated * std + mean
                else:
                    # Standard smoothstep for other levels
                    w_smooth = w * w * (3 - 2 * w)
                    blended[t, j] = (1 - w_smooth) * motion_a_interp[t, j] + w_smooth * motion_b_interp[t, j]
        
        logger.debug(f"[EXIT] _blend_numpy: shape={blended.shape}")
        
        return blended
    
    def _interpolate_motion(self, motion: np.ndarray, target_frames: int) -> np.ndarray:
        """
        Interpolate motion sequence to target frame count.
        
        Args:
            motion: Input motion [T, J, D]
            target_frames: Target number of frames
        
        Returns:
            Interpolated motion [target_frames, J, D]
        """
        T, J, D = motion.shape
        
        if T == target_frames:
            return motion
        
        # Linear interpolation along time axis
        src_indices = np.linspace(0, T - 1, target_frames)
        output = np.zeros((target_frames, J, D), dtype=motion.dtype)
        
        for i, src_idx in enumerate(src_indices):
            low = int(np.floor(src_idx))
            high = min(low + 1, T - 1)
            alpha = src_idx - low
            output[i] = (1 - alpha) * motion[low] + alpha * motion[high]
        
        return output


# =============================================================================
# Service Singleton
# =============================================================================

_service_instance: Optional[SPADEBlendService] = None


def get_spade_service(config: Optional[SPADEConfig] = None) -> SPADEBlendService:
    """
    Get or create SPADE blend service singleton.
    
    Args:
        config: Optional configuration (uses default if None)
    
    Returns:
        SPADEBlendService instance
    """
    global _service_instance
    
    if _service_instance is None:
        _service_instance = SPADEBlendService(config)
    
    return _service_instance


def reset_spade_service() -> None:
    """Reset the service singleton (for testing)."""
    global _service_instance
    _service_instance = None
