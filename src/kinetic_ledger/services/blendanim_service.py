"""
blendanim Integration Service

Implements motion blending algorithms from the blendanim repository:
https://github.com/RydlrCS/blendanim

Core Metrics:
- Coverage: Motion space coverage evaluation
- LocalDiversity: Short-term variation (15-frame windows)
- GlobalDiversity: Long-term variation (30-frame windows via NN-DP)
- L2_velocity: Velocity smoothness metric
- L2_acceleration: Jerk minimization metric
"""

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class MotionSequence:
    """Represents a motion capture sequence."""
    positions: np.ndarray  # Shape: [T, J, 3] - Time, Joints, XYZ
    velocities: Optional[np.ndarray] = None  # Shape: [T-1, J, 3]
    accelerations: Optional[np.ndarray] = None  # Shape: [T-2, J, 3]
    fps: int = 30
    joint_names: Optional[List[str]] = None
    
    def __post_init__(self):
        """Calculate velocities and accelerations if not provided."""
        if self.velocities is None:
            self.velocities = self.positions[1:] - self.positions[:-1]
        if self.accelerations is None and self.velocities is not None:
            self.accelerations = self.velocities[1:] - self.velocities[:-1]


@dataclass
class BlendMetrics:
    """Motion blend quality metrics aligned with blendanim."""
    coverage: float  # 0-1, higher is better
    local_diversity: float  # 0+, context-dependent
    global_diversity: float  # 0+, higher for diverse motions
    l2_velocity: float  # 0+, lower is smoother
    l2_acceleration: float  # 0+, lower is smoother
    blend_area_smoothness: float  # 0-1, higher is better
    per_joint_metrics: Optional[Dict[str, Dict[str, float]]] = None
    quality_tier: str = "medium"
    
    def to_dict(self) -> dict:
        """Convert to dictionary for API responses."""
        return {
            "coverage": float(self.coverage),
            "local_diversity": float(self.local_diversity),
            "global_diversity": float(self.global_diversity),
            "l2_velocity": float(self.l2_velocity),
            "l2_acceleration": float(self.l2_acceleration),
            "blend_area_smoothness": float(self.blend_area_smoothness),
            "quality_tier": self.quality_tier,
            "per_joint_metrics": self.per_joint_metrics
        }


class BlendAnimService:
    """
    Core blending service implementing blendanim algorithms.
    
    Based on:
    - GANimator: Neural Motion Synthesis from a Single Sequence
    - Single-shot temporal conditioning for motion blending
    """
    
    def __init__(self, use_gpu: bool = True):
        """Initialize blending service."""
        if TORCH_AVAILABLE:
            self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
            logger.info(f"BlendAnimService initialized on device: {self.device}")
        else:
            self.device = "cpu"
            logger.info("BlendAnimService initialized (PyTorch not available, using NumPy only)")
        
        # Metric calculation parameters (from blendanim)
        self.coverage_tmin = 30  # Window size for coverage
        self.coverage_threshold = 2.0  # Cost threshold
        self.local_diversity_tmin = 15  # Window for local diversity
        self.global_diversity_tmin = 30  # Window for global diversity
        
        # Key joints for per-joint analysis (SMPLH skeleton)
        self.key_joints = ["Pelvis", "LeftWrist", "RightWrist", "LeftFoot", "RightFoot"]
    
    def _group_cost_from_tensors(
        self, 
        pred, 
        gt
    ):
        """
        Calculate pairwise frame costs using L2 distance.
        
        From blendanim: src/metrics/generation/ganimator_metrics.py
        
        Args:
            pred: Predicted motion [T1, J, 3] (torch.Tensor or np.ndarray)
            gt: Ground truth motion [T2, J, 3] (torch.Tensor or np.ndarray)
        
        Returns:
            Cost matrix [T1, T2]
        """
        if not TORCH_AVAILABLE:
            # Fallback to numpy implementation
            pred = np.array(pred) if not isinstance(pred, np.ndarray) else pred
            gt = np.array(gt) if not isinstance(gt, np.ndarray) else gt
            
            # Reshape for pairwise distance calculation
            pred_flat = pred.reshape(pred.shape[0], -1)  # [T1, J*3]
            gt_flat = gt.reshape(gt.shape[0], -1)  # [T2, J*3]
            
            # Calculate pairwise L2 distances
            # Broadcasting: [T1, 1, J*3] - [1, T2, J*3] -> [T1, T2, J*3]
            diff = pred_flat[:, np.newaxis, :] - gt_flat[np.newaxis, :, :]
            cost = np.sqrt(np.sum(diff ** 2, axis=2))
            return cost
        
        # Reshape for pairwise distance calculation
        pred_flat = pred.reshape(pred.shape[0], -1)  # [T1, J*3]
        gt_flat = gt.reshape(gt.shape[0], -1)  # [T2, J*3]
        
        # Calculate pairwise L2 distances
        if TORCH_AVAILABLE:
            cost_matrix = torch.cdist(pred_flat, gt_flat, p=2.0)  # [T1, T2]
        else:
            # NumPy fallback already handled above
            diff = pred_flat[:, np.newaxis, :] - gt_flat[np.newaxis, :, :]
            cost_matrix = np.sqrt(np.sum(diff ** 2, axis=2))
        
        return cost_matrix
    
    def _calc_perwindow_cost(
        self, 
        group_cost, 
        tmin: int = 15
    ) -> float:
        """
        Calculate per-window cost for diversity metrics.
        
        From blendanim: LocalDiversity implementation
        
        Args:
            group_cost: Pairwise cost matrix [T, T] (torch.Tensor or np.ndarray)
            tmin: Minimum window size
        
        Returns:
            Mean cost across windows
        """
        if isinstance(group_cost, np.ndarray):
            T = group_cost.shape[0]
        else:
            T = group_cost.shape[0]
        
        if T < tmin:
            return 0.0
        
        costs = []
        num_windows = T - tmin + 1
        if num_windows <= 0:
            return 0.0
        
        for i in range(num_windows):
            # Window goes from i to i+tmin
            start_idx = i
            end_idx = i + tmin
            
            # Strict bounds checking
            if start_idx < 0 or start_idx >= T:
                continue
            if end_idx <= start_idx or end_idx > T:
                continue
            if start_idx >= group_cost.shape[0] or end_idx > group_cost.shape[1]:
                continue
                
            # Extract window
            try:
                window = group_cost[start_idx, start_idx:end_idx]
            except (IndexError, ValueError) as e:
                logger.warning(f"Index error at i={i}, start={start_idx}, end={end_idx}, T={T}, shape={group_cost.shape}: {e}")
                continue
            
            if window.size == 0:
                continue
                
            if TORCH_AVAILABLE and not isinstance(group_cost, np.ndarray):
                window_cost = torch.min(window) / tmin
                costs.append(window_cost.item())
            else:
                window_cost = np.min(window) / tmin
                costs.append(float(window_cost))
        
        return float(np.mean(costs)) if costs else 0.0
    
    def _nn_dp_fast(
        self, 
        group_cost, 
        tmin: int = 30
    ) -> float:
        """
        Nearest-neighbor dynamic programming for global diversity.
        
        From blendanim: GlobalDiversity implementation using NN-DP
        
        Args:
            group_cost: Pairwise cost matrix [T, T] (torch.Tensor or np.ndarray)
            tmin: Minimum sequence length
        
        Returns:
            NN-DP alignment cost
        """
        T = group_cost.shape[0]
        if T < tmin:
            return 0.0
        
        # Ensure group_cost is square
        if group_cost.shape[0] != group_cost.shape[1]:
            logger.warning(f"Non-square cost matrix: {group_cost.shape}, using minimum dimension")
            T = min(group_cost.shape[0], group_cost.shape[1])
        
        # Initialize DP table
        if TORCH_AVAILABLE and not isinstance(group_cost, np.ndarray):
            dp = torch.full((T, T), float('inf'), device=group_cost.device)
        else:
            dp = np.full((T, T), float('inf'))
        
        if T == 0:
            return 0.0
            
        dp[0, 0] = group_cost[0, 0]
        
        # Fill DP table with bounds checking
        for i in range(1, T):
            for j in range(1, T):
                # Bounds check
                if i >= group_cost.shape[0] or j >= group_cost.shape[1]:
                    continue
                    
                # Find minimum cost path
                candidates = [
                    dp[i-1, j-1],  # Diagonal (match)
                    dp[i-1, j],    # Insert in pred
                    dp[i, j-1]     # Insert in gt
                ]
                dp[i, j] = min(candidates) + group_cost[i, j]
        
        # Return normalized alignment cost
        alignment_cost = dp[T-1, T-1] / (2 * T)
        if TORCH_AVAILABLE and not isinstance(alignment_cost, (int, float, np.number)):
            return float(alignment_cost.item())
        return float(alignment_cost)
    
    def calculate_coverage(
        self, 
        motion: MotionSequence,
        reference: Optional[MotionSequence] = None
    ) -> float:
        """
        Calculate motion space coverage.
        
        From blendanim: Coverage metric with tmin=30, threshold=2.0
        
        Args:
            motion: Motion sequence to evaluate
            reference: Optional reference motion (uses self if None)
        
        Returns:
            Coverage score (0-1, higher is better)
        """
        if reference is None:
            reference = motion
        
        if TORCH_AVAILABLE:
            pred_tensor = torch.from_numpy(motion.positions).float().to(self.device)
            gt_tensor = torch.from_numpy(reference.positions).float().to(self.device)
        else:
            pred_tensor = motion.positions
            gt_tensor = reference.positions
        
        # Calculate pairwise costs
        group_cost = self._group_cost_from_tensors(pred_tensor, gt_tensor)
        
        T = group_cost.shape[0]
        if T < self.coverage_tmin:
            return 0.0
        
        # Count windows where min cost < threshold
        covered_windows = 0
        
        if T < self.coverage_tmin:
            return 0.0
        
        # Calculate how many full windows we can fit
        num_windows = T - self.coverage_tmin + 1
        if num_windows <= 0:
            return 0.0
        
        for i in range(num_windows):
            # Window goes from i to i+coverage_tmin
            start_idx = i
            end_idx = i + self.coverage_tmin
            
            # Strict bounds checking - ensure we never exceed array bounds
            if start_idx < 0 or start_idx >= T:
                continue
            if end_idx <= start_idx or end_idx > T:
                continue
            if start_idx >= group_cost.shape[0] or end_idx > group_cost.shape[1]:
                continue
                
            # Extract window from cost matrix
            try:
                window = group_cost[start_idx, start_idx:end_idx]
            except (IndexError, ValueError) as e:
                logger.warning(f"Index error at i={i}, start={start_idx}, end={end_idx}, T={T}, shape={group_cost.shape}: {e}")
                continue
            
            if window.size == 0:
                continue
                
            if TORCH_AVAILABLE and not isinstance(window, np.ndarray):
                min_cost = torch.min(window) / self.coverage_tmin
            else:
                min_cost = np.min(window) / self.coverage_tmin
            if min_cost < self.coverage_threshold:
                covered_windows += 1
        
        coverage = covered_windows / num_windows if num_windows > 0 else 0.0
        return float(coverage)
    
    def calculate_local_diversity(
        self, 
        motion: MotionSequence,
        reference: Optional[MotionSequence] = None
    ) -> float:
        """
        Calculate local diversity (15-frame windows).
        
        From blendanim: LocalDiversity metric
        
        Args:
            motion: Motion sequence to evaluate
            reference: Optional reference motion
        
        Returns:
            Local diversity score (0+, lower for similar motions)
        """
        if reference is None:
            reference = motion
        
        if TORCH_AVAILABLE:
            pred_tensor = torch.from_numpy(motion.positions).float().to(self.device)
            gt_tensor = torch.from_numpy(reference.positions).float().to(self.device)
        else:
            pred_tensor = motion.positions
            gt_tensor = reference.positions
        
        group_cost = self._group_cost_from_tensors(pred_tensor, gt_tensor)
        local_div = self._calc_perwindow_cost(group_cost, tmin=self.local_diversity_tmin)
        
        return float(local_div)
    
    def calculate_global_diversity(
        self, 
        motion: MotionSequence,
        reference: Optional[MotionSequence] = None
    ) -> float:
        """
        Calculate global diversity using NN-DP alignment.
        
        From blendanim: GlobalDiversity metric with tmin=30
        
        Args:
            motion: Motion sequence to evaluate
            reference: Optional reference motion
        
        Returns:
            Global diversity score (0+, higher for diverse motions)
        """
        if reference is None:
            reference = motion
        
        if TORCH_AVAILABLE:
            pred_tensor = torch.from_numpy(motion.positions).float().to(self.device)
            gt_tensor = torch.from_numpy(reference.positions).float().to(self.device)
        else:
            pred_tensor = motion.positions
            gt_tensor = reference.positions
        
        group_cost = self._group_cost_from_tensors(pred_tensor, gt_tensor)
        global_div = self._nn_dp_fast(group_cost, tmin=self.global_diversity_tmin)
        
        return float(global_div)
    
    def calculate_l2_velocity(
        self, 
        motion: MotionSequence,
        joint_indices: Optional[List[int]] = None,
        focus_middle: bool = True
    ) -> float:
        """
        Calculate L2 velocity smoothness metric.
        
        From blendanim: L2_velocity for blend area (middle 30 frames)
        
        Args:
            motion: Motion sequence with velocities
            joint_indices: Specific joints to analyze (None = all)
            focus_middle: Focus on middle 30 frames (blend area)
        
        Returns:
            L2 velocity metric (0+, lower is smoother)
        """
        if motion.velocities is None:
            raise ValueError("Motion sequence must have velocities calculated")
        
        velocities = motion.velocities  # [T-1, J, 3]
        
        # Select joints
        if joint_indices is not None:
            velocities = velocities[:, joint_indices, :]
        
        # Calculate L2 norms per frame
        l2_norms = np.linalg.norm(velocities, axis=-1)  # [T-1, J]
        
        # Calculate delta velocities
        delta_v = np.abs(l2_norms[1:] - l2_norms[:-1])  # [T-2, J]
        
        # Focus on blend area (middle 30 frames)
        if focus_middle and delta_v.shape[0] >= 30:
            middle = delta_v.shape[0] // 2
            delta_v = delta_v[middle-15:middle+15, :]
        
        # Mean across frames and joints
        l2_vel = float(np.mean(delta_v))
        return l2_vel
    
    def calculate_l2_acceleration(
        self, 
        motion: MotionSequence,
        joint_indices: Optional[List[int]] = None,
        focus_middle: bool = True
    ) -> float:
        """
        Calculate L2 acceleration (jerk) metric.
        
        From blendanim: L2_acceleration for blend area smoothness
        
        Args:
            motion: Motion sequence with accelerations
            joint_indices: Specific joints to analyze
            focus_middle: Focus on middle 30 frames
        
        Returns:
            L2 acceleration metric (0+, lower is smoother)
        """
        if motion.accelerations is None:
            raise ValueError("Motion sequence must have accelerations calculated")
        
        accelerations = motion.accelerations  # [T-2, J, 3]
        
        # Select joints
        if joint_indices is not None:
            accelerations = accelerations[:, joint_indices, :]
        
        # Calculate L2 norms per frame
        l2_norms = np.linalg.norm(accelerations, axis=-1)  # [T-2, J]
        
        # Calculate delta accelerations (jerk)
        delta_a = np.abs(l2_norms[1:] - l2_norms[:-1])  # [T-3, J]
        
        # Focus on blend area
        if focus_middle and delta_a.shape[0] >= 30:
            middle = delta_a.shape[0] // 2
            delta_a = delta_a[middle-15:middle+15, :]
        
        # Mean across frames and joints
        l2_acc = float(np.mean(delta_a))
        return l2_acc
    
    def calculate_all_metrics(
        self, 
        motion: MotionSequence,
        reference: Optional[MotionSequence] = None,
        calculate_per_joint: bool = False
    ) -> BlendMetrics:
        """
        Calculate all blendanim metrics for a motion sequence.
        
        Args:
            motion: Motion sequence to evaluate
            reference: Optional reference motion
            calculate_per_joint: Calculate metrics per joint
        
        Returns:
            Complete BlendMetrics object
        """
        # Calculate core metrics
        coverage = self.calculate_coverage(motion, reference)
        local_diversity = self.calculate_local_diversity(motion, reference)
        global_diversity = self.calculate_global_diversity(motion, reference)
        l2_velocity = self.calculate_l2_velocity(motion)
        l2_acceleration = self.calculate_l2_acceleration(motion)
        
        # Calculate smoothness (inverse of velocity + acceleration)
        blend_area_smoothness = 1.0 / (1.0 + l2_velocity + l2_acceleration)
        
        # Determine quality tier
        quality_tier = self._determine_quality_tier(
            coverage, l2_velocity, l2_acceleration, blend_area_smoothness
        )
        
        # Per-joint metrics (optional, expensive)
        per_joint_metrics = None
        if calculate_per_joint and motion.joint_names is not None:
            per_joint_metrics = self._calculate_per_joint_metrics(motion)
        
        return BlendMetrics(
            coverage=coverage,
            local_diversity=local_diversity,
            global_diversity=global_diversity,
            l2_velocity=l2_velocity,
            l2_acceleration=l2_acceleration,
            blend_area_smoothness=blend_area_smoothness,
            per_joint_metrics=per_joint_metrics,
            quality_tier=quality_tier
        )
    
    def _determine_quality_tier(
        self, 
        coverage: float,
        l2_vel: float,
        l2_acc: float,
        smoothness: float
    ) -> str:
        """
        Determine quality tier based on metrics.
        
        From BLEND_METRICS.md quality thresholds
        """
        if (coverage >= 0.90 and l2_vel <= 0.03 and 
            l2_acc <= 0.015 and smoothness >= 0.94):
            return "ultra"
        elif (coverage >= 0.85 and l2_vel <= 0.07 and 
              l2_acc <= 0.04 and smoothness >= 0.86):
            return "high"
        elif (coverage >= 0.75 and l2_vel <= 0.10 and 
              l2_acc <= 0.05 and smoothness >= 0.80):
            return "medium"
        else:
            return "low"
    
    def _calculate_per_joint_metrics(
        self, 
        motion: MotionSequence
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate metrics for individual key joints.
        
        Returns dict mapping joint names to their metrics.
        """
        per_joint = {}
        
        # Find indices of key joints
        if motion.joint_names is None:
            return per_joint
        
        for joint_name in self.key_joints:
            if joint_name in motion.joint_names:
                idx = motion.joint_names.index(joint_name)
                
                l2_vel = self.calculate_l2_velocity(motion, joint_indices=[idx])
                l2_acc = self.calculate_l2_acceleration(motion, joint_indices=[idx])
                
                per_joint[joint_name] = {
                    "l2_velocity": l2_vel,
                    "l2_acceleration": l2_acc
                }
        
        return per_joint
    
    def blend_motions(
        self,
        motions: List[MotionSequence],
        weights: List[float],
        method: str = "linear"
    ) -> Tuple[MotionSequence, BlendMetrics]:
        """
        Blend multiple motion sequences.
        
        Args:
            motions: List of motion sequences to blend
            weights: Blend weights (must sum to 1.0)
            method: Blending method ("linear", "slerp", "temporal_conditioning")
        
        Returns:
            Tuple of (blended_motion, metrics)
        """
        if len(motions) != len(weights):
            raise ValueError("Number of motions must match number of weights")
        
        if not np.isclose(sum(weights), 1.0):
            raise ValueError(f"Weights must sum to 1.0, got {sum(weights)}")
        
        if method == "linear":
            blended = self._linear_blend(motions, weights)
        elif method == "slerp":
            blended = self._slerp_blend(motions, weights)
        elif method == "temporal_conditioning":
            blended = self._temporal_conditioning_blend(motions, weights)
        else:
            raise ValueError(f"Unknown blending method: {method}")
        
        # Calculate metrics for blended motion
        reference = motions[0] if len(motions) > 0 else None
        metrics = self.calculate_all_metrics(blended, reference)
        
        return blended, metrics
    
    def _linear_blend(
        self, 
        motions: List[MotionSequence], 
        weights: List[float]
    ) -> MotionSequence:
        """Simple linear interpolation of motion positions."""
        # Find common length (use shortest)
        min_length = min(m.positions.shape[0] for m in motions)
        
        # Blend positions
        blended_positions = np.zeros((min_length, motions[0].positions.shape[1], 3))
        
        for motion, weight in zip(motions, weights):
            blended_positions += weight * motion.positions[:min_length]
        
        return MotionSequence(
            positions=blended_positions,
            fps=motions[0].fps,
            joint_names=motions[0].joint_names
        )
    
    def _slerp_blend(
        self, 
        motions: List[MotionSequence], 
        weights: List[float]
    ) -> MotionSequence:
        """
        Spherical linear interpolation for rotations.
        Note: Requires quaternion representation (simplified for positions).
        """
        # For position-based data, fall back to linear blend
        # Full SLERP would require quaternion joint rotations
        return self._linear_blend(motions, weights)
    
    def _temporal_conditioning_blend(
        self, 
        motions: List[MotionSequence], 
        weights: List[float]
    ) -> MotionSequence:
        """
        Temporal conditioning blend using single-shot approach.
        
        From blendanim: Single-shot temporal conditioning
        This is a simplified version - full implementation requires
        the trained generator model.
        """
        # For now, use linear blend with temporal smoothing
        linear_blend = self._linear_blend(motions, weights)
        
        # Apply temporal smoothing to blend area (middle 30 frames)
        T = linear_blend.positions.shape[0]
        if T >= 60:
            middle_start = T // 2 - 15
            middle_end = T // 2 + 15
            
            # Gaussian smoothing kernel
            kernel_size = 5
            kernel = np.exp(-np.linspace(-2, 2, kernel_size)**2)
            kernel = kernel / kernel.sum()
            
            # Apply to middle section
            for j in range(linear_blend.positions.shape[1]):  # Each joint
                for dim in range(3):  # X, Y, Z
                    linear_blend.positions[middle_start:middle_end, j, dim] = np.convolve(
                        linear_blend.positions[middle_start:middle_end, j, dim],
                        kernel,
                        mode='same'
                    )
        
        return linear_blend
    
    def generate_transition_artifacts(
        self,
        motions: List[MotionSequence],
        weights: List[float],
        crosshatch_offsets: Optional[List[float]] = None,
        transition_frames: int = 30,
        blend_mode: str = "smoothstep"
    ) -> Tuple[List[Dict], BlendMetrics]:
        """
        Generate complete blended sequence with motion segments and transitions.
        
        Creates full sequence: Motion1 → Transition1 → Motion2 → Transition2 → Motion3
        For N motions, generates (N-1) transitions of 30 frames each.
        
        Args:
            motions: List of 2-3 motion sequences to blend
            weights: Blend weights determining duration of each motion
            crosshatch_offsets: Optional frame offsets (0-1) for each motion
            transition_frames: Number of frames per transition (default: 30)
            blend_mode: Temporal blend function ("smoothstep" recommended)
        
        Returns:
            Tuple of (artifact_list, aggregate_metrics)
            - artifact_list: List of dicts with frame data for FULL sequence
            - aggregate_metrics: Overall blend quality metrics
        """
        if len(motions) < 2:
            raise ValueError("Need at least 2 motions for transition generation")
        
        if crosshatch_offsets is None:
            crosshatch_offsets = [0.0] * len(motions)
        
        # Normalize weights
        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]
        
        artifacts = []  # Full sequence artifacts
        all_blended_frames = []  # For calculating aggregate metrics
        frame_counter = 0
        actual_motion_segments = []  # Track actual segment metadata
        
        # Calculate total blend duration based on weights
        # Formula: For each motion, contribution = weight * motion_duration
        # Total duration = sum of all weighted contributions
        total_weighted_duration = 0
        for i, motion in enumerate(motions):
            motion_duration = motion.positions.shape[0] / motion.fps
            total_weighted_duration += normalized_weights[i] * motion_duration
        
        logger.info(f"Total blend duration: {total_weighted_duration:.2f}s across {len(motions)} motions")
        
        # Process each motion pair
        for i in range(len(motions) - 1):
            motion_a = motions[i]
            motion_b = motions[i + 1]
            offset_a = crosshatch_offsets[i]
            offset_b = crosshatch_offsets[i + 1]
            
            # Frame calculation formula:
            # 1. frames_to_use = total_frames * normalized_weight
            # 2. max_start = total_frames - frames_to_use (available sliding window)
            # 3. start_frame = max_start * crosshatch_offset (0.0 = earliest, 1.0 = latest)
            # 4. end_frame = start_frame + frames_to_use
            
            frames_a_total = motion_a.positions.shape[0]
            frames_b_total = motion_b.positions.shape[0]
            
            # Apply weight to determine contribution
            frames_a_to_use = max(1, int(frames_a_total * normalized_weights[i]))
            frames_b_to_use = max(1, int(frames_b_total * normalized_weights[i + 1]))
            
            # Crosshatch offset slides the selection window
            max_start_a = max(0, frames_a_total - frames_a_to_use)
            max_start_b = max(0, frames_b_total - frames_b_to_use)
            
            start_frame_a = int(max_start_a * offset_a)
            start_frame_b = int(max_start_b * offset_b)
            
            end_frame_a = min(start_frame_a + frames_a_to_use, frames_a_total)
            end_frame_b = min(start_frame_b + frames_b_to_use, frames_b_total)
            
            logger.info(f"Motion {i+1}: frames [{start_frame_a}:{end_frame_a}] ({end_frame_a-start_frame_a} frames, weight={normalized_weights[i]:.2f}, offset={offset_a:.2f})")
            logger.info(f"Motion {i+2}: frames [{start_frame_b}:{end_frame_b}] ({end_frame_b-start_frame_b} frames, weight={normalized_weights[i+1]:.2f}, offset={offset_b:.2f})")
            
            # Extract motion segments
            segment_a = MotionSequence(
                positions=motion_a.positions[start_frame_a:end_frame_a],
                fps=motion_a.fps,
                joint_names=motion_a.joint_names
            )
            
            segment_b = MotionSequence(
                positions=motion_b.positions[start_frame_b:end_frame_b],
                fps=motion_b.fps,
                joint_names=motion_b.joint_names
            )
            
            # Add motion A segment frames as artifacts
            segment_start_frame = frame_counter
            for j, frame_pos in enumerate(segment_a.positions):
                artifacts.append({
                    "frame_index": frame_counter,
                    "omega": 0.0,  # Pure motion A
                    "positions": frame_pos,
                    "blend_mode": "motion_segment",
                    "t_normalized": 0.0,
                    "segment_type": f"motion_{i+1}",
                    "segment_frame": j
                })
                all_blended_frames.append(frame_pos)
                frame_counter += 1
            
            # Track actual segment metadata
            actual_motion_segments.append({
                "motion_index": i,
                "weight": float(normalized_weights[i]),
                "crosshatch_offset": float(offset_a),
                "start_frame": start_frame_a,
                "end_frame": end_frame_a,
                "frame_count": frame_counter - segment_start_frame,
                "artifact_start_index": segment_start_frame,
                "artifact_end_index": frame_counter - 1
            })
            
            # Generate and add transition frames
            transition_artifacts = self._generate_transition_frames(
                segment_a,
                segment_b,
                transition_frames,
                blend_mode
            )
            
            # Update frame indices and add to main artifacts list
            for artifact in transition_artifacts:
                artifact["frame_index"] = frame_counter
                all_blended_frames.append(artifact["positions"])
                frame_counter += 1
            
            artifacts.extend(transition_artifacts)
        
        # Add final motion segment using same formula
        final_motion = motions[-1]
        final_offset = crosshatch_offsets[-1]
        frames_final_total = final_motion.positions.shape[0]
        
        # Apply formula: frames = total * weight, with bounds checking
        frames_final_to_use = max(1, int(frames_final_total * normalized_weights[-1]))
        
        max_start_final = max(0, frames_final_total - frames_final_to_use)
        start_frame_final = int(max_start_final * final_offset)
        end_frame_final = min(start_frame_final + frames_final_to_use, frames_final_total)
        
        logger.info(f"Motion {len(motions)}: frames [{start_frame_final}:{end_frame_final}] ({end_frame_final-start_frame_final} frames, weight={normalized_weights[-1]:.2f}, offset={final_offset:.2f})")
        
        final_segment = final_motion.positions[start_frame_final:end_frame_final]
        all_blended_frames.extend(final_segment)
        
        # Add final motion segment frames as artifacts
        segment_start_frame = frame_counter
        for j, frame_pos in enumerate(final_segment):
            artifacts.append({
                "frame_index": frame_counter,
                "omega": 1.0,  # Pure motion B/C
                "positions": frame_pos,
                "blend_mode": "motion_segment",
                "t_normalized": 1.0,
                "segment_type": f"motion_{len(motions)}",
                "segment_frame": j
            })
            frame_counter += 1
        
        # Track final segment metadata
        actual_motion_segments.append({
            "motion_index": len(motions) - 1,
            "weight": float(normalized_weights[-1]),
            "crosshatch_offset": float(final_offset),
            "start_frame": start_frame_final,
            "end_frame": end_frame_final,
            "frame_count": frame_counter - segment_start_frame,
            "artifact_start_index": segment_start_frame,
            "artifact_end_index": frame_counter - 1
        })
        
        # Verification: Validate frame count against weights and crosshatch positions
        expected_total_frames = self._verify_frame_allocation(
            motions=motions,
            normalized_weights=normalized_weights,
            crosshatch_offsets=crosshatch_offsets,
            transition_frames=transition_frames,
            actual_frame_count=frame_counter
        )
        
        logger.info(f"Frame verification: Expected {expected_total_frames}, Actual {frame_counter}, Match: {expected_total_frames == frame_counter}")
        
        if expected_total_frames != frame_counter:
            logger.warning(f"Frame count mismatch! Expected {expected_total_frames} but got {frame_counter}")
        
        # Calculate aggregate metrics for entire blended sequence
        full_sequence = MotionSequence(
            positions=np.array(all_blended_frames),
            fps=motions[0].fps,
            joint_names=motions[0].joint_names
        )
        
        aggregate_metrics = self.calculate_all_metrics(full_sequence, reference=motions[0])
        
        logger.info(f"Generated {len(artifacts)} total frames with {len(motions)-1} transitions")
        logger.info(f"  Motion segment breakdown: {[seg['frame_count'] for seg in actual_motion_segments]}")
        
        return artifacts, aggregate_metrics, actual_motion_segments
    
    def _verify_frame_allocation(
        self,
        motions: List[MotionSequence],
        normalized_weights: List[float],
        crosshatch_offsets: List[float],
        transition_frames: int,
        actual_frame_count: int
    ) -> int:
        """
        Verify frame allocation based on blend weights and crosshatch positions.
        
        This validation ensures:
        1. Each motion contributes frames proportional to its weight
        2. Crosshatch offset correctly selects frame window
        3. Transition frames are correctly inserted between motions
        4. Total frame count matches expected calculation
        
        Args:
            motions: List of motion sequences
            normalized_weights: Normalized blend weights
            crosshatch_offsets: Frame selection offsets (0-1)
            transition_frames: Frames per transition
            actual_frame_count: Actual number of frames generated
        
        Returns:
            Expected total frame count based on formula
        """
        expected_motion_frames = 0
        
        # Calculate expected frames from each motion
        for i, motion in enumerate(motions):
            frames_total = motion.positions.shape[0]
            weight = normalized_weights[i]
            offset = crosshatch_offsets[i]
            
            # Apply formula
            frames_to_use = max(1, int(frames_total * weight))
            max_start = max(0, frames_total - frames_to_use)
            start_frame = int(max_start * offset)
            end_frame = min(start_frame + frames_to_use, frames_total)
            
            actual_frames = end_frame - start_frame
            expected_motion_frames += actual_frames
            
            logger.debug(f"  Motion {i+1}: {frames_total} total × {weight:.3f} weight = {frames_to_use} expected, "
                        f"offset {offset:.3f} → [{start_frame}:{end_frame}] = {actual_frames} actual frames")
        
        # Calculate expected transition frames
        num_transitions = len(motions) - 1
        expected_transition_frames = num_transitions * transition_frames
        
        # Total expected
        expected_total = expected_motion_frames + expected_transition_frames
        
        logger.info(f"Frame allocation verification:")
        logger.info(f"  Motion frames: {expected_motion_frames} (across {len(motions)} motions)")
        logger.info(f"  Transition frames: {expected_transition_frames} ({num_transitions} × {transition_frames})")
        logger.info(f"  Expected total: {expected_total}")
        logger.info(f"  Actual total: {actual_frame_count}")
        logger.info(f"  Difference: {actual_frame_count - expected_total}")
        
        return expected_total
    
    def _generate_transition_frames(
        self,
        motion_a: MotionSequence,
        motion_b: MotionSequence,
        transition_frames: int,
        blend_mode: str
    ) -> List[Dict]:
        """
        Generate smooth transition frames between two motions using temporal conditioning.
        
        Args:
            motion_a: Starting motion (blend from this)
            motion_b: Ending motion (blend to this)
            transition_frames: Number of intermediate frames
            blend_mode: "smoothstep" (C²), "smootherstep" (C³), "linear", or "step"
        
        Returns:
            List of artifact dicts with positions, omega, and per-frame metrics
        """
        artifacts = []
        
        # Get last frame of motion A and first frame of motion B
        frame_a = motion_a.positions[-1]  # Shape: [J, 3]
        frame_b = motion_b.positions[0]   # Shape: [J, 3]
        
        # Generate each transition frame
        for t in range(transition_frames):
            # Calculate normalized time parameter (0 to 1)
            t_normalized = t / (transition_frames - 1) if transition_frames > 1 else 0.5
            
            # Calculate blend weight omega(t) using smoothstep
            omega = self._calculate_omega(t_normalized, blend_mode)
            
            # Blend positions: pos = (1 - omega) * A + omega * B
            blended_positions = (1 - omega) * frame_a + omega * frame_b
            
            # Create motion sequence for this single frame (for metrics)
            frame_sequence = MotionSequence(
                positions=np.expand_dims(blended_positions, axis=0),  # Add time dimension
                fps=motion_a.fps,
                joint_names=motion_a.joint_names
            )
            
            # Store artifact
            artifact = {
                "frame_index": t,
                "omega": float(omega),
                "positions": blended_positions,  # Shape: [J, 3]
                "blend_mode": blend_mode,
                "t_normalized": float(t_normalized)
            }
            
            artifacts.append(artifact)
        
        return artifacts
    
    def _calculate_omega(self, t: float, mode: str) -> float:
        """
        Calculate temporal blend weight omega(t) for time parameter t in [0, 1].
        
        From BlendAnim paper (arXiv:2508.18525) - smoothstep provides C² continuity.
        
        Args:
            t: Normalized time parameter (0 to 1)
            mode: Blend mode
        
        Returns:
            Blend weight (0 = fully motion A, 1 = fully motion B)
        """
        t = np.clip(t, 0.0, 1.0)
        
        if mode == "step":
            # C⁰ - Hard boundary (Heaviside)
            return 1.0 if t >= 0.5 else 0.0
        
        elif mode == "linear":
            # C¹ - Constant blend rate
            return t
        
        elif mode == "smoothstep":
            # C² - Smooth position, velocity, AND acceleration (RECOMMENDED)
            # omega(t) = t² (3 - 2t) = 3t² - 2t³
            return t * t * (3.0 - 2.0 * t)
        
        elif mode == "smootherstep":
            # C³ - Extra smooth
            # omega(t) = t³ (t(6t - 15) + 10) = 6t⁵ - 15t⁴ + 10t³
            return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
        
        else:
            # Default to smoothstep
            return t * t * (3.0 - 2.0 * t)


# Singleton instance
_blendanim_service: Optional[BlendAnimService] = None


def get_blendanim_service() -> BlendAnimService:
    """Get or create singleton blendanim service."""
    global _blendanim_service
    if _blendanim_service is None:
        _blendanim_service = BlendAnimService()
    return _blendanim_service
