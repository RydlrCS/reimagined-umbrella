"""
Motion Blending Quality Metrics Module.

Implements evaluation metrics for SPADE motion blending:
- FID (Fréchet Inception Distance) for distribution similarity
- Coverage for reference distribution coverage
- Diversity for generation variety
- Smoothness for motion quality

References:
- FID: "GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium"
- Motion Metrics: "Action-Conditioned 3D Human Motion Synthesis with Transformer VAE"
"""
import time
import logging
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass

import numpy as np

# Conditional scipy import for matrix operations
try:
    import scipy.linalg
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

from ..utils.logging import get_logger

# Setup logging
logger = get_logger(__name__)


# =============================================================================
# FID (Fréchet Inception Distance)
# =============================================================================

def compute_fid(
    generated_features: np.ndarray,
    reference_features: np.ndarray,
    eps: float = 1e-6,
) -> float:
    """
    Compute Fréchet Inception Distance between two feature distributions.
    
    FID measures the similarity between two distributions by comparing
    their means and covariances in feature space. Lower FID = more similar.
    
    Formula:
        FID = ||μ_g - μ_r||² + Tr(Σ_g + Σ_r - 2(Σ_g Σ_r)^{1/2})
    
    Args:
        generated_features: Generated motion features [N_gen, D]
        reference_features: Reference motion features [N_ref, D]
        eps: Small constant for numerical stability
    
    Returns:
        FID score (lower is better, 0 = identical distributions)
    """
    logger.info(
        f"[ENTRY] compute_fid: "
        f"generated={generated_features.shape}, reference={reference_features.shape}"
    )
    
    start_time = time.perf_counter()
    
    if not SCIPY_AVAILABLE:
        logger.warning("scipy not available, using simplified FID")
        return _compute_fid_simple(generated_features, reference_features)
    
    # Compute statistics
    mu_gen = np.mean(generated_features, axis=0)
    mu_ref = np.mean(reference_features, axis=0)
    
    # Covariance matrices
    sigma_gen = np.cov(generated_features, rowvar=False)
    sigma_ref = np.cov(reference_features, rowvar=False)
    
    # Handle single sample case
    if generated_features.shape[0] == 1:
        sigma_gen = np.zeros_like(sigma_gen)
    if reference_features.shape[0] == 1:
        sigma_ref = np.zeros_like(sigma_ref)
    
    # Ensure positive semi-definite
    sigma_gen = sigma_gen + eps * np.eye(sigma_gen.shape[0])
    sigma_ref = sigma_ref + eps * np.eye(sigma_ref.shape[0])
    
    # Mean difference term
    diff = mu_gen - mu_ref
    mean_term = np.sum(diff ** 2)
    
    # Covariance term: Tr(Σ_g + Σ_r - 2 * sqrt(Σ_g @ Σ_r))
    try:
        # Compute matrix square root of product
        covmean = scipy.linalg.sqrtm(sigma_gen @ sigma_ref)
        
        # Handle numerical issues (imaginary components)
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        
        cov_term = np.trace(sigma_gen + sigma_ref - 2 * covmean)
        
    except Exception as e:
        logger.warning(f"Matrix sqrt failed, using trace approximation: {e}")
        # Fallback: just use trace difference
        cov_term = np.abs(np.trace(sigma_gen) - np.trace(sigma_ref))
    
    fid = float(mean_term + cov_term)
    
    # Ensure non-negative (numerical issues can cause small negatives)
    fid = max(0.0, fid)
    
    elapsed_ms = (time.perf_counter() - start_time) * 1000
    
    logger.info(f"[EXIT] compute_fid: score={fid:.4f}, time={elapsed_ms:.2f}ms")
    
    return fid


def _compute_fid_simple(
    generated_features: np.ndarray,
    reference_features: np.ndarray,
) -> float:
    """Simplified FID using only mean difference (no scipy)."""
    mu_gen = np.mean(generated_features, axis=0)
    mu_ref = np.mean(reference_features, axis=0)
    
    # Just use squared Euclidean distance of means
    return float(np.sum((mu_gen - mu_ref) ** 2))


# =============================================================================
# Coverage Metric
# =============================================================================

def compute_coverage(
    generated_features: np.ndarray,
    reference_features: np.ndarray,
    threshold: Optional[float] = None,
) -> float:
    """
    Compute coverage of reference distribution by generated samples.
    
    Coverage measures what fraction of reference samples have at least
    one "close" generated sample. Higher coverage = better diversity.
    
    Args:
        generated_features: Generated motion features [N_gen, D]
        reference_features: Reference motion features [N_ref, D]
        threshold: Distance threshold for "close" (auto-computed if None)
    
    Returns:
        Coverage score in [0, 1] (higher is better)
    """
    logger.info(
        f"[ENTRY] compute_coverage: "
        f"generated={generated_features.shape}, reference={reference_features.shape}"
    )
    
    start_time = time.perf_counter()
    
    N_gen = generated_features.shape[0]
    N_ref = reference_features.shape[0]
    
    if N_gen == 0 or N_ref == 0:
        logger.warning("[EXIT] compute_coverage: Empty input, returning 0")
        return 0.0
    
    # Auto-compute threshold based on reference distribution
    if threshold is None:
        # Use median pairwise distance in reference set
        if N_ref > 1:
            ref_dists = _pairwise_distances(reference_features, reference_features)
            # Exclude self-distances (diagonal)
            mask = ~np.eye(N_ref, dtype=bool)
            threshold = np.median(ref_dists[mask])
        else:
            threshold = 1.0
    
    # Compute distances from each reference to nearest generated
    covered = 0
    
    for i in range(N_ref):
        ref_sample = reference_features[i]
        
        # Distance to all generated samples
        dists = np.linalg.norm(generated_features - ref_sample, axis=1)
        min_dist = np.min(dists)
        
        if min_dist <= threshold:
            covered += 1
    
    coverage = covered / N_ref
    
    elapsed_ms = (time.perf_counter() - start_time) * 1000
    
    logger.info(
        f"[EXIT] compute_coverage: "
        f"score={coverage:.4f}, threshold={threshold:.4f}, time={elapsed_ms:.2f}ms"
    )
    
    return coverage


def _pairwise_distances(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Compute pairwise L2 distances between X and Y."""
    # ||x - y||² = ||x||² + ||y||² - 2<x, y>
    X_sqnorm = np.sum(X ** 2, axis=1, keepdims=True)
    Y_sqnorm = np.sum(Y ** 2, axis=1, keepdims=True)
    
    dists_sq = X_sqnorm + Y_sqnorm.T - 2 * X @ Y.T
    dists_sq = np.maximum(dists_sq, 0)  # Numerical stability
    
    return np.sqrt(dists_sq)


# =============================================================================
# Diversity Metric
# =============================================================================

def compute_diversity(
    generated_features: np.ndarray,
    num_samples: int = 100,
) -> float:
    """
    Compute diversity of generated motion samples.
    
    Diversity measures the average pairwise distance between generated
    samples. Higher diversity = more variety in generations.
    
    Args:
        generated_features: Generated motion features [N, D]
        num_samples: Max samples to use (for efficiency)
    
    Returns:
        Diversity score (higher is better)
    """
    logger.info(
        f"[ENTRY] compute_diversity: features={generated_features.shape}"
    )
    
    start_time = time.perf_counter()
    
    N = generated_features.shape[0]
    
    if N < 2:
        logger.warning("[EXIT] compute_diversity: Need >=2 samples, returning 0")
        return 0.0
    
    # Subsample if too many
    if N > num_samples:
        indices = np.random.choice(N, num_samples, replace=False)
        features = generated_features[indices]
    else:
        features = generated_features
    
    # Compute average pairwise distance
    N_sub = features.shape[0]
    total_dist = 0.0
    count = 0
    
    for i in range(N_sub):
        for j in range(i + 1, N_sub):
            dist = np.linalg.norm(features[i] - features[j])
            total_dist += dist
            count += 1
    
    diversity = total_dist / max(count, 1)
    
    elapsed_ms = (time.perf_counter() - start_time) * 1000
    
    logger.info(
        f"[EXIT] compute_diversity: "
        f"score={diversity:.4f}, pairs={count}, time={elapsed_ms:.2f}ms"
    )
    
    return diversity


# =============================================================================
# Motion Smoothness Metric
# =============================================================================

def compute_smoothness(
    motion_sequence: np.ndarray,
    fps: float = 30.0,
) -> float:
    """
    Compute motion smoothness based on acceleration.
    
    Smoothness is inversely related to acceleration magnitude.
    A perfectly smooth motion has constant velocity (zero acceleration).
    
    Args:
        motion_sequence: Motion tensor [T, J, D]
        fps: Frames per second
    
    Returns:
        Smoothness score in [0, 1] (higher is smoother)
    """
    logger.info(
        f"[ENTRY] compute_smoothness: motion={motion_sequence.shape}, fps={fps}"
    )
    
    start_time = time.perf_counter()
    
    T = motion_sequence.shape[0]
    
    if T < 3:
        logger.warning("[EXIT] compute_smoothness: Need >=3 frames, returning 1.0")
        return 1.0
    
    dt = 1.0 / fps
    
    # Compute velocity (first derivative)
    velocity = np.diff(motion_sequence, axis=0) / dt  # [T-1, J, D]
    
    # Compute acceleration (second derivative)
    acceleration = np.diff(velocity, axis=0) / dt  # [T-2, J, D]
    
    # Compute acceleration magnitude per frame
    accel_magnitude = np.linalg.norm(acceleration, axis=(1, 2))  # [T-2]
    
    # Mean acceleration
    mean_accel = np.mean(accel_magnitude)
    
    # Convert to smoothness score [0, 1]
    # Use exponential decay: smoothness = exp(-k * accel)
    # k chosen so that "normal" motion has smoothness ~0.8
    k = 0.01
    smoothness = float(np.exp(-k * mean_accel))
    
    elapsed_ms = (time.perf_counter() - start_time) * 1000
    
    logger.info(
        f"[EXIT] compute_smoothness: "
        f"score={smoothness:.4f}, mean_accel={mean_accel:.2f}, time={elapsed_ms:.2f}ms"
    )
    
    return smoothness


# =============================================================================
# Foot Sliding Metric
# =============================================================================

def compute_foot_sliding(
    motion_sequence: np.ndarray,
    foot_joint_indices: Optional[List[int]] = None,
    contact_threshold: float = 0.05,
) -> float:
    """
    Compute foot sliding distance during ground contact.
    
    Foot sliding is a common artifact in motion blending where
    feet move horizontally while supposedly in contact with ground.
    
    Args:
        motion_sequence: Motion tensor [T, J, D]
        foot_joint_indices: Indices of foot joints (default: [16, 17, 20, 21])
        contact_threshold: Height threshold for ground contact detection
    
    Returns:
        Total foot sliding distance in motion units (lower is better)
    """
    logger.info(
        f"[ENTRY] compute_foot_sliding: motion={motion_sequence.shape}"
    )
    
    start_time = time.perf_counter()
    
    # Default foot joints for Mixamo skeleton
    if foot_joint_indices is None:
        foot_joint_indices = [16, 17, 20, 21]  # LeftFoot, LeftToeBase, RightFoot, RightToeBase
    
    T, J, D = motion_sequence.shape
    
    if T < 2:
        logger.warning("[EXIT] compute_foot_sliding: Need >=2 frames, returning 0")
        return 0.0
    
    total_sliding = 0.0
    
    for foot_idx in foot_joint_indices:
        if foot_idx >= J:
            continue
        
        foot_positions = motion_sequence[:, foot_idx, :]  # [T, D]
        
        for t in range(1, T):
            # Check if foot is in contact (low height)
            # Assume Y is up, position at index 1
            height_idx = 1 if D >= 2 else 0
            current_height = foot_positions[t, height_idx]
            prev_height = foot_positions[t - 1, height_idx]
            
            # Both frames in contact
            if current_height < contact_threshold and prev_height < contact_threshold:
                # Compute horizontal displacement
                if D >= 3:
                    # XZ plane displacement
                    dx = foot_positions[t, 0] - foot_positions[t - 1, 0]
                    dz = foot_positions[t, 2] - foot_positions[t - 1, 2]
                    sliding = np.sqrt(dx ** 2 + dz ** 2)
                else:
                    # 2D: just X displacement
                    sliding = abs(foot_positions[t, 0] - foot_positions[t - 1, 0])
                
                total_sliding += sliding
    
    elapsed_ms = (time.perf_counter() - start_time) * 1000
    
    logger.info(
        f"[EXIT] compute_foot_sliding: "
        f"distance={total_sliding:.4f}, time={elapsed_ms:.2f}ms"
    )
    
    return total_sliding


# =============================================================================
# Transition Quality Metric
# =============================================================================

def compute_transition_quality(
    motion_sequence: np.ndarray,
    transition_start: int,
    transition_end: int,
) -> float:
    """
    Compute quality of blend transition region.
    
    Measures smoothness and continuity specifically in the transition
    window where two motions are blended together.
    
    Args:
        motion_sequence: Blended motion [T, J, D]
        transition_start: Start frame of transition
        transition_end: End frame of transition
    
    Returns:
        Transition quality score in [0, 1] (higher is better)
    """
    logger.info(
        f"[ENTRY] compute_transition_quality: "
        f"motion={motion_sequence.shape}, transition=[{transition_start}, {transition_end}]"
    )
    
    start_time = time.perf_counter()
    
    T = motion_sequence.shape[0]
    
    # Validate range
    transition_start = max(0, transition_start)
    transition_end = min(T, transition_end)
    
    if transition_end <= transition_start + 2:
        logger.warning("[EXIT] compute_transition_quality: Transition too short")
        return 1.0
    
    # Extract transition region
    transition_region = motion_sequence[transition_start:transition_end]
    
    # Compute smoothness in transition region
    transition_smoothness = compute_smoothness(transition_region)
    
    # Compute velocity continuity at boundaries
    if transition_start >= 1 and transition_end < T - 1:
        # Velocity before transition
        vel_before = motion_sequence[transition_start] - motion_sequence[transition_start - 1]
        vel_after = motion_sequence[transition_end] - motion_sequence[transition_end - 1]
        
        # Velocity at transition boundaries
        vel_start = motion_sequence[transition_start + 1] - motion_sequence[transition_start]
        vel_end = motion_sequence[transition_end - 1] - motion_sequence[transition_end - 2]
        
        # Continuity score based on velocity alignment
        cos_start = np.sum(vel_before * vel_start) / (
            np.linalg.norm(vel_before) * np.linalg.norm(vel_start) + 1e-8
        )
        cos_end = np.sum(vel_after * vel_end) / (
            np.linalg.norm(vel_after) * np.linalg.norm(vel_end) + 1e-8
        )
        
        continuity = (cos_start + cos_end + 2) / 4  # Map [-1, 1] to [0, 1]
    else:
        continuity = 0.5
    
    # Combined quality score
    quality = 0.7 * transition_smoothness + 0.3 * continuity
    
    elapsed_ms = (time.perf_counter() - start_time) * 1000
    
    logger.info(
        f"[EXIT] compute_transition_quality: "
        f"score={quality:.4f}, smoothness={transition_smoothness:.4f}, "
        f"continuity={continuity:.4f}, time={elapsed_ms:.2f}ms"
    )
    
    return quality


# =============================================================================
# L2 Velocity and Acceleration per Joint
# =============================================================================

# Standard Mixamo joint names for visualization
VISUALIZATION_JOINTS = {
    'Pelvis': 0,      # Hips (root)
    'LeftWrist': 13,  # Left hand
    'RightWrist': 17, # Right hand
    'LeftFoot': 7,    # Left foot (LeftToeBase parent)
    'RightFoot': 11,  # Right foot (RightToeBase parent)
}


def compute_l2_velocity_per_joint(
    motion: np.ndarray,
    joint_indices: Optional[Dict[str, int]] = None,
    fps: float = 30.0,
) -> Dict[str, np.ndarray]:
    """
    Compute L2 velocity magnitude per frame for selected joints.
    
    Based on GANimator evaluation: tracks key joints (Pelvis, Wrists, Feet)
    to evaluate motion dynamics and blend quality.
    
    Args:
        motion: Motion tensor [T, J, 3] (positions)
        joint_indices: Dict mapping joint names to indices. 
                      Defaults to VISUALIZATION_JOINTS.
        fps: Frames per second for velocity scaling
    
    Returns:
        Dict mapping joint names to L2 velocity arrays [T-1]
    """
    logger.info(
        f"[ENTRY] compute_l2_velocity_per_joint: motion={motion.shape}, fps={fps}"
    )
    
    start_time = time.perf_counter()
    
    if joint_indices is None:
        joint_indices = VISUALIZATION_JOINTS
    
    T, J, D = motion.shape
    
    # Compute velocity: v[t] = (p[t+1] - p[t]) * fps
    velocity = np.diff(motion, axis=0) * fps  # [T-1, J, 3]
    
    # Compute L2 magnitude per joint per frame
    # ||v||_2 = sqrt(vx^2 + vy^2 + vz^2)
    l2_velocity = np.linalg.norm(velocity, axis=-1)  # [T-1, J]
    
    # Extract per-joint velocities
    result = {}
    for joint_name, joint_idx in joint_indices.items():
        if joint_idx < J:
            result[joint_name] = l2_velocity[:, joint_idx].copy()
        else:
            logger.warning(
                f"[WARNING] compute_l2_velocity_per_joint: "
                f"Joint {joint_name} index {joint_idx} >= num_joints {J}"
            )
            result[joint_name] = np.zeros(T - 1)
    
    elapsed_ms = (time.perf_counter() - start_time) * 1000
    
    logger.info(
        f"[EXIT] compute_l2_velocity_per_joint: "
        f"joints={list(result.keys())}, frames={T-1}, time={elapsed_ms:.2f}ms"
    )
    
    return result


def compute_l2_acceleration_per_joint(
    motion: np.ndarray,
    joint_indices: Optional[Dict[str, int]] = None,
    fps: float = 30.0,
) -> Dict[str, np.ndarray]:
    """
    Compute L2 acceleration magnitude per frame for selected joints.
    
    Acceleration reveals motion smoothness and identifies jerky transitions.
    High acceleration spikes often indicate poor blend quality.
    
    Args:
        motion: Motion tensor [T, J, 3] (positions)
        joint_indices: Dict mapping joint names to indices.
                      Defaults to VISUALIZATION_JOINTS.
        fps: Frames per second for acceleration scaling
    
    Returns:
        Dict mapping joint names to L2 acceleration arrays [T-2]
    """
    logger.info(
        f"[ENTRY] compute_l2_acceleration_per_joint: motion={motion.shape}, fps={fps}"
    )
    
    start_time = time.perf_counter()
    
    if joint_indices is None:
        joint_indices = VISUALIZATION_JOINTS
    
    T, J, D = motion.shape
    
    # Compute velocity: v[t] = (p[t+1] - p[t]) * fps
    velocity = np.diff(motion, axis=0) * fps  # [T-1, J, 3]
    
    # Compute acceleration: a[t] = (v[t+1] - v[t]) * fps
    acceleration = np.diff(velocity, axis=0) * fps  # [T-2, J, 3]
    
    # Compute L2 magnitude per joint per frame
    l2_accel = np.linalg.norm(acceleration, axis=-1)  # [T-2, J]
    
    # Extract per-joint accelerations
    result = {}
    for joint_name, joint_idx in joint_indices.items():
        if joint_idx < J:
            result[joint_name] = l2_accel[:, joint_idx].copy()
        else:
            logger.warning(
                f"[WARNING] compute_l2_acceleration_per_joint: "
                f"Joint {joint_name} index {joint_idx} >= num_joints {J}"
            )
            result[joint_name] = np.zeros(max(0, T - 2))
    
    elapsed_ms = (time.perf_counter() - start_time) * 1000
    
    logger.info(
        f"[EXIT] compute_l2_acceleration_per_joint: "
        f"joints={list(result.keys())}, frames={T-2}, time={elapsed_ms:.2f}ms"
    )
    
    return result


def compute_joint_dynamics(
    motion: np.ndarray,
    transition_start: Optional[int] = None,
    transition_end: Optional[int] = None,
    joint_indices: Optional[Dict[str, int]] = None,
    fps: float = 30.0,
) -> Dict[str, any]:
    """
    Compute comprehensive joint dynamics for visualization.
    
    Returns L2 velocity and acceleration for selected joints,
    along with transition boundary markers for chart rendering.
    
    Args:
        motion: Motion tensor [T, J, 3]
        transition_start: Frame index where transition begins (for vertical line)
        transition_end: Frame index where transition ends (for vertical line)
        joint_indices: Dict mapping joint names to indices
        fps: Frames per second
    
    Returns:
        Dict with 'velocity', 'acceleration', 'transition_bounds', 'frames'
    """
    logger.info(
        f"[ENTRY] compute_joint_dynamics: motion={motion.shape}, "
        f"transition=[{transition_start}, {transition_end}], fps={fps}"
    )
    
    start_time = time.perf_counter()
    
    T = motion.shape[0]
    
    # Compute per-joint velocity and acceleration
    velocity_data = compute_l2_velocity_per_joint(motion, joint_indices, fps)
    accel_data = compute_l2_acceleration_per_joint(motion, joint_indices, fps)
    
    # Prepare frame indices for x-axis
    velocity_frames = list(range(T - 1))
    accel_frames = list(range(T - 2))
    
    # Convert numpy arrays to lists for JSON serialization
    velocity_dict = {k: v.tolist() for k, v in velocity_data.items()}
    accel_dict = {k: v.tolist() for k, v in accel_data.items()}
    
    result = {
        'velocity': velocity_dict,
        'acceleration': accel_dict,
        'velocity_frames': velocity_frames,
        'acceleration_frames': accel_frames,
        'transition_bounds': {
            'start': transition_start,
            'end': transition_end,
        },
        'total_frames': T,
        'fps': fps,
        'joint_names': list(velocity_data.keys()),
    }
    
    elapsed_ms = (time.perf_counter() - start_time) * 1000
    
    logger.info(
        f"[EXIT] compute_joint_dynamics: "
        f"velocity_frames={len(velocity_frames)}, "
        f"accel_frames={len(accel_frames)}, "
        f"time={elapsed_ms:.2f}ms"
    )
    
    return result


# =============================================================================
# Full Metrics Computation
# =============================================================================

@dataclass
class MotionMetricsResult:
    """Container for all motion quality metrics."""
    
    fid_score: float
    coverage: float
    diversity: float
    smoothness: float
    foot_sliding: float
    transition_quality: float
    computation_time_ms: float
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "fid_score": self.fid_score,
            "coverage": self.coverage,
            "diversity": self.diversity,
            "smoothness": self.smoothness,
            "foot_sliding": self.foot_sliding,
            "transition_quality": self.transition_quality,
            "computation_time_ms": self.computation_time_ms,
        }


def compute_all_metrics(
    generated_motion: np.ndarray,
    reference_motions: Optional[np.ndarray] = None,
    transition_start: int = 0,
    transition_end: Optional[int] = None,
    fps: float = 30.0,
) -> MotionMetricsResult:
    """
    Compute all motion quality metrics.
    
    Args:
        generated_motion: Generated/blended motion [T, J, D]
        reference_motions: Optional reference motions [N, T, J, D] for FID/coverage
        transition_start: Start of blend transition
        transition_end: End of blend transition (default: T)
        fps: Frames per second
    
    Returns:
        MotionMetricsResult with all metrics
    """
    logger.info(
        f"[ENTRY] compute_all_metrics: "
        f"generated={generated_motion.shape}, "
        f"has_reference={reference_motions is not None}"
    )
    
    start_time = time.perf_counter()
    
    T = generated_motion.shape[0]
    if transition_end is None:
        transition_end = T
    
    # Extract compact features for FID/coverage (downsample to avoid memory issues)
    # Instead of full T*J*D, use per-frame joint statistics [T, J*D] then temporal downsample
    MAX_FEATURE_DIM = 2048  # Reasonable size for covariance computation
    T_gen, J_gen, D_gen = generated_motion.shape
    
    # Downsample temporally if needed  
    temporal_stride = max(1, T_gen // 64)  # Max ~64 frames for features
    gen_downsampled = generated_motion[::temporal_stride]  # [T', J, D]
    gen_features = gen_downsampled.reshape(1, -1)  # [1, T'*J*D]
    
    # Further reduce if still too large
    if gen_features.shape[1] > MAX_FEATURE_DIM:
        # Use PCA-like dimensionality reduction via random projection
        np.random.seed(42)  # Reproducible
        proj_matrix = np.random.randn(gen_features.shape[1], MAX_FEATURE_DIM) / np.sqrt(MAX_FEATURE_DIM)
        gen_features = gen_features @ proj_matrix  # [1, MAX_FEATURE_DIM]
    
    # Compute individual metrics
    smoothness = compute_smoothness(generated_motion, fps)
    foot_sliding = compute_foot_sliding(generated_motion)
    transition_quality = compute_transition_quality(
        generated_motion, transition_start, transition_end
    )
    
    # FID and coverage (if reference provided)
    if reference_motions is not None and len(reference_motions) > 0:
        # Extract features from reference with same downsampling
        N_ref = reference_motions.shape[0]
        T_ref = reference_motions.shape[1] if reference_motions.ndim == 4 else reference_motions.shape[0]
        
        if reference_motions.ndim == 4:  # [N, T, J, D]
            ref_stride = max(1, T_ref // 64)
            ref_downsampled = reference_motions[:, ::ref_stride]  # [N, T', J, D]
            ref_features = ref_downsampled.reshape(N_ref, -1)  # [N, T'*J*D]
        else:  # [T, J, D] - single reference
            ref_stride = max(1, T_ref // 64)
            ref_downsampled = reference_motions[::ref_stride]
            ref_features = ref_downsampled.reshape(1, -1)  # [1, T'*J*D]
            N_ref = 1
        
        # Apply same projection if needed
        if ref_features.shape[1] > MAX_FEATURE_DIM:
            np.random.seed(42)  # Same projection
            proj_matrix = np.random.randn(ref_features.shape[1], MAX_FEATURE_DIM) / np.sqrt(MAX_FEATURE_DIM)
            ref_features = ref_features @ proj_matrix
        
        fid_score = compute_fid(gen_features, ref_features)
        coverage = compute_coverage(gen_features, ref_features)
        diversity = compute_diversity(ref_features)
    else:
        # No reference, use defaults
        fid_score = 0.0
        coverage = 1.0
        diversity = compute_diversity(gen_features)
    
    elapsed_ms = (time.perf_counter() - start_time) * 1000
    
    result = MotionMetricsResult(
        fid_score=fid_score,
        coverage=coverage,
        diversity=diversity,
        smoothness=smoothness,
        foot_sliding=foot_sliding,
        transition_quality=transition_quality,
        computation_time_ms=elapsed_ms,
    )
    
    logger.info(
        f"[EXIT] compute_all_metrics: "
        f"fid={fid_score:.4f}, coverage={coverage:.4f}, "
        f"smoothness={smoothness:.4f}, time={elapsed_ms:.2f}ms"
    )
    
    return result


# =============================================================================
# Reference Feature Extraction
# =============================================================================

def extract_motion_features(
    motion: np.ndarray,
    feature_type: str = "flatten",
) -> np.ndarray:
    """
    Extract features from motion sequence for metric computation.
    
    Args:
        motion: Motion tensor [T, J, D]
        feature_type: Type of features to extract
            - "flatten": Just flatten the motion
            - "velocity": Use velocity statistics
            - "combined": Combine position and velocity
    
    Returns:
        Feature vector [D_features]
    """
    logger.debug(f"[ENTRY] extract_motion_features: motion={motion.shape}, type={feature_type}")
    
    if feature_type == "flatten":
        features = motion.flatten()
        
    elif feature_type == "velocity":
        # Compute velocity
        velocity = np.diff(motion, axis=0)
        
        # Statistics
        vel_mean = np.mean(velocity, axis=0).flatten()
        vel_std = np.std(velocity, axis=0).flatten()
        
        features = np.concatenate([vel_mean, vel_std])
        
    elif feature_type == "combined":
        # Position statistics
        pos_mean = np.mean(motion, axis=0).flatten()
        pos_std = np.std(motion, axis=0).flatten()
        
        # Velocity statistics
        velocity = np.diff(motion, axis=0)
        vel_mean = np.mean(velocity, axis=0).flatten()
        vel_std = np.std(velocity, axis=0).flatten()
        
        features = np.concatenate([pos_mean, pos_std, vel_mean, vel_std])
        
    else:
        raise ValueError(f"Unknown feature_type: {feature_type}")
    
    logger.debug(f"[EXIT] extract_motion_features: features={features.shape}")
    
    return features


def build_reference_features(
    motions: List[np.ndarray],
    feature_type: str = "combined",
) -> np.ndarray:
    """
    Build reference feature matrix from list of motions.
    
    Args:
        motions: List of motion tensors, each [T_i, J, D]
        feature_type: Feature extraction type
    
    Returns:
        Reference features [N, D_features]
    """
    logger.info(f"[ENTRY] build_reference_features: n_motions={len(motions)}")
    
    features = []
    
    for motion in motions:
        feat = extract_motion_features(motion, feature_type)
        features.append(feat)
    
    # Stack with padding if needed
    max_len = max(f.shape[0] for f in features)
    padded = []
    
    for f in features:
        if f.shape[0] < max_len:
            f = np.pad(f, (0, max_len - f.shape[0]))
        padded.append(f)
    
    reference = np.stack(padded, axis=0)
    
    logger.info(f"[EXIT] build_reference_features: shape={reference.shape}")
    
    return reference
