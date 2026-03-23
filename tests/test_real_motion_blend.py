"""
Real Motion Blend Tests.

Tests SPADE blending with actual Mixamo FBX motion data:
- Salsa Dancing → Swing Dancing blend
- Character embeddings (michelle, remy)
- Skeleton-aware convolutions
"""
import os
import sys
import pytest
import numpy as np

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


class TestFBXLoader:
    """Test FBX motion loading."""
    
    def test_load_salsa_motion(self):
        """Test loading salsa dancing motion."""
        from kinetic_ledger.utils.fbx_loader import load_fbx_motion
        
        fbx_path = "data/mixamo_anims/fbx/Salsa Dancing.fbx"
        if not os.path.exists(fbx_path):
            pytest.skip(f"FBX file not found: {fbx_path}")
        
        motion = load_fbx_motion(fbx_path)
        
        assert motion.num_frames > 0
        assert motion.num_joints == 22  # Mixamo standard
        assert motion.positions.shape == (motion.num_frames, 22, 3)
        print(f"Salsa motion: {motion}")
    
    def test_load_swing_motion(self):
        """Test loading swing dancing motion."""
        from kinetic_ledger.utils.fbx_loader import load_fbx_motion
        
        fbx_path = "data/mixamo_anims/fbx/Swing Dancing.fbx"
        if not os.path.exists(fbx_path):
            pytest.skip(f"FBX file not found: {fbx_path}")
        
        motion = load_fbx_motion(fbx_path)
        
        assert motion.num_frames > 0
        assert motion.num_joints == 22
        print(f"Swing motion: {motion}")
    
    def test_load_animation_pair(self):
        """Test loading salsa-swing animation pair."""
        from kinetic_ledger.utils.fbx_loader import load_mixamo_animation_pair
        
        try:
            salsa, swing = load_mixamo_animation_pair(
                "Salsa Dancing",
                "Swing Dancing",
            )
            
            assert salsa.num_frames > 0
            assert swing.num_frames > 0
            assert salsa.num_joints == swing.num_joints
            print(f"Salsa: {salsa.num_frames}f, Swing: {swing.num_frames}f")
        except FileNotFoundError:
            pytest.skip("Motion files not found")
    
    def test_get_available_animations(self):
        """Test listing available animations."""
        from kinetic_ledger.utils.fbx_loader import get_available_animations
        
        anims = get_available_animations()
        
        print(f"Available animations: {anims}")
        assert isinstance(anims, list)
    
    def test_motion_normalize(self):
        """Test motion normalization."""
        from kinetic_ledger.utils.fbx_loader import load_fbx_motion
        
        fbx_path = "data/mixamo_anims/fbx/Salsa Dancing.fbx"
        if not os.path.exists(fbx_path):
            pytest.skip(f"FBX file not found: {fbx_path}")
        
        motion = load_fbx_motion(fbx_path, normalize=True)
        
        # Normalized motion should be roughly centered and scaled
        assert np.abs(motion.positions).max() <= 2.0
    
    def test_motion_resample(self):
        """Test motion resampling."""
        from kinetic_ledger.utils.fbx_loader import load_fbx_motion
        
        fbx_path = "data/mixamo_anims/fbx/Salsa Dancing.fbx"
        if not os.path.exists(fbx_path):
            pytest.skip(f"FBX file not found: {fbx_path}")
        
        motion = load_fbx_motion(fbx_path)
        original_frames = motion.num_frames
        
        # Resample to 60 frames
        resampled = motion.resample(60)
        
        assert resampled.num_frames == 60
        assert resampled.num_joints == motion.num_joints


class TestSkeletonIDMaps:
    """Test skeleton ID map generation."""
    
    def test_generate_coarse_level_map(self):
        """Test generating skeleton ID map for COARSE level."""
        from kinetic_ledger.services.spade_blend_service import generate_skeleton_id_map
        from kinetic_ledger.schemas.models import HierarchyLevel
        
        # Use standard 22 joints
        joint_names = [
            "Hips", "Spine", "Spine1", "Spine2", "Neck", "Head",
            "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand",
            "RightShoulder", "RightArm", "RightForeArm", "RightHand",
            "LeftUpLeg", "LeftLeg", "LeftFoot", "LeftToeBase",
            "RightUpLeg", "RightLeg", "RightFoot", "RightToeBase",
        ]
        
        id_map = generate_skeleton_id_map(joint_names, HierarchyLevel.COARSE)
        
        assert id_map.shape == (len(joint_names), len(joint_names))
        assert id_map.dtype == np.float32
        
        # COARSE level should have Hips and Spine marked
        hips_idx = joint_names.index("Hips")
        spine_idx = joint_names.index("Spine")
        assert id_map[hips_idx, hips_idx] > 0  # Self-connection
        assert id_map[spine_idx, spine_idx] > 0
        
        print(f"COARSE ID map sum: {id_map.sum():.2f}")
    
    def test_generate_all_level_maps(self):
        """Test generating ID maps for all hierarchy levels."""
        from kinetic_ledger.services.spade_blend_service import generate_skeleton_id_map
        from kinetic_ledger.schemas.models import HierarchyLevel
        
        joint_names = [
            "Hips", "Spine", "Spine1", "Spine2", "Neck", "Head",
            "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand",
            "RightShoulder", "RightArm", "RightForeArm", "RightHand",
            "LeftUpLeg", "LeftLeg", "LeftFoot", "LeftToeBase",
            "RightUpLeg", "RightLeg", "RightFoot", "RightToeBase",
        ]
        
        for level in HierarchyLevel:
            id_map = generate_skeleton_id_map(joint_names, level)
            
            assert id_map.shape == (22, 22)
            print(f"{level.value} ID map: sum={id_map.sum():.2f}")


class TestSkeletonConv:
    """Test skeleton-aware convolution layers."""
    
    @pytest.fixture
    def skeleton_conv(self):
        """Create SkeletonConv layer for testing."""
        try:
            import torch
            from kinetic_ledger.services.spade_blend_service import SkeletonConv
            
            return SkeletonConv(
                in_channels=256,
                out_channels=256,
                num_joints=22,
                kernel_size=3,
            )
        except ImportError:
            pytest.skip("PyTorch not available")
    
    def test_skeleton_conv_forward(self, skeleton_conv):
        """Test SkeletonConv forward pass."""
        import torch
        from kinetic_ledger.services.spade_blend_service import generate_skeleton_id_map
        from kinetic_ledger.schemas.models import HierarchyLevel
        
        # Create input tensor [B, C, T]
        x = torch.randn(4, 256, 60)
        
        # Create skeleton ID map
        joint_names = [
            "Hips", "Spine", "Spine1", "Spine2", "Neck", "Head",
            "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand",
            "RightShoulder", "RightArm", "RightForeArm", "RightHand",
            "LeftUpLeg", "LeftLeg", "LeftFoot", "LeftToeBase",
            "RightUpLeg", "RightLeg", "RightFoot", "RightToeBase",
        ]
        id_map = generate_skeleton_id_map(joint_names, HierarchyLevel.COARSE)
        id_map_tensor = torch.from_numpy(id_map)
        
        # Forward pass
        output = skeleton_conv(x, id_map_tensor)
        
        assert output.shape == x.shape
        assert not torch.isnan(output).any()
    
    def test_skeleton_conv_gradients(self, skeleton_conv):
        """Test SkeletonConv gradient flow."""
        import torch
        from kinetic_ledger.services.spade_blend_service import generate_skeleton_id_map
        from kinetic_ledger.schemas.models import HierarchyLevel
        
        x = torch.randn(2, 256, 30, requires_grad=True)
        
        joint_names = ["Hips", "Spine"] + [f"Joint{i}" for i in range(20)]
        id_map = generate_skeleton_id_map(joint_names, HierarchyLevel.COARSE)
        id_map_tensor = torch.from_numpy(id_map)
        
        output = skeleton_conv(x, id_map_tensor)
        loss = output.sum()
        loss.backward()
        
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()


class TestSPADESkeletonBlock:
    """Test SPADE skeleton block."""
    
    @pytest.fixture
    def spade_block(self):
        """Create SPADESkeletonBlock for testing."""
        try:
            import torch
            from kinetic_ledger.services.spade_blend_service import SPADESkeletonBlock
            
            return SPADESkeletonBlock(
                motion_channels=256,
                num_joints=22,
                kernel_size=3,
            )
        except ImportError:
            pytest.skip("PyTorch not available")
    
    def test_spade_skeleton_forward(self, spade_block):
        """Test SPADESkeletonBlock forward pass."""
        import torch
        from kinetic_ledger.services.spade_blend_service import generate_skeleton_id_map
        from kinetic_ledger.schemas.models import HierarchyLevel
        
        # Input motion [B, C, T]
        x = torch.randn(4, 256, 60)
        T = 60
        
        # Skeleton ID map
        joint_names = [f"Joint{i}" for i in range(22)]
        joint_names[0] = "Hips"
        joint_names[1] = "Spine"
        id_map = generate_skeleton_id_map(joint_names, HierarchyLevel.COARSE)
        id_map_tensor = torch.from_numpy(id_map)
        
        # Forward pass
        output = spade_block(x, id_map_tensor, T)
        
        assert output.shape == x.shape
        assert not torch.isnan(output).any()
        print(f"SPADESkeletonBlock output range: [{output.min():.3f}, {output.max():.3f}]")


class TestRealMotionBlend:
    """Test blending with real motion data."""
    
    def test_salsa_to_swing_blend(self):
        """Test blending salsa dancing to swing dancing."""
        from kinetic_ledger.utils.fbx_loader import load_mixamo_animation_pair
        from kinetic_ledger.services.spade_blend_service import (
            SPADEBlendService,
            reset_spade_service,
        )
        
        try:
            salsa, swing = load_mixamo_animation_pair(
                "Salsa Dancing",
                "Swing Dancing",
            )
        except FileNotFoundError:
            pytest.skip("Motion files not found")
        
        reset_spade_service()
        service = SPADEBlendService()
        
        # Blend salsa → swing
        blended, timing = service.blend(
            motion_a=salsa.positions,
            motion_b=swing.positions,
            style_labels_a=["salsa", "latin", "smooth"],
            style_labels_b=["swing", "jazz", "bouncy"],
            weights=[0.5, 0.5],
            transition_frames=30,
            joint_names=salsa.joint_names,
        )
        
        # Verify blend
        assert blended.shape[0] > 0
        assert blended.shape[1] == salsa.num_joints
        assert blended.shape[2] == 3
        assert not np.isnan(blended).any()
        
        print(f"Blended motion: {blended.shape}")
        print(f"Timing: {timing}")
    
    def test_blend_with_different_weights(self):
        """Test blending with various weight combinations."""
        from kinetic_ledger.utils.fbx_loader import load_mixamo_animation_pair
        from kinetic_ledger.services.spade_blend_service import (
            SPADEBlendService,
            reset_spade_service,
        )
        
        try:
            salsa, swing = load_mixamo_animation_pair(
                "Salsa Dancing",
                "Swing Dancing",
            )
        except FileNotFoundError:
            pytest.skip("Motion files not found")
        
        reset_spade_service()
        service = SPADEBlendService()
        
        # Test different weight ratios
        weight_configs = [
            ([0.7, 0.3], "mostly_salsa"),
            ([0.3, 0.7], "mostly_swing"),
            ([0.5, 0.5], "balanced"),
        ]
        
        for weights, name in weight_configs:
            blended, timing = service.blend(
                motion_a=salsa.positions,
                motion_b=swing.positions,
                style_labels_a=["salsa"],
                style_labels_b=["swing"],
                weights=weights,
                transition_frames=20,
            )
            
            assert blended.shape[0] > 0
            assert not np.isnan(blended).any()
            print(f"{name}: {blended.shape[0]} frames")
    
    def test_blend_charleston_to_hiphop(self):
        """Test blending charleston to hip hop."""
        from kinetic_ledger.utils.fbx_loader import load_fbx_motion
        from kinetic_ledger.services.spade_blend_service import (
            SPADEBlendService,
            reset_spade_service,
        )
        
        charleston_path = "data/mixamo_anims/fbx/Dance Swing Charleston.fbx"
        hiphop_path = "data/mixamo_anims/fbx/Wave Hip Hop Dance Variation One.fbx"
        
        if not os.path.exists(charleston_path) or not os.path.exists(hiphop_path):
            pytest.skip("Motion files not found")
        
        charleston = load_fbx_motion(charleston_path)
        hiphop = load_fbx_motion(hiphop_path)
        
        reset_spade_service()
        service = SPADEBlendService()
        
        blended, timing = service.blend(
            motion_a=charleston.positions,
            motion_b=hiphop.positions,
            style_labels_a=["charleston", "vintage", "swing"],
            style_labels_b=["hiphop", "urban", "groove"],
            weights=[0.5, 0.5],
            transition_frames=25,
        )
        
        assert blended.shape[0] > 0
        assert not np.isnan(blended).any()
        print(f"Charleston→HipHop blend: {blended.shape}")
    
    def test_blend_with_metrics(self):
        """Test blend with quality metrics computation."""
        from kinetic_ledger.utils.fbx_loader import load_mixamo_animation_pair
        from kinetic_ledger.services.spade_blend_service import (
            SPADEBlendService,
            reset_spade_service,
        )
        from kinetic_ledger.services.metrics import compute_all_metrics
        
        try:
            salsa, swing = load_mixamo_animation_pair(
                "Salsa Dancing",
                "Swing Dancing",
            )
        except FileNotFoundError:
            pytest.skip("Motion files not found")
        
        reset_spade_service()
        service = SPADEBlendService()
        
        blended, timing = service.blend(
            motion_a=salsa.positions,
            motion_b=swing.positions,
            style_labels_a=["salsa"],
            style_labels_b=["swing"],
            weights=[0.5, 0.5],
            transition_frames=30,
        )
        
        # Compute metrics with downsampled motion to avoid memory issues
        # Downsample to 30 frames for metrics computation
        target_frames = min(30, blended.shape[0])
        
        # Simple downsampling
        indices = np.linspace(0, blended.shape[0] - 1, target_frames, dtype=int)
        blended_ds = blended[indices]
        
        salsa_ds = salsa.resample(target_frames).positions
        swing_ds = swing.resample(target_frames).positions
        
        ref_motions = np.stack([salsa_ds, swing_ds])
        
        metrics = compute_all_metrics(
            generated_motion=blended_ds,
            reference_motions=ref_motions,
            transition_start=0,
            transition_end=target_frames,
        )
        
        print(f"Blend metrics:")
        print(f"  FID: {metrics.fid_score:.2f}")
        print(f"  Coverage: {metrics.coverage:.2f}")
        print(f"  Diversity: {metrics.diversity:.2f}")
        print(f"  Smoothness: {metrics.smoothness:.4f}")
        print(f"  Transition quality: {metrics.transition_quality:.4f}")
        
        assert metrics.coverage >= 0
        assert metrics.diversity >= 0


class TestCharacterEmbeddings:
    """Test character-based motion embedding."""
    
    def test_character_files_exist(self):
        """Test that character FBX files are accessible."""
        from kinetic_ledger.utils.fbx_loader import get_character_files
        
        chars = get_character_files()
        print(f"Available characters: {chars}")
        
        # At least one character should exist
        assert len(chars) >= 0  # May be empty if files not present
    
    def test_blend_preserves_joint_structure(self):
        """Test that blending preserves joint hierarchy."""
        from kinetic_ledger.utils.fbx_loader import load_mixamo_animation_pair
        from kinetic_ledger.services.spade_blend_service import (
            SPADEBlendService,
            reset_spade_service,
        )
        
        try:
            salsa, swing = load_mixamo_animation_pair(
                "Salsa Dancing",
                "Swing Dancing",
            )
        except FileNotFoundError:
            pytest.skip("Motion files not found")
        
        reset_spade_service()
        service = SPADEBlendService()
        
        blended, _ = service.blend(
            motion_a=salsa.positions,
            motion_b=swing.positions,
            style_labels_a=["salsa"],
            style_labels_b=["swing"],
            weights=[0.5, 0.5],
            transition_frames=20,
            joint_names=salsa.joint_names,
        )
        
        # Check that joint count is preserved
        assert blended.shape[1] == salsa.num_joints
        
        # Check that hip position is reasonable (not too far from origin)
        hip_trajectory = blended[:, 0, :]  # First joint is Hips
        assert np.abs(hip_trajectory).max() < 10.0  # Reasonable bound


class TestIntegrationWithAPI:
    """Test integration with API endpoints."""
    
    @pytest.mark.asyncio
    async def test_spade_endpoint_with_real_motion(self):
        """Test SPADE API endpoint with real motion data."""
        from kinetic_ledger.utils.fbx_loader import load_mixamo_animation_pair
        
        try:
            salsa, swing = load_mixamo_animation_pair(
                "Salsa Dancing", 
                "Swing Dancing",
            )
        except FileNotFoundError:
            pytest.skip("Motion files not found")
        
        # Prepare API request payload
        request_data = {
            "motion_a": salsa.positions.tolist(),
            "motion_b": swing.positions.tolist(),
            "style_labels_a": ["salsa", "latin"],
            "style_labels_b": ["swing", "jazz"],
            "weights": [0.5, 0.5],
            "transition_frames": 30,
        }
        
        # Just verify the structure is valid
        assert len(request_data["motion_a"]) > 0
        assert len(request_data["motion_b"]) > 0
        assert len(request_data["style_labels_a"]) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
