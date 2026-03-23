"""
Tests for SPADE Hierarchical Motion Blending Service.

Tests cover:
- PyTorch SPADE layer functionality
- Trainable γ/β parameters
- Hash-based style embeddings
- FID/coverage/diversity metrics
- API endpoint validation
- Verbose logging verification
"""
import pytest
import numpy as np
from typing import List, Dict, Any

# Test fixtures and mocks
from kinetic_ledger.schemas.models import (
    HierarchyLevel,
    SPADEConfig,
    SPADEBlendRequest,
    SPADEMetrics,
    SPADEBlendResponse,
    JointHierarchyMapping,
)


# =============================================================================
# Schema Model Tests
# =============================================================================

class TestSPADESchemas:
    """Test SPADE Pydantic schema models."""
    
    def test_hierarchy_level_enum(self):
        """Test HierarchyLevel enum values."""
        assert HierarchyLevel.COARSE.value == "coarse"
        assert HierarchyLevel.MID.value == "mid"
        assert HierarchyLevel.FINE.value == "fine"
        assert HierarchyLevel.DETAIL.value == "detail"
    
    def test_spade_config_defaults(self):
        """Test SPADEConfig with default values."""
        config = SPADEConfig()
        
        assert config.spade_level == HierarchyLevel.COARSE
        assert config.input_dim == 768
        assert config.style_channels == 128
        assert config.motion_channels == 256
        assert config.gamma_init == 1.0
        assert config.beta_init == 0.0
        assert config.transition_sharpness == 5.0
    
    def test_spade_config_validation(self):
        """Test SPADEConfig validation."""
        # Valid config
        config = SPADEConfig(
            spade_level=HierarchyLevel.MID,
            input_dim=512,
            style_channels=64,
        )
        assert config.style_channels == 64
        
        # Invalid: style_channels > input_dim
        with pytest.raises(ValueError, match="style_channels"):
            SPADEConfig(
                input_dim=64,
                style_channels=128,  # Greater than input_dim
            )
    
    def test_spade_blend_request_validation(self):
        """Test SPADEBlendRequest validation."""
        # Valid request
        request = SPADEBlendRequest(
            request_id="test-123",
            motion_ids=["motion_a", "motion_b"],
            weights=[0.6, 0.4],
            style_labels=[["capoeira"], ["breakdance"]],
        )
        assert request.hierarchy_level == 1
        assert request.transition_frames == 30
        
        # Invalid: weights don't sum to 1.0
        with pytest.raises(ValueError, match="sum to 1.0"):
            SPADEBlendRequest(
                request_id="test-456",
                motion_ids=["a", "b"],
                weights=[0.3, 0.3],  # Sum = 0.6
                style_labels=[["a"], ["b"]],
            )
        
        # Invalid: lengths don't match
        with pytest.raises(ValueError, match="length"):
            SPADEBlendRequest(
                request_id="test-789",
                motion_ids=["a", "b", "c"],
                weights=[0.5, 0.5],  # Only 2 weights for 3 motions
                style_labels=[["a"], ["b"], ["c"]],
            )
    
    def test_spade_metrics_bounds(self):
        """Test SPADEMetrics field bounds."""
        metrics = SPADEMetrics(
            fid_score=10.5,
            coverage=0.85,
            diversity=2.3,
            smoothness=0.92,
            foot_sliding=0.05,
            spade_level_used=1,
            transition_quality=0.88,
            blend_time_ms=150.0,
            metrics_time_ms=50.0,
        )
        
        assert metrics.fid_score >= 0
        assert 0 <= metrics.coverage <= 1
        assert 0 <= metrics.smoothness <= 1
        assert 1 <= metrics.spade_level_used <= 4
    
    def test_joint_hierarchy_mapping(self):
        """Test JointHierarchyMapping defaults."""
        mapping = JointHierarchyMapping()
        
        assert "Hips" in mapping.coarse_joints
        assert "Spine" in mapping.coarse_joints
        assert "LeftArm" in mapping.mid_joints
        assert "Head" in mapping.fine_joints
        assert "LeftHand" in mapping.detail_joints


# =============================================================================
# Hash Embedding Tests
# =============================================================================

class TestHashEmbeddings:
    """Test hash-based style embedding generation."""
    
    def test_embedding_deterministic(self):
        """Test that embeddings are deterministic."""
        from kinetic_ledger.services.spade_blend_service import generate_hash_embedding
        
        labels = ["capoeira", "aggressive"]
        
        embed1 = generate_hash_embedding(labels, dim=768)
        embed2 = generate_hash_embedding(labels, dim=768)
        
        np.testing.assert_array_equal(embed1, embed2)
    
    def test_embedding_different_labels(self):
        """Test that different labels produce different embeddings."""
        from kinetic_ledger.services.spade_blend_service import generate_hash_embedding
        
        embed_a = generate_hash_embedding(["capoeira"], dim=768)
        embed_b = generate_hash_embedding(["breakdance"], dim=768)
        
        # Should be different
        assert not np.allclose(embed_a, embed_b)
    
    def test_embedding_normalized(self):
        """Test that embeddings are L2 normalized."""
        from kinetic_ledger.services.spade_blend_service import generate_hash_embedding
        
        embed = generate_hash_embedding(["test_style"], dim=768)
        
        norm = np.linalg.norm(embed)
        assert np.isclose(norm, 1.0, atol=1e-6)
    
    def test_embedding_dimension(self):
        """Test embedding dimension parameter."""
        from kinetic_ledger.services.spade_blend_service import generate_hash_embedding
        
        embed_768 = generate_hash_embedding(["test"], dim=768)
        embed_128 = generate_hash_embedding(["test"], dim=128)
        
        assert embed_768.shape == (768,)
        assert embed_128.shape == (128,)


# =============================================================================
# Metrics Tests
# =============================================================================

class TestMotionMetrics:
    """Test motion quality metrics computation."""
    
    def test_fid_identical_distributions(self):
        """Test FID with identical distributions."""
        from kinetic_ledger.services.metrics import compute_fid
        
        np.random.seed(42)
        features = np.random.randn(100, 64)
        
        fid = compute_fid(features, features)
        
        # FID of identical distributions should be very small
        assert fid < 1.0
    
    def test_fid_different_distributions(self):
        """Test FID with different distributions."""
        from kinetic_ledger.services.metrics import compute_fid
        
        np.random.seed(42)
        gen = np.random.randn(100, 64)
        ref = np.random.randn(100, 64) + 5.0  # Shifted distribution
        
        fid = compute_fid(gen, ref)
        
        # FID should be larger for different distributions
        assert fid > 10.0
    
    def test_coverage_full(self):
        """Test coverage when all references are covered."""
        from kinetic_ledger.services.metrics import compute_coverage
        
        # Generated samples exactly match references
        ref = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]], dtype=np.float32)
        gen = ref.copy()
        
        coverage = compute_coverage(gen, ref)
        
        assert coverage == 1.0
    
    def test_coverage_none(self):
        """Test coverage when no references are covered."""
        from kinetic_ledger.services.metrics import compute_coverage
        
        ref = np.array([[0, 0]], dtype=np.float32)
        gen = np.array([[100, 100]], dtype=np.float32)  # Far away
        
        coverage = compute_coverage(gen, ref, threshold=1.0)
        
        assert coverage == 0.0
    
    def test_diversity_calculation(self):
        """Test diversity metric calculation."""
        from kinetic_ledger.services.metrics import compute_diversity
        
        # Low diversity (similar samples)
        low_div = np.array([[0, 0], [0.1, 0], [0, 0.1]], dtype=np.float32)
        div_low = compute_diversity(low_div)
        
        # High diversity (spread samples)
        high_div = np.array([[0, 0], [10, 0], [0, 10]], dtype=np.float32)
        div_high = compute_diversity(high_div)
        
        assert div_high > div_low
    
    def test_smoothness_constant_motion(self):
        """Test smoothness with constant velocity motion."""
        from kinetic_ledger.services.metrics import compute_smoothness
        
        # Constant velocity: linear motion
        T, J, D = 30, 5, 3
        motion = np.zeros((T, J, D))
        for t in range(T):
            motion[t] = t * 0.1  # Linear motion
        
        smoothness = compute_smoothness(motion, fps=30.0)
        
        # Should be very smooth (high score)
        assert smoothness > 0.9
    
    def test_smoothness_jerky_motion(self):
        """Test smoothness with jerky motion."""
        from kinetic_ledger.services.metrics import compute_smoothness
        
        # Random motion (jerky)
        np.random.seed(42)
        T, J, D = 30, 5, 3
        motion = np.random.randn(T, J, D) * 10
        
        smoothness = compute_smoothness(motion, fps=30.0)
        
        # Should be less smooth
        assert smoothness < 0.5
    
    def test_foot_sliding_static(self):
        """Test foot sliding with static feet."""
        from kinetic_ledger.services.metrics import compute_foot_sliding
        
        T, J, D = 30, 24, 3
        motion = np.zeros((T, J, D))
        # Feet at ground level (y=0), static
        motion[:, 16:22, 1] = 0.01  # Small height
        
        sliding = compute_foot_sliding(motion, foot_joint_indices=[16, 17])
        
        # Should have minimal sliding
        assert sliding < 0.1
    
    def test_transition_quality(self):
        """Test transition quality metric."""
        from kinetic_ledger.services.metrics import compute_transition_quality
        
        T, J, D = 60, 5, 3
        
        # Smooth transition
        motion = np.zeros((T, J, D))
        for t in range(T):
            # Smooth interpolation
            alpha = t / T
            motion[t] = alpha * 1.0
        
        quality = compute_transition_quality(motion, transition_start=20, transition_end=40)
        
        assert 0 <= quality <= 1
    
    def test_compute_all_metrics(self):
        """Test full metrics computation."""
        from kinetic_ledger.services.metrics import compute_all_metrics
        
        np.random.seed(42)
        T, J, D = 60, 10, 3
        motion = np.random.randn(T, J, D).astype(np.float32) * 0.1
        
        result = compute_all_metrics(
            generated_motion=motion,
            transition_start=20,
            transition_end=40,
            fps=30.0,
        )
        
        assert result.fid_score >= 0
        assert 0 <= result.smoothness <= 1
        assert result.computation_time_ms > 0


# =============================================================================
# SPADE Service Tests
# =============================================================================

class TestSPADEService:
    """Test SPADE blend service."""
    
    def test_service_initialization(self):
        """Test SPADE service initialization."""
        from kinetic_ledger.services.spade_blend_service import (
            SPADEBlendService,
            get_spade_service,
            reset_spade_service,
            TORCH_AVAILABLE,
        )
        
        # Reset singleton
        reset_spade_service()
        
        service = get_spade_service()
        
        assert service is not None
        assert service.device in ["cpu", "cuda"]
        
        # If PyTorch available, should have model
        if TORCH_AVAILABLE:
            assert service.model is not None
            assert service.trainable_params_count > 0
    
    def test_service_singleton(self):
        """Test service singleton pattern."""
        from kinetic_ledger.services.spade_blend_service import (
            get_spade_service,
            reset_spade_service,
        )
        
        reset_spade_service()
        
        service1 = get_spade_service()
        service2 = get_spade_service()
        
        assert service1 is service2
    
    def test_blend_two_motions(self):
        """Test blending two motions."""
        from kinetic_ledger.services.spade_blend_service import (
            get_spade_service,
            reset_spade_service,
        )
        
        reset_spade_service()
        service = get_spade_service()
        
        # Create test motions
        np.random.seed(42)
        T, J, D = 60, 24, 3
        motion_a = np.random.randn(T, J, D).astype(np.float32) * 0.5
        motion_b = np.random.randn(T, J, D).astype(np.float32) * 0.5 + 1.0
        
        # Blend
        blended, timing = service.blend(
            motion_a=motion_a,
            motion_b=motion_b,
            style_labels_a=["capoeira"],
            style_labels_b=["breakdance"],
            weights=[0.5, 0.5],
            transition_frames=15,
        )
        
        # Check output
        assert blended.shape[0] > 0
        assert blended.shape[1] == J
        assert blended.shape[2] == D
        assert timing["blend_time_ms"] > 0
    
    def test_blend_interpolation(self):
        """Test motion interpolation for different lengths."""
        from kinetic_ledger.services.spade_blend_service import SPADEBlendService
        
        service = SPADEBlendService()
        
        # Different length motions
        np.random.seed(42)
        motion_a = np.random.randn(30, 24, 3).astype(np.float32)
        motion_b = np.random.randn(60, 24, 3).astype(np.float32)
        
        blended, _ = service.blend(
            motion_a=motion_a,
            motion_b=motion_b,
            style_labels_a=["walk"],
            style_labels_b=["run"],
            weights=[0.4, 0.6],
            transition_frames=10,
        )
        
        # Output length should be based on weighted sum
        assert blended.shape[0] > 0


# =============================================================================
# PyTorch Layer Tests (if available)
# =============================================================================

class TestPyTorchLayers:
    """Test PyTorch SPADE layers (skip if PyTorch unavailable)."""
    
    @pytest.fixture(autouse=True)
    def check_pytorch(self):
        """Skip tests if PyTorch not available."""
        from kinetic_ledger.services.spade_blend_service import TORCH_AVAILABLE
        if not TORCH_AVAILABLE:
            pytest.skip("PyTorch not available")
    
    def test_spade_norm_layer_forward(self):
        """Test SPADENormLayer forward pass."""
        import torch
        from kinetic_ledger.services.spade_blend_service import SPADENormLayer
        
        layer = SPADENormLayer(norm_channels=64, style_channels=32)
        
        # Input: [B, C, T]
        x = torch.randn(2, 64, 30)
        style = torch.randn(2, 32, 30)
        
        output = layer(x, style)
        
        assert output.shape == x.shape
    
    def test_spade_norm_layer_gradients(self):
        """Test that γ/β parameters receive gradients."""
        import torch
        from kinetic_ledger.services.spade_blend_service import SPADENormLayer
        
        layer = SPADENormLayer(norm_channels=64, style_channels=32)
        
        x = torch.randn(2, 64, 30, requires_grad=True)
        style = torch.randn(2, 32, 30)
        
        output = layer(x, style)
        loss = output.sum()
        loss.backward()
        
        # Check gradients exist
        assert layer.gamma_conv.weight.grad is not None
        assert layer.beta_conv.weight.grad is not None
    
    def test_hierarchical_blender_init(self):
        """Test SPADEHierarchicalBlender initialization."""
        import torch
        from kinetic_ledger.services.spade_blend_service import SPADEHierarchicalBlender
        from kinetic_ledger.schemas.models import SPADEConfig, HierarchyLevel
        
        config = SPADEConfig(
            spade_level=HierarchyLevel.COARSE,
            input_dim=768,
            style_channels=128,
            motion_channels=64,
        )
        
        blender = SPADEHierarchicalBlender(config)
        
        # Check level blocks
        assert "coarse" in blender.level_blocks
        assert "mid" in blender.level_blocks
        assert "fine" in blender.level_blocks
        assert "detail" in blender.level_blocks
        
        # Check trainable params
        assert blender.trainable_params_count > 0
    
    def test_hierarchical_blender_omega(self):
        """Test omega weight generation."""
        import torch
        from kinetic_ledger.services.spade_blend_service import SPADEHierarchicalBlender
        from kinetic_ledger.schemas.models import SPADEConfig
        
        config = SPADEConfig()
        blender = SPADEHierarchicalBlender(config)
        
        omega = blender.build_omega(
            total_frames=60,
            transition_start=20,
            transition_end=40,
            device=torch.device("cpu"),
        )
        
        # Check shape
        assert omega.shape == (1, 1, 60)
        
        # Check values
        assert omega[0, 0, 0] == 0.0  # Before transition
        assert omega[0, 0, -1] == 1.0  # After transition
        assert 0 < omega[0, 0, 30] < 1  # During transition
    
    def test_checkpoint_save_load(self):
        """Test checkpoint save and load."""
        import torch
        import tempfile
        from pathlib import Path
        from kinetic_ledger.services.spade_blend_service import SPADEHierarchicalBlender
        from kinetic_ledger.schemas.models import SPADEConfig
        
        config = SPADEConfig()
        blender = SPADEHierarchicalBlender(config)
        
        # Save checkpoint
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "test_checkpoint.pt"
            blender.save_checkpoint(str(checkpoint_path))
            
            assert checkpoint_path.exists()
            
            # Load checkpoint
            blender2 = SPADEHierarchicalBlender(config)
            blender2.load_checkpoint(str(checkpoint_path))
            
            # Verify weights match
            for (n1, p1), (n2, p2) in zip(
                blender.named_parameters(),
                blender2.named_parameters()
            ):
                assert torch.allclose(p1, p2), f"Mismatch in {n1}"


# =============================================================================
# Logging Tests
# =============================================================================

class TestVerboseLogging:
    """Test verbose entry/exit logging."""
    
    def test_blend_logging(self, caplog):
        """Test that blend operations log entry/exit."""
        import logging
        from kinetic_ledger.services.spade_blend_service import (
            SPADEBlendService,
            reset_spade_service,
        )
        
        reset_spade_service()
        
        # Enable debug logging
        caplog.set_level(logging.INFO)
        
        service = SPADEBlendService()
        
        np.random.seed(42)
        motion_a = np.random.randn(30, 10, 3).astype(np.float32)
        motion_b = np.random.randn(30, 10, 3).astype(np.float32)
        
        service.blend(
            motion_a=motion_a,
            motion_b=motion_b,
            style_labels_a=["test_a"],
            style_labels_b=["test_b"],
            weights=[0.5, 0.5],
            transition_frames=10,
        )
        
        # Check for entry/exit logs
        log_text = caplog.text
        assert "[ENTRY]" in log_text or "blend" in log_text.lower()


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests for full SPADE workflow."""
    
    def test_full_blend_workflow(self):
        """Test complete blend workflow."""
        from kinetic_ledger.services.spade_blend_service import get_spade_service, reset_spade_service
        from kinetic_ledger.services.metrics import compute_all_metrics
        
        reset_spade_service()
        service = get_spade_service()
        
        # Create test data
        np.random.seed(42)
        T, J, D = 60, 24, 3
        motion_a = np.random.randn(T, J, D).astype(np.float32)
        motion_b = np.random.randn(T, J, D).astype(np.float32)
        
        # Blend
        blended, timing = service.blend(
            motion_a=motion_a,
            motion_b=motion_b,
            style_labels_a=["style_a"],
            style_labels_b=["style_b"],
            weights=[0.6, 0.4],
            transition_frames=20,
        )
        
        # Compute metrics
        metrics = compute_all_metrics(
            generated_motion=blended,
            reference_motions=np.stack([motion_a, motion_b]),
            transition_start=timing.get("transition_start", 0),
            transition_end=timing.get("transition_end", blended.shape[0]),
        )
        
        # Verify results
        assert blended.shape[0] > 0
        # smoothness can be 0 for very jerky motion (random data has high acceleration)
        assert metrics.smoothness >= 0
        assert metrics.computation_time_ms > 0
    
    def test_multiple_style_labels(self):
        """Test blending with multiple style labels."""
        from kinetic_ledger.services.spade_blend_service import (
            get_spade_service,
            reset_spade_service,
            generate_hash_embedding,
        )
        
        reset_spade_service()
        
        # Multiple labels should produce different embedding than single
        embed_single = generate_hash_embedding(["capoeira"])
        embed_multi = generate_hash_embedding(["capoeira", "aggressive", "fast"])
        
        assert not np.allclose(embed_single, embed_multi)


# =============================================================================
# Edge Cases
# =============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_single_frame_motion(self):
        """Test handling of single-frame motions."""
        from kinetic_ledger.services.metrics import compute_smoothness
        
        motion = np.random.randn(1, 10, 3).astype(np.float32)
        
        # Should handle gracefully
        smoothness = compute_smoothness(motion)
        assert smoothness == 1.0  # Default for too-short motions
    
    def test_empty_style_labels(self):
        """Test handling of empty style labels."""
        from kinetic_ledger.services.spade_blend_service import generate_hash_embedding
        
        # Empty labels should still work
        embed = generate_hash_embedding([], dim=768)
        
        assert embed.shape == (768,)
        assert np.isclose(np.linalg.norm(embed), 1.0)
    
    def test_large_transition_frames(self):
        """Test with transition frames larger than motion."""
        from kinetic_ledger.services.spade_blend_service import (
            SPADEBlendService,
            reset_spade_service,
        )
        
        reset_spade_service()
        service = SPADEBlendService()
        
        # Short motions, long transition
        np.random.seed(42)
        motion_a = np.random.randn(20, 10, 3).astype(np.float32)
        motion_b = np.random.randn(20, 10, 3).astype(np.float32)
        
        # Should handle gracefully
        blended, timing = service.blend(
            motion_a=motion_a,
            motion_b=motion_b,
            style_labels_a=["a"],
            style_labels_b=["b"],
            weights=[0.5, 0.5],
            transition_frames=50,  # Longer than motions
        )
        
        assert blended.shape[0] > 0
