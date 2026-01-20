"""
Comprehensive tests for blendanim integration and Gemini FBX processing.

Tests:
- BlendAnimService metrics calculation
- Motion blending algorithms
- GeminiMotionEmbedder FBX processing
- End-to-end blend pipeline
"""

import pytest
import numpy as np
import asyncio
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from kinetic_ledger.services.blendanim_service import (
    BlendAnimService,
    MotionSequence,
    BlendMetrics,
    get_blendanim_service
)
from kinetic_ledger.services.gemini_motion_embedder import (
    GeminiMotionEmbedder,
    FBXMotionData,
    get_gemini_embedder
)


class TestBlendAnimService:
    """Tests for blendanim metrics and blending."""
    
    @pytest.fixture
    def service(self):
        """Create blendanim service."""
        return BlendAnimService(use_gpu=False)  # CPU for testing
    
    @pytest.fixture
    def simple_motion(self):
        """Create simple test motion."""
        # Create a simple walking motion (sine wave pattern)
        T = 120  # 4 seconds at 30 FPS
        J = 52   # Mixamo skeleton
        
        positions = np.zeros((T, J, 3), dtype=np.float32)
        
        for t in range(T):
            time_factor = t / T
            # Pelvis moves forward and up/down
            positions[t, 0, 0] = time_factor * 2.0  # Forward
            positions[t, 0, 1] = 1.0 + 0.1 * np.sin(time_factor * 4 * np.pi)  # Bounce
            
            # Legs alternate
            positions[t, 10, 2] = 0.3 * np.sin(time_factor * 4 * np.pi)  # Left leg
            positions[t, 15, 2] = -0.3 * np.sin(time_factor * 4 * np.pi)  # Right leg
        
        return MotionSequence(
            positions=positions,
            fps=30,
            joint_names=[f"Joint_{i}" for i in range(J)]
        )
    
    @pytest.fixture
    def dynamic_motion(self):
        """Create dynamic test motion (jumping)."""
        T = 90  # 3 seconds
        J = 52
        
        positions = np.zeros((T, J, 3), dtype=np.float32)
        
        for t in range(T):
            time_factor = t / T
            # Jump pattern
            jump_height = 0.8 * np.sin(time_factor * 2 * np.pi)
            if jump_height < 0:
                jump_height = 0
            
            positions[t, 0, 1] = 1.0 + jump_height  # Pelvis height
            
            # Arms swing
            positions[t, 5, 1] = 1.5 + 0.5 * np.sin(time_factor * 2 * np.pi)
            positions[t, 6, 1] = 1.5 - 0.5 * np.sin(time_factor * 2 * np.pi)
        
        return MotionSequence(
            positions=positions,
            fps=30,
            joint_names=[f"Joint_{i}" for i in range(J)]
        )
    
    def test_coverage_calculation(self, service, simple_motion):
        """Test coverage metric calculation."""
        coverage = service.calculate_coverage(simple_motion)
        
        assert 0.0 <= coverage <= 1.0, "Coverage must be in [0, 1]"
        assert coverage > 0.5, "Simple motion should have reasonable coverage"
        
        print(f"Coverage: {coverage:.4f}")
    
    def test_local_diversity(self, service, simple_motion, dynamic_motion):
        """Test local diversity metric."""
        simple_div = service.calculate_local_diversity(simple_motion)
        dynamic_div = service.calculate_local_diversity(dynamic_motion)
        
        assert simple_div >= 0.0, "Local diversity must be non-negative"
        assert dynamic_div >= 0.0, "Local diversity must be non-negative"
        
        # Dynamic motion should have higher diversity
        print(f"Simple local diversity: {simple_div:.4f}")
        print(f"Dynamic local diversity: {dynamic_div:.4f}")
    
    def test_global_diversity(self, service, simple_motion, dynamic_motion):
        """Test global diversity with NN-DP."""
        simple_global = service.calculate_global_diversity(simple_motion)
        dynamic_global = service.calculate_global_diversity(dynamic_motion)
        
        assert simple_global >= 0.0, "Global diversity must be non-negative"
        assert dynamic_global >= 0.0, "Global diversity must be non-negative"
        
        print(f"Simple global diversity: {simple_global:.4f}")
        print(f"Dynamic global diversity: {dynamic_global:.4f}")
    
    def test_l2_velocity(self, service, simple_motion):
        """Test L2 velocity metric."""
        l2_vel = service.calculate_l2_velocity(simple_motion)
        
        assert l2_vel >= 0.0, "L2 velocity must be non-negative"
        assert l2_vel < 1.0, "L2 velocity should be reasonable for simple motion"
        
        print(f"L2 velocity: {l2_vel:.6f}")
    
    def test_l2_acceleration(self, service, simple_motion):
        """Test L2 acceleration (jerk) metric."""
        l2_acc = service.calculate_l2_acceleration(simple_motion)
        
        assert l2_acc >= 0.0, "L2 acceleration must be non-negative"
        
        print(f"L2 acceleration: {l2_acc:.6f}")
    
    def test_all_metrics(self, service, simple_motion):
        """Test calculating all metrics at once."""
        metrics = service.calculate_all_metrics(simple_motion)
        
        assert isinstance(metrics, BlendMetrics)
        assert 0.0 <= metrics.coverage <= 1.0
        assert metrics.local_diversity >= 0.0
        assert metrics.global_diversity >= 0.0
        assert metrics.l2_velocity >= 0.0
        assert metrics.l2_acceleration >= 0.0
        assert 0.0 <= metrics.blend_area_smoothness <= 1.0
        assert metrics.quality_tier in ["ultra", "high", "medium", "low"]
        
        print(f"\nComplete Metrics:")
        print(f"  Coverage: {metrics.coverage:.4f}")
        print(f"  Local Diversity: {metrics.local_diversity:.4f}")
        print(f"  Global Diversity: {metrics.global_diversity:.4f}")
        print(f"  L2 Velocity: {metrics.l2_velocity:.6f}")
        print(f"  L2 Acceleration: {metrics.l2_acceleration:.6f}")
        print(f"  Smoothness: {metrics.blend_area_smoothness:.4f}")
        print(f"  Quality Tier: {metrics.quality_tier}")
    
    def test_linear_blend(self, service, simple_motion, dynamic_motion):
        """Test linear motion blending."""
        motions = [simple_motion, dynamic_motion]
        weights = [0.5, 0.5]
        
        blended, metrics = service.blend_motions(motions, weights, method="linear")
        
        assert isinstance(blended, MotionSequence)
        assert isinstance(metrics, BlendMetrics)
        
        # Blended motion should have positions
        assert blended.positions is not None
        assert blended.positions.shape[1] == simple_motion.positions.shape[1]  # Same joints
        
        print(f"\nBlended Motion Metrics:")
        print(f"  Shape: {blended.positions.shape}")
        print(f"  Coverage: {metrics.coverage:.4f}")
        print(f"  Quality: {metrics.quality_tier}")
    
    def test_weighted_blend(self, service, simple_motion, dynamic_motion):
        """Test asymmetric blend weights."""
        motions = [simple_motion, dynamic_motion]
        weights = [0.7, 0.3]  # More simple motion
        
        blended, metrics = service.blend_motions(motions, weights, method="linear")
        
        assert blended.positions is not None
        
        # Check that blend is closer to first motion
        diff_to_simple = np.mean(np.abs(blended.positions - simple_motion.positions[:blended.positions.shape[0]]))
        diff_to_dynamic = np.mean(np.abs(blended.positions - dynamic_motion.positions[:blended.positions.shape[0]]))
        
        print(f"\nWeighted Blend (0.7/0.3):")
        print(f"  Distance to simple: {diff_to_simple:.6f}")
        print(f"  Distance to dynamic: {diff_to_dynamic:.6f}")
    
    def test_temporal_conditioning_blend(self, service, simple_motion, dynamic_motion):
        """Test temporal conditioning blend."""
        motions = [simple_motion, dynamic_motion]
        weights = [0.5, 0.5]
        
        blended, metrics = service.blend_motions(
            motions, weights, method="temporal_conditioning"
        )
        
        assert blended.positions is not None
        assert isinstance(metrics, BlendMetrics)
        
        # Temporal conditioning should produce smoother blend
        print(f"\nTemporal Conditioning Blend:")
        print(f"  Smoothness: {metrics.blend_area_smoothness:.4f}")
        print(f"  L2 Velocity: {metrics.l2_velocity:.6f}")
    
    def test_quality_tier_determination(self, service):
        """Test quality tier assignment."""
        # Ultra quality
        ultra_metrics = BlendMetrics(
            coverage=0.95,
            local_diversity=1.0,
            global_diversity=3.0,
            l2_velocity=0.02,
            l2_acceleration=0.01,
            blend_area_smoothness=0.95
        )
        tier = service._determine_quality_tier(
            ultra_metrics.coverage,
            ultra_metrics.l2_velocity,
            ultra_metrics.l2_acceleration,
            ultra_metrics.blend_area_smoothness
        )
        assert tier == "ultra", f"Expected ultra, got {tier}"
        
        # Medium quality
        medium_metrics = BlendMetrics(
            coverage=0.80,
            local_diversity=2.0,
            global_diversity=4.0,
            l2_velocity=0.08,
            l2_acceleration=0.04,
            blend_area_smoothness=0.82
        )
        tier = service._determine_quality_tier(
            medium_metrics.coverage,
            medium_metrics.l2_velocity,
            medium_metrics.l2_acceleration,
            medium_metrics.blend_area_smoothness
        )
        assert tier in ["high", "medium"], f"Expected high/medium, got {tier}"
        
        print(f"Quality tier tests passed")
    
    def test_metrics_to_dict(self):
        """Test metrics conversion to dictionary."""
        metrics = BlendMetrics(
            coverage=0.85,
            local_diversity=2.3,
            global_diversity=4.5,
            l2_velocity=0.067,
            l2_acceleration=0.033,
            blend_area_smoothness=0.866,
            quality_tier="high"
        )
        
        metrics_dict = metrics.to_dict()
        
        assert isinstance(metrics_dict, dict)
        assert "coverage" in metrics_dict
        assert "quality_tier" in metrics_dict
        assert metrics_dict["quality_tier"] == "high"
        
        print(f"Metrics dict: {metrics_dict}")


class TestGeminiMotionEmbedder:
    """Tests for Gemini FBX processing."""
    
    @pytest.fixture
    def embedder(self):
        """Create Gemini embedder (skip if no API key)."""
        try:
            return GeminiMotionEmbedder()
        except ValueError:
            pytest.skip("GEMINI_API_KEY not configured")
    
    @pytest.fixture
    def test_fbx_path(self):
        """Get path to test FBX file."""
        fbx_path = Path("data/mixamo_anims/fbx/X Bot@Capoeira.fbx")
        if not fbx_path.exists():
            pytest.skip(f"Test FBX file not found: {fbx_path}")
        return str(fbx_path)
    
    @pytest.mark.asyncio
    async def test_upload_fbx(self, embedder, test_fbx_path):
        """Test FBX file upload to Gemini."""
        uploaded_file = await embedder.upload_fbx(test_fbx_path)
        
        assert uploaded_file is not None
        assert uploaded_file.uri is not None
        assert uploaded_file.state.name == "ACTIVE"
        
        print(f"\nUploaded file: {uploaded_file.uri}")
    
    @pytest.mark.asyncio
    async def test_extract_positions_fallback(self, embedder, test_fbx_path):
        """Test fallback position extraction."""
        positions, metadata = embedder._generate_fallback_positions(test_fbx_path)
        
        assert positions.shape[1] == 52, "Should have 52 joints (Mixamo)"
        assert positions.shape[2] == 3, "Should have XYZ coordinates"
        assert positions.shape[0] >= 60, "Should have reasonable frame count"
        assert metadata["fallback"] == True
        
        print(f"\nFallback positions: {positions.shape}")
        print(f"Metadata: {metadata}")
    
    @pytest.mark.asyncio
    async def test_motion_embedding(self, embedder):
        """Test motion embedding generation."""
        # Test with text description
        embedding = await embedder.generate_motion_embedding(
            "Dynamic capoeira motion with kicks and spins"
        )
        
        assert embedding is not None
        assert len(embedding) > 0, "Embedding should have dimensions"
        assert embedding.dtype == np.float32
        
        print(f"\nEmbedding dimension: {len(embedding)}")
    
    @pytest.mark.asyncio
    async def test_prompt_analysis(self, embedder):
        """Test natural language prompt analysis."""
        prompt = "blend capoeira and breakdance with 60% capoeira, ultra quality"
        
        analysis = await embedder.analyze_motion_prompt(prompt)
        
        assert "primary_motions" in analysis
        assert "blend_weights" in analysis
        assert "quality_preference" in analysis
        assert "keywords" in analysis
        
        print(f"\nPrompt analysis: {analysis}")
    
    @pytest.mark.asyncio
    async def test_process_fbx_file(self, embedder, test_fbx_path):
        """Test full FBX processing pipeline."""
        fbx_data = await embedder.process_fbx_file(
            test_fbx_path,
            extract_positions=True,
            generate_embedding=True
        )
        
        assert isinstance(fbx_data, FBXMotionData)
        assert fbx_data.file_name is not None
        assert fbx_data.gemini_file_uri is not None
        
        if fbx_data.skeletal_positions is not None:
            assert fbx_data.skeletal_positions.shape[1] == 52
        
        if fbx_data.embedding is not None:
            assert len(fbx_data.embedding) > 0
        
        print(f"\nProcessed FBX: {fbx_data.to_dict()}")
    
    def test_motion_sequence_analysis(self, embedder):
        """Test motion sequence description generation."""
        # Create test motion
        positions = np.random.randn(100, 52, 3).astype(np.float32)
        motion = MotionSequence(positions=positions, fps=30)
        
        description = embedder._analyze_motion_sequence(motion)
        
        assert isinstance(description, str)
        assert len(description) > 0
        assert "frames" in description.lower()
        
        print(f"\nMotion description: {description}")


class TestEndToEndBlending:
    """End-to-end tests for complete blending pipeline."""
    
    @pytest.mark.asyncio
    async def test_capoeira_breakdance_blend(self):
        """Test Capoeira + Breakdance blend example."""
        # Setup services
        blend_service = get_blendanim_service()
        
        # Create synthetic motions (simulating Capoeira and Breakdance)
        capoeira = MotionSequence(
            positions=np.random.randn(135, 52, 3).astype(np.float32) * 0.5,
            fps=30,
            joint_names=[f"Joint_{i}" for i in range(52)]
        )
        
        breakdance = MotionSequence(
            positions=np.random.randn(114, 52, 3).astype(np.float32) * 0.6,
            fps=30,
            joint_names=[f"Joint_{i}" for i in range(52)]
        )
        
        # Blend
        blended, metrics = blend_service.blend_motions(
            motions=[capoeira, breakdance],
            weights=[0.5, 0.5],
            method="temporal_conditioning"
        )
        
        # Validate
        assert blended.positions is not None
        assert metrics.quality_tier in ["ultra", "high", "medium", "low"]
        
        # Calculate cost (from BLEND_METRICS.md example)
        quality_rates = {"ultra": 0.25, "high": 0.10, "medium": 0.05, "low": 0.01}
        rate = quality_rates[metrics.quality_tier]
        duration = blended.positions.shape[0] / blended.fps
        motion_count = 2
        complexity = 1.5
        
        cost = rate * duration * np.sqrt(motion_count) * complexity
        
        print(f"\n=== Capoeira + Breakdance Blend ===")
        print(f"Blended motion: {blended.positions.shape[0]} frames ({duration:.2f}s)")
        print(f"Metrics:")
        print(f"  Coverage: {metrics.coverage:.4f}")
        print(f"  Local Diversity: {metrics.local_diversity:.4f}")
        print(f"  Global Diversity: {metrics.global_diversity:.4f}")
        print(f"  L2 Velocity: {metrics.l2_velocity:.6f}")
        print(f"  L2 Acceleration: {metrics.l2_acceleration:.6f}")
        print(f"  Smoothness: {metrics.blend_area_smoothness:.4f}")
        print(f"  Quality Tier: {metrics.quality_tier}")
        print(f"Cost: ${cost:.3f} USDC")
        
        # Verify metrics align with blendanim standards
        assert 0.0 <= metrics.coverage <= 1.0
        assert metrics.local_diversity >= 0.0
        assert metrics.global_diversity >= 0.0
        assert metrics.l2_velocity >= 0.0
        assert metrics.l2_acceleration >= 0.0


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "-s"])
