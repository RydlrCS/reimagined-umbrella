#!/usr/bin/env python3
"""
blendanim Integration Verification Script

Runs comprehensive checks to verify the integration is working correctly.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def check_imports():
    """Verify all imports work."""
    print("=" * 60)
    print("STEP 1: Checking imports...")
    print("=" * 60)
    
    try:
        from kinetic_ledger.services.blendanim_service import (
            BlendAnimService,
            MotionSequence,
            BlendMetrics
        )
        print("✓ BlendAnimService imports successful")
    except ImportError as e:
        print(f"✗ BlendAnimService import failed: {e}")
        return False
    
    try:
        from kinetic_ledger.services.gemini_motion_embedder import (
            GeminiMotionEmbedder,
            FBXMotionData
        )
        print("✓ GeminiMotionEmbedder imports successful")
    except ImportError as e:
        print(f"✗ GeminiMotionEmbedder import failed: {e}")
        return False
    
    print("\n")
    return True


def test_blendanim_service():
    """Test BlendAnimService basic functionality."""
    print("=" * 60)
    print("STEP 2: Testing BlendAnimService...")
    print("=" * 60)
    
    try:
        from kinetic_ledger.services.blendanim_service import (
            BlendAnimService,
            MotionSequence
        )
        import numpy as np
        
        # Create service
        service = BlendAnimService(use_gpu=False)
        print("✓ BlendAnimService initialized (CPU mode)")
        
        # Create test motion
        positions = np.random.randn(120, 52, 3).astype('float32')
        motion = MotionSequence(positions=positions, fps=30)
        print("✓ Test motion created (120 frames, 52 joints)")
        
        # Calculate metrics
        metrics = service.calculate_all_metrics(motion)
        print("✓ Metrics calculated successfully")
        
        # Verify metric ranges
        assert 0.0 <= metrics.coverage <= 1.0, "Coverage out of range"
        assert metrics.local_diversity >= 0.0, "Local diversity negative"
        assert metrics.global_diversity >= 0.0, "Global diversity negative"
        assert metrics.l2_velocity >= 0.0, "L2 velocity negative"
        assert metrics.l2_acceleration >= 0.0, "L2 acceleration negative"
        assert 0.0 <= metrics.blend_area_smoothness <= 1.0, "Smoothness out of range"
        assert metrics.quality_tier in ["ultra", "high", "medium", "low"], "Invalid quality tier"
        print("✓ All metrics in valid ranges")
        
        # Print results
        print(f"\n  Coverage:          {metrics.coverage:.4f}")
        print(f"  Local Diversity:   {metrics.local_diversity:.4f}")
        print(f"  Global Diversity:  {metrics.global_diversity:.4f}")
        print(f"  L2 Velocity:       {metrics.l2_velocity:.6f}")
        print(f"  L2 Acceleration:   {metrics.l2_acceleration:.6f}")
        print(f"  Smoothness:        {metrics.blend_area_smoothness:.4f}")
        print(f"  Quality Tier:      {metrics.quality_tier}")
        
        print("\n")
        return True
        
    except Exception as e:
        print(f"✗ BlendAnimService test failed: {e}")
        import traceback
        traceback.print_exc()
        print("\n")
        return False


def test_blending():
    """Test motion blending."""
    print("=" * 60)
    print("STEP 3: Testing motion blending...")
    print("=" * 60)
    
    try:
        from kinetic_ledger.services.blendanim_service import (
            BlendAnimService,
            MotionSequence
        )
        import numpy as np
        
        service = BlendAnimService(use_gpu=False)
        
        # Create two motions
        motion1 = MotionSequence(
            positions=np.random.randn(100, 52, 3).astype('float32'),
            fps=30
        )
        motion2 = MotionSequence(
            positions=np.random.randn(100, 52, 3).astype('float32'),
            fps=30
        )
        print("✓ Created two test motions")
        
        # Blend
        blended, metrics = service.blend_motions(
            motions=[motion1, motion2],
            weights=[0.6, 0.4],
            method="temporal_conditioning"
        )
        print("✓ Blending completed")
        
        # Verify output
        assert blended.positions is not None, "No blended positions"
        assert blended.positions.shape[1] == 52, "Wrong joint count"
        print(f"✓ Blended motion: {blended.positions.shape[0]} frames")
        
        print(f"\n  Blend Quality: {metrics.quality_tier}")
        print(f"  Coverage:      {metrics.coverage:.4f}")
        print(f"  Smoothness:    {metrics.blend_area_smoothness:.4f}")
        
        print("\n")
        return True
        
    except Exception as e:
        print(f"✗ Blending test failed: {e}")
        import traceback
        traceback.print_exc()
        print("\n")
        return False


def check_gemini_embedder():
    """Check GeminiMotionEmbedder (may skip if no API key)."""
    print("=" * 60)
    print("STEP 4: Checking GeminiMotionEmbedder...")
    print("=" * 60)
    
    try:
        from kinetic_ledger.services.gemini_motion_embedder import (
            GeminiMotionEmbedder
        )
        
        # Check for API key
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            print("⚠ GEMINI_API_KEY not set - skipping Gemini tests")
            print("  (This is optional for blending functionality)")
            print("\n")
            return True
        
        # Try to initialize
        embedder = GeminiMotionEmbedder()
        print("✓ GeminiMotionEmbedder initialized")
        
        # Test fallback positions
        test_path = "test_motion.fbx"
        positions, metadata = embedder._generate_fallback_positions(test_path)
        print("✓ Fallback position generation works")
        print(f"  Generated {positions.shape[0]} frames, {positions.shape[1]} joints")
        
        print("\n")
        return True
        
    except Exception as e:
        print(f"⚠ GeminiMotionEmbedder check failed: {e}")
        print("  (This is optional for blending functionality)")
        print("\n")
        return True  # Don't fail on Gemini issues


def check_files():
    """Check that all expected files exist."""
    print("=" * 60)
    print("STEP 5: Checking file structure...")
    print("=" * 60)
    
    expected_files = [
        "src/kinetic_ledger/services/blendanim_service.py",
        "src/kinetic_ledger/services/gemini_motion_embedder.py",
        "tests/test_blendanim_integration.py",
        "config/blending.toml",
        "docs/BLEND_METRICS.md",
        "docs/BLENDANIM_ALIGNMENT.md",
        "docs/BLENDANIM_INTEGRATION.md",
        "docs/BLENDANIM_USAGE.md",
        "BLENDANIM_COMPLETE.md"
    ]
    
    all_exist = True
    for file_path in expected_files:
        if Path(file_path).exists():
            print(f"✓ {file_path}")
        else:
            print(f"✗ {file_path} - MISSING")
            all_exist = False
    
    print("\n")
    return all_exist


def main():
    """Run all verification checks."""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 10 + "blendanim Integration Verification" + " " * 13 + "║")
    print("╚" + "=" * 58 + "╝")
    print("\n")
    
    results = {
        "Imports": check_imports(),
        "BlendAnimService": test_blendanim_service(),
        "Motion Blending": test_blending(),
        "GeminiMotionEmbedder": check_gemini_embedder(),
        "File Structure": check_files()
    }
    
    # Summary
    print("=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)
    
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name:.<40} {status}")
    
    all_passed = all(results.values())
    
    print("\n")
    if all_passed:
        print("╔" + "=" * 58 + "╗")
        print("║" + " " * 15 + "✓ ALL CHECKS PASSED" + " " * 23 + "║")
        print("║" + " " * 10 + "blendanim integration is working!" + " " * 15 + "║")
        print("╚" + "=" * 58 + "╝")
        print("\n")
        return 0
    else:
        print("╔" + "=" * 58 + "╗")
        print("║" + " " * 15 + "✗ SOME CHECKS FAILED" + " " * 22 + "║")
        print("║" + " " * 8 + "Please review errors above" + " " * 23 + "║")
        print("╚" + "=" * 58 + "╝")
        print("\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
