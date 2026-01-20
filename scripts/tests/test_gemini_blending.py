"""
Test script for Gemini-powered intelligent motion blending.

This script tests the complete workflow from frame rendering to Gemini analysis.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

async def test_intelligent_blending():
    """Test the intelligent blending workflow."""
    
    print("🧪 Testing Gemini Intelligent Motion Blending")
    print("=" * 60)
    
    # Check environment
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        print("❌ GEMINI_API_KEY not set")
        print("   Set with: export GEMINI_API_KEY='your_key_here'")
        return
    
    print(f"✅ Gemini API key found: {gemini_api_key[:8]}...")
    
    # Test 1: Import all modules
    print("\n📦 Test 1: Module Imports")
    try:
        from src.kinetic_ledger.schemas.gemini_motion_schemas import (
            MotionBlendAnalysis,
            MotionCharacteristics,
            BlendParameters
        )
        print("  ✅ Schemas imported")
        
        from src.kinetic_ledger.services.gemini_motion_analyzer import GeminiMotionAnalyzer
        print("  ✅ GeminiMotionAnalyzer imported")
        
        from src.kinetic_ledger.utils.frame_renderer import (
            MotionFrameRenderer,
            create_standard_skeleton_hierarchy
        )
        print("  ✅ MotionFrameRenderer imported")
        
    except ImportError as e:
        print(f"  ❌ Import failed: {e}")
        return
    
    # Test 2: Initialize services
    print("\n🔧 Test 2: Service Initialization")
    try:
        analyzer = GeminiMotionAnalyzer(
            api_key=gemini_api_key,
            model_name="gemini-2.0-flash-exp",
            frame_sample_rate=5
        )
        print(f"  ✅ GeminiMotionAnalyzer initialized: {analyzer.model_name}")
        
        renderer = MotionFrameRenderer(width=640, height=480)
        print(f"  ✅ MotionFrameRenderer initialized: {renderer.width}x{renderer.height}")
        
    except Exception as e:
        print(f"  ❌ Initialization failed: {e}")
        return
    
    # Test 3: Load test FBX files
    print("\n📂 Test 3: Load FBX Files")
    
    # Find available FBX files
    fbx_dir = Path("data/mixamo_anims/fbx")
    if not fbx_dir.exists():
        print(f"  ⚠️ FBX directory not found: {fbx_dir}")
        print("     Skipping FBX-dependent tests")
        return
    
    fbx_files = list(fbx_dir.glob("*.fbx"))[:2]  # Get first 2 files
    
    if len(fbx_files) < 2:
        print(f"  ⚠️ Need at least 2 FBX files, found {len(fbx_files)}")
        return
    
    print(f"  ✅ Found FBX files:")
    for fbx in fbx_files:
        print(f"     - {fbx.name}")
    
    # Test 4: Parse FBX and render frames
    print("\n🎨 Test 4: Render Frames")
    try:
        from src.kinetic_ledger.utils.fbx_parser import get_fbx_parser
        
        fbx_parser = get_fbx_parser()
        motion_frames = []
        
        for idx, fbx_path in enumerate(fbx_files):
            print(f"  Processing {fbx_path.name}...")
            
            # Parse FBX
            positions, metadata = fbx_parser.parse_fbx(str(fbx_path))
            print(f"    Shape: {positions.shape}, FPS: {metadata.get('fps', 30)}")
            
            # Render frames
            hierarchy = create_standard_skeleton_hierarchy(positions.shape[1])
            frames = renderer.render_frames_from_positions(
                positions,
                skeleton_hierarchy=hierarchy,
                frame_indices=list(range(0, min(len(positions), 100), 10))  # Sample 10 frames
            )
            motion_frames.append(frames)
            
            print(f"    ✅ Rendered {len(frames)} frames")
        
    except Exception as e:
        print(f"  ❌ Frame rendering failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test 5: Gemini Analysis
    print("\n🤖 Test 5: Gemini Motion Analysis")
    print("  This may take 5-10 seconds...")
    
    try:
        analysis = await analyzer.analyze_motion_pair(
            motion_a_frames=motion_frames[0],
            motion_b_frames=motion_frames[1],
            motion_a_name=fbx_files[0].stem,
            motion_b_name=fbx_files[1].stem
        )
        
        print(f"\n  ✨ Analysis Complete!")
        print(f"  {'='*56}")
        print(f"  Motion A: {analysis.motion_a_characteristics.motion_type} ({analysis.motion_a_characteristics.energy_level} energy)")
        print(f"  Motion B: {analysis.motion_b_characteristics.motion_type} ({analysis.motion_b_characteristics.energy_level} energy)")
        print(f"  Compatibility: {analysis.compatibility.overall_score:.2%}")
        print(f"  Reasoning: {analysis.compatibility.reasoning}")
        print(f"\n  Recommended Parameters:")
        print(f"    - Transition frames: {analysis.recommended_parameters.transition_frames}")
        print(f"    - Omega curve: {analysis.recommended_parameters.omega_curve_type}")
        print(f"    - Velocity matching: {analysis.recommended_parameters.apply_velocity_matching}")
        print(f"    - Root correction: {analysis.recommended_parameters.apply_root_motion_correction}")
        print(f"\n  Quality Prediction:")
        print(f"    - Coverage: {analysis.quality_prediction.predicted_coverage:.2%}")
        print(f"    - Diversity: {analysis.quality_prediction.predicted_diversity:.2%}")
        print(f"    - Smoothness: {analysis.quality_prediction.predicted_smoothness:.2%}")
        print(f"    - Confidence: {analysis.quality_prediction.confidence:.2%}")
        
        if analysis.quality_prediction.potential_issues:
            print(f"\n  ⚠️ Potential Issues:")
            for issue in analysis.quality_prediction.potential_issues:
                print(f"    - {issue}")
        
        print(f"\n  Overall Recommendation:")
        print(f"    {analysis.overall_recommendation}")
        
        print("\n  ✅ Gemini analysis successful!")
        
    except Exception as e:
        print(f"  ❌ Gemini analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test 6: Validate schema
    print("\n✅ Test 6: Schema Validation")
    try:
        # Convert to dict and back to ensure serialization works
        analysis_dict = analysis.model_dump()
        print(f"  ✅ Model can be serialized to dict ({len(str(analysis_dict))} chars)")
        
        # Validate round-trip
        analysis_copy = MotionBlendAnalysis.model_validate(analysis_dict)
        print(f"  ✅ Model can be deserialized from dict")
        
        # Check JSON serialization
        import json
        analysis_json = analysis.model_dump_json()
        print(f"  ✅ Model can be serialized to JSON ({len(analysis_json)} chars)")
        
    except Exception as e:
        print(f"  ❌ Schema validation failed: {e}")
        return
    
    print("\n" + "="*60)
    print("🎉 All tests passed!")
    print("="*60)
    print("\nNext steps:")
    print("  1. Start the API server: python -m src.kinetic_ledger.api.server")
    print("  2. Open UI: http://localhost:8000")
    print("  3. Enable '✨ AI Analysis' checkbox in Blend tab")
    print("  4. Generate intelligent artifacts and see Gemini insights!")


if __name__ == "__main__":
    asyncio.run(test_intelligent_blending())
