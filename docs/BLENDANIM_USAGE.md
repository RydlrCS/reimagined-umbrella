# blendanim Integration - Complete Usage Guide

This is the comprehensive usage guide for the blendanim integration with Gemini 3.0 Pro Preview.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Component Reference](#component-reference)
3. [Usage Examples](#usage-examples)
4. [Configuration](#configuration)
5. [Testing](#testing)
6. [Troubleshooting](#troubleshooting)
7. [API Integration](#api-integration)

---

## Quick Start

### Installation

```bash
cd /workspaces/reimagined-umbrella
source .venv/bin/activate

# Install dependencies
pip install torch numpy google-generativeai

# Set API key
export GEMINI_API_KEY="your-gemini-api-key"
```

### Verify Installation

```bash
python -c "
from src.kinetic_ledger.services.blendanim_service import BlendAnimService
from src.kinetic_ledger.services.gemini_motion_embedder import GeminiMotionEmbedder
print('✓ Services imported successfully')
"
```

---

## Component Reference

### BlendAnimService

Location: `src/kinetic_ledger/services/blendanim_service.py`

#### Key Methods

```python
from kinetic_ledger.services.blendanim_service import BlendAnimService, MotionSequence
import numpy as np

# Initialize service
service = BlendAnimService(use_gpu=False)

# Create motion sequence
positions = np.random.randn(120, 52, 3).astype('float32')
motion = MotionSequence(positions=positions, fps=30)

# Calculate metrics
metrics = service.calculate_all_metrics(motion)
print(f"Coverage: {metrics.coverage:.4f}")
print(f"Quality Tier: {metrics.quality_tier}")

# Blend motions
blended, blend_metrics = service.blend_motions(
    motions=[motion1, motion2],
    weights=[0.6, 0.4],
    method="temporal_conditioning"
)
```

### GeminiMotionEmbedder

Location: `src/kinetic_ledger/services/gemini_motion_embedder.py`

#### Key Methods

```python
from kinetic_ledger.services.gemini_motion_embedder import GeminiMotionEmbedder
import asyncio

async def process_fbx():
    embedder = GeminiMotionEmbedder()
    
    # Process single FBX file
    fbx_data = await embedder.process_fbx_file(
        "data/mixamo_anims/fbx/X Bot@Capoeira.fbx",
        extract_positions=True,
        generate_embedding=True
    )
    
    print(f"Frames: {fbx_data.frame_count}")
    print(f"Embedding dim: {len(fbx_data.embedding)}")
    
    return fbx_data

fbx_data = asyncio.run(process_fbx())
```

---

## Usage Examples

### Example 1: Calculate Metrics for Motion

```python
from kinetic_ledger.services.blendanim_service import (
    BlendAnimService,
    MotionSequence
)
import numpy as np

# Create service
service = BlendAnimService(use_gpu=False)

# Load motion data (from FBX or create synthetic)
positions = np.load("capoeira_positions.npy")  # [T, J, 3]
motion = MotionSequence(positions=positions, fps=30)

# Calculate all metrics
metrics = service.calculate_all_metrics(motion)

print(f"=== Motion Quality Metrics ===")
print(f"Coverage:          {metrics.coverage:.4f}")
print(f"Local Diversity:   {metrics.local_diversity:.4f}")
print(f"Global Diversity:  {metrics.global_diversity:.4f}")
print(f"L2 Velocity:       {metrics.l2_velocity:.6f}")
print(f"L2 Acceleration:   {metrics.l2_acceleration:.6f}")
print(f"Smoothness:        {metrics.blend_area_smoothness:.4f}")
print(f"Quality Tier:      {metrics.quality_tier}")
```

### Example 2: Blend Two Motions

```python
from kinetic_ledger.services.blendanim_service import (
    BlendAnimService,
    MotionSequence
)
import numpy as np

service = BlendAnimService(use_gpu=False)

# Load two motions
capoeira = MotionSequence(
    positions=np.load("capoeira.npy"),
    fps=30
)

breakdance = MotionSequence(
    positions=np.load("breakdance.npy"),
    fps=30
)

# Blend with 50/50 weights
blended, metrics = service.blend_motions(
    motions=[capoeira, breakdance],
    weights=[0.5, 0.5],
    method="temporal_conditioning"
)

# Save result
np.save("blended_output.npy", blended.positions)

print(f"Blended motion: {blended.positions.shape}")
print(f"Quality: {metrics.quality_tier}")
```

### Example 3: Process FBX with Gemini

```python
import asyncio
from kinetic_ledger.services.gemini_motion_embedder import GeminiMotionEmbedder

async def process_mixamo_fbx():
    embedder = GeminiMotionEmbedder()
    
    # Process FBX file
    fbx_data = await embedder.process_fbx_file(
        file_path="data/mixamo_anims/fbx/X Bot@Capoeira.fbx",
        extract_positions=True,
        generate_embedding=True
    )
    
    print(f"File: {fbx_data.file_name}")
    print(f"Duration: {fbx_data.duration_seconds:.2f}s")
    print(f"Frames: {fbx_data.frame_count}")
    print(f"FPS: {fbx_data.fps}")
    print(f"Joints: {fbx_data.joint_count}")
    print(f"Gemini URI: {fbx_data.gemini_file_uri}")
    
    # Access skeletal positions
    if fbx_data.skeletal_positions is not None:
        print(f"Position shape: {fbx_data.skeletal_positions.shape}")
    
    # Access embedding
    if fbx_data.embedding is not None:
        print(f"Embedding dim: {len(fbx_data.embedding)}")
    
    return fbx_data

# Run
fbx_data = asyncio.run(process_mixamo_fbx())
```

### Example 4: End-to-End Pipeline

```python
import asyncio
import numpy as np
from kinetic_ledger.services.blendanim_service import BlendAnimService, MotionSequence
from kinetic_ledger.services.gemini_motion_embedder import GeminiMotionEmbedder

async def full_blend_pipeline():
    # Initialize
    embedder = GeminiMotionEmbedder()
    blender = BlendAnimService(use_gpu=False)
    
    # Step 1: Process FBX files with Gemini
    print("Step 1: Processing FBX files...")
    capoeira_fbx = await embedder.process_fbx_file(
        "data/mixamo_anims/fbx/X Bot@Capoeira.fbx"
    )
    
    breakdance_fbx = await embedder.process_fbx_file(
        "data/mixamo_anims/fbx/X Bot@Breakdance Freeze Var 2.fbx"
    )
    
    # Step 2: Convert to MotionSequence
    print("Step 2: Converting to motion sequences...")
    capoeira = MotionSequence(
        positions=capoeira_fbx.skeletal_positions,
        fps=capoeira_fbx.fps
    )
    
    breakdance = MotionSequence(
        positions=breakdance_fbx.skeletal_positions,
        fps=breakdance_fbx.fps
    )
    
    # Step 3: Blend motions
    print("Step 3: Blending motions...")
    blended, metrics = blender.blend_motions(
        motions=[capoeira, breakdance],
        weights=[0.6, 0.4],  # 60% Capoeira, 40% Breakdance
        method="temporal_conditioning"
    )
    
    # Step 4: Calculate cost
    print("Step 4: Calculating cost...")
    quality_rates = {
        "ultra": 0.25,
        "high": 0.10,
        "medium": 0.05,
        "low": 0.01
    }
    
    rate = quality_rates[metrics.quality_tier]
    duration = blended.positions.shape[0] / blended.fps
    motion_count = 2
    complexity = 1.5
    
    cost = rate * duration * np.sqrt(motion_count) * complexity
    
    # Step 5: Report results
    print(f"\n{'=' * 60}")
    print(f"BLEND RESULTS: Capoeira (60%) + Breakdance (40%)")
    print(f"{'=' * 60}")
    print(f"Duration:       {duration:.2f} seconds")
    print(f"Frames:         {blended.positions.shape[0]}")
    print(f"Quality Tier:   {metrics.quality_tier}")
    print(f"\nMetrics:")
    print(f"  Coverage:          {metrics.coverage:.4f}")
    print(f"  Local Diversity:   {metrics.local_diversity:.4f}")
    print(f"  Global Diversity:  {metrics.global_diversity:.4f}")
    print(f"  L2 Velocity:       {metrics.l2_velocity:.6f}")
    print(f"  L2 Acceleration:   {metrics.l2_acceleration:.6f}")
    print(f"  Smoothness:        {metrics.blend_area_smoothness:.4f}")
    print(f"\nCost:           ${cost:.3f} USDC")
    print(f"{'=' * 60}\n")
    
    # Step 6: Save output
    np.save("capoeira_breakdance_blend.npy", blended.positions)
    print("✓ Saved blended motion to: capoeira_breakdance_blend.npy")
    
    return blended, metrics, cost

# Run pipeline
blended, metrics, cost = asyncio.run(full_blend_pipeline())
```

### Example 5: Batch Process Multiple FBX Files

```python
import asyncio
from kinetic_ledger.services.gemini_motion_embedder import GeminiMotionEmbedder

async def batch_process():
    embedder = GeminiMotionEmbedder()
    
    # Process all FBX files in directory
    fbx_data_list = await embedder.batch_process_fbx_directory(
        directory="data/mixamo_anims/fbx",
        pattern="*.fbx",
        max_files=5  # Process first 5 files
    )
    
    print(f"Processed {len(fbx_data_list)} files:")
    for fbx_data in fbx_data_list:
        print(f"  - {fbx_data.file_name}: {fbx_data.frame_count} frames")
    
    return fbx_data_list

fbx_list = asyncio.run(batch_process())
```

---

## Configuration

### Environment Variables

```bash
# Required
export GEMINI_API_KEY="your-gemini-api-key"

# Optional
export BLENDING_USE_GPU="false"
export BLENDING_MAX_WORKERS="4"
```

### Configuration File

Edit `config/blending.toml`:

```toml
[blending]
use_gpu = false
device = "cpu"

[metrics]
coverage_tmin = 30
coverage_threshold = 2.0
local_diversity_tmin = 15
global_diversity_tmin = 30

[quality_tiers.ultra]
min_coverage = 0.90
max_l2_velocity = 0.03
max_l2_acceleration = 0.015
price_per_second = 0.25

[gemini]
model_name = "gemini-2.0-flash-exp"
embedding_model = "text-embedding-004"
```

---

## Testing

### Run All Tests

```bash
pytest tests/test_blendanim_integration.py -v
```

### Run Specific Tests

```bash
# Test metrics calculation
pytest tests/test_blendanim_integration.py::TestBlendAnimService -v

# Test Gemini integration
pytest tests/test_blendanim_integration.py::TestGeminiMotionEmbedder -v

# Test end-to-end
pytest tests/test_blendanim_integration.py::TestEndToEndBlending -v
```

### Quick Validation

```bash
python -c "
from src.kinetic_ledger.services.blendanim_service import BlendAnimService, MotionSequence
import numpy as np

service = BlendAnimService(use_gpu=False)
positions = np.random.randn(120, 52, 3).astype('float32')
motion = MotionSequence(positions=positions, fps=30)
metrics = service.calculate_all_metrics(motion)

print(f'Coverage: {metrics.coverage:.4f}')
print(f'Quality: {metrics.quality_tier}')
print('✓ Test passed!')
"
```

---

## Troubleshooting

### Issue: "GEMINI_API_KEY not found"

```bash
# Set environment variable
export GEMINI_API_KEY="your-api-key"

# Or create .env file
echo "GEMINI_API_KEY=your-api-key" > .env
```

### Issue: "CUDA out of memory"

```python
# Use CPU mode
service = BlendAnimService(use_gpu=False)
```

### Issue: "FBX extraction failed"

The system automatically uses fallback positions if Gemini extraction fails:

```python
# Check if fallback was used
if fbx_data.metadata.get("fallback"):
    print("Using fallback positions")
```

---

## API Integration

Add to `src/kinetic_ledger/api/server.py`:

```python
from kinetic_ledger.services.blendanim_service import get_blendanim_service
from kinetic_ledger.services.gemini_motion_embedder import get_gemini_embedder

@app.post("/api/motions/blend")
async def blend_motions_endpoint(
    motion_ids: List[str],
    weights: List[float],
    quality: str = "medium"
):
    embedder = get_gemini_embedder()
    blender = get_blendanim_service()
    
    # Process motions
    motions = []
    for motion_id in motion_ids:
        fbx_data = await embedder.process_fbx_file(
            f"data/mixamo_anims/fbx/{motion_id}.fbx"
        )
        motion = MotionSequence(
            positions=fbx_data.skeletal_positions,
            fps=fbx_data.fps
        )
        motions.append(motion)
    
    # Blend
    blended, metrics = blender.blend_motions(
        motions=motions,
        weights=weights,
        method="temporal_conditioning"
    )
    
    return {
        "status": "success",
        "metrics": metrics.to_dict(),
        "frame_count": blended.positions.shape[0]
    }
```

---

## References

- [BLEND_METRICS.md](BLEND_METRICS.md) - Metric specifications
- [BLENDANIM_ALIGNMENT.md](BLENDANIM_ALIGNMENT.md) - Alignment summary
- [blendanim GitHub](https://github.com/RydlrCS/blendanim) - Original repository

---

**Last Updated**: January 9, 2026
