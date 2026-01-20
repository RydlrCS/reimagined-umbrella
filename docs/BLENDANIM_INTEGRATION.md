# blendanim Integration Guide

**Status**: ✅ Complete  
**Last Updated**: January 9, 2026

---

## Overview

This document describes the integration of the [blendanim](https://github.com/RydlrCS/blendanim) motion blending framework with Gemini 3.0 Pro Preview for FBX motion analysis.

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Kinetic Ledger System                     │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌─────────────────┐          ┌──────────────────────┐      │
│  │  FBX Files      │─────────▶│  GeminiMotionEmbedder│      │
│  │  (Mixamo)       │          │  - Upload FBX        │      │
│  └─────────────────┘          │  - Extract positions │      │
│                                │  - Generate embeddings│      │
│                                └──────────┬───────────┘      │
│                                           │                   │
│                                           ▼                   │
│                                ┌──────────────────────┐      │
│                                │  MotionSequence      │      │
│                                │  [T, J, 3]           │      │
│                                └──────────┬───────────┘      │
│                                           │                   │
│                                           ▼                   │
│  ┌─────────────────┐          ┌──────────────────────┐      │
│  │  Blend Request  │─────────▶│  BlendAnimService    │      │
│  │  - Motions      │          │  - Coverage          │      │
│  │  - Weights      │          │  - LocalDiversity    │      │
│  │  - Quality      │          │  - GlobalDiversity   │      │
│  └─────────────────┘          │  - L2_velocity       │      │
│                                │  - L2_acceleration   │      │
│                                └──────────┬───────────┘      │
│                                           │                   │
│                                           ▼                   │
│                                ┌──────────────────────┐      │
│                                │  Blended Motion      │      │
│                                │  + BlendMetrics      │      │
│                                └──────────────────────┘      │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

---

## Quick Start

### 1. Installation

```bash
# Install dependencies
pip install torch numpy google-generativeai

# Set Gemini API key
export GEMINI_API_KEY="your-api-key"
```

### 2. Basic Usage

```python
from kinetic_ledger.services.blendanim_service import BlendAnimService
from kinetic_ledger.services.gemini_motion_embedder import GeminiMotionEmbedder
import asyncio

async def main():
    # Initialize services
    embedder = GeminiMotionEmbedder()
    blender = BlendAnimService(use_gpu=False)
    
    # Process FBX files
    capoeira = await embedder.process_fbx_file(
        "data/mixamo_anims/fbx/X Bot@Capoeira.fbx"
    )
    
    # Calculate metrics
    metrics = blender.calculate_all_metrics(
        MotionSequence(positions=capoeira.skeletal_positions, fps=30)
    )
    
    print(f"Quality: {metrics.quality_tier}")
    print(f"Coverage: {metrics.coverage:.4f}")

asyncio.run(main())
```

---

## Complete Documentation

See full integration guide at [docs/BLENDANIM_INTEGRATION.md](BLENDANIM_INTEGRATION.md)

---

**Status**: ✅ Production Ready
