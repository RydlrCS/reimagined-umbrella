# Motion Blend Quality Metrics

**Aligned with:** [blendanim](https://github.com/RydlrCS/blendanim) evaluation framework  
**Date:** January 9, 2026

---

## Overview

Our motion blend evaluation system aligns with the academic `blendanim` repository's metrics for measuring blend quality, diversity, and smoothness. These metrics provide quantitative assessment of generated motion blends.

## Core Metrics

### 1. **Coverage** (Range: 0.0 - 1.0, Higher is Better)

**Definition:** Measures how well the generated blend covers the motion space compared to ground truth.

**Calculation:**
```python
# From blendanim/src/metrics/generation/ganimator_metrics.py
def forward(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    group_cost = _group_cost_from_tensors(pred, gt)
    res = []
    for i in range(group_cost.shape[0] - tmin):
        cost = torch.min(group_cost[i, i + tmin]) / tmin
        res.append(1.0 if cost < threshold else 0.0)
    return torch.mean(torch.Tensor([res]))
```

**Parameters:**
- `tmin`: Minimum window size (default: 30 frames)
- `threshold`: Cost threshold for coverage (default: 2.0)

**Interpretation:**
- **0.9 - 1.0**: Excellent coverage, blend fully represents motion space
- **0.7 - 0.9**: Good coverage, most motion space represented
- **0.5 - 0.7**: Moderate coverage, some gaps in motion space
- **< 0.5**: Poor coverage, significant gaps

**Our Implementation:**
```python
# Higher for more diverse motion combinations
coverage = min(0.95, 0.6 + (motion_count - 1) * 0.15)
```

---

### 2. **Local Diversity** (Range: 0.0+, Lower for Similar Motions)

**Definition:** Measures short-term motion variation within 15-frame windows.

**Calculation:**
```python
# From blendanim/src/metrics/generation/ganimator_metrics.py
def forward(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    group_cost = _group_cost_from_tensors(pred, gt)
    res = _calc_perwindow_cost(group_cost, tmin=15)
    return torch.mean(torch.from_numpy(np.array(res)))
```

**Parameters:**
- `tmin`: Window size (default: 15 frames, ~0.5s at 30fps)

**Interpretation:**
- **< 1.0**: Very smooth, minimal local variation
- **1.0 - 2.0**: Moderate local variation
- **2.0 - 3.0**: High local variation (dance, acrobatics)
- **> 3.0**: Very high variation (may appear jerky)

**Our Implementation:**
```python
local_diversity = complexity * 1.8
```

**Example Values:**
- Idle: ~0.4
- Walk: ~0.6
- Dance: ~1.8
- Capoeira: ~2.2

---

### 3. **Global Diversity** (Range: 0.0+, Higher for Diverse Blends)

**Definition:** Measures long-term motion variation across 30-frame windows using dynamic programming alignment.

**Calculation:**
```python
# From blendanim/src/metrics/generation/ganimator_metrics.py
def forward(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    group_cost = _group_cost_from_tensors(pred, gt)
    val, label = _nn_dp_fast(group_cost, tmin=30)
    res = val / label.shape[0]
    return torch.from_numpy(np.array(res))
```

**Parameters:**
- `tmin`: Window size (default: 30 frames, ~1s at 30fps)

**Interpretation:**
- **< 2.0**: Low global diversity (repetitive motion)
- **2.0 - 4.0**: Moderate diversity
- **4.0 - 6.0**: High diversity (complex choreography)
- **> 6.0**: Very high diversity (may lack coherence)

**Our Implementation:**
```python
global_diversity = complexity * 3.2
```

**Example Values:**
- Idle: ~0.6
- Walk: ~1.0
- Dance: ~3.0
- Capoeira: ~4.8

---

### 4. **L2 Velocity** (Range: 0.0+, Lower is Better)

**Definition:** Measures smoothness of velocity transitions between frames. Lower values indicate smoother motion.

**Calculation:**
```python
# From blendanim/src/metrics/generation/ganimator_metrics.py
def forward(velocities: torch.Tensor) -> torch.Tensor:
    # [B, T, J, 3] -> L2 norm per joint
    l2 = torch.norm(velocities, dim=-1)
    # Calculate delta velocities
    delta_velocities = torch.abs(l2[:, 1:, :] - l2[:, :-1, :])
    return delta_velocities
```

**Focus Area:**
- Evaluates middle 30 frames (blend transition area)
- Measured per joint: Pelvis, LeftWrist, RightWrist, LeftFoot, RightFoot

**Interpretation:**
- **< 0.05**: Very smooth (ultra quality blends)
- **0.05 - 0.10**: Smooth (high quality blends)
- **0.10 - 0.15**: Moderate smoothness (medium quality)
- **> 0.15**: Rough transitions (low quality or intentional style)

**Our Implementation:**
```python
# Inverse relationship with quality tier
quality_multipliers = {"low": 1.5, "medium": 1.0, "high": 0.7, "ultra": 0.4}
smoothness_factor = quality_multipliers[quality]
l2_velocity = 0.10 * smoothness_factor / complexity
```

**Example Values:**
- Ultra quality: ~0.02
- High quality: ~0.07
- Medium quality: ~0.10
- Low quality: ~0.15

---

### 5. **L2 Acceleration** (Range: 0.0+, Lower is Better)

**Definition:** Measures jerkiness (rate of change of velocity). Lower values indicate smoother acceleration profiles.

**Calculation:**
```python
# From blendanim/src/metrics/generation/ganimator_metrics.py
def forward(velocities: torch.Tensor) -> torch.Tensor:
    l2 = torch.norm(velocities, dim=-1)
    delta_velocities = torch.abs(l2[:, 1:, :] - l2[:, :-1, :])
    # Acceleration = Δ(Δv)
    acceleration = torch.abs(delta_velocities[:, 1:, :] - delta_velocities[:, :-1, :])
    return acceleration
```

**Focus Area:**
- Evaluates middle 30 frames (blend transition area)
- Measured per joint: Pelvis, LeftWrist, RightWrist, LeftFoot, RightFoot

**Interpretation:**
- **< 0.02**: Extremely smooth (professional quality)
- **0.02 - 0.05**: Smooth (high quality)
- **0.05 - 0.08**: Moderate (acceptable for games)
- **> 0.08**: Jerky (needs improvement or stylistic choice)

**Our Implementation:**
```python
l2_acceleration = 0.05 * smoothness_factor / complexity
```

**Example Values:**
- Ultra quality: ~0.01
- High quality: ~0.035
- Medium quality: ~0.05
- Low quality: ~0.08

---

### 6. **Blend Area Smoothness** (Range: 0.0 - 1.0, Higher is Better)

**Definition:** Derived metric measuring smoothness specifically in the blend transition region (middle 30 frames).

**Calculation:**
```python
blend_area_smoothness = 1.0 - (l2_velocity * 2.0)
```

**Interpretation:**
- **0.9 - 1.0**: Excellent blend smoothness
- **0.7 - 0.9**: Good blend smoothness
- **0.5 - 0.7**: Moderate smoothness
- **< 0.5**: Poor blend quality

---

## Quality Tiers

### Ultra Quality (`ultra`)
- **Price**: 0.25 USDC/second
- **Coverage**: ≥ 0.90
- **L2 Velocity**: ≤ 0.03
- **L2 Acceleration**: ≤ 0.015
- **Blend Area Smoothness**: ≥ 0.94
- **Use Case**: Cinematic animations, hero moments

### High Quality (`high`)
- **Price**: 0.10 USDC/second
- **Coverage**: ≥ 0.85
- **L2 Velocity**: ≤ 0.07
- **L2 Acceleration**: ≤ 0.04
- **Blend Area Smoothness**: ≥ 0.86
- **Use Case**: AAA game cutscenes, close-up shots

### Medium Quality (`medium`)
- **Price**: 0.05 USDC/second
- **Coverage**: ≥ 0.75
- **L2 Velocity**: ≤ 0.10
- **L2 Acceleration**: ≤ 0.05
- **Blend Area Smoothness**: ≥ 0.80
- **Use Case**: Standard gameplay animations

### Low Quality (`low`)
- **Price**: 0.01 USDC/second
- **Coverage**: ≥ 0.65
- **L2 Velocity**: ≤ 0.15
- **L2 Acceleration**: ≤ 0.08
- **Blend Area Smoothness**: ≥ 0.70
- **Use Case**: Background NPCs, distant characters

---

## Capoeira to Breakdance Example

### Motion A: Capoeira
- **Duration**: 4.5s
- **Novelty**: 0.75
- **Tags**: dance, acrobatic, martial-arts
- **Coverage**: 0.88
- **Local Diversity**: 2.25
- **Global Diversity**: 4.2
- **L2 Velocity**: 0.12
- **L2 Acceleration**: 0.06

### Motion B: Breakdance Freeze
- **Duration**: 3.8s
- **Novelty**: 0.82
- **Tags**: dance, acrobatic, freeze
- **Coverage**: 0.92
- **Local Diversity**: 2.46
- **Global Diversity**: 4.8
- **L2 Velocity**: 0.14
- **L2 Acceleration**: 0.07

### Blend Result (50/50, Medium Quality)
- **Duration**: 8.0s (estimated combined)
- **Complexity**: 1.5
- **Coverage**: 0.75 (moderate, two diverse motions)
- **Local Diversity**: 2.7 (high, both motions are dynamic)
- **Global Diversity**: 4.8 (very diverse blend)
- **L2 Velocity**: 0.067 (good smoothness)
- **L2 Acceleration**: 0.033 (good smoothness)
- **Blend Area Smoothness**: 0.866 (good quality)
- **Cost**: 0.05 × 8.0 × √2 × 1.5 = **0.849 USDC**

---

## API Response Format

### Motion Library Entry
```json
{
  "id": "capoeira-001",
  "name": "Capoeira",
  "tags": ["dance", "acrobatic", "martial-arts"],
  "duration": 4.5,
  "novelty": 0.75,
  "filepath": "/data/mixamo_anims/fbx/X Bot@Capoeira.fbx",
  "metrics": {
    "coverage": 0.880,
    "local_diversity": 2.250,
    "global_diversity": 4.200,
    "l2_velocity": 0.120,
    "l2_acceleration": 0.060
  }
}
```

### Blend Generation Response
```json
{
  "status": "success",
  "blend": {
    "id": "blend-abc123",
    "motions": ["capoeira", "breakdance"],
    "weights": [0.5, 0.5],
    "duration": 8.0,
    "complexity": 1.5,
    "quality_metrics": {
      "coverage": 0.750,
      "local_diversity": 2.700,
      "global_diversity": 4.800,
      "l2_velocity": 0.0667,
      "l2_acceleration": 0.0333,
      "blend_area_smoothness": 0.866,
      "quality_tier": "medium"
    }
  },
  "settlement": {
    "tx_hash": "0x...",
    "amount_usdc": "0.849000",
    "routing": "off-chain",
    "timestamp": 1736467200
  }
}
```

---

## Implementation Notes

### Current Implementation (Stub)
Our current implementation uses **heuristic estimates** based on motion metadata:
- Novelty scores from motion tags
- Complexity from motion count and prompt analysis
- Quality tier from user selection

### Production Implementation
For production, calculate metrics from **actual skeletal data**:

```python
import torch
import numpy as np

def calculate_l2_velocity(positions: torch.Tensor) -> float:
    """
    positions: [T, J, 3] (Time, Joints, XYZ)
    """
    # Calculate velocities
    velocities = positions[1:] - positions[:-1]
    
    # L2 norm per joint
    l2 = torch.norm(velocities, dim=-1)  # [T-1, J]
    
    # Delta velocities
    delta_v = torch.abs(l2[1:] - l2[:-1])  # [T-2, J]
    
    # Focus on blend area (middle 30 frames)
    total_frames = delta_v.shape[0]
    middle = total_frames // 2
    blend_area = delta_v[middle-15:middle+15]
    
    # Mean across blend area and joints
    return blend_area.mean().item()

def calculate_coverage(pred: torch.Tensor, gt: torch.Tensor, 
                      tmin: int = 30, threshold: float = 2.0) -> float:
    """
    pred: [T1, J, 3] predicted motion
    gt: [T2, J, 3] ground truth motion
    """
    # Calculate pairwise cost matrix
    cost = torch.cdist(pred.reshape(pred.shape[0], -1),
                       gt.reshape(gt.shape[0], -1))
    
    # Dynamic programming for coverage
    covered = 0
    for i in range(cost.shape[0] - tmin):
        min_cost = torch.min(cost[i:i+tmin]).item() / tmin
        if min_cost < threshold:
            covered += 1
    
    return covered / (cost.shape[0] - tmin)
```

### Integration with Gemini
For AI-assisted blend generation:
1. Upload FBX files to Gemini File API
2. Extract skeletal data using Gemini vision/analysis
3. Calculate metrics from skeleton positions
4. Use metrics to guide blend quality prediction

---

## References

- **blendanim Repository**: https://github.com/RydlrCS/blendanim
- **Metrics Implementation**: `single-shot-blending/src/metrics/generation/ganimator_metrics.py`
- **Key Classes**:
  - `Coverage`: Motion space coverage
  - `LocalDiversity`: Short-term variation (15-frame windows)
  - `GlobalDiversity`: Long-term variation (30-frame windows)
  - `L2_velocity`: Velocity smoothness
  - `L2_acceleration`: Acceleration smoothness (jerk minimization)

---

**Last Updated**: January 9, 2026  
**Alignment Status**: ✅ Metrics defined according to blendanim standards
