# BlendAnim Playback Implementation

## Overview
Implementation of animation playback system based on the BlendAnim research paper (https://arxiv.org/html/2508.18525v1).

## Architecture

### Server-Side (BlendAnim Service)
The server generates transition frames using BlendAnim's temporal conditioning approach:

1. **Motion Encoding** (ufbx fallback or Gemini):
   - FBX files → skeletal joint positions [J, 3]
   - Optional Gemini embeddings for semantic understanding
   - Fallback to direct ufbx parser when Gemini unavailable

2. **Transition Generation**:
   - Uses smoothstep temporal conditioning: `ω(t) = 3t² - 2t³`
   - Blends motion segments at specified weights
   - Generates 30-frame transitions by default
   - Output: 189 frames (129 motion + 60 transition)

3. **Quality Metrics**:
   - Coverage: Motion space exploration
   - Local/Global Diversity: Pose variation
   - L2 Velocity/Acceleration: Smoothness
   - Quality tier: Ultra/High/Medium/Low

### Client-Side (3D Viewer)

#### Data Flow
```
Server (BlendAnim) → API Response → loadArtifactsInViewer() → ArtifactsViewer.loadArtifacts()
                                                                        ↓
                                                            createAnimationFromArtifacts()
                                                                        ↓
                                                            THREE.AnimationClip + AnimationMixer
                                                                        ↓
                                                                  Playback Loop
```

#### Implementation Details

**File**: `/src/kinetic_ledger/ui/index-new.html`

**Class**: `ArtifactsViewer`

**Key Methods**:

1. **`loadArtifacts(artifactData)`** (Line ~4550)
   - Receives pre-computed artifact sequence from server
   - Stores motion segments and frame data
   - Initializes playback controls
   - Calls `createAnimationFromArtifacts()`

2. **`createAnimationFromArtifacts()`** (Line ~4570)
   - Creates THREE.js animation from artifact positions
   - Uses `VectorKeyframeTrack` for bone positions
   - Builds animation tracks for each skeletal joint
   - Creates looping animation clip at 30 FPS
   
   **BlendAnim Research Alignment**:
   - Server applies smoothstep ω(t) for temporal blending
   - Client receives interpolated positions (no re-blending needed)
   - Direct skeletal position application matches research approach
   - Maintains temporal coherence from server-side conditioning

3. **`animate()`** (Line ~4745)
   - Render loop using requestAnimationFrame
   - Updates AnimationMixer with delta time
   - Syncs timeline slider with current frame
   - Displays frame metadata (ω, segment type)

#### BlendAnim Temporal Conditioning

The server implements the smoothstep function from the paper:

```python
# Server-side (blendanim_service.py)
def smoothstep(t):
    """BlendAnim temporal conditioning function"""
    return 3 * t**2 - 2 * t**3

# Applied during transition generation
omega = smoothstep(transition_progress)  # t ∈ [0, 1]
blended_position = (1 - omega) * motion_A + omega * motion_B
```

Client receives these pre-blended positions and plays them back linearly.

## Playback Features

### Current Implementation ✅
- **Looping Playback**: 189 frames continuous loop
- **Frame-by-Frame Control**: Timeline slider
- **Play/Pause**: Toggle animation
- **Frame Display**: Shows current frame, ω value, segment type
- **Motion Segments**: Preserves motion 1/2/3 boundaries
- **Transition Visualization**: Crosshatch overlays on blend zones

### Alignment with BlendAnim Research
The implementation follows the paper's approach:

1. **Temporal Conditioning** (Section 3.2):
   - Server applies smoothstep ω(t) during generation
   - Client plays back pre-conditioned frames
   - Maintains smooth transitions without runtime blending

2. **Quality Metrics** (Section 4):
   - Coverage: Measures motion space exploration
   - Diversity: Local/global pose variation
   - Smoothness: L2 velocity/acceleration

3. **Multi-Motion Blending**:
   - Supports 3 motions with configurable weights
   - Crosshatch offset controls segment start position
   - Preserves motion identity in segments

## Testing Workflow

### Auto-Load Demo
**Function**: `autoLoadArtifactsDemo()` (Line ~4060)

Automatically loads test motions on page load:
1. **Motions**:
   - Salsa Dancing (67 frames, weight 0.43)
   - Swing Charleston (74 frames, weight 0.43)
   - Wave Hip Hop (505 frames, weight 0.14)

2. **API Call**: Direct POST to `/api/artifacts/generate`

3. **Preview Generation**: Creates thumbnails + crosshatch overlays

4. **Viewer Loading**: Loads artifacts into 3D viewport

5. **Auto-Switch**: Switches to Artifacts tab after 1 second

## Visual Elements

### Reference Strip
- **Thumbnails**: 3 per motion segment + 3 per blend transition
- **Color Coding**:
  - Motion: Blue/purple gradient
  - Blend: Orange gradient
- **Overlap**: -15px margin for visual continuity
- **Crosshatch**: Colored overlays on blend sequences

### 3D Viewport
- **Character**: Ch03_nonPBR.fbx (Michele model)
- **Scale**: 0.01 (Mixamo → Three.js units)
- **Camera**: FOV 45°, positioned at (0, 1.6, 3)
- **Lighting**: Ambient + directional with shadows
- **Ground**: Grid helper (20x20)

### Playback Controls
- **Play/Pause**: ▶/⏸ button
- **Timeline**: Slider (0 to frame count - 1)
- **Frame Display**: "Frame X/189 (ω=0.xxx, motion/blend)"
- **Speed**: Currently 1.0x (can be adjusted)

## Known Limitations

1. **Character Load Timing**:
   - FBX loading is asynchronous
   - Thumbnail may render before character loads
   - 3-second delay mitigates but doesn't guarantee

2. **Frame Rate**:
   - Locked to 30 FPS
   - Matches BlendAnim training data
   - Browser may drop frames on slower devices

3. **Memory**:
   - Full sequence loaded in memory
   - 189 frames × J joints × 3 coordinates
   - ~60KB for typical sequence

## Future Enhancements

### Planned Features
- [ ] Variable playback speed (0.5x - 2x)
- [ ] Frame-by-frame step (← → arrow keys)
- [ ] Export rendered animation as video
- [ ] Real-time blend parameter adjustment
- [ ] Multiple sequence comparison view

### Research Extensions
- [ ] Gemini embeddings integration when available
- [ ] Advanced similarity metrics visualization
- [ ] Interactive blend zone editing
- [ ] Motion clustering and search

## References

1. **BlendAnim Paper**: https://arxiv.org/html/2508.18525v1
   - Section 3.2: Temporal conditioning (smoothstep)
   - Section 4: Quality metrics
   - Section 5: Multi-motion blending

2. **Three.js Documentation**:
   - AnimationMixer: https://threejs.org/docs/#api/en/animation/AnimationMixer
   - VectorKeyframeTrack: https://threejs.org/docs/#api/en/animation/tracks/VectorKeyframeTrack

3. **Implementation Files**:
   - Frontend: `/src/kinetic_ledger/ui/index-new.html`
   - Backend: `/src/kinetic_ledger/services/blendanim_service.py`
   - Server: `/src/kinetic_ledger/api/server.py`
