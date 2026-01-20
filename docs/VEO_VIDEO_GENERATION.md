# Veo Video Generation Guide

> **Status**: ✅ Production Ready | **Version**: 1.0 | **Last Updated**: January 2026

This document explains the video generation pipeline using Google's Veo 3.1, Nano Banana, and Imagen models to transform blended FBX motion files into high-quality video content.

---

## Overview

**Kinetic Ledger** now integrates Google's latest AI models to generate cinematic videos from motion-blended FBX files:

- **Veo 3.1**: State-of-the-art video generation (720p/1080p/4K, 4-8 seconds, native audio)
- **Nano Banana**: Advanced reasoning model (gemini-2.0-flash-thinking-exp) with <thinking> extraction
- **Imagen**: Keyframe generation (gemini-2.5-flash-image) from text descriptions
- **Gemini Files API**: FBX file upload and processing (up to 2GB, 48-hour retention)

---

## Architecture

### Hybrid AI Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                    Motion Blend Generation                       │
│                  (blendanim academic metrics)                    │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│              Step 1: Upload to Gemini Files API                  │
│  • Upload motion_a.fbx (capoeira) and motion_b.fbx (breakdance) │
│  • 2GB limit per file, 48-hour retention                        │
│  • Returns file URIs for AI processing                          │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│         Step 2: Nano Banana Metadata Extraction                  │
│  • gemini-2.0-flash-thinking-exp analyzes FBX files             │
│  • Extracts <thinking> blocks showing reasoning process         │
│  • Generates cinematic description of blended motion            │
│  • Example: "A dynamic fusion of capoeira's fluid..."           │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│            Step 3: Imagen First Frame Generation                 │
│  • gemini-2.5-flash-image creates keyframe image                │
│  • Based on cinematic description from Nano Banana              │
│  • Provides visual anchor for Veo video synthesis               │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                Step 4: Veo 3.1 Video Synthesis                   │
│  • veo-3.1-generate-preview model                                │
│  • Input: first frame image + cinematic prompt                   │
│  • Output: 4-8 second video (720p/1080p/4K)                     │
│  • Async operation: 11s-6min generation time                    │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│              Step 5: Client-Side Video Display                   │
│  • EventSource (SSE) monitors workflow progress                 │
│  • Poll /api/video/status/{operation_name} every 5s             │
│  • Modal auto-displays when status === 'completed'              │
│  • Download/share/replay options available                      │
└─────────────────────────────────────────────────────────────────┘
```

---

## Model Capabilities

### Veo 3.1 (veo-3.1-generate-preview)

**Capabilities**:
- Video generation from text prompts and/or reference images
- Cinematic camera movements, lighting, and effects
- Native audio synthesis (background ambience, motion sounds)
- High-quality output at multiple resolutions

**Specifications**:
| Parameter | Options | Default |
|-----------|---------|---------|
| Resolution | 720p, 1080p, 4K | 1080p |
| Aspect Ratio | 16:9, 9:16, 1:1 | 16:9 |
| Duration | 4-8 seconds | 8s |
| Generation Time | 11s-6min | ~2min (1080p) |

**Limitations**:
- Cannot directly parse FBX binary format (requires text prompt + optional image)
- Preview model: free during beta, may have usage limits
- Async operation: requires polling for completion

### Nano Banana (gemini-2.0-flash-thinking-exp)

**Capabilities**:
- Advanced reasoning with <thinking> tag extraction
- Step-by-step problem-solving transparency
- Enhanced context understanding for complex prompts
- Improved accuracy for creative tasks

**Example Output**:
```xml
<thinking>
Analyzing the two FBX files:
1. Capoeira.fbx shows fluid, circular movements with low center of gravity
2. Breakdance.fbx exhibits explosive power moves and freeze positions
3. Blended motion should transition smoothly between flowing and sharp dynamics
4. Cinematic description should emphasize the contrast and harmony
</thinking>

A dynamic fusion of capoeira's fluid ginga with breakdance's explosive power.
The character flows through sweeping circular movements before suddenly
dropping into a freeze, creating a mesmerizing dance of grace and strength.
```

**Configuration**:
```python
from src.kinetic_ledger.services.gemini_analyzer import GeminiClient

# Enable reasoning mode
client = GeminiClient(enable_reasoning=True)

# Analyze with thinking extraction
result = client.analyze_motion_preview(preview_uri)
print(result.reasoning)  # Contains <thinking> blocks
print(result.analysis)   # Contains final answer
```

### Imagen (gemini-2.5-flash-image)

**Capabilities**:
- High-quality image generation from text prompts
- Keyframe synthesis for video anchoring
- Style consistency across generations
- Fast inference (3-8 seconds)

**Usage in Pipeline**:
```python
# Generate first frame for Veo
first_frame_prompt = f"A character performing {cinematic_description}, professional studio lighting, 3D render"
first_frame_uri = generate_image_with_imagen(first_frame_prompt)
```

---

## API Reference

### Generate Blend with Video

**Endpoint**: `POST /api/motions/blend/blendanim`

**Request**:
```json
{
  "prompt": "blend capoeira and breakdance 60/40, ultra quality",
  "motion_a": "capoeira",
  "motion_b": "breakdance"
}
```

**Response**:
```json
{
  "status": "success",
  "blend_id": "blend-abc123",
  "metrics": {
    "coverage": 0.85,
    "local_diversity": 0.72,
    "global_diversity": 0.68,
    "l2_velocity": 0.067,
    "l2_acceleration": 0.023,
    "quality_tier": "ultra"
  },
  "cost_usdc": 0.849,
  "fbx_paths": {
    "motion_a": "data/mixamo_anims/fbx/Capoeira.fbx",
    "motion_b": "data/mixamo_anims/fbx/Breakdance.fbx"
  },
  "video": {
    "status": "processing",
    "operation_name": "projects/123456/operations/veo-abc123",
    "message": "Video generation started with Veo 3.1"
  }
}
```

### Poll Video Status

**Endpoint**: `GET /api/video/status/{operation_name}`

**Response (Processing)**:
```json
{
  "status": "processing",
  "operation_name": "projects/123456/operations/veo-abc123",
  "progress": "Veo is synthesizing video... (estimated 2-3 minutes)"
}
```

**Response (Completed)**:
```json
{
  "status": "completed",
  "operation_name": "projects/123456/operations/veo-abc123",
  "video_uri": "https://generativelanguage.googleapis.com/v1beta/files/video-abc123",
  "duration": "8s",
  "resolution": "1080p",
  "metadata": {
    "aspect_ratio": "16:9",
    "model": "veo-3.1-generate-preview",
    "created_at": "2026-01-15T10:30:00Z"
  }
}
```

**Response (Failed)**:
```json
{
  "status": "failed",
  "operation_name": "projects/123456/operations/veo-abc123",
  "error": "Video generation timeout after 6 minutes"
}
```

---

## Client-Side Integration

### Feature Flags

Enable video generation via URL parameters:

```bash
# Enable Veo video generation
http://localhost:8000/?veo_enabled=true

# Enable with debug logging
http://localhost:8000/?veo_enabled=true&debug=true

# Timeline-only mode with video
http://localhost:8000/?timeline_mode=true&veo_enabled=true
```

### JavaScript Implementation

**EventSource Monitoring**:
```javascript
// Listen to blend workflow stream
const eventSource = new EventSource(`/api/motions/blend/blendanim/stream`);

eventSource.addEventListener('message', (event) => {
  const data = JSON.parse(event.data);
  
  if (data.status === 'completed' && data.video) {
    // Show video status button
    this.showVideoButton(data.video);
    
    // Start polling if video is processing
    if (data.video.status === 'processing') {
      this.pollVideoStatus(data.video.operation_name);
    }
  }
});
```

**Video Status Polling**:
```javascript
async pollVideoStatus(operationName, onUpdate) {
  const maxAttempts = 60;  // 5 minutes max
  const interval = 5000;   // 5 seconds
  
  for (let i = 0; i < maxAttempts; i++) {
    const response = await fetch(`/api/video/status/${operationName}`);
    const data = await response.json();
    
    onUpdate(data);  // Update UI
    
    if (data.status === 'completed') {
      this.showVideoModal(data);
      break;
    }
    
    if (data.status === 'failed') {
      console.error('Video generation failed:', data.error);
      break;
    }
    
    await new Promise(resolve => setTimeout(resolve, interval));
  }
}
```

**Video Modal Display**:
```javascript
showVideoModal(videoData) {
  const modal = document.getElementById('videoModal');
  const player = document.getElementById('videoPlayer');
  
  if (videoData.status === 'completed') {
    player.src = videoData.video_uri;
    player.load();
    
    // Update metadata
    document.getElementById('videoDuration').textContent = videoData.duration;
    document.getElementById('videoResolution').textContent = videoData.resolution;
  }
  
  modal.classList.add('active');
}
```

---

## Three.js FBXLoader Integration

### Client-Side FBX Loading

Load and visualize FBX files directly in the browser:

```javascript
async loadFBXFile(url) {
  return new Promise((resolve, reject) => {
    const loader = new THREE.FBXLoader();
    
    loader.load(url, (fbx) => {
      // Extract animation data
      const animations = fbx.animations;
      const keyframes = this.extractKeyframes(animations[0]);
      
      resolve({
        fbx: fbx,
        keyframes: keyframes,
        duration: animations[0].duration,
        fps: 30
      });
    }, undefined, reject);
  });
}

extractKeyframes(animation) {
  const keyframes = [];
  const tracks = animation.tracks;
  
  tracks.forEach(track => {
    if (track.name.includes('position')) {
      for (let i = 0; i < track.times.length; i++) {
        keyframes.push({
          time: track.times[i],
          position: track.values.slice(i * 3, i * 3 + 3)
        });
      }
    }
  });
  
  return keyframes;
}
```

### Rendering FBX in Scene

```javascript
// Add FBX to scene
this.scene.add(fbxData.fbx);

// Animate
const mixer = new THREE.AnimationMixer(fbxData.fbx);
const action = mixer.clipAction(fbxData.fbx.animations[0]);
action.play();

// Update loop
function animate() {
  requestAnimationFrame(animate);
  
  const delta = this.clock.getDelta();
  mixer.update(delta);
  
  this.renderer.render(this.scene, this.camera);
}
```

---

## Configuration

### Environment Variables

```bash
# Required for all Gemini features
export GEMINI_API_KEY="your-gemini-api-key"

# Optional: customize video settings
export VEO_RESOLUTION="1080p"        # Options: 720p, 1080p, 4K
export VEO_ASPECT_RATIO="16:9"       # Options: 16:9, 9:16, 1:1
export VEO_DURATION="8"              # Options: 4-8 seconds
```

### Service Configuration

**veo_video_service.py**:
```python
from src.kinetic_ledger.services.veo_video_service import VeoVideoService

# Get singleton instance
veo_service = get_veo_service()

# Generate video with custom settings
video_result = await veo_service.generate_video_from_fbx(
    fbx_path_a="data/mixamo_anims/fbx/Capoeira.fbx",
    fbx_path_b="data/mixamo_anims/fbx/Breakdance.fbx",
    blend_description="Blend of capoeira and breakdance",
    resolution="1080p",
    aspect_ratio="16:9",
    duration=8
)

print(f"Operation: {video_result['operation_name']}")
print(f"Status: {video_result['status']}")
```

---

## Performance & Costs

### Generation Times

| Resolution | Typical Time | Max Time |
|------------|--------------|----------|
| 720p       | 11-30s       | 2min     |
| 1080p      | 30s-2min     | 4min     |
| 4K         | 2-4min       | 6min     |

### API Costs (Preview Model)

| Operation | Cost | Notes |
|-----------|------|-------|
| Veo video generation | Free | Preview model during beta |
| Nano Banana reasoning | Free | Standard Gemini pricing may apply |
| Imagen keyframe | Free | Standard Gemini pricing may apply |
| Files API upload | Free | 2GB limit, 48-hour retention |

**Note**: Pricing may change when models exit preview/beta. Check [Google AI Pricing](https://ai.google.dev/pricing) for latest rates.

### File Limits

| Resource | Limit | Cleanup |
|----------|-------|---------|
| FBX file size | 2GB per file | Auto-deleted after 48 hours |
| Generated video | ~50MB (1080p, 8s) | Stored in Files API |
| Concurrent operations | 10 (recommended) | Rate limiting may apply |

---

## Troubleshooting

### Video Generation Fails

**Problem**: Video status returns `failed` after polling

**Solutions**:
1. Check GEMINI_API_KEY is valid and has Veo access
2. Verify FBX files are under 2GB
3. Ensure FBX files have valid skeletal animation data
4. Check operation timeout (6 minutes max)

**Debug Commands**:
```bash
# Test Gemini API connectivity
python -c "import google.generativeai as genai; genai.configure(api_key='YOUR_KEY'); print(genai.list_models())"

# Verify FBX file integrity
python -c "from FbxCommon import *; sdk = FbxManager.Create(); print('FBX SDK OK')"
```

### Video Not Displaying in Modal

**Problem**: Modal shows loading spinner indefinitely

**Solutions**:
1. Check browser console for CORS errors
2. Verify video_uri is accessible (may require authentication)
3. Ensure video format is supported (MP4/H.264)
4. Test video URL directly in browser

**Debug Code**:
```javascript
// Test video URL
fetch(videoData.video_uri)
  .then(response => console.log('Video accessible:', response.ok))
  .catch(error => console.error('Video error:', error));
```

### Polling Timeout

**Problem**: Video status polling stops after 60 attempts (5 minutes)

**Solutions**:
1. Increase `maxAttempts` in `pollVideoStatus()` method
2. Check Veo service logs for generation issues
3. Use lower resolution (720p) for faster generation
4. Verify FBX complexity (high poly count may slow generation)

---

## Best Practices

### 1. Optimize FBX Files

- Keep file size under 500MB for faster uploads
- Use standard skeleton hierarchies (Mixamo, Unreal, Unity)
- Bake animations to reduce complexity
- Remove unnecessary mesh data (keep only skeleton + animation)

### 2. Craft Effective Prompts

**Good Prompt**:
```
A dynamic fusion of capoeira's fluid ginga with breakdance's explosive power moves.
The character transitions smoothly from flowing circular movements to sharp freeze
positions, creating a mesmerizing dance. Professional studio lighting, 3D render.
```

**Avoid**:
- Vague descriptions: "character moving"
- Overly technical terms: "quaternion interpolation at frame 120"
- Conflicting instructions: "slow and fast simultaneously"

### 3. Handle Async Operations

Always implement:
- Loading states in UI
- Timeout handling (6 minutes max)
- Error messages for failed generations
- Retry logic for network errors

### 4. Cache Generated Videos

```python
# Cache completed videos to avoid regeneration
video_cache = {}

if blend_id in video_cache:
    return video_cache[blend_id]
else:
    video_result = await veo_service.generate_video_from_fbx(...)
    video_cache[blend_id] = video_result
    return video_result
```

### 5. Feature Flag Testing

Test new features with URL parameters before enabling globally:

```bash
# A/B test: 50% of users see Veo videos
if (Math.random() < 0.5 || FEATURE_FLAGS.VEO_ENABLED) {
  generateVideo();
}
```

---

## Examples

### End-to-End Video Generation

```python
from src.kinetic_ledger.services.veo_video_service import get_veo_service
import asyncio

async def generate_blend_video():
    # 1. Get service instance
    veo = get_veo_service()
    
    # 2. Generate video from FBX files
    result = await veo.generate_video_from_fbx(
        fbx_path_a="data/mixamo_anims/fbx/Capoeira.fbx",
        fbx_path_b="data/mixamo_anims/fbx/Breakdance.fbx",
        blend_description="Capoeira-breakdance fusion with fluid transitions",
        resolution="1080p",
        aspect_ratio="16:9",
        duration=8
    )
    
    print(f"Video generation started: {result['operation_name']}")
    
    # 3. Poll for completion
    max_wait = 300  # 5 minutes
    for i in range(max_wait // 5):
        status = await veo.poll_video_status(result['operation_name'])
        
        if status['status'] == 'completed':
            print(f"Video ready: {status['video_uri']}")
            print(f"Duration: {status['duration']}")
            print(f"Resolution: {status['resolution']}")
            break
        
        await asyncio.sleep(5)
    
    return status

# Run
asyncio.run(generate_blend_video())
```

### Client-Side Full Integration

```javascript
class VideoBlendUI {
  async generateAndShowVideo(prompt, motionA, motionB) {
    // 1. Request blend with video generation
    const response = await fetch('/api/motions/blend/blendanim', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({prompt, motion_a: motionA, motion_b: motionB})
    });
    
    const data = await response.json();
    
    // 2. Show blend metrics immediately
    this.displayMetrics(data.metrics);
    
    // 3. Monitor video generation
    if (data.video && data.video.operation_name) {
      this.showVideoButton('processing');
      
      // 4. Poll for completion
      await this.pollVideoStatus(data.video.operation_name, (status) => {
        if (status.status === 'completed') {
          this.showVideoModal(status);
        } else if (status.status === 'failed') {
          this.showError('Video generation failed');
        }
      });
    }
  }
}
```

---

## Future Enhancements

### Planned Features

1. **Multi-Resolution Support**: Generate multiple resolutions simultaneously
2. **Custom Thumbnails**: User-selectable keyframes for video preview
3. **Audio Customization**: Background music selection and volume control
4. **Export Formats**: WebM, GIF, and animated PNG support
5. **Batch Generation**: Queue multiple blends for video generation
6. **Social Sharing**: Direct integration with YouTube, Twitter, Discord

### Research Directions

1. **Longer Videos**: Extend duration beyond 8 seconds using scene composition
2. **Interactive Videos**: Click-to-branch narrative structures
3. **Real-time Generation**: Sub-second video synthesis for game engines
4. **Style Transfer**: Apply artistic styles to generated videos

---

## References

- [Veo 3.1 Documentation](https://ai.google.dev/gemini-api/docs/veo)
- [Gemini Files API](https://ai.google.dev/gemini-api/docs/vision)
- [Three.js FBXLoader](https://threejs.org/docs/#examples/en/loaders/FBXLoader)
- [blendanim Repository](https://github.com/RydlrCS/blendanim)
- [BLEND_METRICS.md](BLEND_METRICS.md) - Academic metric specifications

---

**Built with**: Veo 3.1 • Nano Banana • Imagen • Gemini Files API • Three.js • FastAPI

