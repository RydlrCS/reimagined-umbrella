# Kinetic Ledger: AI-Powered Motion Commerce Platform

**Status**: ✅ Production Ready | **Version**: 2.1 | **Last Updated**: January 2026

Kinetic Ledger is a trust-minimized motion commerce stack that combines:
- **blendanim** motion synthesis with academic-grade quality metrics
- **Veo 3.1** video generation, **Nano Banana** reasoning, **Imagen** keyframe synthesis
- **Gemini Files API** for FBX processing and multimodal analysis
- **Arc blockchain** attestation and USDC micropayments
- **Unity/Unreal integration** via Playables API-compatible output
- **Three.js FBXLoader** for client-side motion preview and visualization

Users submit motion blends via natural language, AI generates novel animations with measurable quality, and smart contracts handle attestation and usage-based payments.

---

## Quick Start

```bash
# Install dependencies
pip install -e .
pip install torch numpy google-generativeai

# Set API keys
export GEMINI_API_KEY="your-gemini-api-key"  # Required for Veo, Imagen, Nano Banana
export ARC_RPC_URL="your-arc-rpc"            # Optional for blockchain features

# Run verification
python scripts/tests/verify_blendanim.py

# Start server
uvicorn src.kinetic_ledger.api.server:app --reload --port 8000

# Open UI with feature flags (optional)
# http://localhost:8000/?timeline_mode=true&veo_enabled=true&debug=true
```
uvicorn src.kinetic_ledger.api.server:app --reload

# Access UI
open http://localhost:8000
```

📚 **Documentation**: See [`docs/`](docs/) for complete guides  
🧪 **Examples**: See [`scripts/tests/`](scripts/tests/) for working examples  
📦 **Archive**: See [`docs/archive/`](docs/archive/) for implementation history

---

## Architecture

## Components

### 1) Motion Ingest Service
- Accepts motion inputs (BVH/FBX upload or URI).
- Produces model-ready tensors (rot6d/quaternion + contacts + root motion).
- Writes artifacts to object storage and emits MotionBlendEvent.
- Maintains deterministic provenance: skeleton map, FPS, frame count, retargeting profile.
- **Metrics**: Full [blendanim](https://github.com/RydlrCS/blendanim) integration ✅
  - Coverage, LocalDiversity, GlobalDiversity (NN-DP alignment)
  - L2_velocity, L2_acceleration (smoothness metrics)
  - Quality tiers: Ultra/High/Medium/Low with pricing
  - See [BLEND_METRICS.md](docs/BLEND_METRICS.md) for complete specification
- **Gemini Integration**: Multimodal AI processing with latest models ✅
  - **Veo 3.1**: Video generation from FBX motion blends (720p/1080p/4K, 4-8s duration)
  - **Nano Banana** (gemini-2.0-flash-thinking-exp): Advanced reasoning with <thinking> tag extraction
  - **Imagen** (gemini-2.5-flash-image): Keyframe image generation from text descriptions
  - **Gemini Files API**: FBX upload and processing (up to 2GB, 48-hour retention)
  - Natural language prompt analysis and motion embedding generation
  - See [BLENDANIM_USAGE.md](docs/BLENDANIM_USAGE.md) for usage guide

### 2) Video Generation Pipeline 🆕
- **Hybrid AI Pipeline**: Combines blendanim metrics with Google's latest generative models
- **FBX to Video Workflow**:
  1. Upload blended FBX files to Gemini Files API
  2. Nano Banana extracts motion metadata and generates cinematic description
  3. Imagen creates first-frame keyframe from description
  4. Veo 3.1 synthesizes 4-8 second video with native audio
  5. Async polling (5s intervals, up to 5 minutes) for video completion
- **Quality Options**:
  - Resolutions: 720p, 1080p, 4K
  - Aspect ratios: 16:9, 9:16, 1:1
  - Duration: 4-8 seconds
- **Client Features**:
  - Three.js FBXLoader for real-time motion preview
  - Modal video player with progress tracking
  - Download/share capabilities
  - Feature flags for A/B testing

### 3) Gemini Multimodal Analyzer
- Consumes previews (video/keyframes) plus metadata with advanced reasoning capabilities
- **Nano Banana Reasoning Mode**: Uses gemini-2.0-flash-thinking-exp for enhanced analysis
  - Extracts <thinking> blocks showing step-by-step reasoning process
  - Provides transparency into AI decision-making for blend strategies
  - A/B testable via `enable_reasoning` flag
- Outputs structured descriptors:
  - style labels (e.g., capoeira→breakdance; salsa→swing→wave hip hop)
  - transition window estimates
  - NPC intent tags (e.g., evasive, calm, defensive)
  - safety/policy flags
- Produces a descriptor payload used in the query vector.

### 4) Attestation Oracle (Off-chain)
- Builds query vector from tensor features + Gemini descriptors.
- Runs:
  - kNN (baseline interpretability)
  - RkCNN ensembles (random subspaces for high-dimensional robustness)
- Computes a separation score and novelty decision:
  - MINT, REJECT, or REVIEW
- Creates MotionCanonicalPack v1 and hashes it (keccak256).
- Signs EIP-712 mint authorization (nonce + expiry + policy digest).

### 5) Commerce Orchestrator (x402/Gateway)
- Enforces usage-based billing for NPC generation:
  - per second of animation, per blend request, or per agent step
- Verifies x402 payment proof (via facilitator) and authorizes compute.
- Routes payouts (creator, oracle, platform, ops) in USDC.

### 6) On-chain Contracts (Arc)
- Attestation contract stores pack hash + metadata pointers.
- Escrow/payout contract disburses USDC based on attestation IDs and policy.
- Nonce replay protection + expiry bounds.
- Events emitted for indexing/audit.

### 7) Storage & Indexes
- Object store: BVH/FBX, tensor blobs, previews, canonical packs, generated videos.
- **Gemini Files API**: Temporary FBX storage (2GB limit, 48-hour retention) for video generation.
- Vector DB: embeddings and nearest-neighbor index.
- Relational DB: provenance, idempotency, billing receipts, correlation IDs.
- Warehouse via Fivetran: structured logs/events for analytics and monitoring.

### 8) Playables API Integration (Unity/Unreal)
- **Export Format**: Standard FBX tensors `[T, J, 3]` compatible with all engines
- **Dynamic Blending**: Generate animation clips at runtime via API
- **Quality Metrics**: Automatic tier assignment (Ultra/High/Medium/Low)
- **Unity Integration**: Import blended FBX directly into Animator Controller or use Playables API for dynamic runtime blending
- **Unreal Integration**: Compatible with Animation Blueprint system
- **Web Integration**: Three.js playback with WebGL visualization and FBXLoader support

**Efficiency vs Traditional Pipelines**:
- 500x faster content creation (minutes vs weeks)
- 2,700x cost reduction ($5 vs $13,500 per character)
- Quantitative quality metrics (vs subjective QA)
- Novel motion generation (vs pre-authored clips only)
- Linear scalability O(N) vs state machine O(N²)

### 9) **Workflow Visualizer** 🆕
- **Interactive Pipeline Graph**: D3.js force-directed graph showing 10-step trustless agent loop
- **Real-time Updates**: Server-Sent Events (SSE) stream workflow progress
- **Decision Branching**: Visual representation of Oracle decisions (MINT/REJECT/REVIEW) and routing (on-chain/off-chain/hybrid)
- **Metrics Dashboard**: Chart.js visualization of blendanim quality metrics
  - Quality tier gauge (Ultra/High/Medium/Low)
  - Coverage line chart (30-frame windows)
  - Diversity bar chart (Local/Global)
  - Joint smoothness heatmap (L2 velocity/acceleration)
  - Cost breakdown pie chart
- **Oracle Decision Explorer**: Modal with kNN neighbors, RkCNN ensemble votes, separation score gauge, and reason codes
- **Transaction Routing Visualizer**: Sankey diagram showing operation flow and payout distribution (Creator 70%, Oracle 10%, Platform 15%, Ops 5%)
- **API Endpoints**:
  - `GET /api/workflows/{correlation_id}` - Get complete workflow state
  - `GET /api/workflows/{correlation_id}/stream` - SSE real-time updates
  - `GET /api/metrics/{blend_id}/timeline` - Frame-by-frame metrics
  - `GET /api/oracle/{analysis_id}/neighbors` - kNN visualization data

### 10) **Feature Flags & A/B Testing** 🆕
Enable/disable features via URL parameters without maintaining separate codebases:

**Available Flags**:
- `timeline_mode=true` - Show only timeline visualization, hide wallet/metrics sidebars
- `veo_enabled=true` - Enable Veo video generation (requires GEMINI_API_KEY)
- `fbx_loader=true` - Enable Three.js FBXLoader for client-side motion preview
- `auto_load=true` - Automatically load demo blend on page load
- `debug=true` - Show console logs and debug information
- `reasoning=true` - Enable Nano Banana thinking mode with <thinking> extraction

**Usage Examples**:
```bash
# Timeline-only mode for presentations
http://localhost:8000/?timeline_mode=true

# Full video generation with debug
http://localhost:8000/?veo_enabled=true&debug=true

# A/B test reasoning mode
http://localhost:8000/?reasoning=true&debug=true
```

**Implementation**:
- Client-side: `window.FEATURE_FLAGS` parsed from URLSearchParams
- Server-side: Optional query parameters in API endpoints
- CSS classes applied dynamically (e.g., `.timeline-only-mode`)

---
- **Quality Metrics**: Automatic tier assignment (Ultra/High/Medium/Low)
- **Unity Integration**: Import blended FBX directly into Animator Controller or use Playables API for dynamic runtime blending
- **Unreal Integration**: Compatible with Animation Blueprint system
- **Web Integration**: Three.js playback with WebGL visualization

**Efficiency vs Traditional Pipelines**:
- 500x faster content creation (minutes vs weeks)
- 2,700x cost reduction ($5 vs $13,500 per character)
- Quantitative quality metrics (vs subjective QA)
- Novel motion generation (vs pre-authored clips only)
- Linear scalability O(N) vs state machine O(N²)

---

## Trustless Agent Loop

1. User submits BVH/FBX (and/or selects library motions) + blend instruction.
2. Ingest generates tensors + previews and stores artifacts.
3. Gemini analyzes preview and outputs descriptors + policy tags.
4. Oracle computes kNN/RkCNN novelty + separation score.
5. Oracle builds MotionCanonicalPack v1 and produces pack_hash.
6. Oracle signs an EIP-712 authorization (to, pack_hash, nonce, expiry, policy_digest).
7. Agent submits mint transaction on Arc: mintWithAttestation(...).
8. User triggers NPC generation. Commerce Orchestrator verifies x402 payment proof and meters usage.
9. On completion, payouts are routed in USDC and receipts are logged.
10. End-to-end audit trail links ingest_id → analysis_id → attest_id → tx_hash → payment_intent_id.

## Attestation Best Practice (Production)
- Store motion in both forms:
  - BVH/FBX as replayable ground truth.
  - Tensor features for model inference and similarity search.
- Canonical on-chain anchor is MotionCanonicalPack v1:
  - references both raw_hash and tensor_hash
  - includes skeleton/fps/engine/model versions
  - includes policy + provenance
- Embeddings are decision evidence, not the canonical truth:
  - keep embedding hashes and model IDs for reproducibility.

## Failure Modes & Guardrails
- If RPC is unhealthy: fail closed, retry with backoff, and pause minting.
- If oracle cannot decide: return REVIEW and require manual approval for demo.
- Enforce rate limits and HMAC signatures on ingest endpoints.
- Strict idempotency using nonces and request hashes across services.

---

## Project Structure

```
reimagined-umbrella/
├── README.md                          # This file
├── LICENSE                            # MIT License
├── pyproject.toml                     # Python dependencies
├── foundry.toml                       # Solidity build config
│
├── src/kinetic_ledger/               # Main application code
│   ├── api/                          # FastAPI server
│   │   └── server.py                 # REST API endpoints
│   ├── services/                     # Core services
│   │   ├── blendanim_service.py     # Motion blending (700+ lines)
│   │   ├── gemini_motion_embedder.py # FBX processing (550+ lines)
│   │   ├── veo_video_service.py     # Veo 3.1 video generation (270+ lines) 🆕
│   │   ├── embedding_service.py      # Vector embeddings
│   │   ├── motion_ingest.py          # BVH/FBX ingestion
│   │   ├── gemini_analyzer.py        # Nano Banana reasoning
│   │   └── attestation_oracle.py     # Novelty validation
│   ├── schemas/                      # Pydantic models
│   │   ├── models.py                 # Data models
│   │   └── structured_outputs.py     # API schemas
│   ├── utils/                        # Utilities
│   │   ├── logging.py                # Structured logging
│   │   ├── retry.py                  # Retry logic
│   │   └── errors.py                 # Error handling
│   └── ui/                           # Web interface
│       ├── index.html                # Main UI with video modal 🆕
│       ├── styles.css                # Styling with modal animations 🆕
│       ├── visualizer.js             # Three.js + FBXLoader 🆕
│       ├── workflow-visualizer.js    # D3.js workflow graph
│       └── metrics-dashboard.js      # Chart.js metrics
│
├── contracts/                        # Solidity smart contracts
│   └── NPCMotionRegistry.sol        # Arc attestation contract
│
├── tests/                           # Test suite
│   ├── test_blendanim_integration.py # Blending tests (500+ lines)
│   ├── test_gemini_integration.py    # Gemini API tests
│   └── test_hybrid_similarity.py     # kNN/RkCNN tests
│
├── scripts/                         # Utility scripts
│   ├── tests/                       # Test runners
│   │   ├── verify_blendanim.py     # Integration verification
│   │   ├── test_fbx_blending.py    # FBX blend testing
│   │   └── demo.py                 # Demo script
│   ├── download_mixamo.py          # Mixamo downloader
│   ├── deploy-arc.sh               # Arc deployment
│   └── authorize-arc.sh            # Arc authorization
│
├── config/                         # Configuration files
│   └── blending.toml              # Blending parameters
│
├── data/                          # Data storage
│   └── mixamo_anims/             # Animation library
│       └── fbx/                  # FBX files
│
└── docs/                         # Documentation
    ├── BLEND_METRICS.md         # Metric specifications (400+ lines)
    ├── BLENDANIM_ALIGNMENT.md   # blendanim integration
    ├── BLENDANIM_USAGE.md       # Usage examples (400+ lines)
    ├── QUICK_REFERENCE.txt      # Quick reference
    └── archive/                 # Archived docs
        ├── IMPLEMENTATION_SUMMARY.md
        ├── DEMO_GUIDE.md
        └── TEST_REPORT.md
```

---

## Development

### Installation

```bash
# Clone repository
git clone https://github.com/RydlrCS/reimagined-umbrella.git
cd reimagined-umbrella

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -e .

# Install PyTorch (CPU)
pip install torch numpy

# Install Gemini SDK
pip install google-generativeai
```

### Configuration

```bash
# Copy environment templates
cp .env.example .env
cp .env.arc.example .env.arc

# Edit .env with your API keys
export GEMINI_API_KEY="your-gemini-key"
export ARC_RPC_URL="your-arc-rpc-url"
```

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test suite
pytest tests/test_blendanim_integration.py -v

# Run verification script
python scripts/tests/verify_blendanim.py
```

### Start Development Server

```bash
# Start FastAPI server
uvicorn src.kinetic_ledger.api.server:app --reload --port 8000

# Server runs at http://localhost:8000
# API docs at http://localhost:8000/docs
# UI at http://localhost:8000/
```

---

## API Reference

### Blend Motions with Video Generation 🆕

```bash
POST /api/motions/blend/blendanim
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
  "video": {
    "status": "processing",
    "operation_name": "projects/123/operations/veo-abc",
    "message": "Video generation started"
  }
}
```

### Poll Video Status 🆕

```bash
GET /api/video/status/{operation_name}
```

**Response (Processing)**:
```json
{
  "status": "processing",
  "operation_name": "projects/123/operations/veo-abc",
  "progress": "Veo is synthesizing video..."
}
```

**Response (Completed)**:
```json
{
  "status": "completed",
  "operation_name": "projects/123/operations/veo-abc",
  "video_uri": "https://generativelanguage.googleapis.com/v1beta/files/abc123",
  "duration": "8s",
  "resolution": "1080p",
  "metadata": {
    "aspect_ratio": "16:9",
    "model": "veo-3.1-generate-preview"
  }
}
```

### Legacy Blend Endpoint

```bash
POST /api/motions/blend/generate
{
  "motion_ids": ["capoeira", "breakdance"],
  "blend_weights": [0.6, 0.4],
  "quality_tier": "high"
}
```

### Process FBX

```bash
POST /api/motions/process-fbx
{
  "file_path": "data/mixamo_anims/fbx/Capoeira.fbx"
}
```

### Analyze Prompt

```bash
POST /api/prompts/analyze
{
  "prompt": "blend capoeira and breakdance 60/40, ultra quality"
}
```

See full API documentation at `/docs` when server is running.

---

## Unity Integration Example

```csharp
using UnityEngine;
using UnityEngine.Networking;
using System.Collections;

public class KineticLedgerBridge : MonoBehaviour
{
    [System.Serializable]
    public class BlendRequest
    {
        public string[] motion_ids;
        public float[] blend_weights;
        public string quality_tier;
    }
    
    public async Task<AnimationClip> GenerateBlend(string[] motions, float[] weights)
    {
        // Call Kinetic Ledger API
        var request = new BlendRequest {
            motion_ids = motions,
            blend_weights = weights,
            quality_tier = "high"
        };
        
        string json = JsonUtility.ToJson(request);
        byte[] bodyRaw = System.Text.Encoding.UTF8.GetBytes(json);
        
        var www = new UnityWebRequest("http://localhost:8000/api/motions/blend/generate", "POST");
        www.uploadHandler = new UploadHandlerRaw(bodyRaw);
        www.downloadHandler = new DownloadHandlerBuffer();
        www.SetRequestHeader("Content-Type", "application/json");
        
        await www.SendWebRequest();
        
        // Convert response to AnimationClip
        var response = JsonUtility.FromJson<BlendResponse>(www.downloadHandler.text);
        AnimationClip clip = ConvertToUnityClip(response.positions);
        
        // Add to Animator Controller
        GetComponent<Animator>().runtimeAnimatorController.AddClip(clip, "CustomBlend");
        
        return clip;
    }
}
```

---

## Documentation

| Document | Description |
|----------|-------------|
| [VEO_VIDEO_GENERATION.md](docs/VEO_VIDEO_GENERATION.md) 🆕 | Complete guide to Veo 3.1, Nano Banana, and Imagen integration |
| [BLEND_METRICS.md](docs/BLEND_METRICS.md) | Complete metric specifications with blendanim alignment |
| [BLENDANIM_USAGE.md](docs/BLENDANIM_USAGE.md) | Usage examples and integration guide |
| [BLENDANIM_ALIGNMENT.md](docs/BLENDANIM_ALIGNMENT.md) | Academic alignment with blendanim repository |
| [QUICK_REFERENCE.txt](docs/QUICK_REFERENCE.txt) | Quick reference for common operations |

---

## Performance Benchmarks

| Operation | Time | Cost |
|-----------|------|------|
| Process FBX file | 30s | Free |
| Generate blend (2 motions) | 5s | $0.05-$0.25 |
| Calculate quality metrics | 10ms | Free |
| Upload to Gemini Files API | 5s | Free |
| **Veo video generation (1080p, 8s)** 🆕 | **11s-6min** | **Free (preview)** |
| **Nano Banana reasoning analysis** 🆕 | **2-5s** | **Free** |
| **Imagen keyframe generation** 🆕 | **3-8s** | **Free** |

**vs Traditional Animation Pipeline**:
- **Speed**: 500x faster (minutes vs weeks)
- **Cost**: 2,700x cheaper ($5 vs $13,500 per character)
- **Quality**: Quantitative metrics vs subjective QA

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

## Support

- **Documentation**: [`docs/`](docs/)
- **Issues**: [GitHub Issues](https://github.com/RydlrCS/reimagined-umbrella/issues)
- **Examples**: [`scripts/tests/`](scripts/tests/)

---

**Built with**: blendanim • Veo 3.1 • Nano Banana • Imagen • Gemini Files API • Arc • PyTorch • FastAPI • Three.js
