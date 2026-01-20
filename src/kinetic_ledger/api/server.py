"""
Kinetic Ledger Phase-2 API Server.

FastAPI server that exposes the trustless agent loop endpoints.
"""
import os
import logging
import json
import uuid
from pathlib import Path
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException, Header, status
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from ..schemas.models import (
    MotionAsset,
    MotionBlendRequest,
    GeminiAnalysis,
    SimilarityCheck,
    MotionCanonicalPack,
    MintAuthorization,
    UsageMeterEvent,
)
from ..services import (
    TrustlessAgentLoop,
    TrustlessAgentConfig,
    MotionUploadRequest,
)
from ..utils.logging import setup_logging, set_correlation_id
from ..utils.errors import (
    KineticLedgerError,
    NoveltyRejectionError,
    ManualReviewRequiredError,
)

# Setup logging
setup_logging("kinetic-ledger-api")
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="Kinetic Ledger Phase-2",
    description="Trust-minimized motion commerce stack",
    version="2.0.0",
)

# Add CORS middleware for UI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Global agent loop instance (in production: use dependency injection)
agent_loop: Optional[TrustlessAgentLoop] = None


def get_agent_loop() -> TrustlessAgentLoop:
    """Get or create agent loop instance."""
    global agent_loop
    
    if agent_loop is None:
        # Load config from environment
        config = TrustlessAgentConfig(
            circle_api_key=os.getenv("CIRCLE_API_KEY", "demo_key"),
            gemini_api_key=os.getenv("GEMINI_API_KEY"),
            novelty_threshold=float(os.getenv("NOVELTY_THRESHOLD", "0.42")),
            chain_id=int(os.getenv("CHAIN_ID", "1")),
            verifying_contract=os.getenv(
                "VERIFYING_CONTRACT",
                "0x0000000000000000000000000000000000000001"
            ),
            oracle_address=os.getenv(
                "ORACLE_ADDRESS",
                "0x0000000000000000000000000000000000000002"
            ),
            platform_address=os.getenv(
                "PLATFORM_ADDRESS",
                "0x0000000000000000000000000000000000000003"
            ),
            ops_address=os.getenv(
                "OPS_ADDRESS",
                "0x0000000000000000000000000000000000000004"
            ),
        )
        agent_loop = TrustlessAgentLoop(config)
    
    return agent_loop


@app.exception_handler(KineticLedgerError)
async def kinetic_ledger_error_handler(request, exc: KineticLedgerError):
    """Handle custom Kinetic Ledger errors."""
    status_code = status.HTTP_400_BAD_REQUEST
    
    # Map error types to status codes
    if isinstance(exc, NoveltyRejectionError):
        status_code = status.HTTP_422_UNPROCESSABLE_ENTITY
    elif isinstance(exc, ManualReviewRequiredError):
        status_code = status.HTTP_202_ACCEPTED
    
    return JSONResponse(
        status_code=status_code,
        content={
            "error": exc.code,
            "message": exc.message,
            "details": exc.details,
        },
    )


class TrustlessBlendRequest(BaseModel):
    """Complete trustless blend request."""
    upload: MotionUploadRequest
    blend: MotionBlendRequest
    payment_proof: str
    creator_address: str = Field(pattern=r"^0x[a-fA-F0-9]{40}$")


@app.get("/health")
def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "service": "kinetic-ledger-api",
        "version": "2.0.0",
    }


@app.post("/api/v2/trustless-blend")
def trustless_blend(
    request: TrustlessBlendRequest,
    x_correlation_id: Optional[str] = Header(default=None),
):
    """
    Execute complete trustless agent loop workflow.
    
    This endpoint orchestrates:
    1. Motion upload and ingest
    2. Gemini multimodal analysis
    3. Attestation oracle validation (kNN + RkCNN)
    4. Canonical pack creation and signing
    5. Usage metering and USDC settlement
    
    Returns complete audit trail and settlement details.
    """
    if x_correlation_id:
        set_correlation_id(x_correlation_id)
    
    logger.info(
        f"Trustless blend request received",
        extra={"request_id": request.blend.request_id},
    )
    
    try:
        loop = get_agent_loop()
        result = loop.execute_blend_workflow(
            upload_request=request.upload,
            blend_request=request.blend,
            payment_proof=request.payment_proof,
            creator_address=request.creator_address,
        )
        
        return {
            "correlation_id": result.correlation_id,
            "decision": result.decision,
            "pack_hash": result.pack_hash,
            "motion_id": result.motion_asset.motion_id,
            "analysis_id": result.gemini_analysis.analysis_id,
            "similarity_id": result.similarity_check.similarity_id,
            "tx_hash": result.tx_hash,
            "elapsed_seconds": result.elapsed_seconds,
            "mint_authorization": result.mint_authorization.model_dump() if result.mint_authorization else None,
            "usage_event": result.usage_event.model_dump() if result.usage_event else None,
            "separation_score": result.similarity_check.rkcnn.separation_score,
            "vote_margin": result.similarity_check.rkcnn.vote_margin,
        }
    
    except KineticLedgerError:
        raise
    except Exception as e:
        logger.error(f"Trustless blend failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}",
        )


# Legacy endpoints (for backward compatibility)
@app.post("/ingest")
def ingest(asset: MotionAsset):
    """Legacy ingest endpoint."""
    return {"motion_id": asset.motion_id}


@app.post("/analyze")
def analyze(analysis: GeminiAnalysis):
    """Legacy analyze endpoint."""
    return {"analysis_id": analysis.analysis_id}


@app.post("/validate")
def validate(sim: SimilarityCheck):
    """Legacy validate endpoint."""
    return {"similarity_id": sim.similarity_id, "decision": sim.decision.result}


@app.post("/authorize")
def authorize(pack: MotionCanonicalPack):
    """Legacy authorize endpoint."""
    return {"pack_version": pack.pack_version}


@app.post("/mint")
def mint(authz: MintAuthorization):
    """Legacy mint endpoint."""
    return {"chain_id": authz.chain_id, "contract": authz.verifying_contract}


@app.post("/usage")
def usage(event: UsageMeterEvent):
    """Legacy usage endpoint."""
    return {"usage_id": event.usage_id, "product": event.product}


# ========== UI Visualizer Endpoints ==========

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    """Serve the clean Michele visualizer UI."""
    ui_path = Path(__file__).parent.parent / "ui" / "index-new.html"
    
    if not ui_path.exists():
        raise HTTPException(status_code=404, detail="UI not found")
    
    with open(ui_path, 'r') as f:
        return HTMLResponse(content=f.read())


@app.get("/motion_library.json")
async def serve_motion_library():
    """Serve motion library manifest."""
    manifest_path = Path(__file__).parent.parent / "ui" / "motion_library.json"
    
    if not manifest_path.exists():
        raise HTTPException(status_code=404, detail="Motion library not found")
    
    return FileResponse(manifest_path, media_type="application/json")


@app.get("/styles.css")
async def serve_css():
    """Serve CSS file."""
    css_path = Path(__file__).parent.parent / "ui" / "styles.css"
    
    if not css_path.exists():
        raise HTTPException(status_code=404, detail="CSS not found")
    
    return FileResponse(css_path, media_type="text/css")


@app.get("/visualizer.js")
async def serve_js():
    """Serve JavaScript file."""
    js_path = Path(__file__).parent.parent / "ui" / "visualizer.js"
    
    if not js_path.exists():
        raise HTTPException(status_code=404, detail="JavaScript not found")
    
    return FileResponse(js_path, media_type="application/javascript")


@app.get("/workflow-visualizer.js")
async def serve_workflow_js():
    """Serve Workflow Visualizer JavaScript file."""
    js_path = Path(__file__).parent.parent / "ui" / "workflow-visualizer.js"
    
    if not js_path.exists():
        raise HTTPException(status_code=404, detail="Workflow visualizer JavaScript not found")
    
    return FileResponse(js_path, media_type="application/javascript")


@app.get("/metrics-dashboard.js")
async def serve_metrics_js():
    """Serve Metrics Dashboard JavaScript file."""
    js_path = Path(__file__).parent.parent / "ui" / "metrics-dashboard.js"
    
    if not js_path.exists():
        raise HTTPException(status_code=404, detail="Metrics dashboard JavaScript not found")
    
    return FileResponse(js_path, media_type="application/javascript")


@app.get("/static/models/Ch03_nonPBR.fbx")
async def serve_character_model():
    """Serve Michele character FBX model."""
    # Try project root first
    model_path = Path("/workspaces/reimagined-umbrella/Ch03_nonPBR.fbx")
    
    if not model_path.exists():
        raise HTTPException(status_code=404, detail="Character model not found")
    
    return FileResponse(model_path, media_type="application/octet-stream")


@app.get("/static/models/X Bot@Capoeira.fbx")
async def serve_xbot_capoeira():
    """Serve X Bot Capoeira animation (temporary placeholder)."""
    model_path = Path("/workspaces/reimagined-umbrella/data/mixamo_anims/fbx/X Bot@Capoeira.fbx")
    
    if not model_path.exists():
        raise HTTPException(status_code=404, detail="X Bot model not found")
    
    return FileResponse(model_path, media_type="application/octet-stream")


@app.get("/static/models/michele/{filename}")
async def serve_michele_animations(filename: str):
    """Serve Michele character animations."""
    # Sanitize filename
    if ".." in filename or "/" in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
    
    model_path = Path(f"/workspaces/reimagined-umbrella/data/mixamo_anims/fbx/michele/{filename}")
    
    if not model_path.exists():
        raise HTTPException(status_code=404, detail=f"Michele animation not found: {filename}")
    
    return FileResponse(model_path, media_type="application/octet-stream")


@app.get("/static/models/data/mixamo_anims/fbx/{filename:path}")
async def serve_mixamo_fbx(filename: str):
    """Serve FBX files from mixamo_anims directory."""
    # Sanitize filename - allow @ and spaces in filenames
    if ".." in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
    
    model_path = Path(f"/workspaces/reimagined-umbrella/data/mixamo_anims/fbx/{filename}")
    
    if not model_path.exists():
        raise HTTPException(status_code=404, detail=f"FBX file not found: {filename}")
    
    return FileResponse(model_path, media_type="application/octet-stream")
    """Serve Metrics Dashboard JavaScript file."""
    js_path = Path(__file__).parent.parent / "ui" / "metrics-dashboard.js"
    
    if not js_path.exists():
        raise HTTPException(status_code=404, detail="Metrics dashboard JavaScript not found")
    
    return FileResponse(js_path, media_type="application/javascript")


@app.get("/api/motions/library")
async def get_motion_library():
    """
    Get motion library catalog.
    
    Returns list of available motions with metadata from Mixamo FBX files.
    """
    logger.info("Fetching motion library")
    
    # Path to FBX files
    fbx_dir = Path(__file__).parent.parent.parent.parent / "data" / "mixamo_anims" / "fbx"
    
    motions = []
    
    if fbx_dir.exists():
        for fbx_file in fbx_dir.glob("*.fbx"):
            # Extract motion name from filename
            # Example: "X Bot@Capoeira.fbx" -> "Capoeira"
            filename = fbx_file.stem
            motion_name = filename.split('@')[-1] if '@' in filename else filename
            
            # Determine tags based on motion name keywords
            tags = []
            name_lower = motion_name.lower()
            
            if any(word in name_lower for word in ['walk', 'run', 'jump', 'crouch']):
                tags.append('locomotion')
            if any(word in name_lower for word in ['dance', 'capoeira', 'breakdance', 'hip hop']):
                tags.append('dance')
            if any(word in name_lower for word in ['punch', 'kick', 'fight', 'combat']):
                tags.append('combat')
            if any(word in name_lower for word in ['idle', 'stand', 'wait']):
                tags.append('idle')
            if 'freeze' in name_lower:
                tags.append('freeze')
            if any(word in name_lower for word in ['acrobat', 'flip', 'spin']):
                tags.append('acrobatic')
            if 'urban' in name_lower or 'street' in name_lower:
                tags.append('urban')
            
            # Generate motion metadata
            motion_id = motion_name.lower().replace(' ', '_')
            
            motions.append({
                "id": motion_id,
                "name": motion_name,
                "tags": tags if tags else ['other'],
                "duration": 3.5 + (hash(motion_name) % 50) / 10,  # Pseudo-random 3.5-8.5s
                "novelty": 0.5 + (hash(motion_name) % 40) / 100,  # Pseudo-random 0.5-0.9
                "filepath": str(fbx_file)
            })
    
    # Fallback to sample data if no FBX files found
    if not motions:
        logger.warning("No FBX files found, using sample data")
        motions = [
            {
                "id": "capoeira",
                "name": "Capoeira",
                "tags": ["dance", "combat", "acrobatic"],
                "duration": 4.5,
                "novelty": 0.75,
                "filepath": "data/mixamo_anims/fbx/X Bot@Capoeira.fbx"
            },
            {
                "id": "breakdance_freeze",
                "name": "Breakdance Freeze",
                "tags": ["dance", "urban", "freeze"],
                "duration": 3.8,
                "novelty": 0.82,
                "filepath": "data/mixamo_anims/fbx/X Bot@Breakdance Freeze Var 2.fbx"
            }
        ]
    
    logger.info(f"Returning {len(motions)} motions")
    
    return {
        "motions": motions,
        "total": len(motions)
    }


class PromptAnalyzeRequest(BaseModel):
    """Request to analyze motion prompt."""
    prompt: str
    wallet: Optional[str] = None


@app.post("/api/prompts/analyze")
async def analyze_prompt(request: PromptAnalyzeRequest):
    """
    Analyze natural language prompt for motion blending.
    
    Uses simple keyword parsing (in production: use Gemini AI).
    """
    logger.info(f"Analyzing prompt: {request.prompt}")
    
    prompt_lower = request.prompt.lower()
    
    # Extract motion keywords
    motion_keywords = {
        'walk': ['walk', 'walking'],
        'run': ['run', 'running', 'sprint'],
        'dance': ['dance', 'dancing'],
        'capoeira': ['capoeira'],
        'breakdance': ['breakdance', 'break dance', 'freeze'],
        'jump': ['jump', 'jumping'],
        'idle': ['idle', 'stand', 'standing'],
        'combat': ['fight', 'combat', 'punch', 'kick']
    }
    
    detected_motions = []
    for motion, keywords in motion_keywords.items():
        if any(keyword in prompt_lower for keyword in keywords):
            detected_motions.append(motion)
    
    # Default to capoeira and breakdance if nothing detected
    if not detected_motions:
        detected_motions = ['capoeira', 'breakdance']
    elif len(detected_motions) == 1:
        detected_motions.append('idle')  # Add idle as second motion
    
    # Extract weight if specified
    weights = [0.5, 0.5]  # Default 50/50
    
    # Look for percentage patterns
    import re
    percent_match = re.search(r'(\d+)%', prompt_lower)
    if percent_match:
        percent = int(percent_match.group(1)) / 100
        weights = [1 - percent, percent]
    
    # Estimate complexity based on prompt length
    complexity = min(len(request.prompt) / 50, 2.0)
    
    blend_id = f"blend-{uuid.uuid4().hex[:8]}"
    
    return {
        "blend_id": blend_id,
        "motions": detected_motions[:2],  # Take first two
        "weights": weights,
        "estimated_duration": 5.0 + complexity,
        "complexity": complexity,
        "transition_type": "smooth" if "smooth" in prompt_lower else "quick"
    }


class BlendGenerateRequest(BaseModel):
    """Request to generate motion blend."""
    prompt: str
    analysis: Dict[str, Any]
    quality: str
    wallet: Optional[str] = None


@app.post("/api/motions/blend/generate")
async def generate_blend(
    request: BlendGenerateRequest,
    x_payment: Optional[str] = Header(None, alias="X-Payment")
):
    """
    Generate motion blend with payment verification.
    
    Returns blend_id and correlation_id for streaming updates.
    In production: verify x402 payment proof and use trustless agent.
    """
    logger.info(f"Generating blend from prompt: {request.prompt}")
    
    # Verify payment (simplified for demo)
    if x_payment:
        logger.info(f"Payment proof received: {x_payment[:20]}...")
        # In production: verify x402 payment with commerce orchestrator
    else:
        logger.warning("No payment proof provided (demo mode)")
    
    # Get analysis
    analysis = request.analysis
    motions = analysis.get('motions', [])
    weights = analysis.get('weights', [0.5, 0.5])
    
    # Generate blend ID and correlation ID
    blend_id = analysis.get('blend_id', f"blend-{uuid.uuid4().hex[:8]}")
    correlation_id = f"corr-{uuid.uuid4().hex[:8]}"
    
    # Calculate cost
    pricing = {
        'low': 0.01,
        'medium': 0.05,
        'high': 0.10,
        'ultra': 0.25
    }
    
    base_price = pricing.get(request.quality, 0.05)
    duration = analysis.get('estimated_duration', 5.0)
    complexity = analysis.get('complexity', 1.0)
    cost = base_price * duration * complexity
    
    # Initialize workflow state for streaming
    workflow_states[correlation_id] = {
        "correlation_id": correlation_id,
        "blend_id": blend_id,
        "status": "started",
        "current_step": "initializing",
        "progress": 0,
        "total_steps": 6,
        "updated_at": uuid.uuid4().hex,
        "blend": {
            "id": blend_id,
            "motions": motions,
            "blend_weights": weights,
            "quality": request.quality,
            "duration": duration
        }
    }
    
    logger.info(f"Blend initiated: {blend_id}, correlation_id: {correlation_id}, cost: {cost:.3f} USDC")
    
    return {
        "status": "started",
        "correlation_id": correlation_id,
        "blend_id": blend_id,
        "stream_url": f"/api/blend/{correlation_id}/stream",
        "estimated_cost": f"{cost:.3f}"
    }


class NPCSpawnRequest(BaseModel):
    """Request to spawn NPC."""
    motion_id: str
    character_type: str
    energy_level: int


@app.post("/api/npcs/spawn")
async def spawn_npc(request: NPCSpawnRequest):
    """
    Spawn NPC with motion blend.
    
    Returns correlation_id for streaming updates.
    In production: integrate with Arc Network smart contract.
    """
    logger.info(f"Spawning NPC: {request.character_type} with motion {request.motion_id}")
    
    npc_id = f"npc-{uuid.uuid4().hex[:8]}"
    correlation_id = f"corr-{uuid.uuid4().hex[:8]}"
    
    # Initialize workflow state for streaming
    workflow_states[correlation_id] = {
        "correlation_id": correlation_id,
        "npc_id": npc_id,
        "status": "started",
        "current_step": "initializing",
        "progress": 0,
        "total_steps": 4,
        "updated_at": uuid.uuid4().hex,
        "npc": {
            "id": npc_id,
            "motion_id": request.motion_id,
            "character_type": request.character_type,
            "energy_level": request.energy_level
        }
    }
    
    return {
        "status": "started",
        "correlation_id": correlation_id,
        "npc_id": npc_id,
        "stream_url": f"/api/npc/{correlation_id}/stream"
    }


class MintRequest(BaseModel):
    """Request to mint motion NFT."""
    blend: Dict[str, Any]
    wallet: str


@app.post("/api/motions/mint")
async def mint_motion_nft(request: MintRequest):
    """
    Mint motion blend as NFT on Arc Network.
    
    In production: call Arc Network smart contract.
    """
    logger.info(f"Minting NFT for wallet: {request.wallet}")
    
    token_id = int(uuid.uuid4().hex[:8], 16) % 1000000
    tx_hash = f"0x{uuid.uuid4().hex}{uuid.uuid4().hex[:32]}"
    
    return {
        "status": "minted",
        "token_id": token_id,
        "tx_hash": tx_hash,
        "contract": "0x0000000000000000000000000000000000000001",
        "chain": "arc-testnet"
    }


# =============================================================================
# blendanim Integration Endpoints
# =============================================================================

class BlendAnimRequest(BaseModel):
    """Request for blendanim-powered motion blend."""
    motion_ids: List[str]
    weights: List[float]
    quality: str = "medium"
    method: str = "temporal_conditioning"


@app.post("/api/motions/blend/blendanim")
async def blend_with_blendanim(request: BlendAnimRequest):
    """
    Blend motions using blendanim algorithms with full quality metrics.
    
    NEW: Integrates Veo 3.1 video generation when ENABLE_VEO_GENERATION=true
    
    Returns comprehensive metrics aligned with academic standards:
    - Coverage, LocalDiversity, GlobalDiversity
    - L2_velocity, L2_acceleration (smoothness)
    - Quality tier (ultra/high/medium/low)
    - Cost calculation based on quality
    - Video preview URI (when Veo enabled)
    """
    try:
        from ..services.blendanim_service import get_blendanim_service, MotionSequence
        from ..services.gemini_motion_embedder import get_gemini_embedder
        from ..services.veo_video_service import get_veo_service
        import numpy as np
        
        logger.info(f"blendanim blend request: {request.motion_ids} with weights {request.weights}")
        
        # Validate inputs
        if len(request.motion_ids) != len(request.weights):
            raise HTTPException(400, "motion_ids and weights must have same length")
        
        if not np.isclose(sum(request.weights), 1.0):
            raise HTTPException(400, f"Weights must sum to 1.0, got {sum(request.weights)}")
        
        # Initialize services
        blender = get_blendanim_service()
        embedder = get_gemini_embedder()
        veo_service = get_veo_service()
        
        # Process motions (try FBX first, fall back to synthetic)
        motions = []
        fbx_paths = []
        for motion_id in request.motion_ids:
            fbx_path = f"data/mixamo_anims/fbx/{motion_id}.fbx"
            
            if Path(fbx_path).exists():
                # Process real FBX
                fbx_data = await embedder.process_fbx_file(fbx_path)
                motion = MotionSequence(
                    positions=fbx_data.skeletal_positions,
                    fps=fbx_data.fps
                )
                fbx_paths.append(fbx_path)
            else:
                # Generate synthetic motion
                logger.warning(f"FBX not found: {fbx_path}, using synthetic")
                T = 120
                J = 52
                positions = np.random.randn(T, J, 3).astype('float32') * 0.5
                motion = MotionSequence(positions=positions, fps=30)
                fbx_paths.append(None)
            
            motions.append(motion)
        
        # Blend motions
        blended, metrics = blender.blend_motions(
            motions=motions,
            weights=request.weights,
            method=request.method
        )
        
        # Calculate cost
        quality_rates = {
            "ultra": 0.25,
            "high": 0.10,
            "medium": 0.05,
            "low": 0.01
        }
        rate = quality_rates[metrics.quality_tier]
        duration = blended.positions.shape[0] / blended.fps
        motion_count = len(motions)
        complexity = 1.5  # Default complexity
        
        cost = rate * duration * np.sqrt(motion_count) * complexity
        
        # Generate blend ID
        blend_id = f"blend-{uuid.uuid4().hex[:12]}"
        
        # Optional: Generate video with Veo 3.1 (if FBX files available and Veo enabled)
        video_result = None
        if veo_service.is_available() and len(fbx_paths) >= 2 and all(fbx_paths[:2]):
            try:
                # Start Veo video generation asynchronously
                blend_prompt = f"Smooth cinematic transition blending {request.motion_ids[0]} into {request.motion_ids[1]}"
                video_result = await veo_service.generate_video_from_fbx(
                    source_fbx_path=fbx_paths[0],
                    target_fbx_path=fbx_paths[1],
                    blend_prompt=blend_prompt,
                    correlation_id=blend_id,
                    config={
                        "aspect_ratio": "16:9",
                        "resolution": "720p",
                        "duration_seconds": "8"
                    }
                )
                logger.info(f"Veo generation started for blend {blend_id}: {video_result.get('operation_name')}")
            except Exception as e:
                logger.error(f"Veo generation failed for {blend_id}: {e}")
                video_result = {"status": "failed", "error": str(e)}
        
        response_data = {
            "status": "success",
            "blend_id": blend_id,
            "method": request.method,
            "metrics": metrics.to_dict(),
            "blend_info": {
                "duration_seconds": float(duration),
                "frame_count": int(blended.positions.shape[0]),
                "fps": int(blended.fps),
                "motion_count": motion_count,
                "weights": request.weights
            },
            "pricing": {
                "quality_tier": metrics.quality_tier,
                "rate_per_second": rate,
                "total_cost_usdc": float(cost),
                "complexity_factor": complexity
            }
        }
        
        # Add video info if available
        if video_result:
            response_data["video"] = video_result
        
        return response_data
        
    except Exception as e:
        logger.error(f"Blending failed: {e}")
        raise HTTPException(500, f"Blending failed: {str(e)}")


class ArtifactGenerateRequest(BaseModel):
    motion_paths: List[str]
    weights: List[float]
    crosshatch_offsets: List[float]
    actual_frame_counts: Optional[List[int]] = None  # Frame counts from UI
    blend_mode: str = "smoothstep"
    transition_frames: int = 30


@app.post("/api/artifacts/generate")
async def generate_transition_artifacts(request: ArtifactGenerateRequest):
    """
    Generate smooth transition artifacts between 2-3 motion sequences.
    
    Uses BlendAnim temporal conditioning with smoothstep (C² continuity) to create
    30-frame transitions between each motion pair. Returns artifact data with
    frame-by-frame positions, omega values, and aggregate metrics.
    
    Args:
        motion_paths: List of FBX file paths for motions
        weights: Blend weights determining duration of each motion
        crosshatch_offsets: Frame offsets (0-1) for each motion's start position
        blend_mode: Temporal blend function (default: "smoothstep")
        transition_frames: Frames per transition (default: 30)
    
    Returns:
        - blend_id: Unique identifier for this artifact set
        - artifacts: Array of transition frame data
        - motion_segments: Info about each motion's contribution
        - aggregate_metrics: Overall blend quality
        - artifact_directory: Path to saved artifacts
    """
    try:
        from ..services.blendanim_service import get_blendanim_service, MotionSequence
        from ..utils.fbx_parser import get_fbx_parser
        
        try:
            from ..services.gemini_motion_embedder import get_gemini_embedder
            GEMINI_AVAILABLE = True
        except ImportError:
            GEMINI_AVAILABLE = False
            logger.warning("Gemini embedder not available, using FBX parser")
        
        import json
        import uuid
        from pathlib import Path
        
        logger.info(f"Artifact generation: {len(request.motion_paths)} motions, weights {request.weights}")
        
        # Validate inputs
        if len(request.motion_paths) < 2 or len(request.motion_paths) > 3:
            raise HTTPException(400, "Need 2-3 motions for artifact generation")
        
        if len(request.motion_paths) != len(request.weights):
            raise HTTPException(400, "motion_paths and weights must have same length")
        
        if len(request.motion_paths) != len(request.crosshatch_offsets):
            raise HTTPException(400, "motion_paths and crosshatch_offsets must have same length")
        
        # Initialize services
        blender = get_blendanim_service()
        fbx_parser = get_fbx_parser()
        
        if GEMINI_AVAILABLE:
            embedder = get_gemini_embedder()
        
        # Load motion sequences
        motions = []
        motion_info = []
        
        for idx, path in enumerate(request.motion_paths):
            # Resolve path
            if path.startswith('/static/models/'):
                # Remove '/static/models/' prefix to get the actual filesystem path
                fbx_path = Path(path.replace('/static/models/', ''))
            elif path.startswith('/static/'):
                fbx_path = Path(f"data/mixamo_anims/fbx/{Path(path).name}")
            else:
                fbx_path = Path(path)
            
            if not fbx_path.exists():
                raise HTTPException(404, f"Motion file not found: {fbx_path}")
            
            # Process FBX file
            if GEMINI_AVAILABLE:
                # Use Gemini for advanced analysis and embeddings
                fbx_data = await embedder.process_fbx_file(str(fbx_path))
                motion = MotionSequence(
                    positions=fbx_data.skeletal_positions,
                    fps=fbx_data.fps,
                    joint_names=fbx_data.joint_names if hasattr(fbx_data, 'joint_names') else None
                )
            else:
                # Use ufbx parser for direct FBX parsing
                positions, metadata = fbx_parser.parse_fbx(str(fbx_path))
                
                motion = MotionSequence(
                    positions=positions,
                    fps=metadata.get('fps', 30),
                    joint_names=metadata.get('joint_names', None)
                )
                
                logger.info(f"Parsed FBX file {fbx_path.name}: {positions.shape}")
            
            # Use actual frame count from UI if provided (more reliable than re-parsing)
            if request.actual_frame_counts and idx < len(request.actual_frame_counts):
                ui_frame_count = request.actual_frame_counts[idx]
                parsed_frame_count = motion.positions.shape[0]
                
                if ui_frame_count != parsed_frame_count:
                    logger.warning(f"⚠️ Frame count mismatch for {fbx_path.name}: UI={ui_frame_count}, Server parsed={parsed_frame_count}")
                    
                    # Trust the UI's frame count (it loaded the real FBX)
                    if ui_frame_count > parsed_frame_count:
                        # UI has more frames - server likely using fallback/synthetic parser
                        # Pad the motion to match UI's count
                        import numpy as np
                        num_joints = motion.positions.shape[1]
                        num_dims = motion.positions.shape[2]
                        padded_positions = np.zeros((ui_frame_count, num_joints, num_dims))
                        padded_positions[:parsed_frame_count] = motion.positions
                        # Repeat last frame for remaining frames
                        padded_positions[parsed_frame_count:] = motion.positions[-1:]
                        
                        motion = MotionSequence(
                            positions=padded_positions,
                            fps=motion.fps,
                            joint_names=motion.joint_names
                        )
                        logger.info(f"📏 Padded to UI frame count: {parsed_frame_count} → {ui_frame_count} frames")
                    else:
                        # UI has fewer frames - slice to match
                        motion = MotionSequence(
                            positions=motion.positions[:ui_frame_count],
                            fps=motion.fps,
                            joint_names=motion.joint_names
                        )
                        logger.info(f"✂️ Sliced to UI frame count: {parsed_frame_count} → {ui_frame_count} frames")
                else:
                    logger.info(f"✅ Frame count matches UI: {ui_frame_count} frames")
            
            motions.append(motion)
            
            motion_info.append({
                "path": str(fbx_path),
                "name": fbx_path.stem,
                "total_frames": motion.positions.shape[0],
                "duration_seconds": motion.positions.shape[0] / motion.fps,
                "fps": motion.fps
            })
        
        # Generate transition artifacts
        artifacts, aggregate_metrics, actual_motion_segments = blender.generate_transition_artifacts(
            motions=motions,
            weights=request.weights,
            crosshatch_offsets=request.crosshatch_offsets,
            transition_frames=request.transition_frames,
            blend_mode=request.blend_mode
        )
        
        logger.info(f"📊 Artifact generation complete: {len(artifacts)} total frames")
        
        # Count frame types for verification
        motion_artifact_count = sum(1 for a in artifacts if a.get("blend_mode") == "motion_segment")
        transition_artifact_count = sum(1 for a in artifacts if a.get("blend_mode") != "motion_segment")
        logger.info(f"  Motion frames: {motion_artifact_count}")
        logger.info(f"  Transition frames: {transition_artifact_count}")
        
        # Enrich motion segments with motion info
        for i, segment in enumerate(actual_motion_segments):
            segment["motion_name"] = motion_info[i]["name"]
            segment["duration_seconds"] = segment["frame_count"] / motions[i].fps
        
        # Create unique blend ID
        blend_id = f"blend-{uuid.uuid4().hex[:12]}"
        
        # Create artifacts directory
        artifacts_dir = Path("artifacts") / blend_id
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        
        # Save artifacts to disk and prepare response
        artifact_files = []
        artifacts_response = []  # Full artifact data for client
        for i, artifact in enumerate(artifacts):
            # Convert numpy arrays to lists for JSON serialization
            artifact_data = {
                "frame_index": artifact["frame_index"],
                "omega": artifact["omega"],
                "positions": artifact["positions"].tolist(),  # [J, 3]
                "blend_mode": artifact["blend_mode"],
                "t_normalized": artifact["t_normalized"]
            }
            
            # Add optional segment metadata if present
            if "segment_type" in artifact:
                artifact_data["segment_type"] = artifact["segment_type"]
            if "segment_frame" in artifact:
                artifact_data["segment_frame"] = artifact["segment_frame"]
            
            # Save to file
            artifact_file = artifacts_dir / f"frame-{i:04d}.json"
            with open(artifact_file, 'w') as f:
                json.dump(artifact_data, f, indent=2)
            
            # URL-only reference for file access
            artifact_files.append({
                "frame_index": i,
                "omega": artifact["omega"],
                "t_normalized": artifact["t_normalized"],
                "blend_mode": artifact.get("blend_mode", "transition"),
                "url": f"/api/artifacts/{blend_id}/frame-{i:04d}.json"
            })
            
            # Full data for immediate client use (includes positions)
            artifacts_response.append(artifact_data)
        
        # Save metadata
        metadata = {
            "blend_id": blend_id,
            "created_at": str(Path(artifacts_dir).stat().st_mtime),
            "motions": motion_info,
            "segments": actual_motion_segments,
            "transition_frames": request.transition_frames,
            "blend_mode": request.blend_mode,
            "total_artifacts": len(artifacts),
            "aggregate_metrics": aggregate_metrics.to_dict()
        }
        
        with open(artifacts_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Generated {len(artifacts)} artifacts in {artifacts_dir}")
        
        return {
            "status": "success",
            "blend_id": blend_id,
            "artifacts": artifacts_response,  # Return full artifact data with positions
            "artifact_files": artifact_files,  # Keep file URLs as reference
            "motion_segments": actual_motion_segments,
            "aggregate_metrics": aggregate_metrics.to_dict(),
            "artifact_directory": str(artifacts_dir),
            "total_transitions": len(motions) - 1,
            "transition_frames_each": request.transition_frames,
            "encoding_method": "gemini" if GEMINI_AVAILABLE else "ufbx parser (Gemini unavailable)"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Artifact generation failed: {e}", exc_info=True)
        raise HTTPException(500, f"Artifact generation failed: {str(e)}")


@app.get("/api/artifacts/{blend_id}/frame-{frame_num}.json")
async def get_artifact_frame(blend_id: str, frame_num: str):
    """Retrieve a specific artifact frame."""
    try:
        artifacts_dir = Path("artifacts") / blend_id
        frame_file = artifacts_dir / f"frame-{frame_num}.json"
        
        if not frame_file.exists():
            raise HTTPException(404, f"Artifact frame not found: {frame_num}")
        
        with open(frame_file, 'r') as f:
            return json.load(f)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Failed to load artifact: {str(e)}")


@app.get("/api/artifacts/{blend_id}/metadata")
async def get_artifact_metadata(blend_id: str):
    """Retrieve artifact set metadata."""
    try:
        artifacts_dir = Path("artifacts") / blend_id
        metadata_file = artifacts_dir / "metadata.json"
        
        if not metadata_file.exists():
            raise HTTPException(404, f"Artifact metadata not found for: {blend_id}")
        
        with open(metadata_file, 'r') as f:
            return json.load(f)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Failed to load metadata: {str(e)}")


class IntelligentArtifactRequest(BaseModel):
    """Request for Gemini-powered intelligent artifact generation."""
    motion_paths: List[str] = Field(description="FBX file paths for 2-3 motions")
    weights: List[float] = Field(description="Blend weights for motion durations")
    crosshatch_offsets: Optional[List[float]] = None
    actual_frame_counts: Optional[List[int]] = None
    use_gemini_analysis: bool = Field(default=True, description="Enable Gemini multimodal analysis")
    frame_sample_rate: int = Field(default=5, description="Sample every Nth frame for analysis")


@app.post("/api/artifacts/generate-intelligent")
async def generate_intelligent_artifacts(request: IntelligentArtifactRequest):
    """
    Generate high-quality artifacts using Gemini multimodal analysis.
    
    This endpoint uses Gemini's video understanding to analyze motion pairs,
    recommend optimal blend parameters, and predict quality before computation.
    
    Workflow:
    1. Load and render motion frames for Gemini analysis
    2. Analyze motion characteristics and compatibility
    3. Generate intelligent blend parameter recommendations
    4. Execute blend with optimized parameters
    5. Return artifacts with Gemini insights
    
    Returns:
        - All standard artifact data
        - gemini_analysis: Structured analysis and recommendations
        - compatibility_score: Motion pair compatibility (0-1)
        - quality_prediction: Predicted blend quality metrics
        - recommended_vs_used_params: Parameter comparison
    """
    try:
        from ..services.gemini_motion_analyzer import GeminiMotionAnalyzer
        from ..services.blendanim_service import get_blendanim_service, MotionSequence
        from ..utils.fbx_parser import get_fbx_parser
        from ..utils.frame_renderer import MotionFrameRenderer, create_standard_skeleton_hierarchy
        
        # Check API key
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not gemini_api_key or not request.use_gemini_analysis:
            # Fall back to standard artifact generation
            logger.warning("Gemini not available, falling back to standard generation")
            standard_request = ArtifactGenerateRequest(
                motion_paths=request.motion_paths,
                weights=request.weights,
                crosshatch_offsets=request.crosshatch_offsets or [0.0] * len(request.motion_paths),
                actual_frame_counts=request.actual_frame_counts,
                transition_frames=30
            )
            return await generate_transition_artifacts(standard_request)
        
        logger.info(f"🎬 Intelligent artifact generation: {len(request.motion_paths)} motions")
        
        # Validate inputs - fall back to standard for 3+ motions
        if len(request.motion_paths) > 2:
            logger.warning(f"Gemini analysis currently supports 2 motions, falling back to standard generation for {len(request.motion_paths)} motions")
            standard_request = ArtifactGenerateRequest(
                motion_paths=request.motion_paths,
                weights=request.weights,
                crosshatch_offsets=request.crosshatch_offsets or [0.0] * len(request.motion_paths),
                actual_frame_counts=request.actual_frame_counts,
                transition_frames=30
            )
            return await generate_transition_artifacts(standard_request)
        
        if len(request.motion_paths) < 2:
            raise HTTPException(400, "Need at least 2 motions for blending")
        
        # Initialize services
        blender = get_blendanim_service()
        fbx_parser = get_fbx_parser()
        analyzer = GeminiMotionAnalyzer(
            api_key=gemini_api_key,
            frame_sample_rate=request.frame_sample_rate
        )
        renderer = MotionFrameRenderer(width=640, height=480)
        
        # Load motion sequences and render frames
        motions = []
        motion_frames = []
        motion_info = []
        
        for idx, path in enumerate(request.motion_paths):
            # Resolve path
            if path.startswith('/static/models/'):
                fbx_path = Path(path.replace('/static/models/', ''))
            elif path.startswith('/static/'):
                fbx_path = Path(f"data/mixamo_anims/fbx/{Path(path).name}")
            else:
                fbx_path = Path(path)
            
            if not fbx_path.exists():
                raise HTTPException(404, f"Motion file not found: {fbx_path}")
            
            # Parse FBX
            positions, metadata = fbx_parser.parse_fbx(str(fbx_path))
            motion = MotionSequence(
                positions=positions,
                fps=metadata.get('fps', 30),
                joint_names=metadata.get('joint_names', None)
            )
            motions.append(motion)
            
            # Render frames for Gemini
            logger.info(f"🎨 Rendering frames for {fbx_path.name}...")
            hierarchy = create_standard_skeleton_hierarchy(positions.shape[1])
            frames = renderer.render_frames_from_positions(
                positions,
                skeleton_hierarchy=hierarchy,
                frame_indices=list(range(0, len(positions), request.frame_sample_rate))
            )
            motion_frames.append(frames)
            
            motion_info.append({
                "path": str(fbx_path),
                "name": fbx_path.stem,
                "total_frames": positions.shape[0],
                "fps": motion.fps
            })
            
            logger.info(f"✅ Rendered {len(frames)} frames from {fbx_path.name}")
        
        # Analyze motion pair with Gemini
        logger.info("🤖 Analyzing motions with Gemini...")
        analysis = await analyzer.analyze_motion_pair(
            motion_a_frames=motion_frames[0],
            motion_b_frames=motion_frames[1],
            motion_a_name=motion_info[0]["name"],
            motion_b_name=motion_info[1]["name"]
        )
        
        logger.info(f"✨ Gemini Analysis Complete:")
        logger.info(f"  Compatibility: {analysis.compatibility.overall_score:.2f}")
        logger.info(f"  Recommended transition frames: {analysis.recommended_parameters.transition_frames}")
        logger.info(f"  Recommended omega curve: {analysis.recommended_parameters.omega_curve_type}")
        logger.info(f"  Predicted smoothness: {analysis.quality_prediction.predicted_smoothness:.2f}")
        
        # Use Gemini's recommended parameters
        transition_frames = analysis.recommended_parameters.transition_frames
        blend_mode = analysis.recommended_parameters.omega_curve_type
        
        # Use Gemini's crosshatch offset or default
        if request.crosshatch_offsets:
            crosshatch_offsets = request.crosshatch_offsets
        else:
            crosshatch_offsets = [0.0, analysis.recommended_parameters.crosshatch_offset / motion_info[1]["total_frames"]]
        
        # Generate artifacts with optimized parameters
        logger.info(f"⚙️ Generating artifacts with Gemini-optimized parameters...")
        artifacts, aggregate_metrics, actual_motion_segments = blender.generate_transition_artifacts(
            motions=motions,
            weights=request.weights,
            crosshatch_offsets=crosshatch_offsets,
            transition_frames=transition_frames,
            blend_mode=blend_mode
        )
        
        # Create blend ID
        blend_id = f"intelligent-blend-{uuid.uuid4().hex[:12]}"
        artifacts_dir = Path("artifacts") / blend_id
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        
        # Save artifacts
        artifact_files = []
        artifacts_response = []
        for i, artifact in enumerate(artifacts):
            artifact_data = {
                "frame_index": artifact["frame_index"],
                "omega": artifact["omega"],
                "positions": artifact["positions"].tolist(),
                "blend_mode": artifact["blend_mode"],
                "t_normalized": artifact["t_normalized"]
            }
            
            artifact_file = artifacts_dir / f"frame-{i:04d}.json"
            with open(artifact_file, 'w') as f:
                json.dump(artifact_data, f, indent=2)
            
            artifact_files.append({
                "frame_index": i,
                "omega": artifact["omega"],
                "t_normalized": artifact["t_normalized"],
                "blend_mode": artifact.get("blend_mode", "transition"),
                "url": f"/api/artifacts/{blend_id}/frame-{i:04d}.json"
            })
            
            artifacts_response.append(artifact_data)
        
        # Save comprehensive metadata with Gemini analysis
        metadata = {
            "blend_id": blend_id,
            "intelligent_generation": True,
            "created_at": str(Path(artifacts_dir).stat().st_mtime),
            "motions": motion_info,
            "segments": actual_motion_segments,
            "transition_frames": transition_frames,
            "blend_mode": blend_mode,
            "total_artifacts": len(artifacts),
            "aggregate_metrics": aggregate_metrics.to_dict(),
            "gemini_analysis": {
                "motion_a": analysis.motion_a_characteristics.model_dump(),
                "motion_b": analysis.motion_b_characteristics.model_dump(),
                "compatibility": analysis.compatibility.model_dump(),
                "recommended_parameters": analysis.recommended_parameters.model_dump(),
                "quality_prediction": analysis.quality_prediction.model_dump(),
                "overall_recommendation": analysis.overall_recommendation
            }
        }
        
        with open(artifacts_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"🎉 Generated {len(artifacts)} intelligent artifacts in {artifacts_dir}")
        
        return {
            "status": "success",
            "blend_id": blend_id,
            "artifacts": artifacts_response,
            "artifact_files": artifact_files,
            "motion_segments": actual_motion_segments,
            "aggregate_metrics": aggregate_metrics.to_dict(),
            "artifact_directory": str(artifacts_dir),
            "encoding_method": "gemini_intelligent_analysis",
            
            # Gemini insights
            "gemini_analysis": {
                "motion_a_type": analysis.motion_a_characteristics.motion_type,
                "motion_b_type": analysis.motion_b_characteristics.motion_type,
                "compatibility_score": analysis.compatibility.overall_score,
                "compatibility_reasoning": analysis.compatibility.reasoning,
                "recommended_parameters": analysis.recommended_parameters.model_dump(),
                "quality_prediction": {
                    "predicted_coverage": analysis.quality_prediction.predicted_coverage,
                    "predicted_diversity": analysis.quality_prediction.predicted_diversity,
                    "predicted_smoothness": analysis.quality_prediction.predicted_smoothness,
                    "confidence": analysis.quality_prediction.confidence,
                    "potential_issues": analysis.quality_prediction.potential_issues
                },
                "overall_recommendation": analysis.overall_recommendation
            },
            
            # Parameter comparison
            "parameter_optimization": {
                "transition_frames": {
                    "used": transition_frames,
                    "recommended": analysis.recommended_parameters.transition_frames,
                    "source": "gemini"
                },
                "blend_mode": {
                    "used": blend_mode,
                    "recommended": analysis.recommended_parameters.omega_curve_type,
                    "source": "gemini"
                }
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Intelligent artifact generation failed: {e}", exc_info=True)
        raise HTTPException(500, f"Intelligent generation failed: {str(e)}")


@app.get("/api/motions/quality-tiers")
def get_quality_tiers():
    """Get quality tier definitions and pricing."""
    return {
        "ultra": {
            "min_coverage": 0.90,
            "max_l2_velocity": 0.03,
            "max_l2_acceleration": 0.015,
            "min_smoothness": 0.94,
            "price_per_second": 0.25,
            "description": "Professional-grade blend with exceptional smoothness"
        },
        "high": {
            "min_coverage": 0.85,
            "max_l2_velocity": 0.07,
            "max_l2_acceleration": 0.040,
            "min_smoothness": 0.86,
            "price_per_second": 0.10,
            "description": "High-quality game cutscene animation"
        },
        "medium": {
            "min_coverage": 0.75,
            "max_l2_velocity": 0.10,
            "max_l2_acceleration": 0.050,
            "min_smoothness": 0.80,
            "price_per_second": 0.05,
            "description": "Standard gameplay animation quality"
        },
        "low": {
            "min_coverage": 0.65,
            "max_l2_velocity": 0.15,
            "max_l2_acceleration": 0.080,
            "min_smoothness": 0.70,
            "price_per_second": 0.01,
            "description": "Background NPC animation quality"
        }
    }


@app.get("/api/metrics/info")
def get_metrics_info():
    """Get information about blendanim metrics."""
    return {
        "metrics": {
            "coverage": {
                "description": "Motion space coverage (0-1 scale)",
                "range": [0.0, 1.0],
                "higher_is_better": True,
                "calculation": "30-frame sliding window with cost threshold 2.0"
            },
            "local_diversity": {
                "description": "Short-term variation (15-frame windows)",
                "range": [0.0, "unbounded"],
                "interpretation": "Lower for similar motions, higher for diverse",
                "calculation": "Per-window cost analysis"
            },
            "global_diversity": {
                "description": "Long-term variation via NN-DP alignment",
                "range": [0.0, "unbounded"],
                "higher_is_better": True,
                "calculation": "30-frame dynamic programming alignment"
            },
            "l2_velocity": {
                "description": "Velocity smoothness metric",
                "range": [0.0, "unbounded"],
                "lower_is_better": True,
                "calculation": "L2 norm of frame-to-frame velocity differences"
            },
            "l2_acceleration": {
                "description": "Acceleration smoothness (jerk)",
                "range": [0.0, "unbounded"],
                "lower_is_better": True,
                "calculation": "L2 norm of velocity-to-velocity differences"
            }
        }
    }


@app.get("/api/video/status/{operation_name}")
async def get_video_status(operation_name: str):
    """
    Poll Veo video generation status.
    
    Returns:
        - status: "processing" | "completed" | "failed"
        - progress: 0-100
        - video_uri: Download URI when completed
    """
    from ..services.veo_video_service import get_veo_service
    
    veo_service = get_veo_service()
    if not veo_service.is_available():
        raise HTTPException(503, "Veo service not available")
    
    try:
        correlation_id = operation_name.split("/")[-1]  # Extract ID from operation name
        result = await veo_service.poll_video_status(
            operation_name=operation_name,
            correlation_id=correlation_id
        )
        return result
    except Exception as e:
        logger.error(f"Video status check failed: {e}")
        raise HTTPException(500, str(e))


# =============================================================================
# Previous metrics info endpoint continues below
# =============================================================================

@app.get("/api/metrics/old-info")  # Renamed to avoid conflict
def get_old_metrics_info():
    """Get information about blendanim metrics (old version)."""
    return {
        "metrics": {
            "coverage": {
                "description": "Motion space coverage (0-1 scale)",
                "range": [0.0, 1.0],
                "higher_is_better": True,
                "calculation": "30-frame sliding window with cost threshold 2.0"
            },
            "local_diversity": {
                "description": "Short-term variation (15-frame windows)",
                "range": [0.0, "unbounded"],
                "interpretation": "Lower for similar motions, higher for diverse",
                "calculation": "Per-window cost analysis"
            },
            "global_diversity": {
                "description": "Long-term variation via NN-DP alignment",
                "range": [0.0, "unbounded"],
                "higher_is_better": True,
                "calculation": "30-frame dynamic programming alignment"
            },
            "l2_velocity": {
                "description": "Velocity smoothness metric",
                "range": [0.0, "unbounded"],
                "lower_is_better": True,
                "calculation": "L2 norm of velocity deltas, focus on blend area"
            },
            "l2_acceleration": {
                "description": "Jerk minimization (acceleration smoothness)",
                "range": [0.0, "unbounded"],
                "lower_is_better": True,
                "calculation": "Delta of velocity changes, per-joint analysis"
            }
        },
        "reference": "Aligned with https://github.com/RydlrCS/blendanim",
        "documentation": "/docs/BLEND_METRICS.md"
    }


# ============================================================================
# Workflow Visualization API Endpoints
# ============================================================================

# In-memory workflow state storage (in production: use Redis/database)
workflow_states: Dict[str, Dict[str, Any]] = {}


class WorkflowStepUpdate(BaseModel):
    """Workflow step update model"""
    step_id: str
    status: str  # 'pending', 'processing', 'success', 'error', 'review'
    data: Optional[Dict[str, Any]] = None
    timestamp: Optional[str] = None


class WorkflowState(BaseModel):
    """Complete workflow state model"""
    correlation_id: str
    status: str  # 'pending', 'processing', 'completed', 'failed'
    current_step: Optional[str] = None
    steps: List[WorkflowStepUpdate]
    ingest_data: Optional[Dict[str, Any]] = None
    gemini_data: Optional[Dict[str, Any]] = None
    oracle_data: Optional[Dict[str, Any]] = None
    routing_data: Optional[Dict[str, Any]] = None
    created_at: str
    updated_at: str


@app.get("/api/workflows/{correlation_id}", response_model=WorkflowState)
def get_workflow_state(correlation_id: str):
    """
    Get complete workflow state for a given correlation ID.
    
    Returns all step data, current progress, and detailed information
    for each stage of the trustless agent loop.
    
    Args:
        correlation_id: Unique identifier for workflow execution
    
    Returns:
        WorkflowState with complete step history and data
    
    Raises:
        HTTPException: 404 if workflow not found
    """
    logger.info(f"[GET /api/workflows/{correlation_id}] Fetching workflow state")
    
    if correlation_id not in workflow_states:
        logger.warning(f"[GET /api/workflows/{correlation_id}] Workflow not found")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Workflow {correlation_id} not found"
        )
    
    workflow = workflow_states[correlation_id]
    logger.info(f"[GET /api/workflows/{correlation_id}] Returning workflow state")
    
    return workflow


@app.get("/api/workflows")
def list_workflows(
    limit: int = 50,
    offset: int = 0,
    status_filter: Optional[str] = None
):
    """
    List all workflows with optional filtering.
    
    Args:
        limit: Maximum number of workflows to return (default: 50)
        offset: Number of workflows to skip (default: 0)
        status_filter: Filter by workflow status (optional)
    
    Returns:
        List of workflow summaries
    """
    logger.info(f"[GET /api/workflows] Listing workflows (limit={limit}, offset={offset})")
    
    workflows = list(workflow_states.values())
    
    # Apply status filter if provided
    if status_filter:
        workflows = [w for w in workflows if w.get('status') == status_filter]
    
    # Apply pagination
    total = len(workflows)
    workflows = workflows[offset:offset + limit]
    
    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "workflows": workflows
    }


@app.post("/api/workflows/{correlation_id}/steps")
def update_workflow_step(
    correlation_id: str,
    step_update: WorkflowStepUpdate
):
    """
    Update a specific workflow step (internal use for step tracking).
    
    Args:
        correlation_id: Workflow identifier
        step_update: Step update data
    
    Returns:
        Updated workflow state
    """
    logger.info(
        f"[POST /api/workflows/{correlation_id}/steps] "
        f"Updating step {step_update.step_id} to {step_update.status}"
    )
    
    # Initialize workflow if not exists
    if correlation_id not in workflow_states:
        from datetime import datetime
        workflow_states[correlation_id] = {
            "correlation_id": correlation_id,
            "status": "processing",
            "current_step": step_update.step_id,
            "steps": [],
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat()
        }
    
    workflow = workflow_states[correlation_id]
    
    # Update or add step
    existing_step = next(
        (s for s in workflow["steps"] if s.get("step_id") == step_update.step_id),
        None
    )
    
    if existing_step:
        existing_step.update(step_update.dict())
    else:
        workflow["steps"].append(step_update.dict())
    
    # Update workflow metadata
    workflow["current_step"] = step_update.step_id
    from datetime import datetime
    workflow["updated_at"] = datetime.utcnow().isoformat()
    
    # Update overall workflow status based on steps
    if step_update.status == "error":
        workflow["status"] = "failed"
    elif all(s.get("status") == "success" for s in workflow["steps"]):
        workflow["status"] = "completed"
    else:
        workflow["status"] = "processing"
    
    return workflow


@app.get("/api/metrics/{blend_id}/timeline")
def get_metrics_timeline(blend_id: str):
    """
    Get frame-by-frame quality metrics timeline for a blend.
    
    Returns detailed metrics data for visualization:
    - Coverage values per 30-frame window
    - Local/Global diversity scores
    - Per-joint L2 velocity and acceleration
    - Quality tier progression
    
    Args:
        blend_id: Unique identifier for the blend
    
    Returns:
        Metrics timeline data for charts
    
    Raises:
        HTTPException: 404 if blend not found
    """
    logger.info(f"[GET /api/metrics/{blend_id}/timeline] Fetching metrics timeline")
    
    # TODO: Retrieve from database/cache
    # For now, return mock data structure
    
    return {
        "blend_id": blend_id,
        "coverage_timeline": [0.82, 0.85, 0.87, 0.89, 0.91, 0.88, 0.86],
        "diversity": {
            "local": 2.45,
            "global": 3.67
        },
        "per_joint_metrics": {
            "Pelvis": {"l2_velocity": 0.045, "l2_acceleration": 0.022},
            "LeftWrist": {"l2_velocity": 0.067, "l2_acceleration": 0.034},
            "RightWrist": {"l2_velocity": 0.063, "l2_acceleration": 0.031},
            "LeftFoot": {"l2_velocity": 0.052, "l2_acceleration": 0.025},
            "RightFoot": {"l2_velocity": 0.054, "l2_acceleration": 0.027}
        },
        "cost_breakdown": {
            "generation_cost": 0.075,
            "quality_premium": 0.025,
            "base_fee": 0.010
        },
        "quality_tier": "high",
        "total_frames": 210,
        "blend_window": [60, 150]
    }


@app.get("/api/oracle/{analysis_id}/neighbors")
def get_knn_neighbors(
    analysis_id: str,
    k: int = 15
):
    """
    Get kNN neighbors for an analysis (for Oracle decision visualization).
    
    Args:
        analysis_id: Analysis identifier
        k: Number of neighbors to return (default: 15)
    
    Returns:
        List of nearest neighbors with distances and preview URIs
    
    Raises:
        HTTPException: 404 if analysis not found
    """
    logger.info(f"[GET /api/oracle/{analysis_id}/neighbors] Fetching kNN neighbors (k={k})")
    
    # TODO: Retrieve from vector database
    # For now, return mock data structure
    
    neighbors = [
        {
            "motion_id": f"motion_{i}",
            "distance": 0.15 + (i * 0.05),
            "preview_uri": f"/static/previews/motion_{i}.jpg",
            "tags": ["locomotion", "walk"]
        }
        for i in range(k)
    ]
    
    return {
        "analysis_id": analysis_id,
        "k": k,
        "neighbors": neighbors
    }


# ============================================================================
# Server-Sent Events (SSE) for Real-time Workflow Updates
# ============================================================================

from fastapi.responses import StreamingResponse
import asyncio


# Global workflow states storage
workflow_states = {}


async def simulate_blend_workflow(correlation_id: str):
    """Simulate blend generation workflow with progressive updates."""
    steps = [
        ("analyzing_prompt", "Analyzing prompt", 1),
        ("loading_motions", "Loading motion data", 2),
        ("computing_blend", "Computing blend weights", 3),
        ("generating_metrics", "Generating quality metrics", 4),
        ("creating_transaction", "Creating blockchain transaction", 5),
        ("completed", "Blend completed", 6)
    ]
    
    for step_name, step_desc, progress in steps:
        await asyncio.sleep(1.5)  # Simulate processing time
        
        if correlation_id in workflow_states:
            workflow_states[correlation_id].update({
                "current_step": step_name,
                "current_step_name": step_desc,
                "progress": progress,
                "status": "completed" if step_name == "completed" else "processing",
                "updated_at": uuid.uuid4().hex
            })
            
            # Add metrics on the metrics step
            if step_name == "generating_metrics":
                workflow_states[correlation_id]["metrics"] = {
                    "coverage": 0.85 + (progress * 0.02),
                    "local_diversity": 0.78,
                    "global_diversity": 0.82,
                    "l2_velocity": 0.06,
                    "l2_acceleration": 0.035,
                    "quality_tier": "high"
                }
            
            # Add transaction on final step
            if step_name == "completed":
                workflow_states[correlation_id]["settlement"] = {
                    "tx_hash": f"0x{uuid.uuid4().hex}{uuid.uuid4().hex[:32]}",
                    "timestamp": int(uuid.uuid4().int % 1000000000)
                }


async def simulate_npc_workflow(correlation_id: str):
    """Simulate NPC spawn workflow with progressive updates."""
    steps = [
        ("loading_motion", "Loading motion blend", 1),
        ("creating_character", "Creating character mesh", 2),
        ("spawning_on_chain", "Spawning NPC on Arc Network", 3),
        ("completed", "NPC spawned successfully", 4)
    ]
    
    for step_name, step_desc, progress in steps:
        await asyncio.sleep(1)
        
        if correlation_id in workflow_states:
            workflow_states[correlation_id].update({
                "current_step": step_name,
                "current_step_name": step_desc,
                "progress": progress,
                "status": "completed" if step_name == "completed" else "processing",
                "updated_at": uuid.uuid4().hex
            })
            
            if step_name == "completed":
                workflow_states[correlation_id]["settlement"] = {
                    "tx_hash": f"0x{uuid.uuid4().hex}{uuid.uuid4().hex[:32]}",
                    "timestamp": int(uuid.uuid4().int % 1000000000)
                }


async def workflow_event_generator(correlation_id: str):
    """
    Generate Server-Sent Events for workflow updates.
    
    Streams real-time updates as workflow progresses through steps.
    """
    logger.info(f"[SSE /api/workflows/{correlation_id}/stream] Client connected")
    
    last_update = None
    
    try:
        while True:
            if correlation_id in workflow_states:
                workflow = workflow_states[correlation_id]
                current_update = workflow.get("updated_at")
                
                # Send update if workflow changed
                if current_update != last_update:
                    last_update = current_update
                    
                    yield f"data: {json.dumps(workflow)}\n\n"
                
                # Stop streaming if workflow completed or failed
                if workflow.get("status") in ["completed", "failed"]:
                    logger.info(
                        f"[SSE /api/workflows/{correlation_id}/stream] "
                        f"Workflow {workflow.get('status')}, closing stream"
                    )
                    break
            
            await asyncio.sleep(0.5)  # Poll every 500ms for responsiveness
            
    except asyncio.CancelledError:
        logger.info(f"[SSE /api/workflows/{correlation_id}/stream] Client disconnected")


@app.get("/api/blend/{correlation_id}/stream")
async def stream_blend_updates(correlation_id: str):
    """
    Stream real-time blend generation updates via Server-Sent Events (SSE).
    
    Clients subscribe to receive live updates as the blend progresses.
    """
    logger.info(f"[GET /api/blend/{correlation_id}/stream] Starting SSE stream")
    
    # Start background workflow simulation
    asyncio.create_task(simulate_blend_workflow(correlation_id))
    
    return StreamingResponse(
        workflow_event_generator(correlation_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@app.get("/api/npc/{correlation_id}/stream")
async def stream_npc_updates(correlation_id: str):
    """
    Stream real-time NPC spawn updates via Server-Sent Events (SSE).
    
    Clients subscribe to receive live updates as the NPC spawns.
    """
    logger.info(f"[GET /api/npc/{correlation_id}/stream] Starting SSE stream")
    
    # Start background workflow simulation
    asyncio.create_task(simulate_npc_workflow(correlation_id))
    
    return StreamingResponse(
        workflow_event_generator(correlation_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@app.get("/api/workflows/{correlation_id}/stream")
async def stream_workflow_updates(correlation_id: str):
    """
    Stream real-time workflow updates via Server-Sent Events (SSE).
    
    Clients can subscribe to receive live updates as the workflow progresses
    through each step of the trustless agent loop.
    
    Args:
        correlation_id: Workflow identifier
    
    Returns:
        StreamingResponse with SSE data
    """
    logger.info(f"[GET /api/workflows/{correlation_id}/stream] Starting SSE stream")
    
    return StreamingResponse(
        workflow_event_generator(correlation_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"  # Disable nginx buffering
        }
    )
