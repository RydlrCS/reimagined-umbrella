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


# GCS integration for Cloud Run
def sync_data_from_gcs():
    """Sync data and config from GCS bucket on startup (for Cloud Run)."""
    if os.getenv("K_SERVICE"):  # Running in Cloud Run
        try:
            from ..services.gcs_storage import sync_directory_from_gcs
            logger.info("Running in Cloud Run, syncing data from GCS...")
            
            # Sync data directory (mixamo animations, etc.)
            sync_directory_from_gcs("data/", "/app/data")
            
            # Sync config directory
            sync_directory_from_gcs("config/", "/app/config")
            
            logger.info("GCS data sync complete")
        except Exception as e:
            logger.warning(f"Failed to sync data from GCS: {e}")


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    logger.info("Starting Kinetic Ledger API...")
    sync_data_from_gcs()
    logger.info("Kinetic Ledger API started")


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


# Cloud Run Health Check Endpoints
@app.get("/ready")
def ready():
    """
    Startup probe endpoint for Cloud Run.
    Returns 200 when the application is ready to receive traffic.
    """
    return {"status": "ready", "service": "kinetic-ledger-api"}


@app.get("/traffic")
def traffic():
    """
    Readiness probe endpoint for Cloud Run.
    Returns 200 when the application can handle traffic.
    """
    return {"status": "accepting_traffic", "service": "kinetic-ledger-api"}


@app.get("/restart")
def restart():
    """
    Liveness probe endpoint for Cloud Run.
    Returns 200 to indicate the application is alive.
    """
    return {"status": "alive", "service": "kinetic-ledger-api"}


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


# ========== Favicon Routes ==========

@app.get("/favicon.ico")
async def serve_favicon() -> FileResponse:
    """Serve favicon.ico file."""
    logger.debug("[ENTRY] serve_favicon: Serving favicon.ico")
    favicon_path = Path(__file__).parent.parent / "ui" / "favicon_io" / "favicon.ico"
    
    if not favicon_path.exists():
        logger.warning("[EXIT] serve_favicon: Favicon not found")
        raise HTTPException(status_code=404, detail="Favicon not found")
    
    logger.debug("[EXIT] serve_favicon: Favicon served successfully")
    return FileResponse(favicon_path, media_type="image/x-icon")


@app.get("/favicon-16x16.png")
async def serve_favicon_16() -> FileResponse:
    """Serve 16x16 favicon PNG."""
    logger.debug("[ENTRY] serve_favicon_16: Serving favicon-16x16.png")
    favicon_path = Path(__file__).parent.parent / "ui" / "favicon_io" / "favicon-16x16.png"
    
    if not favicon_path.exists():
        logger.warning("[EXIT] serve_favicon_16: Favicon not found")
        raise HTTPException(status_code=404, detail="Favicon 16x16 not found")
    
    logger.debug("[EXIT] serve_favicon_16: Favicon served successfully")
    return FileResponse(favicon_path, media_type="image/png")


@app.get("/favicon-32x32.png")
async def serve_favicon_32() -> FileResponse:
    """Serve 32x32 favicon PNG."""
    logger.debug("[ENTRY] serve_favicon_32: Serving favicon-32x32.png")
    favicon_path = Path(__file__).parent.parent / "ui" / "favicon_io" / "favicon-32x32.png"
    
    if not favicon_path.exists():
        logger.warning("[EXIT] serve_favicon_32: Favicon not found")
        raise HTTPException(status_code=404, detail="Favicon 32x32 not found")
    
    logger.debug("[EXIT] serve_favicon_32: Favicon served successfully")
    return FileResponse(favicon_path, media_type="image/png")


@app.get("/apple-touch-icon.png")
async def serve_apple_touch_icon() -> FileResponse:
    """Serve Apple touch icon for iOS devices."""
    logger.debug("[ENTRY] serve_apple_touch_icon: Serving apple-touch-icon.png")
    icon_path = Path(__file__).parent.parent / "ui" / "favicon_io" / "apple-touch-icon.png"
    
    if not icon_path.exists():
        logger.warning("[EXIT] serve_apple_touch_icon: Icon not found")
        raise HTTPException(status_code=404, detail="Apple touch icon not found")
    
    logger.debug("[EXIT] serve_apple_touch_icon: Icon served successfully")
    return FileResponse(icon_path, media_type="image/png")


@app.get("/android-chrome-192x192.png")
async def serve_android_chrome_192() -> FileResponse:
    """Serve Android Chrome 192x192 icon."""
    logger.debug("[ENTRY] serve_android_chrome_192: Serving android-chrome-192x192.png")
    icon_path = Path(__file__).parent.parent / "ui" / "favicon_io" / "android-chrome-192x192.png"
    
    if not icon_path.exists():
        logger.warning("[EXIT] serve_android_chrome_192: Icon not found")
        raise HTTPException(status_code=404, detail="Android Chrome 192 icon not found")
    
    logger.debug("[EXIT] serve_android_chrome_192: Icon served successfully")
    return FileResponse(icon_path, media_type="image/png")


@app.get("/android-chrome-512x512.png")
async def serve_android_chrome_512() -> FileResponse:
    """Serve Android Chrome 512x512 icon."""
    logger.debug("[ENTRY] serve_android_chrome_512: Serving android-chrome-512x512.png")
    icon_path = Path(__file__).parent.parent / "ui" / "favicon_io" / "android-chrome-512x512.png"
    
    if not icon_path.exists():
        logger.warning("[EXIT] serve_android_chrome_512: Icon not found")
        raise HTTPException(status_code=404, detail="Android Chrome 512 icon not found")
    
    logger.debug("[EXIT] serve_android_chrome_512: Icon served successfully")
    return FileResponse(icon_path, media_type="image/png")


@app.get("/site.webmanifest")
async def serve_webmanifest() -> FileResponse:
    """Serve PWA web manifest."""
    logger.debug("[ENTRY] serve_webmanifest: Serving site.webmanifest")
    manifest_path = Path(__file__).parent.parent / "ui" / "favicon_io" / "site.webmanifest"
    
    if not manifest_path.exists():
        logger.warning("[EXIT] serve_webmanifest: Manifest not found")
        raise HTTPException(status_code=404, detail="Web manifest not found")
    
    logger.debug("[EXIT] serve_webmanifest: Manifest served successfully")
    return FileResponse(manifest_path, media_type="application/manifest+json")


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


# ========== Autonomous Commerce UI ==========

@app.get("/commerce", response_class=HTMLResponse)
async def serve_autonomous_commerce():
    """Serve the Autonomous Commerce marketplace UI."""
    ui_path = Path(__file__).parent.parent / "ui" / "autonomous-commerce.html"
    
    if not ui_path.exists():
        raise HTTPException(status_code=404, detail="Autonomous Commerce UI not found")
    
    with open(ui_path, 'r') as f:
        return HTMLResponse(content=f.read())


@app.get("/autonomous-commerce.js")
async def serve_autonomous_commerce_js():
    """Serve Autonomous Commerce JavaScript file."""
    js_path = Path(__file__).parent.parent / "ui" / "autonomous-commerce.js"
    
    if not js_path.exists():
        raise HTTPException(status_code=404, detail="Autonomous Commerce JavaScript not found")
    
    return FileResponse(js_path, media_type="application/javascript")


@app.get("/static/models/Ch03_nonPBR.fbx")
async def serve_character_model():
    """Serve Michele character FBX model (legacy path, redirects to michelle.fbx)."""
    # Try new name first (michelle.fbx), then fall back to old name
    model_paths = [
        Path("/workspaces/reimagined-umbrella/data/mixamo_anims/fbx/michelle.fbx"),
        Path("/workspaces/reimagined-umbrella/Ch03_nonPBR.fbx"),
    ]
    
    for model_path in model_paths:
        if model_path.exists():
            return FileResponse(model_path, media_type="application/octet-stream")
    
    raise HTTPException(status_code=404, detail="Character model not found")


@app.get("/static/models/michelle.fbx")
async def serve_michelle_character():
    """Serve Michelle character FBX model."""
    model_path = Path("/workspaces/reimagined-umbrella/data/mixamo_anims/fbx/michelle.fbx")
    
    if not model_path.exists():
        raise HTTPException(status_code=404, detail="Michelle character model not found")
    
    return FileResponse(model_path, media_type="application/octet-stream")


@app.get("/static/models/remy.fbx")
async def serve_remy_character():
    """Serve Remy character FBX model."""
    # Try both capitalizations
    model_paths = [
        Path("/workspaces/reimagined-umbrella/data/mixamo_anims/fbx/Remy.fbx"),
        Path("/workspaces/reimagined-umbrella/data/mixamo_anims/fbx/remy.fbx"),
    ]
    
    for model_path in model_paths:
        if model_path.exists():
            return FileResponse(model_path, media_type="application/octet-stream")
    
    raise HTTPException(status_code=404, detail="Remy character model not found")


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


@app.get("/static/animations/{filename:path}")
async def serve_animation_fbx(filename: str):
    """Serve animation FBX files from mixamo_anims directory."""
    # Sanitize filename - allow @ and spaces in filenames
    if ".." in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
    
    # URL decode the filename (spaces may be encoded as %20)
    from urllib.parse import unquote
    decoded_filename = unquote(filename)
    
    model_path = Path(f"/workspaces/reimagined-umbrella/data/mixamo_anims/fbx/{decoded_filename}")
    
    if not model_path.exists():
        raise HTTPException(status_code=404, detail=f"Animation FBX file not found: {decoded_filename}")
    
    return FileResponse(model_path, media_type="application/octet-stream")


@app.get("/api/motions/library")
async def get_motion_library():
    """
    Get motion library catalog.
    
    Returns list of available motions with metadata from Mixamo FBX files.
    Filters out character files (michelle.fbx, Remy.fbx) - only returns animations.
    """
    logger.info("Fetching motion library")
    
    # Path to FBX files
    fbx_dir = Path(__file__).parent.parent.parent.parent / "data" / "mixamo_anims" / "fbx"
    
    # Character files to exclude from animation list
    character_files = {'michelle.fbx', 'remy.fbx', 'ch03_nonpbr.fbx'}
    
    # Animations optimized for specific characters
    # Swing Dancing and Salsa Dancing are optimized for Remy
    # All others are mapped for Michelle
    remy_animations = {'swing dancing.fbx', 'salsa dancing.fbx'}
    
    motions = []
    
    if fbx_dir.exists():
        for fbx_file in fbx_dir.glob("*.fbx"):
            # Skip character files - they are not animations
            if fbx_file.name.lower() in character_files:
                continue
                
            # Extract motion name from filename
            # Example: "X Bot@Capoeira.fbx" -> "Capoeira"
            filename = fbx_file.stem
            motion_name = filename.split('@')[-1] if '@' in filename else filename
            
            # Determine character compatibility
            compatible_character = 'remy' if fbx_file.name.lower() in remy_animations else 'michelle'
            
            # Determine tags and icon based on motion name keywords
            tags = []
            icon = '🎭'  # Default icon
            name_lower = motion_name.lower()
            
            if any(word in name_lower for word in ['walk', 'run', 'jump', 'crouch']):
                tags.append('locomotion')
                icon = '🚶'
            if any(word in name_lower for word in ['salsa']):
                tags.append('dance')
                icon = '💃'
            elif any(word in name_lower for word in ['swing', 'charleston']):
                tags.append('dance')
                icon = '🕺'
            elif any(word in name_lower for word in ['hip hop', 'wave']):
                tags.append('dance')
                icon = '🎤'
            elif any(word in name_lower for word in ['breakdance']):
                tags.append('dance')
                icon = '🤸'
            elif any(word in name_lower for word in ['capoeira']):
                tags.append('dance')
                tags.append('combat')
                icon = '🥋'
            elif any(word in name_lower for word in ['dance']):
                tags.append('dance')
                icon = '💃'
            if any(word in name_lower for word in ['punch', 'kick', 'fight', 'combat']):
                tags.append('combat')
                icon = '👊'
            if any(word in name_lower for word in ['idle', 'stand', 'wait']):
                tags.append('idle')
                icon = '🧍'
            if 'freeze' in name_lower:
                tags.append('freeze')
            if any(word in name_lower for word in ['acrobat', 'flip', 'spin']):
                tags.append('acrobatic')
                icon = '🤸'
            if 'urban' in name_lower or 'street' in name_lower:
                tags.append('urban')
            
            # Generate motion metadata
            motion_id = motion_name.lower().replace(' ', '_').replace('-', '_')
            
            # URL path to serve this animation
            url_path = f"/static/animations/{fbx_file.name}"
            
            motions.append({
                "id": motion_id,
                "name": motion_name,
                "tags": tags if tags else ['other'],
                "icon": icon,
                "character": compatible_character,  # Character this animation is optimized for
                "duration": 3.5 + (hash(motion_name) % 50) / 10,  # Pseudo-random 3.5-8.5s
                "novelty": 0.5 + (hash(motion_name) % 40) / 100,  # Pseudo-random 0.5-0.9
                "filepath": str(fbx_file),
                "path": url_path,
                "frames": 100,  # Will be updated when loaded
                "fps": 30
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


# =============================================================================
# SPADE Hierarchical Motion Blending Endpoints
# =============================================================================


class SPADEBlendAPIRequest(BaseModel):
    """API request for SPADE-enhanced motion blending."""
    motion_paths: List[str] = Field(description="FBX file paths for motions")
    weights: List[float] = Field(description="Blend weights for duration allocation")
    style_labels: List[List[str]] = Field(
        description="Style labels for each motion (e.g., [['capoeira'], ['breakdance']])"
    )
    hierarchy_level: int = Field(
        default=1, ge=1, le=4,
        description="Hierarchy level for SPADE conditioning (1=COARSE optimal)"
    )
    transition_frames: int = Field(default=30, ge=1, le=120, description="Frames for transition")
    use_trainable_params: bool = Field(default=True, description="Use PyTorch trainable γ/β")
    checkpoint_path: Optional[str] = Field(default=None, description="Path to pretrained weights")
    # Trim parameters (optional - defaults to full clips)
    start_frames: Optional[List[int]] = Field(
        default=None, 
        description="Start frame for each motion clip (0-indexed). If None, uses frame 0."
    )
    end_frames: Optional[List[int]] = Field(
        default=None, 
        description="End frame for each motion clip (inclusive). If None, uses last frame."
    )


@app.post("/api/motions/blend/spade")
async def blend_with_spade(request: SPADEBlendAPIRequest):
    """
    Blend motions using SPADE (Spatially-Adaptive Denormalization) hierarchical conditioning.
    
    Based on research showing that applying SPADE conditioning at Level 1 (COARSE)
    yields optimal FID/coverage trade-off by establishing overall motion structure
    at the coarse level (Hips, Spine joints).
    
    Features:
    - 4-level hierarchy: COARSE → MID → FINE → DETAIL
    - Trainable γ/β parameters via PyTorch nn.Module
    - Hash-based style embeddings (MVP), Gemini embeddings in Phase 2
    - FID, coverage, diversity, smoothness metrics
    - Verbose entry/exit logging for CI/CD
    
    Args:
        motion_paths: List of FBX file paths (2-8 motions)
        weights: Blend weights (must sum to 1.0)
        style_labels: Style labels per motion for conditioning
        hierarchy_level: SPADE level (1=COARSE recommended)
        transition_frames: Frames for blend transition
        use_trainable_params: Enable PyTorch trainable parameters
        checkpoint_path: Optional pretrained weights path
    
    Returns:
        SPADEBlendResponse with artifacts, metrics, and debug info
    """
    import time
    import json
    import uuid
    from pathlib import Path
    
    correlation_id = str(uuid.uuid4())[:8]
    logger.info(
        f"[ENTRY] blend_with_spade [{correlation_id}]: "
        f"motions={len(request.motion_paths)}, "
        f"level={request.hierarchy_level}, "
        f"transition={request.transition_frames}, "
        f"trainable={request.use_trainable_params}"
    )
    
    start_time = time.perf_counter()
    
    try:
        # Import SPADE services
        from ..services.spade_blend_service import (
            get_spade_service,
            SPADEBlendService,
            TORCH_AVAILABLE,
        )
        from ..services.metrics import compute_all_metrics
        from ..schemas.models import (
            SPADEConfig,
            SPADEMetrics,
            SPADEBlendResponse,
            HierarchyLevel,
        )
        from ..utils.fbx_parser import get_fbx_parser
        import numpy as np
        
        # Validate inputs
        if len(request.motion_paths) < 2:
            raise HTTPException(400, "SPADE blending requires at least 2 motions")
        
        if len(request.motion_paths) > 8:
            raise HTTPException(400, "SPADE blending supports maximum 8 motions")
        
        if len(request.motion_paths) != len(request.weights):
            raise HTTPException(400, "motion_paths and weights must have same length")
        
        if len(request.motion_paths) != len(request.style_labels):
            raise HTTPException(400, "motion_paths and style_labels must have same length")
        
        # Validate trim parameters if provided
        if request.start_frames is not None:
            if len(request.start_frames) != len(request.motion_paths):
                raise HTTPException(400, "start_frames must match motion_paths length")
        if request.end_frames is not None:
            if len(request.end_frames) != len(request.motion_paths):
                raise HTTPException(400, "end_frames must match motion_paths length")
        
        weight_sum = sum(request.weights)
        if abs(weight_sum - 1.0) > 1e-6:
            raise HTTPException(400, f"Weights must sum to 1.0, got {weight_sum}")
        
        # Map hierarchy level
        level_map = {
            1: HierarchyLevel.COARSE,
            2: HierarchyLevel.MID,
            3: HierarchyLevel.FINE,
            4: HierarchyLevel.DETAIL,
        }
        spade_level = level_map[request.hierarchy_level]
        
        logger.info(
            f"[PROGRESS] blend_with_spade [{correlation_id}]: "
            f"Loading {len(request.motion_paths)} motions..."
        )
        
        # Load motions from FBX files
        fbx_parser = get_fbx_parser()
        motions = []
        motion_info = []
        
        for idx, path in enumerate(request.motion_paths):
            # Handle static file paths
            if path.startswith('/static/models/'):
                fbx_path = Path(path.replace('/static/models/', 'data/mixamo_anims/fbx/'))
            else:
                fbx_path = Path(path)
            
            if not fbx_path.exists():
                # Try alternate location
                alt_path = Path("data/mixamo_anims/fbx") / fbx_path.name
                if alt_path.exists():
                    fbx_path = alt_path
                else:
                    logger.warning(f"Motion file not found: {fbx_path}, generating synthetic")
                    # Generate synthetic motion for demo
                    T = 120
                    J = 24  # Mixamo skeleton
                    D = 3
                    positions = np.random.randn(T, J, D).astype(np.float32) * 0.5
                    motions.append(positions)
                    motion_info.append({
                        "path": str(fbx_path),
                        "name": fbx_path.stem,
                        "frames": T,
                        "joints": J,
                        "style_labels": request.style_labels[idx],
                        "synthetic": True,
                        "trim_applied": False,
                    })
                    continue
            
            # Parse real FBX
            positions, metadata = fbx_parser.parse_fbx(str(fbx_path))
            original_frames = positions.shape[0]
            
            # Apply trim slicing if specified
            start_frame = 0
            end_frame = original_frames - 1
            
            if request.start_frames is not None and idx < len(request.start_frames):
                start_frame = max(0, request.start_frames[idx])
            if request.end_frames is not None and idx < len(request.end_frames):
                end_frame = min(original_frames - 1, request.end_frames[idx])
            
            # Validate trim range
            if start_frame > end_frame:
                logger.warning(
                    f"[WARNING] blend_with_spade [{correlation_id}]: "
                    f"Invalid trim range for motion {idx}: start={start_frame} > end={end_frame}. "
                    f"Using full clip."
                )
                start_frame = 0
                end_frame = original_frames - 1
            
            # Slice motion by trim range (end_frame is inclusive)
            trimmed_positions = positions[start_frame:end_frame + 1]
            trim_applied = (start_frame != 0 or end_frame != original_frames - 1)
            
            logger.info(
                f"[PROGRESS] blend_with_spade [{correlation_id}]: "
                f"Motion {idx+1} trim: [{start_frame}:{end_frame+1}] of {original_frames} frames "
                f"-> {trimmed_positions.shape[0]} frames (trim_applied={trim_applied})"
            )
            
            motions.append(trimmed_positions)
            motion_info.append({
                "path": str(fbx_path),
                "name": fbx_path.stem,
                "frames": trimmed_positions.shape[0],
                "original_frames": original_frames,
                "joints": trimmed_positions.shape[1],
                "style_labels": request.style_labels[idx],
                "synthetic": False,
                "trim_applied": trim_applied,
                "start_frame": start_frame,
                "end_frame": end_frame,
            })
            
            logger.debug(
                f"[PROGRESS] blend_with_spade [{correlation_id}]: "
                f"Loaded motion {idx+1}: {fbx_path.stem} ({trimmed_positions.shape})"
            )
        
        # Initialize SPADE service with config
        config = SPADEConfig(
            spade_level=spade_level,
            input_dim=768,
            style_channels=128,
            motion_channels=motions[0].shape[2] if len(motions[0].shape) > 2 else 3,
            transition_sharpness=5.0,
        )
        
        spade_service = get_spade_service(config)
        
        # Load checkpoint if provided
        checkpoint_loaded = False
        if request.checkpoint_path:
            checkpoint_loaded = spade_service.load_checkpoint(request.checkpoint_path)
        
        logger.info(
            f"[PROGRESS] blend_with_spade [{correlation_id}]: "
            f"SPADE service ready (PyTorch={TORCH_AVAILABLE}, "
            f"device={spade_service.device}, "
            f"params={spade_service.trainable_params_count:,})"
        )
        
        # Perform SPADE blending (currently 2-motion support)
        # TODO: Extend to N-motion sequential blending
        if len(motions) == 2:
            blended, timing = spade_service.blend(
                motion_a=motions[0],
                motion_b=motions[1],
                style_labels_a=request.style_labels[0],
                style_labels_b=request.style_labels[1],
                weights=request.weights,
                transition_frames=request.transition_frames,
            )
        else:
            # Sequential pairwise blending for N motions
            logger.info(f"[PROGRESS] blend_with_spade [{correlation_id}]: Sequential N-motion blend")
            blended = motions[0]
            total_weight = request.weights[0]
            
            for i in range(1, len(motions)):
                relative_weight = request.weights[i] / (total_weight + request.weights[i])
                blended, _ = spade_service.blend(
                    motion_a=blended,
                    motion_b=motions[i],
                    style_labels_a=request.style_labels[i-1],
                    style_labels_b=request.style_labels[i],
                    weights=[1.0 - relative_weight, relative_weight],
                    transition_frames=request.transition_frames,
                )
                total_weight += request.weights[i]
            
            timing = {"blend_time_ms": 0, "total_frames": blended.shape[0]}
        
        # Compute quality metrics
        logger.info(f"[PROGRESS] blend_with_spade [{correlation_id}]: Computing metrics...")
        
        metrics_result = compute_all_metrics(
            generated_motion=blended,
            reference_motions=np.stack(motions) if len(motions) > 1 else None,
            transition_start=timing.get("transition_start", 0),
            transition_end=timing.get("transition_end", blended.shape[0]),
            fps=30.0,
        )
        
        # Create SPADE metrics
        spade_metrics = SPADEMetrics(
            fid_score=metrics_result.fid_score,
            coverage=metrics_result.coverage,
            diversity=metrics_result.diversity,
            smoothness=metrics_result.smoothness,
            foot_sliding=metrics_result.foot_sliding,
            spade_level_used=request.hierarchy_level,
            transition_quality=metrics_result.transition_quality,
            blend_time_ms=timing.get("blend_time_ms", 0),
            metrics_time_ms=metrics_result.computation_time_ms,
        )
        
        # Generate blend ID and save artifacts
        blend_id = f"spade-{uuid.uuid4().hex[:12]}"
        artifacts_dir = Path("artifacts") / blend_id
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        
        # Save artifacts (frame-by-frame)
        artifacts = []
        transition_start = timing.get("transition_start", 0)
        transition_end = timing.get("transition_end", blended.shape[0])
        
        for t in range(blended.shape[0]):
            # Determine blend mode for this frame
            if t < transition_start:
                blend_mode = "motion_a"
            elif t >= transition_end:
                blend_mode = "motion_b"
            else:
                blend_mode = f"spade_level_{request.hierarchy_level}"
            
            artifact = {
                "frame_index": t,
                "omega": float(t - transition_start) / max(transition_end - transition_start, 1),
                "positions": blended[t].tolist(),
                "blend_mode": blend_mode,
                "t_normalized": t / max(blended.shape[0] - 1, 1),
            }
            artifacts.append(artifact)
            
            # Save to file
            with open(artifacts_dir / f"frame-{t:04d}.json", 'w') as f:
                json.dump(artifact, f, indent=2)
        
        # Save metadata
        metadata = {
            "blend_id": blend_id,
            "spade_level": request.hierarchy_level,
            "spade_level_name": spade_level.value,
            "source_motions": motion_info,
            "weights": request.weights,
            "transition_frames": request.transition_frames,
            "total_frames": blended.shape[0],
            "metrics": spade_metrics.model_dump(),
            "pytorch_available": TORCH_AVAILABLE,
            "device": spade_service.device,
            "trainable_params": spade_service.trainable_params_count,
            "checkpoint_loaded": checkpoint_loaded,
        }
        
        with open(artifacts_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Compute joint dynamics for velocity/acceleration charts
        from ..services.metrics import compute_joint_dynamics
        
        logger.info(f"[PROGRESS] blend_with_spade [{correlation_id}]: Computing joint dynamics...")
        
        joint_dynamics = compute_joint_dynamics(
            motion=blended,
            transition_start=transition_start,
            transition_end=transition_end,
            fps=30.0,
        )
        
        # Save joint dynamics to artifacts
        with open(artifacts_dir / "joint_dynamics.json", 'w') as f:
            json.dump(joint_dynamics, f, indent=2)
        
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        
        logger.info(
            f"[EXIT] blend_with_spade [{correlation_id}]: "
            f"SUCCESS blend_id={blend_id}, "
            f"frames={blended.shape[0]}, "
            f"fid={spade_metrics.fid_score:.4f}, "
            f"coverage={spade_metrics.coverage:.4f}, "
            f"smoothness={spade_metrics.smoothness:.4f}, "
            f"time={elapsed_ms:.2f}ms"
        )
        
        # Build response
        return {
            "status": "success",
            "blend_id": blend_id,
            "spade_config": {
                "level": request.hierarchy_level,
                "level_name": spade_level.value,
                "transition_frames": request.transition_frames,
                "trainable_params": request.use_trainable_params,
            },
            "source_motions": motion_info,
            "total_frames": blended.shape[0],
            "output_fps": 30,
            "artifact_directory": str(artifacts_dir),
            "artifacts": artifacts[:10],  # Preview first 10 frames
            "artifact_urls": [
                f"/api/artifacts/{blend_id}/frame-{i:04d}.json"
                for i in range(min(100, blended.shape[0]))
            ],
            "joint_dynamics": joint_dynamics,  # L2 velocity/acceleration data
            "metrics": spade_metrics.model_dump(),
            "debug": {
                "pytorch_available": TORCH_AVAILABLE,
                "device": spade_service.device,
                "trainable_params_count": spade_service.trainable_params_count,
                "checkpoint_loaded": checkpoint_loaded,
                "total_time_ms": elapsed_ms,
            },
            "warnings": [] if TORCH_AVAILABLE else ["PyTorch unavailable, using numpy fallback"],
        }
        
    except HTTPException:
        raise
    except Exception as e:
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        logger.error(
            f"[EXIT] blend_with_spade [{correlation_id}]: "
            f"FAILED after {elapsed_ms:.2f}ms - {e}",
            exc_info=True
        )
        raise HTTPException(500, f"SPADE blend failed: {str(e)}")


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


@app.get("/api/artifacts/{blend_id}/joint-dynamics")
async def get_artifact_joint_dynamics(blend_id: str):
    """
    Retrieve joint dynamics data (L2 velocity and acceleration) for visualization.
    
    Returns per-joint velocity and acceleration time series for:
    - Pelvis (root joint)
    - LeftWrist, RightWrist (hands)
    - LeftFoot, RightFoot (feet)
    
    Data format matches reference images with transition boundaries marked.
    """
    logger.info(f"[ENTRY] get_artifact_joint_dynamics: blend_id={blend_id}")
    
    try:
        artifacts_dir = Path("artifacts") / blend_id
        dynamics_file = artifacts_dir / "joint_dynamics.json"
        
        if not dynamics_file.exists():
            logger.warning(f"[EXIT] get_artifact_joint_dynamics: File not found for {blend_id}")
            raise HTTPException(404, f"Joint dynamics not found for: {blend_id}")
        
        with open(dynamics_file, 'r') as f:
            dynamics = json.load(f)
        
        logger.info(
            f"[EXIT] get_artifact_joint_dynamics: blend_id={blend_id}, "
            f"joints={dynamics.get('joint_names', [])}, "
            f"frames={dynamics.get('total_frames', 0)}"
        )
        return dynamics
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[EXIT] get_artifact_joint_dynamics: Failed - {e}", exc_info=True)
        raise HTTPException(500, f"Failed to load joint dynamics: {str(e)}")


@app.get("/api/artifacts/list")
async def list_artifacts():
    """
    List all available blend artifacts in the database.
    
    Returns a list of artifact summaries with metadata for UI display.
    """
    logger.info("[ENTRY] list_artifacts")
    
    try:
        artifacts_dir = Path("artifacts")
        if not artifacts_dir.exists():
            logger.info("[EXIT] list_artifacts: No artifacts directory")
            return {"artifacts": [], "total": 0}
        
        artifacts = []
        for blend_dir in sorted(artifacts_dir.iterdir(), reverse=True):
            if blend_dir.is_dir() and blend_dir.name.startswith("spade-"):
                metadata_file = blend_dir / "metadata.json"
                dynamics_file = blend_dir / "joint_dynamics.json"
                
                if metadata_file.exists():
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                    
                    # Count frame files
                    frame_count = len(list(blend_dir.glob("frame-*.json")))
                    
                    artifacts.append({
                        "blend_id": blend_dir.name,
                        "spade_level": metadata.get("spade_level", 1),
                        "spade_level_name": metadata.get("spade_level_name", "COARSE"),
                        "total_frames": metadata.get("total_frames", frame_count),
                        "source_motions": [m.get("name", "Unknown") for m in metadata.get("source_motions", [])],
                        "weights": metadata.get("weights", []),
                        "metrics": metadata.get("metrics", {}),
                        "has_joint_dynamics": dynamics_file.exists(),
                        "created_at": metadata_file.stat().st_mtime,
                    })
        
        logger.info(f"[EXIT] list_artifacts: Found {len(artifacts)} artifacts")
        return {"artifacts": artifacts, "total": len(artifacts)}
        
    except Exception as e:
        logger.error(f"[EXIT] list_artifacts: Failed - {e}", exc_info=True)
        raise HTTPException(500, f"Failed to list artifacts: {str(e)}")


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

# ===========================
# Arc Network Tokenization
# ===========================

class ArtifactTokenizationRequest(BaseModel):
    """Request to tokenize a blend artifact on Arc Network."""
    artifact_data: Dict[str, Any] = Field(..., description="Complete artifact data from blend generation")
    buyer_address: str = Field(..., pattern=r"^0x[a-fA-F0-9]{40}$", description="Wallet address to receive NFT")
    price_usdc: float = Field(..., ge=0, description="Price in USDC")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


@app.post("/api/arc/tokenize-artifact")
async def tokenize_artifact(request: ArtifactTokenizationRequest):
    """
    Tokenize a blend artifact as an NFT on Arc testnet.
    
    This endpoint:
    1. Creates metadata JSON for the blend artifact
    2. Mints an NFT on Arc Network via NPCMotionRegistry
    3. Assigns ownership to the buyer's wallet
    4. Returns transaction hash and token ID
    
    Args:
        request: Artifact tokenization request with artifact data, buyer address, and price
    
    Returns:
        JSON with token_id, tx_hash, ipfs_uri, and metadata
    """
    try:
        logger.info(f"Tokenizing artifact for buyer: {request.buyer_address}")
        logger.info(f"  Price: {request.price_usdc} USDC")
        logger.info(f"  Artifact frames: {len(request.artifact_data.get('artifacts', []))}")
        
        # For hackathon demo: simulate tokenization (no real blockchain tx)
        # In production: would call ArcNetworkService.mint_motion_pack()
        
        # Generate mock token ID and tx hash
        import hashlib
        import time
        
        token_data = f"{request.buyer_address}{time.time()}{request.price_usdc}"
        token_id = int(hashlib.sha256(token_data.encode()).hexdigest()[:16], 16) % 1000000
        tx_hash = "0x" + hashlib.sha256(f"tx_{token_data}".encode()).hexdigest()
        
        # Create artifact metadata
        artifact_metadata = {
            "name": request.metadata.get("name", "Blend Artifact"),
            "description": f"Motion blend artifact with {len(request.artifact_data.get('artifacts', []))} frames",
            "frames": len(request.artifact_data.get('artifacts', [])),
            "duration_seconds": len(request.artifact_data.get('artifacts', [])) / 30,
            "motion_count": len(request.artifact_data.get('motion_segments', [])),
            "transition_count": request.artifact_data.get('total_transitions', 0),
            "created_at": request.metadata.get("created", ""),
            "creator": request.buyer_address,
            "price_usdc": request.price_usdc,
            "blend_mode": "smoothstep",
            "quality_tier": request.artifact_data.get('aggregate_metrics', {}).get('quality_tier', 'Standard'),
            "attributes": [
                {
                    "trait_type": "Frame Count",
                    "value": len(request.artifact_data.get('artifacts', []))
                },
                {
                    "trait_type": "Motion Sequences",
                    "value": len(request.artifact_data.get('motion_segments', []))
                },
                {
                    "trait_type": "Transitions",
                    "value": request.artifact_data.get('total_transitions', 0)
                },
                {
                    "trait_type": "Quality",
                    "value": request.artifact_data.get('aggregate_metrics', {}).get('quality_tier', 'Standard')
                }
            ]
        }
        
        # Mock IPFS URI (in production: upload to IPFS)
        ipfs_uri = f"ipfs://Qm{hashlib.sha256(token_data.encode()).hexdigest()[:44]}"
        
        logger.info(f"✅ Artifact tokenized successfully")
        logger.info(f"  Token ID: {token_id}")
        logger.info(f"  Transaction: {tx_hash}")
        logger.info(f"  IPFS URI: {ipfs_uri}")
        
        return {
            "success": True,
            "token_id": token_id,
            "tx_hash": tx_hash,
            "ipfs_uri": ipfs_uri,
            "metadata": artifact_metadata,
            "buyer": request.buyer_address,
            "price_usdc": request.price_usdc,
            "chain": "Arc Testnet",
            "chain_id": 52085143,
            "contract_address": os.getenv("NPC_MOTION_REGISTRY_ADDRESS", "0x742d35Cc6634C0532925a3b844Bc9e7595f0bEb0"),
            "message": "Artifact successfully tokenized on Arc testnet! (Demo mode - no real blockchain transaction)"
        }
        
    except Exception as e:
        logger.error(f"Tokenization failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Tokenization failed: {str(e)}"
        )


# ============================================================================
# API V2: Circle Wallets, x402 Payments, and Royalty Chain Endpoints
# ============================================================================

# Import new services (optional - may not be installed)
try:
    from ..services.circle_wallets import CircleWalletsService, CircleWalletsConfig
    from ..services.x402_facilitator import X402FacilitatorService, X402FacilitatorConfig
    from ..services.payment_automation import PaymentAutomationService, PaymentAutomationConfig
    from ..services.gasless_executor import GaslessExecutor, GaslessExecutorConfig
    from ..services.gemini_oracle_attestor import GeminiOracleAttestor, GeminiOracleConfig
    from ..schemas.models import (
        RoyaltyChain,
        RoyaltyNode,
        PaymentTriggerEvent,
        RecursivePayoutResult,
        CircleWallet,
        CircleTransfer,
        DerivativeDetectionResult,
        ROYALTY_DECAY_FACTOR,
        MAX_ROYALTY_CHAIN_DEPTH,
        GAS_SPONSOR_REPLENISH_THRESHOLD,
    )
    HAS_V2_SERVICES = True
except ImportError as e:
    logger.warning(f"V2 services not available: {e}")
    HAS_V2_SERVICES = False


# V2 Service instances (lazy initialization)
_circle_service: Optional[Any] = None
_x402_service: Optional[Any] = None
_payment_service: Optional[Any] = None
_gasless_executor: Optional[Any] = None
_oracle_attestor: Optional[Any] = None


def get_circle_service():
    """Get or create Circle Wallets service instance."""
    global _circle_service
    if _circle_service is None and HAS_V2_SERVICES:
        config = CircleWalletsConfig()
        _circle_service = CircleWalletsService(config)
    return _circle_service


def get_x402_service():
    """Get or create x402 Facilitator service instance."""
    global _x402_service
    if _x402_service is None and HAS_V2_SERVICES:
        config = X402FacilitatorConfig()
        _x402_service = X402FacilitatorService(config)
    return _x402_service


def get_payment_service():
    """Get or create Payment Automation service instance."""
    global _payment_service
    if _payment_service is None and HAS_V2_SERVICES:
        config = PaymentAutomationConfig()
        _payment_service = PaymentAutomationService(config)
    return _payment_service


def get_gasless_executor():
    """Get or create Gasless Executor instance."""
    global _gasless_executor
    if _gasless_executor is None and HAS_V2_SERVICES:
        config = GaslessExecutorConfig()
        _gasless_executor = GaslessExecutor(config)
    return _gasless_executor


def get_oracle_attestor():
    """Get or create Gemini Oracle Attestor instance."""
    global _oracle_attestor
    if _oracle_attestor is None and HAS_V2_SERVICES:
        config = GeminiOracleConfig()
        _oracle_attestor = GeminiOracleAttestor(config)
    return _oracle_attestor


# ---- Request/Response Models for V2 API ----

class CreateWalletRequest(BaseModel):
    """Request to create a new Circle wallet."""
    idempotency_key: Optional[str] = None
    wallet_set_id: Optional[str] = None


class TransferUsdcRequest(BaseModel):
    """Request to transfer USDC between wallets."""
    from_wallet_id: str
    to_address: str = Field(pattern=r"^0x[a-fA-F0-9]{40}$")
    amount_usdc: str
    idempotency_key: Optional[str] = None


class X402SettlementRequest(BaseModel):
    """Request to settle an x402 payment."""
    x402_header: str
    expected_amount: Optional[str] = None


class RoyaltyPreviewRequest(BaseModel):
    """Request to preview royalty distribution."""
    motion_id: str
    base_amount: float = Field(ge=0.01)
    parent_motion_ids: List[str] = []


class DerivativeCheckRequest(BaseModel):
    """Request to check for derivative content."""
    motion_data_uri: str
    threshold: float = Field(default=0.85, ge=0.0, le=1.0)


class GaslessTransferRequest(BaseModel):
    """Request to execute a gasless USDC transfer."""
    to_address: str = Field(pattern=r"^0x[a-fA-F0-9]{40}$")
    amount_usdc: float = Field(ge=0.01)


# ---- V2 API Endpoints ----

@app.post("/api/v2/wallets/create")
async def create_circle_wallet(
    request: CreateWalletRequest,
    x_correlation_id: Optional[str] = Header(None, alias="X-Correlation-ID"),
):
    """
    Create a new Circle Programmable Wallet.
    
    Creates a developer-controlled wallet on the Arc testnet (or mainnet if enabled).
    """
    if not HAS_V2_SERVICES:
        raise HTTPException(status_code=501, detail="V2 services not available")
    
    if x_correlation_id:
        set_correlation_id(x_correlation_id)
    
    logger.info("[API] POST /api/v2/wallets/create")
    
    try:
        service = get_circle_service()
        wallet = await service.create_developer_wallet(
            idempotency_key=request.idempotency_key,
            wallet_set_id=request.wallet_set_id,
        )
        
        return {
            "success": True,
            "wallet": wallet.model_dump() if hasattr(wallet, 'model_dump') else wallet,
        }
    except Exception as e:
        logger.error(f"Wallet creation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v2/wallets/{wallet_id}/balance")
async def get_wallet_balance(
    wallet_id: str,
    x_correlation_id: Optional[str] = Header(None, alias="X-Correlation-ID"),
):
    """
    Get the USDC balance of a Circle wallet.
    """
    if not HAS_V2_SERVICES:
        raise HTTPException(status_code=501, detail="V2 services not available")
    
    if x_correlation_id:
        set_correlation_id(x_correlation_id)
    
    logger.info(f"[API] GET /api/v2/wallets/{wallet_id}/balance")
    
    try:
        service = get_circle_service()
        balance = await service.get_wallet_balance(wallet_id)
        
        return {
            "wallet_id": wallet_id,
            "balance": balance.model_dump() if hasattr(balance, 'model_dump') else balance,
        }
    except Exception as e:
        logger.error(f"Balance fetch failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v2/wallets/{wallet_id}/transactions")
async def get_wallet_transactions(
    wallet_id: str,
    limit: int = 20,
    x_correlation_id: Optional[str] = Header(None, alias="X-Correlation-ID"),
):
    """
    Get recent transactions for a Circle wallet.
    """
    if not HAS_V2_SERVICES:
        raise HTTPException(status_code=501, detail="V2 services not available")
    
    if x_correlation_id:
        set_correlation_id(x_correlation_id)
    
    logger.info(f"[API] GET /api/v2/wallets/{wallet_id}/transactions")
    
    try:
        service = get_circle_service()
        # Get transaction history
        # Note: This would need to be implemented in the CircleWalletsService
        return {
            "wallet_id": wallet_id,
            "transactions": [],
            "message": "Transaction history not yet implemented",
        }
    except Exception as e:
        logger.error(f"Transaction fetch failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v2/wallets/transfer")
async def transfer_usdc(
    request: TransferUsdcRequest,
    x_correlation_id: Optional[str] = Header(None, alias="X-Correlation-ID"),
):
    """
    Transfer USDC between wallets.
    """
    if not HAS_V2_SERVICES:
        raise HTTPException(status_code=501, detail="V2 services not available")
    
    if x_correlation_id:
        set_correlation_id(x_correlation_id)
    
    logger.info(f"[API] POST /api/v2/wallets/transfer: {request.amount_usdc} USDC")
    
    try:
        service = get_circle_service()
        transfer = await service.transfer_usdc(
            from_wallet_id=request.from_wallet_id,
            to_address=request.to_address,
            amount_usdc=request.amount_usdc,
            idempotency_key=request.idempotency_key,
        )
        
        return {
            "success": True,
            "transfer": transfer.model_dump() if hasattr(transfer, 'model_dump') else transfer,
        }
    except Exception as e:
        logger.error(f"Transfer failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v2/payments/verify")
async def verify_x402_payment(
    x_402_payment: str = Header(..., alias="X-402-Payment"),
    expected_amount: Optional[str] = Header(None, alias="X-Expected-Amount"),
    x_correlation_id: Optional[str] = Header(None, alias="X-Correlation-ID"),
):
    """
    Verify an x402 payment proof from HTTP header.
    
    Used for pay-per-request API access and motion content purchases.
    """
    if not HAS_V2_SERVICES:
        raise HTTPException(status_code=501, detail="V2 services not available")
    
    if x_correlation_id:
        set_correlation_id(x_correlation_id)
    
    logger.info("[API] POST /api/v2/payments/verify")
    
    try:
        service = get_x402_service()
        result = await service.verify_payment(
            x402_header=x_402_payment,
            expected_amount=expected_amount,
        )
        
        if not result.valid:
            raise HTTPException(
                status_code=402,
                detail={
                    "error": "payment_required",
                    "message": result.error_message or "Invalid payment proof",
                    "receipt_id": result.receipt_id,
                },
            )
        
        return {
            "valid": True,
            "receipt_id": result.receipt_id,
            "amount": result.amount,
            "timestamp": result.timestamp,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Payment verification failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v2/payments/settle")
async def settle_x402_payment(
    request: X402SettlementRequest,
    x_correlation_id: Optional[str] = Header(None, alias="X-Correlation-ID"),
):
    """
    Settle an x402 payment and execute the USDC transfer on-chain.
    """
    if not HAS_V2_SERVICES:
        raise HTTPException(status_code=501, detail="V2 services not available")
    
    if x_correlation_id:
        set_correlation_id(x_correlation_id)
    
    logger.info("[API] POST /api/v2/payments/settle")
    
    try:
        service = get_x402_service()
        result = await service.settle_payment(
            x402_header=request.x402_header,
        )
        
        if not result.success:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "settlement_failed",
                    "message": result.error_message,
                    "tx_hash": result.tx_hash,
                },
            )
        
        return {
            "success": True,
            "tx_hash": result.tx_hash,
            "chain": result.chain,
            "amount": result.amount,
            "timestamp": result.timestamp,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Payment settlement failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v2/payments/supported")
async def get_supported_payment_methods():
    """
    Get supported x402 payment methods.
    """
    if not HAS_V2_SERVICES:
        raise HTTPException(status_code=501, detail="V2 services not available")
    
    try:
        service = get_x402_service()
        methods = await service.get_supported_methods()
        
        return {
            "methods": [m.model_dump() if hasattr(m, 'model_dump') else m for m in methods],
        }
    except Exception as e:
        logger.error(f"Failed to get payment methods: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v2/royalties/preview")
async def preview_royalty_distribution(
    request: RoyaltyPreviewRequest,
    x_correlation_id: Optional[str] = Header(None, alias="X-Correlation-ID"),
):
    """
    Preview royalty distribution for a motion with derivative chain.
    
    Calculates the recursive payout distribution using:
    - ROYALTY_DECAY_FACTOR: 50% per level
    - MAX_ROYALTY_CHAIN_DEPTH: 10 levels
    """
    if not HAS_V2_SERVICES:
        raise HTTPException(status_code=501, detail="V2 services not available")
    
    if x_correlation_id:
        set_correlation_id(x_correlation_id)
    
    logger.info(f"[API] POST /api/v2/royalties/preview: {request.base_amount} USDC")
    
    try:
        service = get_payment_service()
        
        # Build a simple royalty chain for preview
        # In production, this would fetch from the registry
        nodes = []
        for i, parent_id in enumerate(request.parent_motion_ids[:MAX_ROYALTY_CHAIN_DEPTH]):
            royalty_rate = 0.10 * (ROYALTY_DECAY_FACTOR ** i)  # 10% base, decayed
            nodes.append({
                "motion_id": parent_id,
                "wallet_address": f"0x{'0' * 39}{i + 1}",  # Placeholder
                "royalty_percentage": royalty_rate,
                "depth": i,
            })
        
        # Calculate payouts
        payouts = []
        total_royalties = 0
        
        for i, node in enumerate(nodes):
            amount = request.base_amount * node["royalty_percentage"]
            if amount >= 0.01:  # Minimum $0.01
                payouts.append({
                    "motion_id": node["motion_id"],
                    "wallet_address": node["wallet_address"],
                    "amount_usdc": round(amount, 6),
                    "depth": i,
                    "percentage": round(node["royalty_percentage"] * 100, 2),
                })
                total_royalties += amount
        
        creator_amount = request.base_amount - total_royalties
        
        return {
            "motion_id": request.motion_id,
            "base_amount": request.base_amount,
            "royalty_payouts": payouts,
            "total_royalties": round(total_royalties, 6),
            "creator_amount": round(creator_amount, 6),
            "chain_depth": len(payouts),
            "decay_factor": ROYALTY_DECAY_FACTOR,
            "max_depth": MAX_ROYALTY_CHAIN_DEPTH,
        }
    except Exception as e:
        logger.error(f"Royalty preview failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v2/derivatives/check")
async def check_derivative_content(
    request: DerivativeCheckRequest,
    x_correlation_id: Optional[str] = Header(None, alias="X-Correlation-ID"),
):
    """
    Check if motion content is a derivative of existing registered motions.
    
    Uses KNN similarity search to find potential parent motions.
    """
    if not HAS_V2_SERVICES:
        raise HTTPException(status_code=501, detail="V2 services not available")
    
    if x_correlation_id:
        set_correlation_id(x_correlation_id)
    
    logger.info("[API] POST /api/v2/derivatives/check")
    
    try:
        attestor = get_oracle_attestor()
        result = await attestor.detect_derivative(
            motion_data_uri=request.motion_data_uri,
            threshold=request.threshold,
        )
        
        return {
            "is_derivative": result.is_derivative,
            "similarity_score": result.similarity_score,
            "parent_motion_id": result.parent_motion_id,
            "parent_creator": result.parent_creator,
            "recommended_royalty": result.recommended_royalty,
            "confidence": result.confidence,
        }
    except Exception as e:
        logger.error(f"Derivative check failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v2/gasless/transfer")
async def execute_gasless_transfer(
    request: GaslessTransferRequest,
    x_correlation_id: Optional[str] = Header(None, alias="X-Correlation-ID"),
):
    """
    Execute a gasless USDC transfer via EIP-7702.
    
    Gas is sponsored by the platform gas budget (40% replenish threshold).
    """
    if not HAS_V2_SERVICES:
        raise HTTPException(status_code=501, detail="V2 services not available")
    
    if x_correlation_id:
        set_correlation_id(x_correlation_id)
    
    logger.info(f"[API] POST /api/v2/gasless/transfer: {request.amount_usdc} USDC")
    
    try:
        executor = get_gasless_executor()
        result = await executor.send_gasless_tx(
            to_address=request.to_address,
            amount_usdc=request.amount_usdc,
        )
        
        return {
            "success": result.success,
            "tx_hash": result.tx_hash,
            "gas_sponsored": result.gas_sponsored,
            "gas_cost_usdc": result.gas_cost_usdc,
        }
    except Exception as e:
        logger.error(f"Gasless transfer failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v2/gasless/budget")
async def get_gas_sponsor_budget(
    x_correlation_id: Optional[str] = Header(None, alias="X-Correlation-ID"),
):
    """
    Get the current gas sponsor budget status.
    
    Returns current balance and whether replenishment is needed (40% threshold).
    """
    if not HAS_V2_SERVICES:
        raise HTTPException(status_code=501, detail="V2 services not available")
    
    if x_correlation_id:
        set_correlation_id(x_correlation_id)
    
    logger.info("[API] GET /api/v2/gasless/budget")
    
    try:
        executor = get_gasless_executor()
        balance = await executor.get_gas_budget_balance()
        threshold = executor.config.initial_budget * GAS_SPONSOR_REPLENISH_THRESHOLD
        
        return {
            "current_balance": balance,
            "initial_budget": executor.config.initial_budget,
            "threshold": threshold,
            "threshold_percentage": GAS_SPONSOR_REPLENISH_THRESHOLD * 100,
            "needs_replenishment": balance < threshold,
        }
    except Exception as e:
        logger.error(f"Failed to get gas budget: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v2/config/constants")
async def get_global_constants():
    """
    Get global configuration constants for the payment system.
    """
    if not HAS_V2_SERVICES:
        return {
            "royalty_decay_factor": 0.5,
            "max_royalty_chain_depth": 10,
            "gas_sponsor_replenish_threshold": 0.40,
            "v2_services_available": False,
        }
    
    return {
        "royalty_decay_factor": ROYALTY_DECAY_FACTOR,
        "max_royalty_chain_depth": MAX_ROYALTY_CHAIN_DEPTH,
        "gas_sponsor_replenish_threshold": GAS_SPONSOR_REPLENISH_THRESHOLD,
        "v2_services_available": True,
    }


# ========== Marketplace Endpoints ==========

class BlendRegistrationRequest(BaseModel):
    """Request to register a new blend strip."""
    motion_id: str
    creator: str
    metadata: Dict[str, Any]
    price_usdc: float
    uniqueness_score: int


@app.get("/api/v2/marketplace/blends")
async def get_marketplace_blends(
    limit: int = 50,
    offset: int = 0,
    x_correlation_id: Optional[str] = Header(None, alias="X-Correlation-ID"),
):
    """
    Get blend strips available in the marketplace.
    
    Returns a list of minted blend strips with metadata, pricing, and uniqueness scores.
    """
    if x_correlation_id:
        set_correlation_id(x_correlation_id)
    
    logger.info(f"[API] GET /api/v2/marketplace/blends limit={limit} offset={offset}")
    
    # In production, this would query a database of registered blends
    # For now, return sample data for UI demonstration
    sample_blends = []
    
    return {
        "blends": sample_blends,
        "total": len(sample_blends),
        "limit": limit,
        "offset": offset,
    }


@app.post("/api/v2/motions/register")
async def register_blend_motion(
    request: BlendRegistrationRequest,
    x_correlation_id: Optional[str] = Header(None, alias="X-Correlation-ID"),
):
    """
    Register a new blend strip motion on Arc Network.
    
    Mints the blend as an NFT with royalty chain configuration.
    """
    if x_correlation_id:
        set_correlation_id(x_correlation_id)
    
    logger.info(f"[API] POST /api/v2/motions/register: {request.motion_id}")
    
    try:
        # Generate transaction hash (in production, this would be the actual mint tx)
        import hashlib
        tx_data = f"{request.motion_id}{request.creator}{request.price_usdc}".encode()
        tx_hash = "0x" + hashlib.sha256(tx_data).hexdigest()
        
        return {
            "success": True,
            "motion_id": request.motion_id,
            "tx_hash": tx_hash,
            "price_usdc": request.price_usdc,
            "uniqueness_score": request.uniqueness_score,
            "creator": request.creator,
            "chain": "arc-testnet",
            "chain_id": 1301,
        }
    except Exception as e:
        logger.error(f"Motion registration failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))