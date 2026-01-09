"""
Kinetic Ledger Phase-2 API Server.

FastAPI server that exposes the trustless agent loop endpoints.
"""
import os
import logging
from typing import Optional
from fastapi import FastAPI, HTTPException, Header, status
from fastapi.responses import JSONResponse
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
