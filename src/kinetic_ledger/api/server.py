from fastapi import FastAPI
from ..schemas.models import (
    MotionAsset,
    MotionBlendRequest,
    GeminiAnalysis,
    SimilarityCheck,
    MotionCanonicalPack,
    MintAuthorization,
    UsageMeterEvent,
)

app = FastAPI(title="Kinetic Ledger Phase-2")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/ingest")
def ingest(asset: MotionAsset):
    return {"motion_id": asset.motion_id}

@app.post("/analyze")
def analyze(analysis: GeminiAnalysis):
    return {"analysis_id": analysis.analysis_id}

@app.post("/validate")
def validate(sim: SimilarityCheck):
    return {"similarity_id": sim.similarity_id, "decision": sim.decision.result}

@app.post("/authorize")
def authorize(pack: MotionCanonicalPack):
    return {"pack_version": pack.pack_version}

@app.post("/mint")
def mint(authz: MintAuthorization):
    return {"chain_id": authz.chain_id, "contract": authz.verifying_contract}

@app.post("/usage")
def usage(event: UsageMeterEvent):
    return {"usage_id": event.usage_id, "product": event.product}
