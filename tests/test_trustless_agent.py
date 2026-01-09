"""
Integration tests for the trustless agent loop.
"""
import base64
import pytest
from kinetic_ledger.services import (
    TrustlessAgentLoop,
    TrustlessAgentConfig,
    MotionUploadRequest,
)
from kinetic_ledger.schemas.models import MotionBlendRequest, BlendPlan, BlendSegment


@pytest.fixture
def agent_config():
    """Create test agent configuration."""
    return TrustlessAgentConfig(
        circle_api_key="test_key",
        gemini_api_key="test_gemini_key",
        novelty_threshold=0.42,
        chain_id=1,
        verifying_contract="0x1234567890123456789012345678901234567890",
        oracle_address="0x0000000000000000000000000000000000000001",
        platform_address="0x0000000000000000000000000000000000000002",
        ops_address="0x0000000000000000000000000000000000000003",
    )


@pytest.fixture
def agent_loop(agent_config):
    """Create agent loop instance."""
    return TrustlessAgentLoop(agent_config)


@pytest.fixture
def sample_upload_request():
    """Create sample upload request."""
    # Create fake BVH content
    fake_bvh = b"HIERARCHY\nROOT Hips\n{\n  OFFSET 0.0 0.0 0.0\n}\n"
    content_b64 = base64.b64encode(fake_bvh).decode("utf-8")
    
    return MotionUploadRequest(
        filename="capoeira_to_breakdance.bvh",
        content_base64=content_b64,
        content_type="model/bvh",
        owner_wallet="0xA1b2c3D4e5F60718293aBcDeF1234567890aBCdE",
        skeleton_id="mixamo_24j_v1",
    )


@pytest.fixture
def sample_blend_request():
    """Create sample blend request."""
    return MotionBlendRequest(
        request_id="5a55e015-5b54-4f6b-9b75-7f0df919c2f3",
        created_at=1761905422,
        user_wallet="0xA1b2c3D4e5F60718293aBcDeF1234567890aBCdE",
        inputs=[
            {"motion_id": "11111111-1111-1111-1111-111111111111", "label": "capoeira"},
            {"motion_id": "22222222-2222-2222-2222-222222222222", "label": "breakdance"},
        ],
        blend_plan=BlendPlan(
            type="single_shot_temporal_conditioning",
            segments=[
                BlendSegment(label="capoeira", start_frame=0, end_frame=124),
                BlendSegment(label="breakdance", start_frame=125, end_frame=249),
            ],
        ),
        npc_context={
            "game": "biomimicry_multi_agent_sim",
            "intent": ["de_escalation", "triage"],
            "environment": "chaotic_crowd_scene",
        },
        policy={
            "allowed_use": "npc_generation",
            "max_seconds": 10,
            "safety_level": "standard",
        },
    )


def test_agent_loop_initialization(agent_loop):
    """Test agent loop initializes correctly."""
    assert agent_loop is not None
    assert agent_loop.ingest_service is not None
    assert agent_loop.gemini_service is not None
    assert agent_loop.attestation_oracle is not None
    assert agent_loop.commerce_orchestrator is not None


def test_complete_workflow(agent_loop, sample_upload_request, sample_blend_request):
    """Test complete trustless agent loop workflow."""
    result = agent_loop.execute_blend_workflow(
        upload_request=sample_upload_request,
        blend_request=sample_blend_request,
        payment_proof="x402_test_proof_12345",
        creator_address="0xCreator123456789012345678901234567890ABC",
    )
    
    # Verify all artifacts were created
    assert result.correlation_id is not None
    assert result.motion_asset is not None
    assert result.gemini_analysis is not None
    assert result.similarity_check is not None
    assert result.canonical_pack is not None
    assert result.decision in ["MINT", "REJECT", "REVIEW"]
    assert result.pack_hash.startswith("0x")
    
    # Verify decision workflow
    if result.decision == "MINT":
        assert result.mint_authorization is not None
        assert result.usage_event is not None
        assert result.tx_hash is not None
        assert result.tx_hash.startswith("0x")
    
    # Verify timing
    assert result.elapsed_seconds > 0
    
    # Verify Gemini analysis
    assert len(result.gemini_analysis.outputs.style_labels) > 0
    assert result.gemini_analysis.outputs.transition_window is not None
    
    # Verify similarity check
    assert result.similarity_check.knn.k > 0
    assert result.similarity_check.rkcnn.ensemble_size > 0
    assert 0 <= result.similarity_check.rkcnn.separation_score <= 1
    assert 0 <= result.similarity_check.rkcnn.vote_margin <= 1


def test_motion_ingest_service(agent_loop, sample_upload_request):
    """Test motion ingest service."""
    asset = agent_loop.ingest_service.ingest_upload(
        request=sample_upload_request,
    )
    
    assert asset.motion_id is not None
    assert asset.owner_wallet == sample_upload_request.owner_wallet
    assert asset.source.sha256.startswith("0x")
    assert asset.tensor.sha256.startswith("0x")
    assert asset.tensor.frame_count > 0
    assert asset.tensor.joint_count == 24  # Mixamo standard
    assert asset.preview.sha256.startswith("0x")


def test_gemini_analyzer_service(agent_loop):
    """Test Gemini analyzer service."""
    from kinetic_ledger.services.gemini_analyzer import GeminiAnalysisRequest
    
    request = GeminiAnalysisRequest(
        request_id="test_request_123",
        preview_uri="s3://test/preview.mp4",
        blend_segments=[
            {"label": "capoeira", "start_frame": 0, "end_frame": 124},
            {"label": "breakdance", "start_frame": 125, "end_frame": 249},
        ],
    )
    
    analysis = agent_loop.gemini_service.analyze(request=request)
    
    assert analysis.analysis_id is not None
    assert analysis.request_id == "test_request_123"
    assert len(analysis.outputs.style_labels) >= 2
    assert "capoeira" in analysis.outputs.style_labels
    assert "breakdance" in analysis.outputs.style_labels
    assert analysis.outputs.transition_window is not None
    assert len(analysis.outputs.npc_tags) > 0


def test_attestation_oracle(agent_loop):
    """Test attestation oracle similarity check."""
    similarity_check = agent_loop.attestation_oracle.validate_similarity(
        analysis_id="test_analysis_123",
        tensor_hash="0x1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef",
        gemini_descriptors={
            "style_labels": ["capoeira", "breakdance"],
            "npc_tags": ["agile", "dynamic"],
        },
        safety_flags=[],
    )
    
    assert similarity_check.similarity_id is not None
    assert similarity_check.decision.result in ["MINT", "REJECT", "REVIEW"]
    assert similarity_check.knn.k > 0
    assert similarity_check.rkcnn.ensemble_size >= 32  # Natural minimum
    assert similarity_check.rkcnn.subspace_dim >= 16  # Natural minimum


def test_commerce_orchestrator(agent_loop):
    """Test commerce orchestrator metering."""
    from kinetic_ledger.services.commerce_orchestrator import PaymentIntentRequest
    
    request = PaymentIntentRequest(
        user_wallet="0x1234567890123456789012345678901234567890",
        product="npc_generation",
        unit="seconds_generated",
        quantity=10.0,
        unit_price_usdc="0.50",
        attestation_pack_hash="0x1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef",
    )
    
    usage_event = agent_loop.commerce_orchestrator.meter_usage(
        request=request,
        payment_proof="x402_proof_test",
        creator_address="0xCreator123456789012345678901234567890ABC",
    )
    
    assert usage_event.usage_id is not None
    assert usage_event.metering.unit == "seconds_generated"
    assert usage_event.metering.quantity == 10.0
    assert float(usage_event.metering.total_usdc) == 5.0  # 10 * 0.50
    assert usage_event.x402.verified is True
    assert usage_event.settlement.tx_hash.startswith("0x")
    assert len(usage_event.payout_split) == 4  # creator, oracle, platform, ops
    
    # Verify payout percentages sum to 1.0
    total_payout = sum(float(p.amount_usdc) for p in usage_event.payout_split)
    assert abs(total_payout - 5.0) < 0.01  # Within rounding tolerance


def test_knn_similarity():
    """Test kNN similarity search."""
    import numpy as np
    from kinetic_ledger.services.similarity import knn
    
    query = np.array([1.0, 2.0, 3.0])
    items = [
        ("a", np.array([1.1, 2.1, 3.1])),
        ("b", np.array([5.0, 6.0, 7.0])),
        ("c", np.array([1.0, 2.0, 3.0])),
    ]
    
    neighbors = knn(query, items, k=2, distance_metric="euclidean")
    
    assert len(neighbors) == 2
    assert neighbors[0][0] == "c"  # Exact match
    assert neighbors[0][1] < 0.01  # Near-zero distance
    assert neighbors[1][0] == "a"  # Second closest


def test_rkcnn_ensemble():
    """Test RkCNN ensemble."""
    import numpy as np
    from kinetic_ledger.services.similarity import rkcnn
    
    np.random.seed(42)
    query = np.random.randn(256)
    items = [
        (f"item_{i}", np.random.randn(256))
        for i in range(10)
    ]
    
    separation, vote_margin, ensemble_results = rkcnn(
        query=query,
        items=items,
        k=3,
        seed=42,
    )
    
    assert 0 <= separation <= 1
    assert 0 <= vote_margin <= 1
    assert len(ensemble_results) >= 32  # Natural minimum ensemble size
    assert all(len(result) <= 3 for result in ensemble_results)  # k=3


def test_canonical_pack_creation(agent_loop, sample_upload_request, sample_blend_request):
    """Test canonical pack creation and hashing."""
    # First ingest motion
    motion_asset = agent_loop.ingest_service.ingest_upload(
        request=sample_upload_request,
    )
    
    # Then analyze with Gemini
    from kinetic_ledger.services.gemini_analyzer import GeminiAnalysisRequest
    gemini_request = GeminiAnalysisRequest(
        request_id=sample_blend_request.request_id,
        preview_uri=motion_asset.preview.uri,
        blend_segments=[seg.model_dump() for seg in sample_blend_request.blend_plan.segments],
    )
    gemini_analysis = agent_loop.gemini_service.analyze(request=gemini_request)
    
    # Create canonical pack
    pack = agent_loop.attestation_oracle.create_canonical_pack(
        motion_asset=motion_asset,
        blend_request=sample_blend_request,
        gemini_analysis=gemini_analysis,
    )
    
    assert pack.pack_version == "MotionCanonicalPack/v1"
    assert pack.request_id == sample_blend_request.request_id
    assert pack.owner_wallet == motion_asset.owner_wallet
    
    # Verify pack hash is deterministic
    from kinetic_ledger.utils.canonicalize import keccak256_json
    pack_hash1 = keccak256_json(pack.model_dump())
    pack_hash2 = keccak256_json(pack.model_dump())
    assert pack_hash1 == pack_hash2
    assert pack_hash1.startswith("0x")
    assert len(pack_hash1) == 66  # 0x + 64 hex chars
