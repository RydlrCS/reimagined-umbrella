#!/usr/bin/env python3
"""
Demo script for Kinetic Ledger Phase-2 Trustless Agent Loop.

This demonstrates the complete workflow from motion upload to USDC settlement.
"""
import base64
import os
from kinetic_ledger.services import (
    TrustlessAgentLoop,
    TrustlessAgentConfig,
    MotionUploadRequest,
)
from kinetic_ledger.schemas.models import (
    MotionBlendRequest,
    BlendPlan,
    BlendSegment,
)


def main():
    print("🚀 Kinetic Ledger Phase-2 Trustless Agent Loop Demo\n")
    
    # 1. Configure agent
    print("1️⃣  Configuring trustless agent...")
    
    # Get Gemini API key from environment or use demo key
    gemini_api_key = os.environ.get("GEMINI_API_KEY", "demo_gemini_key")
    
    config = TrustlessAgentConfig(
        circle_api_key="demo_circle_key",
        gemini_api_key=gemini_api_key,
        novelty_threshold=0.42,
        chain_id=1,
        verifying_contract="0x1234567890123456789012345678901234567890",
        oracle_address="0x0000000000000000000000000000000000000001",
        platform_address="0x0000000000000000000000000000000000000002",
        ops_address="0x0000000000000000000000000000000000000003",
    )
    
    agent = TrustlessAgentLoop(config)
    
    if gemini_api_key != "demo_gemini_key":
        print(f"   ✅ Agent configured with real Gemini API")
    else:
        print(f"   ⚠️  Agent using fallback mode (set GEMINI_API_KEY env var for real API)")
    print()
    
    # 2. Create upload request
    print("2️⃣  Creating motion upload request...")
    fake_bvh = b"""HIERARCHY
ROOT Hips
{
  OFFSET 0.0 0.0 0.0
  CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation
  JOINT Spine
  {
    OFFSET 0.0 0.0 0.0
    CHANNELS 3 Zrotation Xrotation Yrotation
  }
}
MOTION
Frames: 120
Frame Time: 0.033333
0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
"""
    
    upload = MotionUploadRequest(
        filename="capoeira_to_breakdance.bvh",
        content_base64=base64.b64encode(fake_bvh).decode("utf-8"),
        content_type="model/bvh",
        owner_wallet="0xA1b2c3D4e5F60718293aBcDeF1234567890aBCdE",
        skeleton_id="mixamo_24j_v1",
    )
    print("   ✅ Upload request created\n")
    
    # 3. Create blend request
    print("3️⃣  Creating blend request...")
    blend = MotionBlendRequest(
        request_id="demo-request-001",
        created_at=1761905422,
        user_wallet="0xA1b2c3D4e5F60718293aBcDeF1234567890aBCdE",
        inputs=[
            {"motion_id": "capoeira-001", "label": "capoeira"},
            {"motion_id": "breakdance-002", "label": "breakdance_freezes"},
        ],
        blend_plan=BlendPlan(
            type="single_shot_temporal_conditioning",
            segments=[
                BlendSegment(label="capoeira", start_frame=0, end_frame=124),
                BlendSegment(label="breakdance_freezes", start_frame=125, end_frame=249),
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
    print("   ✅ Blend request created\n")
    
    # 4. Execute trustless workflow
    print("4️⃣  Executing trustless agent loop workflow...\n")
    print("   → Step 1/6: Motion Ingest")
    print("   → Step 2/6: Gemini Multimodal Analysis")
    print("   → Step 3/6: Attestation Oracle (kNN + RkCNN)")
    print("   → Step 4/6: Canonical Pack Creation")
    print("   → Step 5/6: Mint Authorization Signing")
    print("   → Step 6/6: Usage Metering & USDC Settlement\n")
    
    result = agent.execute_blend_workflow(
        upload_request=upload,
        blend_request=blend,
        payment_proof="x402_demo_proof_123456",
        creator_address="0xCreator123456789012345678901234567890ABC",
    )
    
    # 5. Display results
    print("✅ Workflow Complete!\n")
    print("=" * 70)
    print(f"📋 Correlation ID:     {result.correlation_id}")
    print(f"🎯 Decision:           {result.decision}")
    print(f"📦 Pack Hash:          {result.pack_hash[:20]}...")
    print(f"🎬 Motion ID:          {result.motion_asset.motion_id}")
    print(f"🧠 Analysis ID:        {result.gemini_analysis.analysis_id}")
    print(f"🔍 Similarity ID:      {result.similarity_check.similarity_id}")
    print("=" * 70)
    
    # Gemini Analysis
    print("\n🧠 Gemini Analysis:")
    print(f"   Style Labels:       {', '.join(result.gemini_analysis.outputs.style_labels)}")
    print(f"   NPC Tags:           {', '.join(result.gemini_analysis.outputs.npc_tags)}")
    print(f"   Transition Window:  frames {result.gemini_analysis.outputs.transition_window}")
    print(f"   Safety Flags:       {result.gemini_analysis.outputs.safety_flags or 'None'}")
    
    # Similarity Check
    print("\n🔍 Similarity Check:")
    print(f"   kNN Neighbors:      {len(result.similarity_check.knn.neighbors)}")
    print(f"   RkCNN Ensembles:    {result.similarity_check.rkcnn.ensemble_size}")
    print(f"   Separation Score:   {result.similarity_check.rkcnn.separation_score:.4f}")
    print(f"   Vote Margin:        {result.similarity_check.rkcnn.vote_margin:.4f}")
    print(f"   Threshold:          {result.similarity_check.decision.novelty_threshold}")
    
    if result.mint_authorization:
        print("\n🔐 Mint Authorization:")
        print(f"   Chain ID:           {result.mint_authorization.chain_id}")
        print(f"   Contract:           {result.mint_authorization.verifying_contract}")
        print(f"   Nonce:              {result.mint_authorization.message.nonce}")
        print(f"   Expiry:             {result.mint_authorization.message.expiry}")
    
    if result.usage_event:
        print("\n💰 Usage & Settlement:")
        print(f"   Product:            {result.usage_event.product}")
        print(f"   Quantity:           {result.usage_event.metering.quantity} {result.usage_event.metering.unit}")
        print(f"   Total USDC:         ${result.usage_event.metering.total_usdc}")
        print(f"   TX Hash:            {result.tx_hash[:20]}...")
        print("\n   Payout Split:")
        for payout in result.usage_event.payout_split:
            print(f"      {payout.label:12} ${payout.amount_usdc:>8} → {payout.to}")
    
    print(f"\n⏱️  Total Time:          {result.elapsed_seconds:.2f} seconds")
    print("\n" + "=" * 70)
    print("🎉 Demo Complete! All systems operational.")
    print("=" * 70)


if __name__ == "__main__":
    main()
