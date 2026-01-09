# Demo Flow (Phase-2 Trustless Agent Loop)

This is a step-by-step script for the Phase-2 hackathon demo: usage-based NPC generation via x402/Gateway + Gemini multimodal + off-chain validation + on-chain attestation + USDC settlement on Arc.

## Prereqs
- Circle Wallets for payer, agent treasury, creator
- Object storage bucket (GCS/S3) for BVH/FBX, tensors, previews
- FastAPI services: ingest, analysis, oracle, commerce
- Arc RPC access and verifying contract address

## Steps
1. Upload capoeira→breakdance BVH/FBX to object store and submit MotionBlendRequest.
2. Motion Ingest stores raw, generates tensors (rot6d/quaternion + contacts + root motion), produces preview, emits MotionBlendEvent.
3. Gemini service consumes preview + metadata and outputs descriptors (style labels, transition window, npc tags, safety flags).
4. Oracle builds query vector (tensor features + Gemini descriptors), runs kNN + RkCNN ensembles, computes separation score and decision (MINT/REJECT/REVIEW).
5. Build MotionCanonicalPack v1, compute pack_hash = keccak256(canonical_bytes).
6. Oracle signs EIP-712 mint authorization (to, pack_hash, nonce, expiry, policy_digest).
7. Agent calls Arc contract: mintWithAttestation(...), receives tx_hash.
8. User triggers NPC generation; commerce orchestrator verifies x402 payment proof and meters usage.
9. Route payouts in USDC (creator/oracle/platform/ops). Log receipts and correlation IDs.
10. Verify audit trail in warehouse via Fivetran (ingest_id → analysis_id → attest_id → tx_hash → payment_intent_id).

## Verification checkpoints
- Gemini descriptors include two style labels and transition window.
- RkCNN separation_score > novelty_threshold (e.g., 0.57 > 0.42) ⇒ MINT.
- pack_hash present on-chain via event; tx_hash recorded.
- UsageMeterEvent shows unit seconds_generated and total_usdc split.

## Curl examples
- See docs/DATA_SCHEMAS.md for exact payloads.
