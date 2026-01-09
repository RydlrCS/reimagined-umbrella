# Architecture

Kinetic Ledger (Phase-2) is a trust-minimized motion commerce stack: users submit motion blends, Gemini extracts multimodal intent, an off-chain oracle validates novelty with kNN/RkCNN, then Arc smart contracts mint an attestation and settle usage-based USDC payments via Circle infrastructure.

## Components

### 1) Motion Ingest Service
- Accepts motion inputs (BVH/FBX upload or URI).
- Produces model-ready tensors (rot6d/quaternion + contacts + root motion).
- Writes artifacts to object storage and emits MotionBlendEvent.
- Maintains deterministic provenance: skeleton map, FPS, frame count, retargeting profile.

### 2) Gemini Multimodal Analyzer
- Consumes previews (video/keyframes) plus metadata.
- Outputs structured descriptors:
  - style labels (e.g., capoeira→breakdance; salsa→swing→wave hip hop)
  - transition window estimates
  - NPC intent tags (e.g., evasive, calm, defensive)
  - safety/policy flags
- Produces a descriptor payload used in the query vector.

### 3) Attestation Oracle (Off-chain)
- Builds query vector from tensor features + Gemini descriptors.
- Runs:
  - kNN (baseline interpretability)
  - RkCNN ensembles (random subspaces for high-dimensional robustness)
- Computes a separation score and novelty decision:
  - MINT, REJECT, or REVIEW
- Creates MotionCanonicalPack v1 and hashes it (keccak256).
- Signs EIP-712 mint authorization (nonce + expiry + policy digest).

### 4) Commerce Orchestrator (x402/Gateway)
- Enforces usage-based billing for NPC generation:
  - per second of animation, per blend request, or per agent step
- Verifies x402 payment proof (via facilitator) and authorizes compute.
- Routes payouts (creator, oracle, platform, ops) in USDC.

### 5) On-chain Contracts (Arc)
- Attestation contract stores pack hash + metadata pointers.
- Escrow/payout contract disburses USDC based on attestation IDs and policy.
- Nonce replay protection + expiry bounds.
- Events emitted for indexing/audit.

### 6) Storage & Indexes
- Object store: BVH/FBX, tensor blobs, previews, canonical packs.
- Vector DB: embeddings and nearest-neighbor index.
- Relational DB: provenance, idempotency, billing receipts, correlation IDs.
- Warehouse via Fivetran: structured logs/events for analytics and monitoring.

## Trustless Agent Loop (Phase-2)

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


