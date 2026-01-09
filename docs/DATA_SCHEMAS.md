# Data Schemas

All timestamps are Unix seconds. All IDs are UUIDv4 unless stated. All hashes are 0x-prefixed hex. JSON objects are canonicalized before hashing (sorted keys, stable float rounding, UTF-8).

---

## 1) MotionAsset (raw + tensor artifacts)

```json
{
  "motion_id": "b6d23a79-78c7-45d0-a15d-9c9d3c533f7d",
  "created_at": 1761905400,
  "owner_wallet": "0xA1b2c3D4e5F60718293aBcDeF1234567890aBCdE",
  "source": {
    "type": "upload",
    "filename": "capoeira_to_breakdance.fbx",
    "content_type": "model/fbx",
    "uri": "s3://kinetic-ledger/raw/b6d23a79/capoeira_to_breakdance.fbx",
    "sha256": "0x8f2f1f9a2d63b9e8c1e2b7c3d4e5f60718293abcdeffedcba0123456789abcd"
  },
  "tensor": {
    "representation": "quaternion",
    "fps": 30,
    "frame_count": 250,
    "joint_count": 24,
    "features": ["joint_rot_quat", "foot_contacts", "root_vel_xy", "root_height_z"],
    "uri": "s3://kinetic-ledger/tensors/b6d23a79/tensor_v1.npz",
    "sha256": "0x1a9b3c5d7e8f90123456789abcdeffedcba0987654321123456789abcdeff0"
  },
  "preview": {
    "uri": "s3://kinetic-ledger/previews/b6d23a79/preview.mp4",
    "sha256": "0x0f0e0d0c0b0a09080706050403020100ffeeddccbbaa99887766554433221100"
  },
  "skeleton": {
    "skeleton_id": "mixamo_24j_v1",
    "retarget_profile": "humanoid_unity_v2",
    "joint_map_uri": "s3://kinetic-ledger/skeletons/mixamo_24j_v1/joint_map.json"
  }
}
```

## 2) MotionBlendRequest (user intent)

```json
{
  "request_id": "5a55e015-5b54-4f6b-9b75-7f0df919c2f3",
  "created_at": 1761905422,
  "user_wallet": "0xA1b2c3D4e5F60718293aBcDeF1234567890aBCdE",
  "inputs": [
    { "motion_id": "11111111-1111-1111-1111-111111111111", "label": "capoeira" },
    { "motion_id": "22222222-2222-2222-2222-222222222222", "label": "breakdance_freezes" }
  ],
  "blend_plan": {
    "type": "single_shot_temporal_conditioning",
    "segments": [
      { "label": "capoeira", "start_frame": 0, "end_frame": 124 },
      { "label": "breakdance_freezes", "start_frame": 125, "end_frame": 249 }
    ]
  },
  "npc_context": {
    "game": "biomimicry_multi_agent_sim",
    "intent": ["de_escalation", "triage", "routing"],
    "environment": "chaotic_crowd_scene"
  },
  "policy": {
    "allowed_use": "npc_generation",
    "max_seconds": 12,
    "safety_level": "standard"
  }
}
```

## 3) GeminiAnalysis (multimodal descriptors)

```json
{
  "analysis_id": "a8d0b0fa-5d2d-4c8a-9a08-3d29dc76d1c2",
  "request_id": "5a55e015-5b54-4f6b-9b75-7f0df919c2f3",
  "created_at": 1761905431,
  "model": { "provider": "google", "name": "gemini", "version": "latest" },
  "inputs": {
    "preview_uri": "s3://kinetic-ledger/previews/b6d23a79/preview.mp4",
    "metadata_ref": "s3://kinetic-ledger/metadata/5a55e015.json"
  },
  "outputs": {
    "style_labels": ["capoeira", "breakdance_freezes"],
    "transition_window": { "start_frame": 110, "end_frame": 140 },
    "npc_tags": ["agile", "evasive", "high_energy"],
    "safety_flags": [],
    "summary": "Blend transitions from capoeira flow into breakdance freezes with a short transition window."
  }
}
```

## 4) SimilarityCheck (kNN + RkCNN)

```json
{
  "similarity_id": "f5e7b6b8-4c38-41a4-98f2-7e5b39b8e6f1",
  "analysis_id": "a8d0b0fa-5d2d-4c8a-9a08-3d29dc76d1c2",
  "created_at": 1761905440,
  "feature_space": {
    "embedding_dim": 768,
    "embedding_model_id": "motionblend_embed_v3",
    "distance": "euclidean"
  },
  "knn": {
    "k": 10,
    "neighbors": [
      { "motion_id": "33333333-3333-3333-3333-333333333333", "dist": 0.91 },
      { "motion_id": "44444444-4444-4444-4444-444444444444", "dist": 0.94 }
    ],
    "min_dist": 0.91
  },
  "rkcnn": {
    "k": 10,
    "ensemble_size": 64,
    "subspace_dim": 96,
    "vote_margin": 0.28,
    "separation_score": 0.57
  },
  "decision": {
    "novelty_threshold": 0.42,
    "result": "MINT",
    "reason": "Separation score exceeds novelty threshold; no policy violations."
  }
}
```

## 5) MotionCanonicalPack v1 (canonical attestation bytes)

This is the canonical package used for on-chain anchoring. The canonical_bytes are built from the JSON below after canonicalization rules (sorted keys, stable number formatting).

```json
{
  "pack_version": "MotionCanonicalPack/v1",
  "request_id": "5a55e015-5b54-4f6b-9b75-7f0df919c2f3",
  "owner_wallet": "0xA1b2c3D4e5F60718293aBcDeF1234567890aBCdE",
  "raw_ref": {
    "uri": "s3://kinetic-ledger/raw/b6d23a79/capoeira_to_breakdance.fbx",
    "sha256": "0x8f2f1f9a2d63b9e8c1e2b7c3d4e5f60718293abcdeffedcba0123456789abcd"
  },
  "tensor_ref": {
    "uri": "s3://kinetic-ledger/tensors/b6d23a79/tensor_v1.npz",
    "sha256": "0x1a9b3c5d7e8f90123456789abcdeffedcba0987654321123456789abcdeff0",
    "representation": "quaternion",
    "fps": 30,
    "frame_count": 250,
    "joint_count": 24
  },
  "skeleton": {
    "skeleton_id": "mixamo_24j_v1",
    "retarget_profile": "humanoid_unity_v2"
  },
  "versions": {
    "engine_id": "unity_6_0",
    "blend_model_id": "single_shot_tc_v1",
    "embedding_model_id": "motionblend_embed_v3"
  },
  "gemini": {
    "style_labels": ["capoeira", "breakdance_freezes"],
    "transition_window": { "start_frame": 110, "end_frame": 140 },
    "npc_tags": ["agile", "evasive", "high_energy"]
  },
  "policy": {
    "allowed_use": "npc_generation",
    "safety_level": "standard"
  }
}
```

Derived values:

- `pack_hash = keccak256(canonical_bytes)`
- optional: `raw_hash`, `tensor_hash` are already embedded
- optional: `embedding_hash = keccak256(embedding_bytes)` stored as evidence

## 6) MintAuthorization (EIP-712 payload)

```json
{
  "chain_id": 421614,
  "verifying_contract": "0xDeaDbeefdEAdbeefdEadbEEFdeadbeEFdEaDbeeF",
  "message": {
    "to": "0xA1b2c3D4e5F60718293aBcDeF1234567890aBCdE",
    "pack_hash": "0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    "nonce": 1042,
    "expiry": 1761991800,
    "policy_digest": "0xbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
  },
  "signature": "0x<65-byte-signature>"
}
```

## 7) UsageMeterEvent (x402 usage-based billing)

```json
{
  "usage_id": "e7a6f19a-4186-4c7c-8c4c-8d2d0cbe2d9c",
  "created_at": 1761905505,
  "user_wallet": "0xA1b2c3D4e5F60718293aBcDeF1234567890aBCdE",
  "attestation_pack_hash": "0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
  "product": "npc_generation",
  "metering": {
    "unit": "seconds_generated",
    "quantity": 8.4,
    "unit_price_usdc": "0.03",
    "total_usdc": "0.252"
  },
  "x402": {
    "payment_proof": "x402:<opaque-proof>",
    "facilitator_receipt_id": "fac_9c1b7e",
    "verified": true
  },
  "settlement": {
    "chain": "arc",
    "token": "USDC",
    "tx_hash": "0xcccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc"
  },
  "payout_split": [
    { "to": "0xCreator000000000000000000000000000000000000", "amount_usdc": "0.1512", "label": "creator" },
    { "to": "0xOracle0000000000000000000000000000000000000", "amount_usdc": "0.0252", "label": "oracle_fee" },
    { "to": "0xPlatform00000000000000000000000000000000000", "amount_usdc": "0.0504", "label": "platform" },
    { "to": "0xOps000000000000000000000000000000000000000", "amount_usdc": "0.0252", "label": "ops" }
  ]
}
```

---

## Embedding Hash (best practice)

- Hash: `keccak256` over canonicalized `embedding_bytes`.
- Canonicalization metadata must accompany the hash:
  - `dtype`: one of [`float32`, `float16`, `bfloat16`, `int8`]
  - `endianness`: `little` (default)
  - `ordering`: `row_major`
  - `float_rounding`: `round_to_nearest_ties_to_even`
  - `normalization`: [`none`, `l2_unit`, `zscore`, `max_abs`]
  - `dim`: integer dimension after normalization
  - `model_id`: embedding model identifier
- Batches: optionally compute a `merkle_root` over per-embedding leaf hashes with index-ascending ordering.
- Store as decision evidence in the canonical pack; keep `pack_hash` as the on-chain anchor.

## RkCNN Subspace Parameters (natural mathematics)

- Let `d` be embedding_dim.
- Subspace dimension `m = min(d, max(16, round(4*sqrt(d))))`.
- Ensemble size `E = max(32, min(128, 8 * ceil(log2(d))))`.
- Projections: `achlioptas_sparse` (faster, norm-preserving) or `gaussian` (highest accuracy).
- Metric: `cosine` if embeddings are L2-normalized; otherwise `euclidean`.
- Vote margin: `vote_margin = (V_top - V_second)/E`; default threshold `0.10`.

## Payout Policy Configuration (exposed via policy)

Example policy block additions:

```json
{
  "policy": {
    "allowed_use": "npc_generation",
    "safety_level": "standard",
    "base_rate": 0.0005,
    "unit": "USD/s",
    "max_seconds": 3600,
    "payout_split": {
      "creator": 0.50,
      "oracle": 0.20,
      "platform": 0.20,
      "ops": 0.10
    },
    "ethics_modifiers": {
      "multipliers": { "fair_use": 1.10, "climate": 0.95 },
      "caps": { "sensitive_content": 0.80 }
    }
  }
}
```

Validation rules:
- Payout percentages each in [0.0, 1.0]; sum equals 1.0 with tolerance ≤ 1e-6.
- Multipliers in [0.0, 2.0]; caps in [0.0, 1.0].
- `base_rate` non-negative; `unit` ∈ {`USD/s`, `ETH/s`, `USD/h`, `ETH/h`}.
- `max_seconds` ≥ 0.

---

## Judge script (90–120 seconds, plain language)

Here’s a tight script you can read on stage:

> Hi judges—this is **Kinetic Ledger**, a trustless way to turn motion into paid, verifiable building blocks for NPC behavior.  
>   
> A user uploads a motion blend—like **capoeira transitioning into breakdance**, or a three-part blend **salsa to swing to wave hip hop**. We store the original motion file for replay—BVH or FBX—and we also store a normalized tensor version that our models can process fast.  
>   
> Next, **Gemini** watches a preview of the motion and outputs simple descriptors: what styles it sees, where the transition happens, and what the motion “feels like” for an NPC—agile, evasive, calm, and so on.  
>   
> Then we run **off-chain validation**. We compare the motion to our library using **k-nearest neighbors**, and we add **Random k-Conditional NN** to stay reliable in high-dimensional space. That gives us a separation score—basically, “is this motion truly new or just a duplicate?”  
>   
> If it’s novel and policy-safe, we create a **canonical attestation package** that references both the raw file hash and the tensor hash, plus the skeleton and model versions. We hash that package and the oracle signs it.  
>   
> On-chain, Arc only needs that signature to mint an attestation and anchor the proof—no trust in our servers required.  
>   
> Finally, we monetize it **usage-based**: every second of NPC animation generation triggers a micropayment via **x402/Gateway**, and we settle in **USDC on Arc**. Payouts are split automatically between the creator, the oracle, and the platform.  
>   
> In short: Gemini understands motion, the oracle proves novelty, and Arc + USDC make motion generation a measurable, programmable commerce flow.
