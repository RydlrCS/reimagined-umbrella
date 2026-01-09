# Kinetic Ledger Monorepo (Phase-2 Hackathon Blueprint)

Below is a **Phase-2 hackathon-ready blueprint** for **Kinetic Ledger** that adds:

- **Gemini multimodal understanding** for motion blends (image/video/sequence metadata)
- **Agent-driven decisions** (policy + risk + pricing)
- **Arc settlement with USDC** using **Circle infrastructure** (Wallets / Gateway / CCTP / x402)
- **Off-chain validation** that produces **on-chain attestations** (hashes + signatures) and can reference **existing on-chain data**

You’ll get:

1. **Repo file schema** (monorepo structure)
2. **System architecture** (services + data flow)
3. **Code standards** (Python prod-grade, linting, typing, tests, logging, retries)
4. **Python production skeleton** (well-commented) + **tests**

---

## 1) Repository file schema (monorepo)

This assumes your existing scaffold (web-dapp, agent-service, api-gateway, contracts) stays, and we **add** Phase-2 services + shared libs.

```text
kinetic-ledger/
  README.md
  pnpm-workspace.yaml
  package.json
  .gitignore
  .editorconfig
  .env.example

  apps/
    web-dapp/                       # Next.js UI (Polkadot-like UX / motion timeline)
    agent-service/                  # Node/TS agent runner (optional: keep for EVM ops)
    api-gateway/                    # FastAPI edge ingress (already present)

  services/                         # NEW: python-first “production” services
    motion-ingest/                  # motion data + blend metadata ingestion
      pyproject.toml
      src/motion_ingest/
        __init__.py
        main.py                     # FastAPI (upload / event intake)
        config.py
        logging.py
        schemas.py                  # Pydantic DTOs (MotionBlend, AttestationRequest)
        storage.py                  # S3/GCS/local abstraction
        queue.py                    # Redis/CloudTasks abstraction
      tests/
        test_health.py
        test_ingest.py

    multimodal-gemini/              # Gemini analysis + embedding + explanation
      pyproject.toml
      src/gemini_mm/
        __init__.py
        main.py                     # FastAPI (analyze motion preview / keyframes)
        config.py
        logging.py
        schemas.py                  # prompts, results, safety flags
        gemini_client.py            # wrapper (timeouts, retries, typed responses)
        features.py                 # feature vectors, query vectors
      tests/
        test_gemini_client.py
        test_features.py

    attestation-oracle/             # kNN/RkCNN validation, novelty checks, signing
      pyproject.toml
      src/attest_oracle/
        __init__.py
        main.py                     # FastAPI (validate -> sign attestation)
        config.py
        logging.py
        schemas.py                  # Attestation, SimilarityReport, Decision
        knn.py                      # exact/approx kNN
        rkcnn.py                    # randomized conditional NN ensembles
        thresholds.py               # bias/variance tuning, separation score
        fid_metrics.py              # optional quality scoring hooks
        signer.py                   # EIP-712 typed-data signing payload (off-chain)
        db.py                       # vector store interface (Qdrant/pgvector)
      tests/
        test_knn.py
        test_rkcnn.py
        test_thresholds.py

    vector-database/                # Elasticsearch vector search for motion analysis
      pyproject.toml
      src/vector_db/
        __init__.py
        main.py                     # FastAPI (search endpoints)
        config.py
        logging.py
        schemas.py                  # SearchQuery, SearchResult, HybridSearch
        elasticsearch_connector.py  # typed ES client with retry logic
        embedding_service.py        # SentenceTransformers (all-MiniLM-L6-v2)
        search_engine.py            # k-NN, semantic, hybrid search orchestration
        indexing.py                 # bulk index, document upsert
        fallback.py                 # mock search when ES unavailable
      tests/
        test_elasticsearch_connector.py
        test_embedding_service.py
        test_search_integration.py

    commerce-orchestrator/          # Circle wallets/gateway + Arc settlement coordinator
      pyproject.toml
      src/commerce/
        __init__.py
        main.py                     # FastAPI (create subscription, usage charge, tip)
        config.py
        logging.py
        schemas.py                  # PaymentIntent, Charge, Subscription
        circle_client.py            # typed API wrapper with retries
        x402.py                     # x402 verification + facilitator hook
        arc_client.py               # EVM tx submit (via RPC); USDC transfers
        policies.py                 # agent guardrails (limits, allowlists, KYC tags)
        ledger.py                   # state machine (idempotent workflows)
      tests/
        test_policies.py
        test_ledger_idempotency.py

  packages/
    contracts/                      # AttestedMotion.sol, RewardsEscrow.sol (already)
    sdk/                            # shared TS helpers (hashing, typed-data)
    schemata/                       # shared types

  libs/                             # NEW: shared python utilities (importable)
    py/
      kinetic_lib/
        __init__.py
        logging.py                  # JSON structured logging + correlation IDs
        errors.py                   # typed domain errors
        retry.py                    # tenacity policies
        idempotency.py              # request keys, nonce handling
        security.py                 # HMAC verify, request signing helpers
        time.py                     # UTC helpers
        observability.py            # optional OpenTelemetry wiring
        vector_utils.py             # embedding normalization, cosine similarity

  infra/
    docker/
      docker-compose.dev.yml        # local Redis/Qdrant + services
    k8s/
      base/                         # manifests for services
    terraform/
      README.md

  docs/
    ARCHITECTURE.md
    GETTING_STARTED.md
    RUNBOOK.md
    SECURITY.md
    API.md                          # endpoints overview
    DATA_SCHEMAS.md                 # motion, attestation, payments
```

---

## 2) System architecture (Phase-2)

### Core idea

**Gemini interprets motion blend sequences** (e.g., capoeira→breakdance) to produce:

- multimodal labels (style, quality flags, transition segments)
- an explanation payload for the user/judges
- structured “motion descriptors” used by the validation oracle

**Attestation Oracle** validates novelty + authenticity off-chain using:

- **kNN** for nearest neighbors
- **RkCNN** (Random-k Conditional NN ensembles) for high-dimensional robustness
  Then it **signs a mint authorization** (EIP-712 style) consumed by the on-chain contract.

**Commerce Orchestrator** handles Circle flows:

- create user/agent wallets, policy limits, subscriptions, usage-based billing
- settle USDC on Arc (and optionally move USDC cross-chain via Gateway/CCTP/Bridge Kit)

### Data flow (end-to-end)

1. **Web DApp** uploads motion blend metadata (+ optional preview frames/video clip)
2. **motion-ingest** stores raw payload off-chain (object storage) and emits event to queue
3. **multimodal-gemini** analyzes preview → outputs structured tags + confidence
4. **attestation-oracle** builds query vectors, runs kNN + RkCNN, computes **separation score**, decides **MINT / REJECT / REVIEW**
5. If MINT: oracle signs **Mint Authorization** (nonce+expiry)
6. **commerce-orchestrator**:
   - creates a **PaymentIntent** (tip, usage charge, subscription event)
   - executes settlement in USDC on Arc (and records receipt)
7. **agent-service** (optional) can run autonomous workflows:
   - auto-mint when novelty high and policy passes
   - auto-pay creators upon mint confirmation
8. Everything emits **correlated logs** + idempotent keys for replay safety.
### Vector Database Integration (Elasticsearch)

**Purpose**: Semantic search and similarity matching for motion analysis data using Elasticsearch Cloud with dense_vector support.

**Architecture** (based on MotionBlendAI patterns):

- **Elasticsearch 8.x+** with k-NN plugin for vector similarity
- **384-dimensional embeddings** using SentenceTransformers (`all-MiniLM-L6-v2`)
- **Hybrid search** combining vector similarity + text relevance (RRF ranking)
- **ELSER semantic model** for natural language queries
- **Lazy initialization** with connection pooling and fallback to mock data

**Index Structure**:

```python
Index: "kinetic-motion-analysis"
Mappings:
  - motion_vector: dense_vector (384 dims, cosine similarity)
  - query_descriptor: text + semantic_text (ELSER)
  - style_labels: keyword (from Gemini analysis)
  - npc_tags: keyword (character tags)
  - gemini_summary: text + semantic_text
  - source_motion/target_motion: text + keyword + semantic
  - attestation metadata: validation_score, novelty_score, knn_distance
  - blend metadata: blend_ratio, transition_start/end, blend_method
  - timestamps: created_at, updated_at (date)
```

**Search Modes**:

1. **Vector Similarity (k-NN)**: Generate embedding → cosine similarity search → top-k matches
2. **Semantic Text Search**: Multi-field search with ELSER + fuzzy matching + highlights
3. **Hybrid Search**: Weighted vector + text with RRF ranking

**Implementation**:

```python
# Lazy connector initialization
connector = ElasticsearchConnector.get_instance(
    cloud_url=os.getenv("ES_CLOUD_URL"),
    api_key=os.getenv("ES_API_KEY"),
    index_name="kinetic-motion-analysis"
)

# Generate embedding from Gemini analysis
motion_vector = embed_motion_descriptor(
    style_labels=["capoeira", "breakdance"],
    npc_tags=["warrior", "athletic"],
    summary="Dynamic blend with explosive energy"
)

# Index with embedding
connector.index_document({
    "analysis_id": "motion_123",
    "motion_vector": motion_vector,
    "style_labels": ["capoeira", "breakdance"],
    "validation_score": 0.92
})

# Hybrid search (60% vector, 40% text)
results = connector.search_hybrid(
    query_vector=motion_vector,
    query_text="explosive athletic martial arts",
    k=10,
    vector_weight=0.6
)
```

**Configuration**:

```bash
ES_CLOUD_URL=https://my-deployment.es.us-central1.gcp.elastic.cloud:443
ES_API_KEY=<base64_encoded_api_key>
ES_INDEX_NAME=kinetic-motion-analysis  # Optional
ES_TIMEOUT=30  # Optional, seconds
```
---

## 3) Code standards (Python “production by default”)

**Runtime**

- Python: `>=3.11`
- Web: FastAPI + Uvicorn
- Validation: Pydantic v2
- HTTP: httpx (timeouts + retries)
- Retries: tenacity (exponential backoff + jitter)
- Logging: JSON structured logs (`structlog` or stdlib JSON formatter)
- Testing: pytest + pytest-asyncio
- Lint: ruff
- Type check: mypy (strict-ish)

**Non-negotiables**

- Every request gets a **correlation_id**
- Every workflow step is **idempotent** (idempotency_key / nonce)
- All external calls (Gemini, Circle, Arc RPC) have:
  - timeouts
  - retries with caps
  - circuit-breaker style “fail fast” thresholds (optional)
- Errors are **typed** and mapped to stable API error codes

**Suggested exit codes / failure taxonomy**

- `E_CFG_*` configuration missing/invalid
- `E_DEP_*` dependency down (Gemini/Circle/RPC)
- `E_VAL_*` validation failure
- `E_POLICY_*` policy rejection
- `E_TX_*` chain settlement failure

---

## 4) Core data schemas (canonical)

### MotionBlendEvent (ingest)

```python
# docs/DATA_SCHEMAS.md -> canonical
MotionBlendEvent:
  event_id: str (uuid)
  user_id: str
  wallet: str (0x..)
  created_at: datetime
  sequence:
    sources: list[str]                 # ["capoeira", "breakdance_freezes"]
    timeline:
      - start_frame: int
        end_frame: int
        label: str                     # "capoeira"
      - start_frame: int
        end_frame: int
        label: str                     # "transition"
      - start_frame: int
        end_frame: int
        label: str                     # "breakdance"
  artifacts:
    preview_image_urls: list[str]      # keyframes (optional)
    preview_video_url: str | null
    motion_bytes_b64: str | null       # optional raw (usually stored in object store)
  metadata:
    fps: int
    skeleton: str                      # e.g. "mixamo_24j"
    embedding_dim: int                 # e.g. 512/1024
```

### AttestationDecision (oracle output)

```python
AttestationDecision:
  decision_id: str
  event_id: str
  data_hash: str                       # keccak256 of canonical bytes
  query_vector_hash: str               # hash of embedding / descriptor
  knn:
    k: int
    nearest: list[{id, distance}]
  rkcnn:
    ensembles: int
    k: int
    separation_score: float
  novelty:
    threshold: float
    is_novel: bool
  decision: "MINT" | "REJECT" | "REVIEW"
  reason_codes: list[str]
  signed_mint_auth: {nonce, expiry, sig} | null
```

### PaymentIntent (commerce)

```python
PaymentIntent:
  intent_id: str
  event_id: str | null
  payer_wallet_id: str                 # Circle wallet (or user wallet ref)
  payee_address: str                   # Arc address
  amount_usdc: str                     # decimal string
  purpose: "TIP" | "USAGE" | "SUBSCRIPTION" | "PAYOUT"
  settlement:
    chain: "ARC"
    usdc_token_address: str
  status: "CREATED" | "AUTHORIZED" | "SETTLED" | "FAILED"
  idempotency_key: str
```

---

## 5) Python production skeleton (one service example)

Below is a **complete, lint-friendly** FastAPI service skeleton for the **Attestation Oracle** (kNN + RkCNN), showing:

- typed config
- structured logging
- retries
- clear error mapping
- testable pure functions

### `services/attestation-oracle/pyproject.toml`

```toml
[project]
name = "attestation-oracle"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
  "fastapi==0.115.0",
  "uvicorn[standard]==0.30.6",
  "pydantic==2.9.2",
  "httpx==0.27.2",
  "tenacity==9.0.0",
  "numpy==2.1.1"
]

[tool.ruff]
line-length = 100
select = ["E","F","I","B","UP","N","SIM","RUF"]
ignore = ["E501"]

[tool.mypy]
python_version = "3.11"
strict = true
warn_return_any = true
warn_unused_configs = true

[tool.pytest.ini_options]
testpaths = ["tests"]
```

### `services/attestation-oracle/src/attest_oracle/logging.py`

```python
import json
import logging
import os
from typing import Any

class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "level": record.levelname,
            "msg": record.getMessage(),
            "logger": record.name,
        }
        # attach correlation_id if present
        if hasattr(record, "correlation_id"):
            payload["correlation_id"] = getattr(record, "correlation_id")

        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)

        return json.dumps(payload, ensure_ascii=False)

def setup_logging() -> None:
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    root = logging.getLogger()
    root.setLevel(level)

    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    root.handlers.clear()
    root.addHandler(handler)
```

### `services/attestation-oracle/src/attest_oracle/config.py`

```python
from pydantic import BaseModel, Field

class Settings(BaseModel):
    service_name: str = "attestation-oracle"
    # Vector DB / feature store (stubbed; swap with Qdrant/pgvector later)
    vector_store_url: str = Field(default="http://localhost:6333")
    # Similarity params
    knn_k: int = 15
    rkcnn_k: int = 15
    rkcnn_ensembles: int = 24
    subspace_dim: int = 128
    novelty_threshold: float = 0.42  # tuned per dataset
    # Signing parameters (off-chain mint authorization)
    validator_private_key_hex: str = Field(min_length=64)
    attestation_expiry_seconds: int = 600
```

### `services/attestation-oracle/src/attest_oracle/schemas.py`

```python
from datetime import datetime
from pydantic import BaseModel, Field

class MotionBlendEvent(BaseModel):
    event_id: str
    wallet: str = Field(pattern=r"^0x[a-fA-F0-9]{40}$")
    created_at: datetime
    motion_bytes_b64: str
    embedding_dim: int = Field(ge=128, le=4096)

class Neighbor(BaseModel):
    item_id: str
    distance: float

class KNNReport(BaseModel):
    k: int
    nearest: list[Neighbor]

class RKCNNReport(BaseModel):
    ensembles: int
    k: int
    separation_score: float

class SignedMintAuth(BaseModel):
    nonce: int
    expiry: int
    signature_hex: str

class AttestationDecision(BaseModel):
    decision_id: str
    event_id: str
    data_hash: str
    knn: KNNReport
    rkcnn: RKCNNReport
    novelty_threshold: float
    is_novel: bool
    decision: str  # "MINT" | "REJECT" | "REVIEW"
    reason_codes: list[str]
    signed_mint_auth: SignedMintAuth | None = None
```

### `services/attestation-oracle/src/attest_oracle/knn.py`

```python
from __future__ import annotations
import numpy as np

def l2_distance(a: np.ndarray, b: np.ndarray) -> float:
    # NOTE: In high dimensions Euclidean distance can concentrate.
    # We keep it as a baseline and rely on subspace ensembles for robustness.
    return float(np.linalg.norm(a - b))

def knn(query: np.ndarray, items: list[tuple[str, np.ndarray]], k: int) -> list[tuple[str, float]]:
    if k <= 0:
        raise ValueError("k must be > 0")
    if not items:
        return []

    dists = [(item_id, l2_distance(query, vec)) for (item_id, vec) in items]
    dists.sort(key=lambda x: x[1])
    return dists[:k]
```

### `services/attestation-oracle/src/attest_oracle/rkcnn.py`

```python
from __future__ import annotations
import numpy as np
from .knn import knn

def random_subspace_indices(d: int, sub_d: int, rng: np.random.Generator) -> np.ndarray:
    if sub_d <= 0 or sub_d > d:
        raise ValueError("subspace_dim must be in (0, d]")
    return rng.choice(d, size=sub_d, replace=False)

def separation_score(nearest_distances: list[float]) -> float:
    """
    A simple “separation score” heuristic:
      separation = (d_k - d_1) / (d_k + eps)
    - Higher means clearer margin between best match and boundary of neighborhood.
    - Helps in high-dimensional regimes where absolute L2 distances can concentrate.
    """
    if not nearest_distances:
        return 0.0
    d1 = nearest_distances[0]
    dk = nearest_distances[-1]
    eps = 1e-9
    return float((dk - d1) / (dk + eps))

def rkcnn(
    query: np.ndarray,
    items: list[tuple[str, np.ndarray]],
    k: int,
    ensembles: int,
    subspace_dim: int,
    seed: int = 7,
) -> tuple[float, list[list[tuple[str, float]]]]:
    """
    Random-k Conditional kNN (RkCNN-like ensemble):
    - Sample random subspaces of dimensions subspace_dim.
    - Run kNN in each subspace.
    - Aggregate a separation score across ensembles for a robust novelty signal.
    """
    rng = np.random.default_rng(seed)
    d = query.shape[0]

    all_knn: list[list[tuple[str, float]]] = []
    scores: list[float] = []

    for _ in range(ensembles):
        idx = random_subspace_indices(d, subspace_dim, rng)
        q_sub = query[idx]
        items_sub = [(item_id, vec[idx]) for (item_id, vec) in items]
        nn = knn(q_sub, items_sub, k=k)
        all_knn.append(nn)
        scores.append(separation_score([dist for _, dist in nn]))

    return float(np.mean(scores) if scores else 0.0), all_knn
```

### `services/attestation-oracle/src/attest_oracle/thresholds.py`

```python
def decide_novelty(separation: float, novelty_threshold: float) -> tuple[bool, list[str]]:
    """
    Bias-variance trade-off intuition:
    - Lower threshold => more mints (higher recall, more false positives)
    - Higher threshold => fewer mints (higher precision, more false negatives)
    We expose this as a single knob for hackathon demo tuning.
    """
    if separation >= novelty_threshold:
        return True, ["NOVELTY_PASS"]
    return False, ["NOVELTY_FAIL"]
```

### `services/attestation-oracle/src/attest_oracle/main.py`

```python
from __future__ import annotations

import base64
import hashlib
import logging
import os
import time
import uuid

import numpy as np
from fastapi import FastAPI, Header, HTTPException, Request
from pydantic import ValidationError

from .config import Settings
from .logging import setup_logging
from .schemas import AttestationDecision, KNNReport, MotionBlendEvent, Neighbor, RKCNNReport
from .knn import knn
from .rkcnn import rkcnn
from .thresholds import decide_novelty

setup_logging()
log = logging.getLogger("attestation-oracle")

app = FastAPI(title="Kinetic Ledger — Attestation Oracle", version="2.0.0")

def get_settings() -> Settings:
    try:
        return Settings(
            validator_private_key_hex=os.environ["VALIDATOR_PRIVATE_KEY_HEX"],
            vector_store_url=os.getenv("VECTOR_STORE_URL", "http://localhost:6333"),
            knn_k=int(os.getenv("KNN_K", "15")),
            rkcnn_k=int(os.getenv("RKCNN_K", "15")),
            rkcnn_ensembles=int(os.getenv("RKCNN_ENSEMBLES", "24")),
            subspace_dim=int(os.getenv("SUBSPACE_DIM", "128")),
            novelty_threshold=float(os.getenv("NOVELTY_THRESHOLD", "0.42")),
            attestation_expiry_seconds=int(os.getenv("ATTEST_EXPIRY_SECONDS", "600")),
        )
    except KeyError as e:
        log.error({"missing_env": str(e)}, "Missing required env var")
        raise

@app.middleware("http")
async def correlation_mw(request: Request, call_next):
    correlation_id = request.headers.get("x-correlation-id", str(uuid.uuid4()))
    start = time.time()
    try:
        response = await call_next(request)
        return response
    finally:
        elapsed_ms = int((time.time() - start) * 1000)
        log.info(
            f"{request.method} {request.url.path} {elapsed_ms}ms",
            extra={"correlation_id": correlation_id},
        )

@app.get("/health")
def health():
    return {"ok": True, "ts": int(time.time())}

@app.post("/attest", response_model=AttestationDecision)
def attest(ev: MotionBlendEvent, x_correlation_id: str = Header(default="")):
    correlation_id = x_correlation_id or str(uuid.uuid4())
    s = get_settings()

    # 1) Decode bytes and compute canonical hash (on-chain reference anchor)
    try:
        raw = base64.b64decode(ev.motion_bytes_b64)
    except Exception as e:
        raise HTTPException(400, f"E_VAL_BAD_B64: {e}") from e

    data_hash = "0x" + hashlib.sha256(raw).hexdigest()  # demo; swap to keccak for EVM parity

    # 2) Produce a query vector (placeholder)
    # In your pipeline: this comes from MotionBlend embeddings or Gemini descriptors.
    # For hackathon: deterministic vector from hash so results are reproducible.
    h = hashlib.sha256(raw).digest()
    d = ev.embedding_dim
    rng = np.random.default_rng(int.from_bytes(h[:8], "big"))
    query = rng.normal(size=(d,)).astype(np.float32)

    # 3) Fetch candidate vectors from your feature store (stub: empty list => always novel)
    items: list[tuple[str, np.ndarray]] = []  # TODO: integrate Qdrant/pgvector

    # 4) kNN baseline
    nn = knn(query, items, k=s.knn_k)
    knn_report = KNNReport(
        k=s.knn_k,
        nearest=[Neighbor(item_id=i, distance=dist) for (i, dist) in nn],
    )

    # 5) RkCNN ensemble for high-dimensional robustness
    sep, _ = rkcnn(
        query=query,
        items=items,
        k=s.rkcnn_k,
        ensembles=s.rkcnn_ensembles,
        subspace_dim=s.subspace_dim,
    )
    rkcnn_report = RKCNNReport(ensembles=s.rkcnn_ensembles, k=s.rkcnn_k, separation_score=sep)

    # 6) Decision
    is_novel, reasons = decide_novelty(sep, s.novelty_threshold)
    decision = "MINT" if is_novel else "REJECT"

    # NOTE: for demo we skip signing (wire in EIP-712 signing here)
    out = AttestationDecision(
        decision_id=str(uuid.uuid4()),
        event_id=ev.event_id,
        data_hash=data_hash,
        knn=knn_report,
        rkcnn=rkcnn_report,
        novelty_threshold=s.novelty_threshold,
        is_novel=is_novel,
        decision=decision,
        reason_codes=reasons,
        signed_mint_auth=None,
    )

    log.info(
        "attest_complete",
        extra={"correlation_id": correlation_id},
    )
    return out
```

### Tests: `services/attestation-oracle/tests/test_rkcnn.py`

```python
import numpy as np
from attest_oracle.rkcnn import rkcnn

def test_rkcnn_runs():
    q = np.ones((256,), dtype=np.float32)
    items = [("a", np.ones((256,), dtype=np.float32))]
    sep, ensembles = rkcnn(q, items, k=1, ensembles=5, subspace_dim=32)
    assert sep >= 0.0
    assert len(ensembles) == 5
```

---

## 6) “Polkadot-like” UI concept (maps to your brand + motion blends)

For judges, the UI should show **one thing clearly**:
**Upload → Analyze → Validate → Mint → Pay (USDC)**

**Suggested UI modules**

- **Motion Timeline Panel** (like your sample image): stacked tracks for each source + blended output, with color-banded segments and a “transition window”
- **Gemini Insight Card**: “What motion is this?” + style labels + quality warnings (foot contact anomalies, jitter)
- **Novelty Meter**: separation score + nearest neighbors list (kNN + RkCNN)
- **Mint & Pay Drawer**:
  - “Mint Attested Motion”
  - “Tip Creator (USDC)”
  - “Charge Usage (x402 / Gateway)”
  - “Subscribe to Creator Pack”

---

## 7) Hackathon sprint alignment (high ROI / hardest piece)

If you want the **highest ROI** in Phase 2, build the **Trustless Agent loop**:

**Goal:** demonstrate a policy-guarded agent that can **autonomously decide** and **settle USDC** based on motion input.

**Sprint plan (tight)**

1. **Day 1–2:** Motion upload + Gemini analysis endpoint + UI preview
2. **Day 3–4:** Attestation Oracle (kNN + RkCNN separation score) + decision UI
3. **Day 5–6:** Commerce Orchestrator (PaymentIntent + USDC settlement on Arc)
4. **Day 7:** End-to-end demo: capoeira→breakdance blend → MINT → TIP in USDC
5. **Day 8:** Polish UX + logging + runbook + 3-min video
