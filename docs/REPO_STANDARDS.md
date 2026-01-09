# Repository Standards

## Python Style & Tooling
- PEP 8; type hints everywhere.
- Pydantic v2 with `Field` constraints; custom validators for cross-field rules.
- Lint: Flake8 + Black + isort; MyPy strict.
- Tests: pytest with fixtures; mock external services (Gemini, Arc RPC, Circle, storage).

## Fivetran Connector Conventions
- Logging: use `log.fine()`, `log.info()`, `log.warning()`, `log.severe()` (never `log.error()`).
- Retry: exponential backoff; `__MAX_RETRIES = 3`, `__BASE_DELAY_SECONDS = 1`; cap at 60s.
- configuration.json: strings only; cast types in code.
- `schema()` defines tables and primary keys; `update()` performs upserts and checkpoints.

## Attestation Canonicalization
- Sorted JSON keys; stable float rounding (6 decimals, ties-to-even); UTF-8 bytes.
- `pack_hash = keccak256(canonical_bytes)`; optional `raw_hash`, `tensor_hash`.
- Embeddings as decision evidence: store `embedding_hash` + metadata; avoid on-chain bytes.

## RkCNN Defaults (Natural Mathematics)
- `m = min(d, max(16, round(4*sqrt(d))))`; `E = max(32, min(128, 8*ceil(log2(d))))`.
- Projections: `achlioptas_sparse` (default) or `gaussian`; metric: `cosine` if L2-normalized else `euclidean`.
- Vote margin: `(V_top - V_second)/E`; default threshold 0.10.

## Policy & Payout
- `payout_split` must sum to 1.0; percentages in [0.0, 1.0]; tolerance ≤ 1e-6.
- `ethics_modifiers` multipliers in [0.0, 2.0]; caps in [0.0, 1.0].
- `base_rate ≥ 0`; `unit ∈ {USD/s, ETH/s, USD/h, ETH/h}`; `max_seconds ≥ 0`.

## Observability & IDs
- Correlation IDs: `ingest_id → analysis_id → attest_id → tx_hash → payment_intent_id`.
- Structured logs per service; warehouse via Fivetran for analytics and billing.
