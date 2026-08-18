# 07 — Embedding + Reranker Engines

**Tier:** ⭐ (Tier 3)
**Effort:** Medium
**Impact:** Medium (unlocks RAG-over-EXO)
**oMLX source:** `omlx/engine/{embedding,reranker,base}.py`, `omlx/models/{embedding,reranker,qwen2_embedding,xlm_roberta,mlx_embeddings_compat}.py`, `omlx/api/{embedding_models,embedding_utils,rerank_models}.py`
**EXO target:** new `src/exo/worker/engines/mlx/embedding.py`, `reranker.py`; new API routes; `src/exo/api/`

---

## What it is

oMLX serves **embedding** and **reranker** models alongside LLMs/VLMs in the same
process, exposing:
- `POST /v1/embeddings` — text embeddings
- `POST /v1/rerank` — document reranking

Supported model families (oMLX):
- **Embedding:** BERT, BGE-M3, ModernBERT, Qwen2-Embedding (via `mlx-embeddings` compat)
- **Reranker:** ModernBERT, XLM-RoBERTa (SequenceClassification + CausalLM-based)

EXO has **only** `image` + `mlx` (LLM) engines — no embeddings, no rerankers.
This blocks RAG-over-EXO use cases (retrieve-then-generate with a local
embedding model + reranker).

---

## oMLX design

### `engine/embedding.py` — `EmbeddingEngine(BaseNonStreamingEngine)`
- Wraps `MLXEmbeddingModel` (from `omlx/models/embedding.py`, which wraps
  `mlx-embeddings`).
- Single forward pass; no streaming, no chat.
- Batch size configurable; `_input_length` groups similar-size inputs.
- Uses `get_mlx_executor()` to serialize MLX work.

### `engine/reranker.py` — `RerankerEngine(BaseNonStreamingEngine)`
- Wraps `MLXRerankerModel` (SequenceClassification + CausalLM-based rerankers).
- Single forward pass; returns `RerankOutput` (scores + rankings).

### `models/`
- `embedding.py` — `MLXEmbeddingModel`, `EmbeddingOutput`.
- `reranker.py` — `MLXRerankerModel`, `RerankOutput`.
- `qwen2_embedding.py` — Qwen2-Embedding specific.
- `xlm_roberta.py` — XLM-RoBERTa reranker.
- `mlx_embeddings_compat.py` — compatibility shim for `mlx-embeddings`.

### `api/`
- `embedding_models.py` + `embedding_utils.py` — `/v1/embeddings` request/response models.
- `rerank_models.py` — `/v1/rerank` request/response models.

### Auto-detection
`model_discovery.py` auto-detects model type (LLM/VLM/embedding/reranker) from
config. `is_realtime_stt_model` etc. exist for audio.

---

## EXO current state

- Engines: `src/exo/worker/engines/{image,mlx}/` only.
- API routes: chat completions, responses, claude, ollama — **no** `/v1/embeddings`
  or `/v1/rerank`.
- EXO's `mlx` engine is LLM-only (`batch_generate.py` / `generate.py`).
- EXO has no `mlx-embeddings` dependency.

---

## Integration seam in EXO

- **Dependency:** add `mlx-embeddings` to `pyproject.toml` (optional extra
  `[embeddings]` to keep the default install light).
- **Engine classes:** new `src/exo/worker/engines/mlx/embedding.py` +
  `reranker.py`, modeled on oMLX's but conforming to EXO's engine/runner
  patterns (`src/exo/worker/runner/`).
- **Model type detection:** extend EXO's model discovery to classify
  embedding/reranker models from `config.json` (architecture → type). Add a
  `ModelType` literal (`llm` | `vlm` | `embedding` | `reranker`).
- **API routes:** add `/v1/embeddings` and `/v1/rerank` to `src/exo/api/main.py`
  with request/response types in `src/exo/api/types/`.
- **Placement:** embedding/reranker models are single-node (no sharding needed);
  the master places them like any model instance but they don't participate in
  sharded inference.
- **Dashboard:** list embedding/reranker models separately; allow loading them
  alongside an LLM (ties into doc 05's engine pool — or, pre-pool, treat as the
  "one model" with a mode toggle).

---

## Phased plan

### Phase 1 — Embeddings
- Add `mlx-embeddings` optional dep.
- Port `MLXEmbeddingModel` wrapper + `EmbeddingEngine`.
- Add `POST /v1/embeddings` route + types (OpenAI-compatible: `input`,
  `model`, `encoding_format`, `dimensions`).
- Model-type detection for BERT/BGE-M3/ModernBERT/Qwen2-Embedding.
- **Tests:** embedding output shape/dtype; batch handling; OpenAI API parity
  (compare against reference outputs for BGE-M3).

### Phase 2 — Reranker
- Port `MLXRerankerModel` + `RerankerEngine`.
- Add `POST /v1/rerank` route + types (Cohere-style: `query`, `documents`,
  `top_n`).
- Model-type detection for ModernBERT/XLM-RoBERTa.
- **Tests:** rerank score ordering correctness; top_n truncation; API parity.

### Phase 3 — Co-location with LLM (needs doc 05 Scope 2)
- Serve an embedding/reranker model concurrently with an LLM on one node via
  the EnginePool. Pre-pool, this requires unloading the LLM to load the
  embedder — acceptable for Phase 1/2 but not ideal.

---

## Risks & open questions

- **Co-location:** without doc 05's pool, embedding + LLM can't coexist. Phase
  1/2 are still useful standalone (dedicated embedding node in a cluster), but
  the UX is better with co-location. Decide whether to block on doc 05.
- **`mlx-embeddings` version drift:** pin a compatible version; add a
  compatibility test.
- **Cluster routing:** EXO's master routes chat completions to model instances.
  Embedding/rerank requests must route to the embedding/reranker instance. Add
  model-type-aware routing to the master.
- **Quantization:** embedding models are often small; decide whether EXO's
  quant tooling applies or if embeddings stay full-precision.
- **API auth:** EXO's API auth (if any) must cover the new routes.

---

## Definition of done

- [ ] Phase 1: `POST /v1/embeddings` works with BGE-M3; output matches reference
      within tol; batch handling correct.
- [ ] Phase 2: `POST /v1/rerank` works with ModernBERT/XLM-RoBERTa; score
      ordering correct.
- [ ] Model-type auto-detection classifies embedding/reranker correctly.
- [ ] `basedpyright` + `ruff` + `nix fmt` + `pytest` clean.