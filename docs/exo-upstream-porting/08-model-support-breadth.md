# 08 — Model Support Breadth: Unified Catalog + GGUF + Embeddings

**Tier:** ⭐⭐ (Tier 2)
**Effort:** Medium (catalog) / High (GGUF) / Low (embeddings)
**Impact:** Medium-high (UX + RAG + low-quant quality)
**Upstream evidence:**
- PR #2012 `Unified model catalog across HF, LM Studio, Ollama, llama.cpp + content-based resolution`
- PR #2145 `custom model cards + instance links`
- #1695 `[Feature Request] GGUF model loading` (7👍)
- #1047 `Add support for embedding models` (6👍, enhancement)
- #2172 `Opencode mlx community HF models not working`

**Cross-reference:** `docs/omlx-porting/07-embedding-reranker.md` (oMLX embeddings).

---

## What it is

Three related "model support breadth" asks, best tackled together because the **unified catalog** (#2012) is the backbone the others plug into:

### A. Unified model catalog (#2012)
Today EXO's model discovery is HF/MLX-centric. #2012 unifies the catalog across **HuggingFace, LM Studio, Ollama, llama.cpp** with **content-based resolution** (identify a model by its content/hash, not just name) — so the same model reachable via different providers resolves to one logical entry. Plus #2145 (custom model cards + instance links) lets users define their own cards.

### B. GGUF model loading (#1695)
> GGUF models, especially for low quants, are often slightly higher quality than their MLX-community counterparts.
>
> Two approaches: (1) model conversions, (2) a llama.cpp runner. Both have tradeoffs and are large undertakings.

GGUF is the dominant format for CPU/GPU inference outside MLX. Supporting it (likely via a llama.cpp runner engine) unlocks a huge model library + the CUDA path (doc 01, which likely uses llama.cpp anyway).

### C. Embedding models (#1047)
`/v1/embeddings` for RAG. EXO has no embedding engine today. (oMLX doc 07 covers the engine design.)

---

## Why it matters

- **Unified catalog (#2012):** users currently struggle to use non-HF sources (#2172 "Opencode mlx community models not working"). Content-based resolution means a model downloaded via LM Studio is recognized, not re-downloaded from HF. This ties into model distribution (doc 09 — avoid redundant downloads).
- **GGUF (#1695):** low-quant quality + the CUDA/Linux path. If EXO supports CUDA via llama.cpp (doc 01), GGUF comes naturally.
- **Embeddings (#1047):** RAG-over-EXO is a top request; without embeddings, EXO can't be the single local endpoint for a RAG pipeline.

---

## Upstream PR landscape

| PR/Issue | Scope |
|----------|-------|
| #2012 | Unified catalog (HF/LM Studio/Ollama/llama.cpp) + content-based resolution — the backbone |
| #2145 | Custom model cards + instance links |
| #2071 (merged) | Add node backends to model cards |
| #2024/#2025 (merged) | Custom model cards in State + reconcile from State |
| #1695 | GGUF loading (issue; no PR yet) |
| #1047 | Embedding models (issue; no PR yet) |

Custom model cards are already merged upstream; #2012 is the unification on top.

---

## EXO current state (local fork)

- `src/exo/shared/models/model_cards.py` — `ModelCard` exists (with `supports_ring` to add per doc 02, `drafter_model_id` per doc 03).
- Model discovery: HF/MLX-focused (`src/exo/download/`).
- Engines: `mlx` + `image` only — no GGUF/llama.cpp, no embeddings.
- VLM: ✅ done locally (ahead of upstream #1002).
- Local fork has GLM-5.2 model cards.

---

## Integration seam

- **Catalog (#2012):** port the unified catalog + content-based resolution into `src/exo/shared/models/` and the download/discovery layer. Custom cards already in State (merged) — build on that.
- **GGUF (#1695):** add a `llama.cpp` engine under `src/exo/worker/engines/llamacpp/` (or reuse for CUDA, doc 01). Model card `format: gguf`. Decide: wrap llama.cpp server, or use llama-cpp-python. The runner pattern (`src/exo/worker/runner/`) is the seam.
- **Embeddings (#1047):** add `src/exo/worker/engines/mlx/embedding.py` (oMLX doc 07 design) + `/v1/embeddings` route (`src/exo/api/`). `mlx-embeddings` optional dep.
- **Model card fields:** add `format` (mlx | gguf | …), `backend` (mlx | cuda | llamacpp | cpu), `model_type` (llm | vlm | embedding | reranker).

---

## Phased plan

### Phase 1 — Unified catalog (port #2012)
- Port the unified catalog + content-based resolution.
- Reconcile with existing custom-card-in-State (merged #2024/#2025).
- **Tests:** same model via HF + LM Studio resolves to one entry; content hash stable; #2172 Opencode models resolve.

### Phase 2 — Embeddings (#1047)
- `mlx-embeddings` dep; `EmbeddingEngine`; `/v1/embeddings` route.
- Model-type detection for BERT/BGE-M3/ModernBERT.
- **Tests:** BGE-M3 output parity; batch handling; OpenAI API shape.
- (Reranker can follow — see oMLX doc 07.)

### Phase 3 — GGUF via llama.cpp (#1695)
- `engines/llamacpp/` runner; `format: gguf` model cards.
- Decide wrapper (llama.cpp server vs llama-cpp-python).
- **Tests:** GGUF model loads + generates; output parity vs MLX equivalent (within tol); low-quant quality spot-check.

### Phase 4 — Content-based dedup with distribution (compose with doc 09)
- A model already present (any format) isn't re-downloaded; content hash shared across nodes for p2p distribution.

---

## Risks & open questions

- **GGUF is large:** a llama.cpp runner is a big undertaking (#1695 says so). Consider gating on the CUDA path (doc 01) — if CUDA uses llama.cpp, GGUF comes with it; otherwise GGUF-on-Mac is a separate, lower-priority effort.
- **Embedding co-location:** without the engine pool (oMLX doc 05 Scope 2), embedding + LLM can't coexist. Phase 2 is still useful on a dedicated node.
- **Catalog migration:** unifying the catalog may change model ids users reference. Keep backward-compatible aliases.
- **Content-based resolution cost:** hashing large model files is expensive; do it once at download, cache the hash in the card.
- **Format↔backend matrix:** GGUF on MLX isn't a thing; GGUF on CUDA/llama.cpp is. The catalog must filter by node backend capability.

---

## Definition of done

- [ ] Phase 1: unified catalog; content-based resolution; #2172 fixed.
- [ ] Phase 2: `/v1/embeddings` works with BGE-M3.
- [ ] Phase 3: GGUF model loads + generates via llama.cpp runner.
- [ ] `basedpyright` + `ruff` + `nix fmt` + `pytest` clean.