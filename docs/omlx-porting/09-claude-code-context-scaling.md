# 09 — Claude Code Context-Scaling + SSE Keep-Alive

**Tier:** ⭐ (Tier 3)
**Effort:** Low
**Impact:** Medium-high for the agentic use case (Claude Code / Pi / Codex)
**oMLX source:** `omlx/integrations/claude.py`, `omlx/server.py`, `omlx/patches/deepseek_v4/chat_template_v4.py`
**EXO target:** `src/exo/api/adapters/claude.py`, `src/exo/api/keepalive.py`, `src/exo/api/main.py`

---

## What it is

Two small but high-value features for agentic clients (Claude Code in particular):

### A. Context-scaling for small-context models
Scale **reported token counts** so the client's auto-compact triggers at the
right timing. Claude Code (and similar agents) decide when to auto-compact
based on reported usage vs. a context window. When serving a small-context
model, naive reporting causes premature or missed compaction. oMLX scales the
reported counts so auto-compact fires correctly.

From `omlx/integrations/claude.py`:
```python
env["CLAUDE_CODE_AUTO_COMPACT_WINDOW"] = str(ctx.context_window)
# min(detected_window, CLAUDE_CODE_AUTO_COMPACT_WINDOW). Setting ...
```
The integration sets `CLAUDE_CODE_AUTO_COMPACT_WINDOW` and the server scales
reported usage relative to the model's real context window.

### B. SSE keep-alive during long prefill
Clients with read timeouts (Claude Code) disconnect if the server is silent
during a long prefill. oMLX sends SSE keep-alive comments to prevent this.

From `omlx/server.py:2156`:
> causing clients with read timeouts (like Claude Code) to disconnect.

EXO already has a `keepalive` module — `src/exo/api/keepalive.py` — so part of
this may already exist. Verify and extend.

---

## Why it fits EXO

EXO explicitly targets agentic use (AGENTS.md mentions Claude Code, Pi, Codex,
Hermes Agent, Copilot). Both features are small and directly improve those
clients:

- **Context-scaling:** when EXO serves a small-context model to Claude Code,
  reported usage must reflect the real window so auto-compact triggers correctly.
- **SSE keep-alive:** long prefills (especially on large models across a
  cluster) can exceed client read timeouts; keep-alive pings prevent silent
  disconnects.

EXO already has the Anthropic Messages adapter (`api/adapters/claude.py`) and a
keepalive module — this is a polish pass, not new architecture.

---

## oMLX design

### Context-scaling (`integrations/claude.py`)
- Detects the model's real context window.
- Sets `CLAUDE_CODE_AUTO_COMPACT_WINDOW` for the integration.
- Server-side: scales reported `usage` token counts in responses so the client's
  auto-compact math works against the real window, not a default assumption.
- Global settings: `claude_code.mode`, `claude_code.opus_model`,
  `claude_code.sonnet_model`, `claude_code.haiku_model` (model mapping for
  Claude Code's opus/sonnet/haiku tier requests — Claude Code sends
  `model: claude-sonnet-...`, the server maps to a local model).

### SSE keep-alive (`server.py`)
- During long prefill, emit SSE comment frames (e.g. `:keep-alive\n\n`) at a
  configurable interval to keep the connection alive past client read timeouts.

### Chat-template compat (`patches/deepseek_v4/chat_template_v4.py`)
- Claude Code appends a volatile system message immediately after the user;
  converts `tool_result` blocks into `tool` role messages so Claude Code's
  format works with DeepSeek-V4's template. Relevant only if EXO serves
  DeepSeek-V4 to Claude Code.

---

## EXO current state

- `src/exo/api/keepalive.py` — **already exists.** Read it first; verify it
  covers SSE comment keep-alive during prefill. Extend if needed.
- `src/exo/api/adapters/claude.py` — Anthropic Messages adapter exists. Check
  whether it scales reported usage or maps opus/sonnet/haiku model ids.
- EXO reports `Usage` / `GenerationStats` / `PromptTokensDetails` /
  `CompletionTokensDetails` (see `src/exo/api/types/api.py`) — the scaling hook
  goes here.
- No Claude Code integration settings (model mapping, auto-compact window).

---

## Integration seam in EXO

### A. Context-scaling
- In `api/adapters/claude.py` (and `chat_completions.py` for OpenAI clients),
  scale the reported `usage` fields by a factor derived from
  `model_context_window / client_assumed_window`.
- Add a config: `EXO_CONTEXT_SCALE_WINDOW` (the client's assumed window, e.g.
  Claude Code's default) and the model's real window (from model config).
- Optional: opus/sonnet/haiku → local model mapping (a small alias table in
  config) so Claude Code's `model: claude-sonnet-4...` routes to a chosen
  local model.

### B. SSE keep-alive
- Read `src/exo/api/keepalive.py`. If it already emits SSE comments during
  prefill, just verify the interval and that it fires *before* the first token
  (the silent prefill window).
- If not, add a preflight keep-alive task that emits `:keep-alive\n\n` every N
  seconds until the first token streams.
- Make the interval configurable (`EXO_SSE_KEEPALIVE_INTERVAL`, default ~5s).

### C. Chat-template compat (optional, DeepSeek-V4 only)
- If EXO serves DeepSeek-V4 to Claude Code, port the `tool_result` → `tool`
  conversion from `chat_template_v4.py`. Otherwise skip.

---

## Phased plan

### Phase 1 — Audit + SSE keep-alive
- Read `src/exo/api/keepalive.py`; document current behavior.
- Ensure SSE comment keep-alive fires during the prefill silent window for both
  OpenAI and Anthropic adapters.
- Configurable interval.
- **Tests:** keep-alive frames emitted at expected interval during a synthetic
  slow prefill; no frames after first token; client-style read-timeout doesn't
  disconnect.

### Phase 2 — Context-scaling
- Implement usage scaling in the Anthropic + OpenAI adapters.
- Config: assumed client window + model real window.
- **Tests:** reported usage scales correctly; auto-compact trigger timing
  validated against a mock client.

### Phase 3 — Claude Code model mapping (optional)
- opus/sonnet/haiku → local model alias table.
- Dashboard UI to configure the mapping.
- **Tests:** `model: claude-sonnet-4...` routes to the configured local model.

---

## Risks & open questions

- **Scaling correctness:** over-scaling causes premature compaction (lost
  context); under-scaling causes OOM (model truncates). Validate against real
  Claude Code behavior, not just math.
- **Usage semantics:** Anthropic and OpenAI report usage differently
  (prompt/completion vs. input/output; cache vs. non-cache tokens). Scaling
  must apply to the right fields. EXO's `Usage`/`PromptTokensDetails` types
  need careful field mapping.
- **Keep-alive vs. first-token race:** ensure keep-alive stops cleanly when the
  first token arrives (no stray frames in the stream).
- **Existing keepalive module:** don't duplicate — extend `keepalive.py` rather
  than adding a parallel mechanism.

---

## Definition of done

- [ ] Phase 1: SSE keep-alive fires during prefill for both adapters; configurable
      interval; read-timeout test passes.
- [ ] Phase 2: usage scaling correct; auto-compact timing validated.
- [ ] Phase 3 (optional): Claude Code model mapping works end-to-end.
- [ ] `basedpyright` + `ruff` + `nix fmt` + `pytest` clean.