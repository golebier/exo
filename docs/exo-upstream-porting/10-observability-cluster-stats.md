# 10 — Observability: Prometheus /metrics + Cluster Stats

**Tier:** ⭐ (Tier 3)
**Effort:** Low–medium
**Impact:** Medium (ops/production readiness)
**Upstream evidence:**
- PR #1985 `feat: Prometheus /metrics endpoint` (adurham)
- PR #2144 `move metrics to zenoh Last Value semantics` (Evanev7)
- PR #2126 `feat(telemetry): MVP support for opt-in telemetry of runner crashes` (AndreiCravtov)
- #1700 `[FEATURE REQUEST] Global cluster stats`

---

## What it is

EXO has **no metrics endpoint** and no global cluster stats view. Three open PRs + one issue address observability:

- **#1985** — Prometheus `/metrics` endpoint (standard scraped metrics).
- **#2144** — move metrics to zenoh Last Value semantics (cluster-wide metric aggregation via zenoh).
- **#2126** — opt-in telemetry of runner crashes (anonymous crash reporting).
- **#1700** — global cluster stats (aggregate per-node stats into a cluster view).

Together: per-node Prometheus metrics + zenoh-aggregated cluster stats + opt-in crash telemetry.

---

## Why it matters

- **Production readiness:** any ops team running EXO needs `/metrics` for Prometheus/Grafana dashboards. Without it, EXO is unobservable.
- **Cluster debugging:** #2208-style hangs are far easier to diagnose with per-node prefill/decode TPS, RDMA collective latency, and memory time-series. The RDMA reliability cluster (doc 06) would benefit hugely.
- **Placement feedback:** cluster stats close the loop on placement (doc 05) — measure actual `C_i`/`L_i` vs predicted.
- **Low effort, high leverage:** #1985 is a standard FastAPI `/metrics` endpoint.

---

## Upstream PR landscape

| PR | Scope |
|----|-------|
| #1985 | Prometheus `/metrics` endpoint (the core) |
| #2144 | zenoh Last Value semantics for cluster-wide metrics |
| #2126 | opt-in runner-crash telemetry |

#1985 + #2144 compose: per-node Prometheus metrics + zenoh aggregation for cluster-wide views.

---

## EXO current state (local fork)

- `rg prometheus|/metrics` in `src/exo/api` → **nothing**. No metrics endpoint.
- EXO already has energy/power profiling (merged #2124 "capture energy in prefill and generation separately", #2038 time-weighted power sampling, #2041 ANE power) — rich per-node stats exist internally, just not exposed.
- zenoh in place (good for #2144).
- Dashboard exists (Svelte) — could surface cluster stats (#1700).

---

## Integration seam

- **`/metrics` endpoint:** add to `src/exo/api/main.py` (Prometheus exposition format). Expose: per-model prefill/decode TPS, KV cache hit/evict rates, memory usage, RDMA collective latency, request counts/latencies, energy.
- **zenoh Last Value (#2144):** publish metrics over zenoh with Last Value semantics so any node can read cluster-wide values.
- **Cluster stats (#1700):** aggregate in the master (or any node via zenoh) and surface in the dashboard.
- **Crash telemetry (#2126):** opt-in runner crash reporting (respect privacy; default off).

---

## Phased plan

### Phase 1 — Prometheus `/metrics` (port #1985)
- Add `/metrics` endpoint exposing existing internal stats (energy, TPS, memory, cache).
- **Tests:** `/metrics` returns valid Prometheus format; counters monotonic; gauges sane.

### Phase 2 — zenoh cluster aggregation (port #2144)
- Publish metrics over zenoh Last Value; aggregate cluster-wide.
- **Tests:** 3-node cluster; any node reads all 3 nodes' metrics via zenoh.

### Phase 3 — Dashboard cluster stats (#1700)
- Surface cluster-wide stats in the Svelte dashboard.
- **Tests:** dashboard shows per-node + aggregate TPS/memory.

### Phase 4 — Opt-in crash telemetry (port #2126)
- Runner crash reports, opt-in, anonymized.
- **Tests:** crash captured; opt-in respected; no PII.

---

## Risks & open questions

- **Metric cardinality:** high-cardinality labels (per-request) explode Prometheus. Keep labels low-cardinality (per-model, per-node, per-shard).
- **Performance overhead:** metrics collection must not stall inference. Sample asynchronously; reuse the existing energy/profiling samplers.
- **Privacy (#2126):** crash telemetry must be opt-in and scrub model names/prompts. Default off.
- **zenoh Last Value semantics:** ensure #2144's semantics match EXO's zenoh version (post-#2132 migration).
- **Compose with RDMA debugging (doc 06):** add RDMA collective latency + link health metrics — directly helps diagnose #2208/#1847.

---

## Definition of done

- [ ] Phase 1: `/metrics` live; Prometheus scrapes; covers TPS/memory/cache/energy.
- [ ] Phase 2: cluster-wide metrics via zenoh.
- [ ] Phase 3: dashboard cluster-stats view.
- [ ] Phase 4: opt-in crash telemetry.
- [ ] `basedpyright` + `ruff` + `nix fmt` + `pytest` clean.