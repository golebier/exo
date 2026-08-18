# Gra's Notes & Investigations

Personal working docs for this fork (`golebier/exo`) — build notes and the
GLM-5.2 investigation thread. These are *not* part of upstream `exo-explore/exo`
and live here to keep the repo root aligned with upstream.

## Index

| Doc | Description |
|-----|-------------|
| [EXO-build-and-dependencies.md](EXO-build-and-dependencies.md) | How to build/release the macOS `.app`/DMG on this fork; prerequisites, release pipeline, `v1.0.72-dev1` tag, and how it differs from upstream `v1.0.71`. |
| [GLM-5.2-EXO-PLAN.md](GLM-5.2-EXO-PLAN.md) | The original phased plan for fixing `avlp12/GLM-5.2-Alis-MLX-Dynamic-2.3bpw` (`glm_moe_dsa`) in EXO — phases 1–6. |
| [GLM-5.2-INVESTIGATION-SUMMARY.md](GLM-5.2-INVESTIGATION-SUMMARY.md) | Why the implemented fix wasn't working in practice ("produces 0s/garbled"); reviews the plan and code. |
| [GLM-5.2-DEEP-SUMMARY.md](GLM-5.2-DEEP-SUMMARY.md) | Deep investigation that disproves the "missing `install_local_sharded_load_fallback()`" hypothesis and reframes the problem. |
| [GLM-5.2-RESEARCH-RESULTS.md](GLM-5.2-RESEARCH-RESULTS.md) | Cached-source research verifying EXO's vendored GLM code is a faithful oMLX port; refutes the argpartition/NaN hypothesis; cites sources. |
| [GLM-5.2-FIX-SUMMARY.md](GLM-5.2-FIX-SUMMARY.md) | Full fix summary (every fix applied, in order) for the 2× MSU tensor-parallel config; final artifact `EXO-1.0.72-GLM-5.2-dev9.dmg`. |
| [LATEST-FOUNDINGS-GLM-5.2-issue.md](LATEST-FOUNDINGS-GLM-5.2-issue.md) | Latest research-session findings on the GLM-5.2 "produces 0s" issue and upstream #2208. |

## Related

- [`../omlx-porting/`](../omlx-porting/) — oMLX → EXO feature-porting analysis (9 docs + index).
- [`../exo-upstream-porting/`](../exo-upstream-porting/) — `exo-explore/exo` issues/PRs analysis (12 docs + index); doc `06` covers the **RDMA/networking** side of GLM-5.2 issue #2208 (complementing the model-side work documented here).

## Reading order (GLM-5.2 thread)

The GLM-5.2 docs form an investigation thread; chronological reading order:

1. `GLM-5.2-EXO-PLAN.md` — the plan.
2. `GLM-5.2-INVESTIGATION-SUMMARY.md` — why it didn't work.
3. `GLM-5.2-DEEP-SUMMARY.md` — deeper, disproves the leading hypothesis.
4. `GLM-5.2-RESEARCH-RESULTS.md` — verified evidence (the authoritative reference).
5. `GLM-5.2-FIX-SUMMARY.md` — the fixes that landed.
6. `LATEST-FOUNDINGS-GLM-5.2-issue.md` — most recent session.