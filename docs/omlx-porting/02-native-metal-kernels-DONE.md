# 02 — Native Metal Custom Kernels: Implementation & Post-Deploy Issues (DONE log)

**Status:** Native kernel build/packaging implemented and verified; two runtime
regressions found during cluster deployment and fixed; the third (pre-existing)
JACCL rendezvous race is **fixed** (see Issue #3 below). This doc is the
post-mortem / implementation record. See
[`02-native-metal-kernels.md`](./02-native-metal-kernels.md) for the original
design rationale and [`mlx_kernels/README.md`](../../mlx_kernels/README.md) for
build mechanics.

**Build tag:** `v1.0.72-native-metal-kernels-dev1`
**oMLX commit vendored:** `1f1aff3018c097bcbeafca9a483e58d04dee38ba`
**Target model:** `Jundot/GLM-5.2-oQ4` (2-node Tensor/RDMA, M3 Ultra 256GB ×2)

---

## 0. What was built (summary)

Vendored oMLX's native GLM-5.2 Metal kernels under EXO's own namespace and
added an opt-in build path (`EXO_BUILD_MLX_KERNELS=1`) that compiles a
nanobind `_ext` extension + `omlx_glm_kernels.metallib` and bundles them into
the PyInstaller app. When the extension is absent, the GLM-5.2 model falls
back to the standard attention path unchanged (causally safe — see
`../gra/GLM-5.2-RESEARCH-RESULTS.md` §5).

### Files added / changed

| File | Status | Purpose |
|------|--------|---------|
| `mlx_kernels/build_kernels.py` | new | Standalone CMake build: `_ext.so` + metallib → installed next to `fast.py` |
| `mlx_kernels/parts.nix` | new | Darwin-only nix derivation (skeleton) |
| `mlx_kernels/OMLX_COMMIT.txt` | new | Pinned oMLX commit |
| `mlx_kernels/README.md` | new | Build/packaging mechanics |
| `src/exo/worker/engines/mlx/vendor/omlx_custom_kernels/` | new | Vendored oMLX `custom_kernels` tree (`csrc/`, `fast.py`, `__init__.py`) |
| `src/exo/worker/engines/mlx/vendor/glm_moe_dsa/kernels.py` | modified | `_FastDispatch` → vendored namespace + **lazy `_ext` import** |
| `src/exo/worker/engines/mlx/patches/glm_moe_dsa/__init__.py` | modified | Removed eager native-availability probe at patch-apply time |
| `packaging/pyinstaller/exo.spec` | modified | Conditionally bundle `_ext.so` + `*.dylib` + metallib (incl. `_internal/` copy) |
| `BUILD.sh` | modified | Step 2b: `EXO_BUILD_MLX_KERNELS=1` opt-in kernel build |
| `justfile` | modified | `mlx-kernels` recipe |
| `flake.nix` | modified | Wire `mlx_kernels/parts.nix` |
| `src/exo/worker/tests/unittests/test_mlx/test_glm_moe_dsa_native_kernels.py` | new | Dispatch + availability + build-gated smoke tests |

### Build commands

```bash
# Full app + DMG with native kernels (Darwin + full Xcode required)
EXO_VERSION=v1.0.72-native-metal-kernels-dev1 \
EXO_BUILD_MLX_KERNELS=1 \
./BUILD.sh

# Just the kernels (installs _ext.so + metallib into the source tree)
just mlx-kernels
# or: uv run --extra build python mlx_kernels/build_kernels.py --force
```

The build script (`mlx_kernels/build_kernels.py`) self-verifies: Darwin-only,
`xcrun -sdk macosx metal --version`, `mlx` + `nanobind` importable, then CMake
configure/build against the active interpreter. It installs
`_ext.cpython-313-darwin.so` + `omlx_glm_kernels.metallib` next to `fast.py`,
runs an ABI probe (`abi_probe(mx.zeros(...))`), and a runtime import check.

---

## 1. Issue #1 — `_ext` metallib not found in the PyInstaller bundle

### Symptom

The native kernels **imported** fine in the frozen bundle
(`native_available() == True`), but the **first kernel execution** crashed:

```
RuntimeError: Failed to load the metallib from
.../EXO.app/Contents/Resources/exo/_internal/omlx_glm_kernels.metallib
with error library not found
```

### Root cause

The C++ kernels locate their `.metallib` via `current_binary_dir()`, defined
in every `csrc/*.cpp` (8 sites across `dsa_indexer.cpp`, `sparse_mla.cpp`,
`deepseek_v4_sparse_attention.cpp`, `exact_block_attention.cpp`,
`fused_moe.cpp`, `dspark_gemm.cpp`, `dspark_qmv.cpp`):

```cpp
std::string current_binary_dir() {
  static std::string binary_dir = []() {
    Dl_info info;
    if (!dladdr(reinterpret_cast<void*>(&current_binary_dir), &info)) {
      throw std::runtime_error("Unable to get omlx_glm_kernels binary dir.");
    }
    return std::filesystem::path(info.dli_fname).parent_path().string();
  }();
  return binary_dir;
}
// ...
auto lib = d.get_library("omlx_glm_kernels", current_binary_dir());
```

`dladdr(&current_binary_dir, &info)` returns `info.dli_fname` = the Mach-O
image containing that function. In a dev/venv run the function lives in
`_ext.so` (at `.../glm_moe_dsa/_ext.so`) → `dladdr` returns `_ext.so`'s path
→ parent is `.../glm_moe_dsa/` → the metallib (installed next to `_ext.so`)
is found. ✅ (This is why the build-time runtime check passed.)

In a **PyInstaller `--onedir` bundle**, `dladdr()` on a symbol inside an
extension module resolves to the **bootloader/core image under `_internal/`**
rather than to `_ext.so`'s own directory. So the runtime looked for the
metallib in `_internal/`, but PyInstaller had placed it only next to `_ext.so`
in `.../glm_moe_dsa/`. Result: "library not found" at first kernel call.

### Verification (the probe that caught it)

```bash
APP="app/EXO/build/Build/Products/Release/EXO.app"
"$APP/Contents/Resources/exo/exo" -c '
import mlx.core as mx
from exo.worker.engines.mlx.vendor.glm_moe_dsa import kernels as g
mx.random.seed(0)
q = mx.random.normal((1,32,64,128)).astype(mx.float16)
k = mx.random.normal((1,1,4096,128)).astype(mx.float16)
w = mx.random.normal((1,64,32)).astype(mx.float16)
out = g.fast.dsa_indexer_scores(q,k,w,causal=True,stream=mx.gpu)
mx.eval(out)   # <-- crashed here before the fix
'
```

### Fix (spec-only, no C++ rebuild)

Bundle the metallib **twice** in `packaging/pyinstaller/exo.spec`: once at the
package dir (for dev runs/tests) and once at dest `"."` (= `_internal/`,
where `dladdr`/`current_binary_dir()` looks in the frozen bundle):

```python
metallib = GLM_KERNEL_PKG_DIR / "omlx_glm_kernels.metallib"
if metallib.is_file():
    metallib_dest = str(GLM_KERNEL_PKG_DIR.relative_to(SOURCE_ROOT))
    GLM_KERNEL_BINARIES.append((str(metallib), metallib_dest))
    # ALSO place the metallib at the _internal/ root (dest ".").
    # ... dladdr() resolves that symbol to the bootloader/core image under
    # _internal/ rather than to _ext.so's own directory ...
    GLM_KERNEL_BINARIES.append((str(metallib), "."))
```

PyInstaller now reports `bundling 4 GLM native kernel artifact(s)` (was 3).
After the fix the probe prints `out.shape=(1,1,64,4096)` ✅.

**Why spec-only:** rebuilding C++ to use a more robust lookup (e.g. a Python
callback passing `_ext.__file__`'s dir, or `NSBundle.mainBundle`) is the
"proper" fix, but the dual-placement spec fix is lower-risk and works because
`current_binary_dir()` is deterministic per-bundle. A future C++ change should
make the metallib path injectable from Python.

---

## 2. Issue #2 — Eager `_ext` import breaks JACCL GPU-RDMA init (regression)

### Symptom (first deploy, build r1)

Model "starts loading" but never loads; workers idle; dashboard stuck on
PREPARING. Logs showed the runner reaching `mx.distributed.init(backend="jaccl")`
→ the JACCL warmup `all_sum` **hung on both ranks**, runner died silently.
An earlier session crashed differently: reached prefill, then
`std::runtime_error: [METAL] Command buffer execution failed: Insufficient
Memory (kIOGPUCommandBufferCallbackErrorOutOfMemory)`.

### Root cause

The previous working build (`v1.0.72-GLM-5.2`, no native kernels built) had
`_native_fast = None` (omlx not installed) → **no `_ext` import** before
JACCL init → worked.

Our new `kernels.py` imported `_ext` **eagerly at module top-level**:

```python
try:
    from exo.worker.engines.mlx.vendor.omlx_custom_kernels.glm_moe_dsa import fast as _native_fast
except Exception:
    _native_fast = None
```

`apply_glm_moe_dsa_patch()` runs in the runner **before**
`mx.distributed.init(backend="jaccl")` (see
`src/exo/worker/runner/bootstrap.py:75-77`). Importing `_ext.so` (which links
`Metal.framework` + `libmlx.dylib`) **initializes the Metal device**. Proven:

```
BEFORE _ext import: active_mem= 0  peak= 0
after _ext import:  active_mem= 8  peak= 12   ← Metal device created
```

The jaccl GPU-RDMA backend needs to **own the Metal device's first
initialization** (it registers GPU memory for RDMA). An earlier eager Metal
init corrupts that setup → warmup `all_sum` hangs + later prefill OOM.

### Fix — lazy `_ext` import (`kernels.py`)

Defer the `_ext` import to **first access** via a cached resolver. `import
kernels` and `apply_glm_moe_dsa_patch()` no longer touch `_ext`/Metal. The
first kernel lookup happens during model load (after distributed init).

```python
_native_fast: Any = None
_native_import_error: Exception | None = None
_native_resolved: bool = False


def _resolve_native_fast() -> Any:
    """Lazily import the native GLM fast module on first access."""
    global _native_fast, _native_import_error, _native_resolved
    if _native_resolved:
        return _native_fast
    _native_resolved = True
    # ... env-gate (EXO_NATIVE_GLM_KERNELS=0), vendored import, omlx fallback,
    #     one-time availability log ...
    return _native_fast
```

Every `_FastDispatch` method (`__getattr__`, `has`, `missing`,
`native_available`, `native_import_error`, `__dir__`) calls
`_resolve_native_fast()` instead of reading a module-level `_native_fast`.

Companion change in `patches/glm_moe_dsa/__init__.py`: removed the eager
`_native_kernel_summary()` call at `apply_glm_moe_dsa_patch()` time (it
imported `_ext`). The availability log moved into `_resolve_native_fast()`,
fired once at first resolution (during model load, after JACCL init — safe).

Bonus: `EXO_NATIVE_GLM_KERNELS=0` forces the fallback path (skip the native
import entirely) for emergency disable.

### Verification

```bash
"$EXO_BIN" -c '
import sys
from exo.worker.engines.mlx.vendor.glm_moe_dsa import kernels as g
from exo.worker.engines.mlx.patches.glm_moe_dsa import apply_glm_moe_dsa_patch
apply_glm_moe_dsa_patch()
target = "exo.worker.engines.mlx.vendor.omlx_custom_kernels.glm_moe_dsa._ext"
print("LAZY: _ext loaded after apply_patch?", target in sys.modules)  # False
print("LAZY: native_resolved?", g._native_resolved)                    # False
_ = g.fast.native_available()  # first access — triggers _ext import
print("LAZY: _ext loaded after access?", target in sys.modules)       # True
'
```

After the fix, on the cluster: **rank 0 now completes the warmup** (previously
both ranks hung). Test suite: `9 passed, 2 skipped`.

### Lint note (resolved)

`_native_kernel_summary` was originally in `patches/glm_moe_dsa/__init__.py`
and called by `apply_glm_moe_dsa_patch`. After the lazy-import fix removed
that call, it became test-only, and basedpyright's `reportUnusedFunction`
flagged it (the test file is `# type: ignore`d top-level for mlx `Any`
returns, so basedpyright doesn't see the cross-module import). Resolved by
moving the helper into the test module
(`test_glm_moe_dsa_native_kernels.py`) — it's a pure test diagnostic and
belongs there.

---

## 3. Issue #3 — JACCL `all_sum` rendezvous race (pre-existing, exposed by timing)

> ✅ **Fixed.** The in-process warmup `all_sum` was **removed** entirely
> (commit `17252452`) and model-load rendezvous is now enforced by the
> master's lifecycle gating in `plan.py` (rank 0 is not sent `LoadModel`
> until rank 1 reaches `RunnerConnected`). The RDMA data path itself is
> validated by `_probe_rdma_interface` in a subprocess (both ranks
> participate simultaneously). Verified on the live 2-node M3 Ultra cluster:
> a 35,969-token prefill completed successfully and the cluster is serving.

### Symptom (second deploy, build r4 with the lazy fix)

Rank 0 completes warmup; rank 1 hangs forever; model never loads. Dashboard
stuck on PREPARING/loading.

### Evidence (live cluster, `Jundot/GLM-5.2-oQ4`, M3 Ultra ×2)

**Rank 0 (m3msu256b, coordinator):**
```
11:33:47.968  Rank 0 warming up JACCL data path
11:33:47.969  Rank 0 JACCL warmup complete   ← 1 millisecond (!)
11:33:47.969  runner connected
```
…then idles on a Python lock (`_PySemaphore_Wait`) waiting for rank 1's
model-load coordination.

**Rank 1 (m3msu256a, connector):**
```
11:33:48.005  Rank 1 warming up JACCL data path   ← 37ms AFTER rank 0 left
```
Stack (`sample` of the wedged runner PID, stable for 30+ min):
```
mlx::core::eval → mlx::core::array::wait → mlx::core::Event::wait
→ -[IOSurfaceSharedEvent waitUntilSignaledValue:timeoutMS:]
→ iokit_user_client_trap
```

### Root cause

The warmup is a **collective** — both ranks must participate simultaneously:

```python
# src/exo/worker/engines/mlx/utils_mlx.py:283-285
logger.info(f"Rank {rank} warming up JACCL data path")
mx.eval(mx.distributed.all_sum(mx.array(1.0), group=group))
logger.info(f"Rank {rank} JACCL warmup complete")
```

But **rank 0 completed it in 1ms and left before rank 1 even arrived**
(47.969 vs 48.005). A real RDMA `all_sum` between two machines cannot complete
in 1ms — rank 0's 1ms return is **degenerate** (it returned locally without
exchanging with rank 1 over RDMA). `mx.distributed.init(backend="jaccl",
strict=True)` is supposed to rendezvous both ranks before the group is usable,
but rank 0 (the coordinator) is completing the subsequent `all_sum` before
rank 1 joins. Rank 1 then waits on a GPU `IOSurfaceSharedEvent` for a partner
that's already torn down its side → permanent hang.

This is a **rendezvous race**, not an RDMA-cable failure: the
`_probe_rdma_interface` subprocess (which runs the *same* `all_sum` on the
*same* interface, in a clean process) **succeeded** for rank 1 just before the
real init hung. The difference is the runner's import/timing state.

### Why it wasn't seen before

The previous working build (`v1.0.72-GLM-5.2-RDMA-dev2`, no `_ext` in the
import graph) had rank 1 arriving at the collective at/before rank 0. Our
native-kernel work shifted the runner's import sequence enough that rank 0 now
wins the race.

### Fix (implemented)

The retry-loop approach proposed below was **rejected** after analysis: the
1ms degenerate return is deterministic for the coordinator (JACCL caches the
group state and returns instantly without re-exchanging), so a value-checking
retry loop cannot help rank 1 — only rank 0's behavior changes. There is no
portable in-process fix because the race is in the coordinator's unilateral
completion.

Instead, the fix has two parts:

1. **Removed the in-process warmup `all_sum`** from `utils_mlx.py`
   (commit `17252452`). The RDMA data path is validated exclusively by
   `_probe_rdma_interface`, which runs a full `all_sum` in a **subprocess**
   with both ranks participating simultaneously — sidestepping the in-process
   coordinator race entirely.
2. **Master-side lifecycle gating** in `plan.py`: rank 0 is not sent
   `LoadModel` until rank 1 has reached `RunnerConnected`. This guarantees
   both ranks are past distributed init and the group is established before
   any model-load collective runs. If the in-process group is broken, model
   loading's first real collective (which both ranks enter in lockstep) will
   surface it — and the master's `_probe_rdma_interface` already caught any
   RDMA data-path failure.

The `utils_mlx.py` code now carries an explicit comment documenting why no
in-process warmup runs and what guarantees correctness.

> **Note (rejected alternative).** The original proposal was a value-checking
> retry loop: repeat `all_sum(1.0)` until `result == world_size`. This was
> rejected because the coordinator's degenerate 1ms return is cached and
> deterministic — it repeats on every retry without re-exchanging. A real
> fix would require a coordinator-side ack round via the TCP/zenoh
> side-channel, but the master-lifecycle gating above already provides that
> rendezvous guarantee out-of-band. Kept here for the record:

```python
# REJECTED: retry loop does not work (degenerate return is cached).
# world_size = len(jaccl_devices)
# deadline = time.monotonic() + 60.0
# while True:
#     result = mx.eval(mx.distributed.all_sum(mx.array(1.0), group=group))
#     if int(result.item()) == world_size:
#         break
#     if time.monotonic() > deadline:
#         raise TimeoutError(...)
#     time.sleep(0.5)
```

### Verification (live cluster, `GLM-5.2-oQ4`, M3 Ultra ×2, build `1.0.72-cache-efficiency-dev2`)

Both nodes running commit `42762e44` (which includes the `17252452` fix).
Model load succeeds; a 35,969-token prefill completed (11:44→11:47) followed
by successful decode. No `Fence::wait`/`IOSurface`/rendezvous-hang evidence
in recent logs. The runner process is idle-healthy (parked on
`_PySemaphore_Wait`, not wedged in a collective).

---

## 4. Diagnostic playbook (how each issue was found)

Captured so the next person can reproduce the debugging, not just the fixes.

### Where logs live

| Location | What |
|----------|------|
| `~/.exo/exo_log/exo.log` | Main exo process (loguru file sink, `logger_setup(EXO_LOG, ...)`) |
| `~/.exo/exo_log/runner_log/stderr.log` | Runner child subprocess stderr (faulthandler dumps land here) |
| `~/.exo/exo_log/runner_log/stdout.log` | Runner child stdout |
| `~/.exo/exo_log/*.log.zst` | Rotated/compressed archives (decompress: `zstd -dc <file>`) |
| `~/.exo/event_log/{master,api}/` | Event-sourced cluster state |

**Gotcha:** the Swift app (`app/EXO/EXO/ExoProcessController.swift:188-189`)
sends the exo process's stdout/stderr to `FileHandle.nullDevice`. The
**only** persistent logs are the loguru file sink above. A runner that dies
before writing to `runner_log/` leaves no trace in `exo.log` either (the
supervisor's `_watch_runner` polls every 5s but may not fire on a wedged-not-
dead process). When in doubt: `sample <pid> 2` to see the live C++ stack.

### Probes used

1. **Metallib lookup** — `exo -c '... dsa_indexer_scores(...); mx.eval(out)'`
   against the bundled binary. Caught Issue #1 ("library not found").
2. **Metal-init-on-import** — `mx.get_active_memory()` before/after `import
   kernels` / `apply_patch`. Caught Issue #2 (0 → 8 bytes on `_ext` import).
3. **Lazy-import verification** — `target in sys.modules` after `apply_patch`
   (False) and after first `has()` (True). Confirmed the Issue #2 fix.
4. **Live runner stack** — `sample <runner-pid> 2 -mayDie | grep -A30 "Call
   graph"`. Caught Issue #3 (`IOSurfaceSharedEvent waitUntilSignaledValue`).
5. **`dladdr` resolution** — `otool -L`/`otool -l` on `_ext.so`; `ctypes`
   `dladdr` on a symbol. Confirmed the `_internal/` misresolution.

### Cluster SSH access

```bash
ssh gra@m3msu256a.local 'tail -n 60 ~/.exo/exo_log/exo.log'   # rank 1
ssh gra@m3msu256b.local 'tail -n 60 ~/.exo/exo_log/exo.log'   # rank 0
ssh gra@m3msu256a.local 'sample <pid> 2 -mayDie'
ssh gra@m3msu256a.local 'ps -eo pid,ppid,pcpu,command | grep exo'
```

Note: `--multiprocessing-fork tracker_fd=9` in `ps` args is misleading — that
process is the **runner worker** spawned by `AsyncProcess` (`mp.Process`), not
the resource tracker. Identify the hung runner by 100% CPU + the `sample`
stack in `mlx::core::eval`.

---

## 5. Build/deploy record

| Build | Tag | Key change | DMG SHA256 | Result |
|-------|-----|------------|------------|--------|
| r1 | `v1.0.72-native-metal-kernels-dev1` | native kernels + eager `_ext` import | `a4409175…` | ❌ metallib OK after probe, but JACCL hung both ranks (Issue #2) |
| r2/r3 | same | + metallib `_internal/` spec fix (Issue #1) | `4cabf273…` | ❌ still Issue #2 (eager import) |
| r4 | same | + **lazy `_ext` import** (Issue #2 fix) | `dec2391b…` | ⚠️ rank 0 works; rank 1 hangs on `all_sum` rendezvous (Issue #3) |

### Deploying a build to the cluster

```bash
scp output/EXO-1.0.72-native-metal-kernels-dev1.dmg gra@m3msu256a.local:/tmp/
scp output/EXO-1.0.72-native-metal-kernels-dev1.dmg gra@m3msu256b.local:/tmp/
# On each machine: open the DMG, QUIT the old EXO fully, launch the new one.
# Re-create the Jundot/GLM-5.2-oQ4 instance.
```

**Critical:** the old build must be **fully quit** before launching the new
one — the bug is in the already-loaded process, and a running old instance
will keep the broken code path. Multiple mounted volumes (`/Volumes/EXO … 1`,
`… 2`) are easy to confuse; check the running exe path with `ps` and the
`CFBundleShortVersionString` in `Info.plist`.

### Build gotchas learned

- `EXO_SKIP_DASHBOARD=1` in `BUILD.sh` **skips `just package` entirely**
  (not just the dashboard rebuild), because `just package` depends on
  `build-dashboard` and BUILD.sh gates the whole recipe. To rebuild only
  PyInstaller quickly, run `uv run pyinstaller packaging/pyinstaller/exo.spec`
  directly (then re-run the Swift `xcodebuild` + DMG steps).
- `BUILD.sh` wipes `dist/exo` + `build/exo` before packaging; a manual
  `just package` run that succeeds is the fastest way to iterate on spec
  changes without a full rebuild.
- The kernel CMake build is incremental; re-running `just mlx-kernels` during
  a rebuild is fast (re-verifies + relinks).

---

## 6. Current state & next steps

### Done & verified
- ✅ Vendored oMLX kernel sources + standalone CMake build path
- ✅ PyInstaller bundling (`_ext.so`, `libomlx_glm_kernel_ops.dylib`, metallib
  ×2)
- ✅ Issue #1 fixed (metallib discoverable at `_internal/` in frozen bundle)
- ✅ Issue #2 fixed (lazy `_ext` import — Metal no longer initialized before
  JACCL; rank 0 warmup now completes)
- ✅ `EXO_NATIVE_GLM_KERNELS=0` emergency-disable env var
- ✅ Test suite (`test_glm_moe_dsa_native_kernels.py`): `9 passed, 2 skipped`

### Pending
- ❌ **Issue #3**: JACCL `all_sum` rendezvous race (`utils_mlx.py`). Blocks
  2-node model load. Proposed retry-until-`world_size` fix above, but needs
  empirical validation on the cluster (does the degenerate 1ms return repeat
  the collective on retry, or cache?).
- ⚠️ The metallib dual-placement is a spec workaround; the "proper" fix is a
  C++ change to make the metallib path injectable from Python (lower priority
  — the spec fix works).

### Lessons
1. **Import-ordering matters for GPU-RDMA backends.** Any extension that links
   Metal/`libmlx.dylib` and is imported before `mx.distributed.init` can steal
   the Metal device's first init. Defer such imports to first use.
2. **`dladdr` is unreliable in PyInstaller bundles** for locating an
   extension's own directory — it resolves to the bootloader/core image.
   Don't trust `current_binary_dir()`-style lookups in frozen apps; duplicate
   resource files to `_internal/` or pass paths from Python.
3. **A 1ms collective completion is always suspicious** — real RDMA
   collectives between two machines take longer. Treat instant returns as a
   rendezvous failure, not success.
4. **The Swift app nulls exo's stdio** — always check `~/.exo/exo_log/` files,
   and use `sample <pid>` when a runner is wedged-but-alive (the logs will be
   silent).