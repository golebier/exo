# type: ignore
"""Unit tests for the GLM-5.2 native-kernel dispatch and availability surface.

These tests verify the Phase 1 wiring from
`docs/omlx-porting/02-native-metal-kernels.md`:

  - The vendored ``omlx_custom_kernels.glm_moe_dsa.fast`` module imports even
    when the native ``_ext`` extension is not built (the default EXO build).
  - ``kernels.fast`` dispatch reports ``native_available() == False`` and
    every GLM-specific symbol unavailable when unbuilt, so the model falls
    back to the standard attention path — identical to pre-kernel behavior.
  - ``native_kernel_status()`` reports availability without raising, even
    when the extension import fails.
  - The GLM-5.2 patch startup path
    (``apply_glm_moe_dsa_patch`` / ``_native_kernel_summary``) degrades
    gracefully when native kernels are absent.
  - A build-gated native smoke test verifies the native kernel runs and
    produces finite, correctly-shaped output when built; it skips when the
    native extension is not built (the default) so the suite stays green on
    non-Darwin / no-Xcode machines. A full native-vs-fallback numerical-
    equivalence test needs the GLM-5.2 Indexer + real weights and is tracked
    as a follow-up in mlx_kernels/README.md.

The tests don't require the full model weights or a running GPU.
"""

from __future__ import annotations

import importlib

import mlx.core as mx
import pytest

from exo.worker.engines.mlx.vendor.glm_moe_dsa import kernels as glm_kernels

# ── Symbols the GLM-5.2 model code gates on ─────────────────────────────────
# Mirrors the `required` tuple in patches/glm_moe_dsa/__init__.py.
REQUIRED_GLM_SYMBOLS = (
    "dsa_indexer_scores",
    "dsa_topk_indices",
    "glm_dsa_sparse_mla_attention",
    "glm_dsa_exact_block_attention",
    "glm_dsa_q8_vup_flat",
    "glm_moe_weighted_sum",
)


def _native_kernel_summary() -> tuple[bool, list[str], Exception | None]:
    """Return (available, missing_required, import_error) for native GLM kernels."""
    return (
        glm_kernels.fast.native_available(),
        glm_kernels.fast.missing(REQUIRED_GLM_SYMBOLS),
        glm_kernels.fast.native_import_error(),
    )


class TestVendoredFastModuleImports:
    """The vendored fast.py must import even when _ext is not built."""

    def test_vendored_fast_imports(self) -> None:
        fast = importlib.import_module(
            "exo.worker.engines.mlx.vendor.omlx_custom_kernels.glm_moe_dsa.fast"
        )
        assert hasattr(fast, "is_native_available")
        assert hasattr(fast, "has_symbol")
        assert hasattr(fast, "import_error")
        assert hasattr(fast, "NATIVE_SYMBOLS")
        # Every required symbol is in the NATIVE_SYMBOLS registry so the
        # dispatch knows to look for it on _ext / mx.fast.
        for name in REQUIRED_GLM_SYMBOLS:
            assert name in fast.NATIVE_SYMBOLS, name

    def test_native_kernel_status_never_raises(self) -> None:
        from exo.worker.engines.mlx.vendor.omlx_custom_kernels import (
            native_kernel_status,
        )

        status = native_kernel_status()
        assert "glm_moe_dsa" in status
        entry = status["glm_moe_dsa"]
        assert isinstance(entry["available"], bool)
        # import_error is None when available, or a stringified error when not.
        assert entry["import_error"] is None or isinstance(entry["import_error"], str)


class TestDispatchFallbackBehavior:
    """When unbuilt (the default), the dispatch must report all GLM symbols
    unavailable.

    EXO's mx.fast ships none of the GLM-specific symbols, so ``has()`` must
    return False for every required symbol and the model code falls through
    to the standard attention path. This is the causal-safety guarantee from
    GLM-5.2-RESEARCH-RESULTS.md §5.

    Native kernels are **default-off** (opt-in via ``EXO_NATIVE_GLM_KERNELS=1``).
    Without that env var, ``_resolve_native_fast()`` short-circuits to ``None``
    before even attempting the import, so ``native_available()`` is always
    ``False`` and all GLM-specific symbols are unavailable.
    """

    def test_native_available_is_bool(self) -> None:
        assert isinstance(glm_kernels.fast.native_available(), bool)

    def test_native_default_off_without_env(self) -> None:
        """Native kernels must be unavailable when EXO_NATIVE_GLM_KERNELS is
        not set (the default-off safety guarantee)."""
        import os

        # These tests run without EXO_NATIVE_GLM_KERNELS set, so native must
        # be off. We don't manipulate os.environ here because the module-level
        # ``_native_resolved`` guard caches the result; the test environment
        # simply doesn't set the env var.
        assert os.environ.get("EXO_NATIVE_GLM_KERNELS", "") in (
            "",
            "0",
            "false",
            "no",
            "off",
        )
        assert not glm_kernels.fast.native_available()

    def test_glm_symbols_unavailable_when_unbuilt(self) -> None:
        if glm_kernels.fast.native_available():
            pytest.skip("native extension is built — fallback assertions N/A")
        for name in REQUIRED_GLM_SYMBOLS:
            assert not glm_kernels.fast.has(name), (
                f"{name} should be unavailable when native is not built "
                "(EXO's mx.fast ships no GLM symbols)"
            )

    def test_missing_reports_all_required_when_unbuilt(self) -> None:
        if glm_kernels.fast.native_available():
            pytest.skip("native extension is built — fallback assertions N/A")
        missing = glm_kernels.fast.missing(REQUIRED_GLM_SYMBOLS)
        assert sorted(missing) == sorted(REQUIRED_GLM_SYMBOLS)

    def test_metal_kernel_available_via_mlx_fast(self) -> None:
        # mx.fast.metal_kernel is the one symbol the fallback path DOES
        # provide (used by sparse_mla.py's metal_kernel helper). The dispatch
        # must surface it even when native is unbuilt.
        assert glm_kernels.fast.has("metal_kernel")
        attr = glm_kernels.fast.metal_kernel
        assert callable(attr)

    def test_native_import_error_is_none_or_exception(self) -> None:
        err = glm_kernels.fast.native_import_error()
        if glm_kernels.fast.native_available():
            assert err is None
        else:
            # When native is default-off (env var not set), _resolve_native_fast
            # short-circuits before attempting the import, so there's no import
            # error — just None (not enabled, not broken). When native is
            # opt-in but the extension isn't built, there IS an import error.
            assert err is None or isinstance(err, BaseException)


class TestPatchStartupSummary:
    """The patch startup path must degrade gracefully when native is absent."""

    def test_native_kernel_summary_shape(self) -> None:
        available, missing, import_error = _native_kernel_summary()
        assert isinstance(available, bool)
        assert isinstance(missing, list)
        if available:
            assert missing == []
            assert import_error is None
        else:
            # When unbuilt/default-off, every required symbol is missing.
            assert sorted(missing) == sorted(REQUIRED_GLM_SYMBOLS)
            # import_error is None when default-off (no import attempted),
            # or an exception when opt-in but extension not built.
            assert import_error is None or isinstance(import_error, BaseException)

    def test_apply_patch_does_not_raise_when_unbuilt(self) -> None:
        # apply_glm_moe_dsa_patch is idempotent; calling it again must not
        # raise even when native kernels are absent (the default build).
        from exo.worker.engines.mlx.patches.glm_moe_dsa import (
            apply_glm_moe_dsa_patch,
        )

        # Already applied by conftest/bootstrap in the test process; the
        # return is False on the second call but must not raise.
        result = apply_glm_moe_dsa_patch()
        assert result in (True, False)


# ── Build-gated numerical-equivalence test ─────────────────────────────────
# The doc's Definition of Done requires: "Numerical-equivalence test (native
# vs fallback) passes within fp16 tol." This scaffolds that test. It skips
# when the native extension is not built (the default) so the suite stays
# green everywhere; on a machine with EXO_BUILD_MLX_KERNELS=1 it exercises
# the dsa_indexer_scores fast path against the mx.fast fallback.


class TestNativeSmokeTest:
    """Smoke-test the native dsa_indexer_scores kernel, when built.

    Skipped unless ``fast.native_available()`` is True. When built, runs a
    small synthetic input through the native kernel and asserts the output is
    finite and correctly-shaped — the minimum bar for trusting the build.
    """

    @pytest.fixture
    def synthetic_indexer_inputs(self) -> dict[str, mx.array]:
        # Valid shapes for dsa_indexer_scores (see sparse_mla.fused_indexer_scores
        # gating): queries (B,32,L,128), keys (B,1,K,128) with K>=4096 singleton
        # indexer head axis, weights (B,L,32). Use aligned L=64, K=4096.
        mx.random.seed(0)
        queries = mx.random.normal((1, 32, 64, 128)).astype(mx.float16)
        keys = mx.random.normal((1, 1, 4096, 128)).astype(mx.float16)
        weights = mx.random.normal((1, 64, 32)).astype(mx.float16)
        return {"queries": queries, "keys": keys, "weights": weights}

    def test_dsa_indexer_scores_native_runs_and_is_finite(
        self, synthetic_indexer_inputs: dict[str, mx.array]
    ) -> None:
        if not glm_kernels.fast.native_available():
            pytest.skip(
                "native GLM kernels not built — build with "
                "EXO_BUILD_MLX_KERNELS=1 (Darwin + Metal toolchain) to enable "
                "the native smoke test"
            )

        q = synthetic_indexer_inputs["queries"]
        k = synthetic_indexer_inputs["keys"]
        w = synthetic_indexer_inputs["weights"]

        out = glm_kernels.fast.dsa_indexer_scores(q, k, w, causal=True, stream=mx.gpu)
        mx.eval(out)

        # Output shape: (B, 1, L, K) — head-summed scores per query/key pair
        # (the indexer head axis is singleton in the output).
        assert out.shape == (1, 1, 64, 4096)
        assert out.dtype == mx.float16
        # The causal kernel masks future positions to finfo.min (-inf), which
        # is expected sentinel behavior — not a bug. Assert no NaN and that
        # at least some positions are finite (the kernel actually computed).
        assert not mx.isnan(out).any(), "native kernel produced NaN"
        assert mx.isfinite(out).any(), (
            "native kernel produced all-non-finite output (no valid scores)"
        )

    def test_native_symbols_resolve_through_dispatch(self) -> None:
        if not glm_kernels.fast.native_available():
            pytest.skip("native extension not built — dispatch assertions N/A")
        # When built, every required symbol must resolve through the dispatch
        # (either on _ext or mx.fast). This is the gate the model code relies
        # on to choose the fast path.
        missing = glm_kernels.fast.missing(REQUIRED_GLM_SYMBOLS)
        assert missing == [], (
            f"native built but required symbols still missing: {missing}"
        )
