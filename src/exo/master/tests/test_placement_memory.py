# type: ignore
"""Tests for placement memory-feasibility (Phase 3: #1709, #2240, #2241).

Covers:
- ``estimate_kv_bytes_per_layer_per_token`` conservative bound (GQA/MQA/MLA
  over-estimation; exact for MHA; 0 when dims missing).
- ``effective_context_tokens`` override (``EXO_PLACEMENT_CONTEXT_TOKENS``).
- ``estimate_node_memory_requirement`` = weights + KV(context) + margin.
- Integration: ``_allocate_and_validate_layers`` rejects a near-limit config
  once KV(context) is reserved, and the ``EXO_PLACEMENT_CONTEXT_TOKENS``
  override enables near-limit placement (#1709 lever).
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from exo.master import placement_memory as pm
from exo.master.placement_utils import _allocate_and_validate_layers
from exo.shared.models.model_cards import ModelCard, ModelId, ModelTask
from exo.shared.types.backends import Backend
from exo.shared.types.common import NodeId
from exo.shared.types.memory import Memory
from exo.shared.types.profiling import MemoryUsage


def _card(
    *,
    n_layers: int = 10,
    hidden_size: int = 1000,
    num_key_value_heads: int | None = 4,
    num_attention_heads: int | None = 8,
    context_length: int = 0,
    storage_kb: int = 1000,
) -> ModelCard:
    return ModelCard(
        model_id=ModelId("test-model"),
        storage_size=Memory.from_kb(storage_kb),
        n_layers=n_layers,
        hidden_size=hidden_size,
        supports_tensor=True,
        num_key_value_heads=num_key_value_heads,
        num_attention_heads=num_attention_heads,
        context_length=context_length,
        tasks=[ModelTask.TextGeneration],
        backends=[Backend.MlxMetal],
    )


class TestEstimateKvBytesPerLayerPerToken:
    def test_exact_for_mha_when_attn_heads_present(self):
        # head_dim = hidden / num_attention_heads = 1000/8 = 125
        # kv = 2 * n_kv * head_dim * precision = 2 * 4 * 125 * 2 = 2000
        c = _card(num_key_value_heads=4, num_attention_heads=8)
        assert pm.estimate_kv_bytes_per_layer_per_token(c) == 2 * 4 * 125 * 2

    def test_conservative_when_attn_heads_absent(self):
        # Falls back to head_dim = hidden / num_kv_heads = 1000/4 = 250
        # (over-estimate vs true 125). kv = 2 * 4 * 250 * 2 = 4000
        c = _card(num_key_value_heads=4, num_attention_heads=None)
        assert pm.estimate_kv_bytes_per_layer_per_token(c) == 2 * 4 * 250 * 2

    def test_mqa_over_estimates_safely(self):
        # MQA: num_kv_heads=1. Without attn heads, head_dim = hidden/1 = 1000
        # (massive over-estimate; safe direction).
        c = _card(num_key_value_heads=1, num_attention_heads=None)
        assert pm.estimate_kv_bytes_per_layer_per_token(c) == 2 * 1 * 1000 * 2

    def test_zero_when_kv_heads_missing(self):
        c = _card(num_key_value_heads=None, num_attention_heads=None)
        assert pm.estimate_kv_bytes_per_layer_per_token(c) == 0

    def test_precision_from_env(self):
        c = _card(num_key_value_heads=4, num_attention_heads=8)
        with patch.object(pm, "_PLACEMENT_KV_PRECISION_BYTES", 1):  # 8-bit KV
            assert pm.estimate_kv_bytes_per_layer_per_token(c) == 2 * 4 * 125 * 1


class TestEffectiveContextTokens:
    def test_uses_model_context_when_env_unset(self):
        c = _card(context_length=32768)
        assert pm.effective_context_tokens(c) == 32768

    def test_env_overrides_model_context(self):
        c = _card(context_length=1_048_576)
        with patch.object(pm, "_PLACEMENT_CONTEXT_TOKENS", 8192):
            assert pm.effective_context_tokens(c) == 8192

    def test_zero_when_neither_set(self):
        c = _card(context_length=0)
        assert pm.effective_context_tokens(c) == 0


class TestEstimateNodeMemoryRequirement:
    def test_weights_only_when_no_context(self):
        # No context => KV skipped, margin=0 => pure weights.
        c = _card(n_layers=10, storage_kb=1000)  # 100 KB/layer
        req = pm.estimate_node_memory_requirement(c, node_layers=3)
        assert req.in_bytes == (1024 * 1000) * 3 // 10  # 307.2 KB

    def test_weights_plus_kv_when_context_set(self):
        c = _card(
            n_layers=10,
            storage_kb=1000,
            num_key_value_heads=4,
            num_attention_heads=8,
            context_length=1024,
        )
        # weights = 100 KB/layer * 3 = 307.2 KB
        # kv_per_layer_per_token = 2*4*125*2 = 2000 B
        # kv = 2000 * 3 layers * 1024 tokens = 6_144_000 B
        weights = (1024 * 1000) * 3 // 10
        kv = 2 * 4 * 125 * 2 * 3 * 1024
        req = pm.estimate_node_memory_requirement(c, node_layers=3)
        assert req.in_bytes == weights + kv  # margin defaults to 0

    def test_margin_added_when_enabled(self):
        c = _card(
            n_layers=10,
            storage_kb=1000,
            num_key_value_heads=4,
            num_attention_heads=8,
            context_length=1024,
        )
        weights = (1024 * 1000) * 3 // 10
        kv = 2 * 4 * 125 * 2 * 3 * 1024
        with patch.object(pm, "_PLACEMENT_ACTIVATION_MARGIN", 0.10):
            req = pm.estimate_node_memory_requirement(c, node_layers=3)
        margin = int((weights + kv) * 0.10)
        assert req.in_bytes == weights + kv + margin


class TestAllocateAndValidateLayers:
    """Integration: the per-node gate now reserves KV(context)."""

    def _node_memory(self, available_bytes: int) -> MemoryUsage:
        return MemoryUsage.from_bytes(
            ram_total=available_bytes,
            ram_available=available_bytes,
            swap_total=0,
            swap_available=0,
        )

    def test_weights_only_passes_exact_fit(self):
        # context_length=0 => weights-only check; exact fit still passes.
        c = _card(n_layers=10, storage_kb=1000)  # 100 KB/layer
        node_id = NodeId()
        node_memory = {node_id: self._node_memory(1024 * 100)}  # exactly 1 layer
        # Single node gets all 10 layers = 1 MB; needs 1 MB. Exact fit.
        node_memory = {node_id: self._node_memory(1024 * 1000)}
        total = Memory.from_bytes(1024 * 1000)
        alloc = _allocate_and_validate_layers([node_id], node_memory, total, c)
        assert alloc == [10]

    def test_rejects_when_kv_context_exceeds_memory(self):
        # With a large context, KV reservation pushes the node over.
        c = _card(
            n_layers=10,
            storage_kb=1000,  # 100 KB/layer weights
            num_key_value_heads=4,
            num_attention_heads=8,
            context_length=1_000_000,  # 1M tokens
        )
        node_id = NodeId()
        # Give just enough for weights (1 MB) but not for 1M-context KV.
        node_memory = {node_id: self._node_memory(2 * 1024 * 1024)}
        total = Memory.from_bytes(2 * 1024 * 1024)
        with pytest.raises(ValueError, match="insufficient memory"):
            _allocate_and_validate_layers([node_id], node_memory, total, c)

    def test_context_override_enables_near_limit_placement(self):
        # #1709 lever: EXO_PLACEMENT_CONTEXT_TOKENS below the model's 1M
        # context reserves less KV, so the same node now fits.
        c = _card(
            n_layers=10,
            storage_kb=1000,
            num_key_value_heads=4,
            num_attention_heads=8,
            context_length=1_000_000,
        )
        node_id = NodeId()
        # Enough for weights + KV(8K) but not KV(1M).
        weights = 1024 * 1000  # 1 MB
        kv_8k = 2 * 4 * 125 * 2 * 10 * 8192  # ~81 MB
        node_memory = {node_id: self._node_memory(weights + kv_8k)}
        total = Memory.from_bytes(weights + kv_8k)
        with patch.object(pm, "_PLACEMENT_CONTEXT_TOKENS", 8192):
            alloc = _allocate_and_validate_layers([node_id], node_memory, total, c)
        assert alloc == [10]

    def test_error_message_mentions_context_lever(self):
        c = _card(
            n_layers=10,
            storage_kb=1000,
            num_key_value_heads=4,
            num_attention_heads=8,
            context_length=1_000_000,
        )
        node_id = NodeId()
        node_memory = {node_id: self._node_memory(1024 * 1000)}
        total = Memory.from_bytes(1024 * 1000)
        with pytest.raises(ValueError, match="EXO_PLACEMENT_CONTEXT_TOKENS"):
            _allocate_and_validate_layers([node_id], node_memory, total, c)
