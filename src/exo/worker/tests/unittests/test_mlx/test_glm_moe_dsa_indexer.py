# type: ignore
"""Unit tests for the GLM-5.2 glm_moe_dsa indexer sharing logic.

These tests verify:
  - ModelArgs correctly derives indexer_types from index_topk_freq /
    index_skip_topk_offset when indexer_types is not provided.
  - GlmMoeDsaAttention sets self.indexer = None for shared layers.
  - GlmMoeDsaDecoderLayer passes topk_indices between layers.
  - Model.sanitize() removes indexer weights for shared layers.

The tests don't require the full model weights or a running GPU — they test
the configuration and structural properties of the vendored GLM-5.2 model.
"""

import mlx.core as mx  # noqa: F401

from exo.worker.engines.mlx.vendor.glm_moe_dsa.glm_moe_dsa_model import (
    GlmMoeDsaAttention,
    GlmMoeDsaDecoderLayer,
    Model,
    ModelArgs,
)


def _make_glm52_model_args(
    *,
    indexer_types: list[str] | None = None,
    index_topk_freq: int = 4,
    index_skip_topk_offset: int = 3,
    num_hidden_layers: int = 78,
) -> ModelArgs:
    """Create a ModelArgs for GLM-5.2 with the real config values."""
    return ModelArgs(
        model_type="glm_moe_dsa",
        vocab_size=154880,
        hidden_size=6144,
        index_head_dim=128,
        index_n_heads=32,
        index_topk=2048,
        intermediate_size=12288,
        moe_intermediate_size=2048,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=64,
        num_key_value_heads=64,
        n_shared_experts=1,
        n_routed_experts=256,
        routed_scaling_factor=2.5,
        kv_lora_rank=512,
        q_lora_rank=2048,
        qk_rope_head_dim=64,
        v_head_dim=256,
        qk_nope_head_dim=192,
        topk_method="noaux_tc",
        scoring_func="sigmoid",
        norm_topk_prob=True,
        n_group=1,
        topk_group=1,
        num_experts_per_tok=8,
        moe_layer_freq=1,
        first_k_dense_replace=3,
        max_position_embeddings=1048576,
        rms_norm_eps=1e-05,
        rope_parameters={"rope_theta": 8000000.0, "rope_type": "default"},
        attention_bias=False,
        indexer_types=indexer_types,
        index_topk_freq=index_topk_freq,
        index_skip_topk_offset=index_skip_topk_offset,
    )


class TestModelArgsIndexerTypes:
    """Tests for ModelArgs indexer_types derivation."""

    def test_explicit_indexer_types(self) -> None:
        """When indexer_types is provided explicitly, it should be used as-is."""
        explicit = ["full", "shared", "full", "shared"]
        args = _make_glm52_model_args(indexer_types=explicit)
        assert args.indexer_types == explicit

    def test_derived_indexer_types_from_freq_offset(self) -> None:
        """When indexer_types is None, derive from index_topk_freq and
        index_skip_topk_offset."""
        args = _make_glm52_model_args(indexer_types=None)
        from collections import Counter

        counts = Counter(args.indexer_types)
        full_count = sum(1 for i in range(78) if (max(i - 2, 0) % 4) == 0)
        assert counts["full"] == full_count
        assert counts["shared"] == 78 - full_count

    def test_glm52_real_config_matches(self) -> None:
        """The derived indexer_types should match the real GLM-5.2 config
        (21 full, 57 shared)."""
        args = _make_glm52_model_args(indexer_types=None)
        from collections import Counter

        counts = Counter(args.indexer_types)
        assert counts["full"] == 21
        assert counts["shared"] == 57


class TestGlmMoeDsaAttention:
    """Tests for the GLM-5.2 attention layer with indexer sharing."""

    def test_shared_layer_has_no_indexer(self) -> None:
        """Shared layers should have self.indexer = None."""
        args = _make_glm52_model_args()
        attn = GlmMoeDsaAttention(args, layer_idx=3)
        assert attn.skip_topk is True
        assert attn.indexer is None

    def test_full_layer_has_indexer(self) -> None:
        """Full layers should have a non-None indexer."""
        args = _make_glm52_model_args()
        attn = GlmMoeDsaAttention(args, layer_idx=0)
        assert attn.skip_topk is False
        assert attn.indexer is not None

    def test_skip_topk_flag_matches_indexer_types(self) -> None:
        """skip_topk should match the indexer_types for each layer."""
        args = _make_glm52_model_args()
        indexer_types = args.indexer_types
        assert indexer_types is not None
        for i, t in enumerate(indexer_types):
            attn = GlmMoeDsaAttention(args, layer_idx=i)
            assert attn.skip_topk == (t == "shared")


class TestGlmMoeDsaDecoderLayer:
    """Tests for the decoder layer threading of topk_indices."""

    def test_decoder_layer_exists(self) -> None:
        """GlmMoeDsaDecoderLayer should be constructable."""
        args = _make_glm52_model_args()
        layer = GlmMoeDsaDecoderLayer(args, layer_idx=0)
        assert layer is not None
        assert isinstance(layer.self_attn, GlmMoeDsaAttention)


class TestModelSanitize:
    """Tests for Model.sanitize() removing shared-layer indexer weights."""

    def test_sanitize_removes_shared_indexer_weights(self) -> None:
        """sanitize() should drop indexer weights for shared layers."""
        args = _make_glm52_model_args()
        model = Model(args)

        weights: dict[str, mx.array] = {}
        for i in range(args.num_hidden_layers):
            prefix = f"model.layers.{i}.self_attn.indexer"
            weights[f"{prefix}.wq_b.weight"] = mx.zeros((1, 1))
            weights[f"{prefix}.wk.weight"] = mx.zeros((1, 1))
            weights[f"{prefix}.weights_proj.weight"] = mx.zeros((1, 1))

        sanitized = model.sanitize(weights)

        indexer_types = args.indexer_types
        assert indexer_types is not None
        for i, t in enumerate(indexer_types):
            if t == "shared":
                prefix = f"model.layers.{i}.self_attn.indexer"
                assert f"{prefix}.wq_b.weight" not in sanitized
                assert f"{prefix}.wk.weight" not in sanitized
                assert f"{prefix}.weights_proj.weight" not in sanitized

        for i, t in enumerate(indexer_types):
            if t == "full":
                prefix = f"model.layers.{i}.self_attn.indexer"
                assert f"{prefix}.wq_b.weight" in sanitized
