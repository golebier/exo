"""Unit tests for EOS token detection for GLM-5.2 and related models.

Verifies that get_eos_token_ids_for_model returns the correct EOS token IDs
for each GLM model family.
"""

from exo.shared.models.model_cards import ModelId
from exo.worker.engines.mlx.utils_mlx import get_eos_token_ids_for_model


class TestGlm52EosTokens:
    """Tests for GLM-5.2 EOS token detection."""

    def test_glm52_alis_dynamic(self) -> None:
        eos = get_eos_token_ids_for_model(
            ModelId("avlp12/GLM-5.2-Alis-MLX-Dynamic-2.3bpw")
        )
        assert eos == [154820, 154827, 154829]

    def test_glm52_mxfp4(self) -> None:
        eos = get_eos_token_ids_for_model(ModelId("mlx-community/GLM-5.2-mxfp4"))
        assert eos == [154820, 154827, 154829]

    def test_glm52_dq4plus(self) -> None:
        eos = get_eos_token_ids_for_model(ModelId("mlx-community/GLM-5.2-DQ4plus-q8"))
        assert eos == [154820, 154827, 154829]


class TestGlm51EosTokens:
    """Tests for GLM-5.1 EOS token detection (same as GLM-5.2)."""

    def test_glm51_bf16(self) -> None:
        eos = get_eos_token_ids_for_model(ModelId("mlx-community/GLM-5.1"))
        assert eos == [154820, 154827, 154829]

    def test_glm51_mxfp4(self) -> None:
        eos = get_eos_token_ids_for_model(ModelId("mlx-community/GLM-5.1-MXFP4-Q8"))
        assert eos == [154820, 154827, 154829]


class TestGlm47EosTokens:
    """Tests for GLM-4.7 EOS token detection (different from GLM-5.2)."""

    def test_glm47_4bit(self) -> None:
        eos = get_eos_token_ids_for_model(ModelId("mlx-community/GLM-4.7-4bit"))
        assert eos == [151336, 151329, 151338]

    def test_glm47_flash(self) -> None:
        eos = get_eos_token_ids_for_model(ModelId("mlx-community/GLM-4.7-Flash-4bit"))
        assert eos == [151336, 151329, 151338]


class TestGlm45EosTokens:
    """Tests for GLM-4.5 EOS token detection."""

    def test_glm45_air(self) -> None:
        eos = get_eos_token_ids_for_model(ModelId("mlx-community/GLM-4.5-Air-8bit"))
        assert eos == [151336, 151329, 151338]


class TestGlm5MatchesGlm52:
    """Verify that 'glm-5' matches both GLM-5.2 and GLM-5.1 but not GLM-4.7."""

    def test_glm5_in_glm52(self) -> None:
        assert "glm-5" in "avlp12/glm-5.2-alis-mlx-dynamic-2.3bpw".lower()

    def test_glm5_in_glm51(self) -> None:
        assert "glm-5" in "mlx-community/glm-5.1".lower()

    def test_glm5_not_in_glm47(self) -> None:
        assert "glm-5" not in "mlx-community/glm-4.7-4bit".lower()

    def test_glm_in_glm47(self) -> None:
        assert "glm" in "mlx-community/glm-4.7-4bit".lower()
