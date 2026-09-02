# SPDX-License-Identifier: Apache-2.0
"""Vendored DeepSeek-V4 runtime support (GLM-5.3 dependency).

Vendored from oMLX ``omlx/patches/deepseek_v4/``:
- ``hyper_connection.py`` — HyperConnection / hc_expand used by GLM-5.3 layers.
- ``cache_extras.py`` — PoolingCache / BatchPoolingCache (mlx-lm PR 1192),
  injected into ``mlx_lm.models.cache`` by the GLM-5.3 registration patch.
- ``decode_consistency.py`` — M=1 verification helper; no-op in EXO (the
  DSpark verification harness is not ported).

These are dependencies of the GLM-5.3 (``glm5_next``) model, not a standalone
DeepSeek-V4 model port.
"""

from __future__ import annotations

__all__: list[str] = []
