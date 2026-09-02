# ruff: noqa
# SPDX-License-Identifier: Apache-2.0
"""Native GLM kernel extensions used by the EXO-vendored GLM-5.2 monkey patch.

Vendored from oMLX ``omlx/custom_kernels/glm_moe_dsa/__init__.py``.
"""

from . import fast

__all__ = ["fast"]
