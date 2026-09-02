from exo.worker.engines.mlx.patches.glm5_next import apply_glm5_next_patch
from exo.worker.engines.mlx.patches.glm_moe_dsa import apply_glm_moe_dsa_patch
from exo.worker.engines.mlx.patches.opt_batch_gen import apply_batch_gen_patch
from exo.worker.engines.mlx.patches.standard_yarn_rope import patch_yarn_rope

_applied = False


def apply_mlx_patches() -> None:
    global _applied
    if _applied:
        return
    _applied = True
    patch_yarn_rope()
    apply_batch_gen_patch()
    apply_glm_moe_dsa_patch()
    # GLM-5.3 (glm5_next) VLM model registration. Runs after the GLM-5.2 DSA
    # patch so the shared DSA components (deepseek_v32 / sparse_mla /
    # switch_layers) the GLM-5.3 language model reuses are already importable.
    apply_glm5_next_patch()
