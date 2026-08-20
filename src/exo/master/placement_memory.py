"""Memory-feasibility for model placement: weights + KV(context) + activation margin.

Phase 3 of the memory-headroom work (issues #1709, #2240, #2241). The placement
validator previously only accounted for weights proportional to layers; a
long-context model placed "near the limit" would then OOM at prefill because no
room was reserved for the KV cache or prefill activations.

This module adds, to the per-node requirement:

1. **KV(context)** — the cache footprint reserved for the *effective* context
   window. ``num_attention_heads`` (when present on the card) gives the exact
   ``head_dim``; otherwise ``head_dim`` is bounded above by
   ``hidden_size / num_key_value_heads`` (an over-estimate for GQA/MQA and
   MLA — every one a *safe* direction for a guard).
2. **Activation margin** — a small, tunable fraction of (weights + KV) reserved
   for the prefill transient. The precise per-chunk check lives in the runtime
   guards (Phase 1/2); placement only needs to reject obviously-too-tight
   configs.

The effective context is configurable via ``EXO_PLACEMENT_CONTEXT_TOKENS`` so a
1M-context model can be placed on a 16 GB node by reserving only, say, 32 K of
KV — this is the lever that makes near-limit placement possible (#1709).
"""

from __future__ import annotations

import os

from exo.shared.models.model_cards import ModelCard
from exo.shared.types.memory import Memory

# KV element precision used for the placement reservation. Defaults to fp16
# (2 bytes), matching mlx-lm's default KV dtype. Override with
# ``EXO_PLACEMENT_KV_BITS`` (e.g. 4 for 4-bit KV caches).
_PLACEMENT_KV_BITS = int(os.environ.get("EXO_PLACEMENT_KV_BITS", "16"))
_PLACEMENT_KV_PRECISION_BYTES = max(1, _PLACEMENT_KV_BITS // 8)

# Activation margin: fraction of (weights + KV) reserved for the prefill
# transient. Defaults to **0** so placement does not reject configs that
# currently fit exactly — the runtime preflight/chunk guards (Phases 1–2)
# are the precise gate. Operators who want a placement-time buffer can set
# ``EXO_PLACEMENT_ACTIVATION_MARGIN`` (e.g. 0.05 for 5%).
_PLACEMENT_ACTIVATION_MARGIN = float(
    os.environ.get("EXO_PLACEMENT_ACTIVATION_MARGIN", "0.0")
)

# Effective context (tokens) reserved for KV at placement time. ``0`` (default)
# means use the model card's ``context_length``; a positive value overrides it
# so a high-context model can be placed on constrained memory by reserving a
# smaller window. This is the #2240/#2241 lever for near-limit placement.
_PLACEMENT_CONTEXT_TOKENS = int(os.environ.get("EXO_PLACEMENT_CONTEXT_TOKENS", "0"))


def effective_context_tokens(model_card: ModelCard) -> int:
    """Context reserved for KV at placement time.

    ``EXO_PLACEMENT_CONTEXT_TOKENS`` overrides the model's max context so a
    1M-context model doesn't reserve 1M of KV on a 16 GB node when the user
    only needs 32 K. Returns 0 when neither is set (disables KV reservation —
    preserves the legacy weights-only check).
    """
    if _PLACEMENT_CONTEXT_TOKENS > 0:
        return _PLACEMENT_CONTEXT_TOKENS
    return model_card.context_length or 0


def estimate_head_dim(model_card: ModelCard) -> int:
    """Per-attention-head dimension, conservative when unknown.

    With ``num_attention_heads`` on the card, ``head_dim = hidden_size /
    num_attention_heads`` (exact). Without it, bound above by ``hidden_size /
    num_key_value_heads`` — an over-estimate for GQA/MQA (fewer KV heads ⇒
    larger quotient ⇒ larger KV reservation) and MLA (compressed KV). Every
    direction is safe for a guard. Returns 0 when dims are missing.
    """
    hidden = model_card.hidden_size
    n_attn = model_card.num_attention_heads
    if n_attn is not None and n_attn > 0:
        return hidden // n_attn
    n_kv = model_card.num_key_value_heads
    if n_kv is not None and n_kv > 0:
        return hidden // n_kv
    return 0


def estimate_kv_bytes_per_layer_per_token(model_card: ModelCard) -> int:
    """KV bytes for one layer, one token (both K and V, all KV heads on that layer).

    ``2 (K+V) * num_kv_heads * head_dim * precision``. Returns 0 when KV-head
    or head-dim info is absent (the reservation is skipped — never blocks).
    """
    n_kv = model_card.num_key_value_heads
    if n_kv is None or n_kv == 0:
        return 0
    head_dim = estimate_head_dim(model_card)
    if head_dim <= 0:
        return 0
    return 2 * n_kv * head_dim * _PLACEMENT_KV_PRECISION_BYTES


def estimate_kv_bytes(
    model_card: ModelCard, *, node_layers: int, context_tokens: int
) -> Memory:
    """KV cache footprint reserved on one node holding ``node_layers`` layers.

    For pipeline parallel each node owns ``node_layers`` contiguous layers and
    their KV. Returns ``Memory(0)`` when context or dims are unset.
    """
    if context_tokens <= 0:
        return Memory.from_bytes(0)
    per_layer_per_token = estimate_kv_bytes_per_layer_per_token(model_card)
    if per_layer_per_token <= 0:
        return Memory.from_bytes(0)
    return Memory.from_bytes(per_layer_per_token * node_layers * context_tokens)


def estimate_activation_margin_bytes(
    model_card: ModelCard, *, weights: Memory, kv: Memory
) -> Memory:
    """Prefill activation headroom reserved at placement time.

    A fraction of (weights + KV); the runtime guards do the precise check.
    ``EXO_PLACEMENT_ACTIVATION_MARGIN=0`` disables it.
    """
    if _PLACEMENT_ACTIVATION_MARGIN <= 0:
        return Memory.from_bytes(0)
    base = Memory.from_bytes(weights.in_bytes + kv.in_bytes)
    return base * _PLACEMENT_ACTIVATION_MARGIN


def estimate_node_memory_requirement(
    model_card: ModelCard, *, node_layers: int
) -> Memory:
    """Total memory a node must have free to host ``node_layers`` layers.

    ``weights(node_layers) + KV(effective_context) + activation_margin``. When
    the model card has no context info this degrades to the legacy
    weights-proportional check.
    """
    total_layers = model_card.n_layers
    weights = (model_card.storage_size * node_layers) // total_layers
    context_tokens = effective_context_tokens(model_card)
    kv = estimate_kv_bytes(
        model_card, node_layers=node_layers, context_tokens=context_tokens
    )
    activation = estimate_activation_margin_bytes(model_card, weights=weights, kv=kv)
    return Memory.from_bytes(weights.in_bytes + kv.in_bytes + activation.in_bytes)
