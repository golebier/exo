"""EXO joint-block wrapper for the pure-MLX Qwen-Image-2.1 block.

Implements the six BlockWrapper abstracts against our own transformer modules.
RoPE layout: adapter passes a single (cos, sin) table over [text | image]
positions (RotaryEmbeddings tuple), text = table[:text_len],
image = table[text_len + patch_start : text_len + patch_end].
"""

from typing import final

import mlx.core as mx

from exo.worker.engines.image.models.base import RotaryEmbeddings
from exo.worker.engines.image.models.qwen21.transformer import (
    HEAD_DIM,
    HIDDEN_DIM,
    NUM_HEADS,
    Qwen21TransformerBlock,
    apply_rope_21,
)
from exo.worker.engines.image.pipeline.block_wrapper import (
    BlockWrapperMode,
    JointBlockWrapper,
)

NEG_INF = -1e9


@final
class Qwen21JointBlockWrapper(JointBlockWrapper[Qwen21TransformerBlock]):
    def __init__(
        self,
        block: Qwen21TransformerBlock,
        text_seq_len: int,
        encoder_hidden_states_mask: mx.array | None = None,
    ):
        super().__init__(block, text_seq_len)
        self._encoder_hidden_states_mask = encoder_hidden_states_mask
        self._num_heads = NUM_HEADS
        self._head_dim = HEAD_DIM

    def set_encoder_mask(self, mask: mx.array | None) -> None:
        self._encoder_hidden_states_mask = mask

    def _text_additive_mask(self, joint_seq_len: int) -> mx.array | None:
        mask = self._encoder_hidden_states_mask
        if mask is None:
            return None
        if mask.ndim > 1:
            mask = mask.reshape(mask.shape[0], -1)
        text_valid = mask.astype(mx.float32)
        img_seq_len = joint_seq_len - self._text_seq_len
        ones_img = mx.ones((text_valid.shape[0], img_seq_len), dtype=mx.float32)
        joint_mask = mx.concatenate([text_valid, ones_img], axis=1)
        if mx.all(joint_mask >= 0.999):
            return None
        additive = (1.0 - joint_mask) * NEG_INF
        return additive.reshape((additive.shape[0], 1, 1, additive.shape[1]))

    def _text_causal_mask(self, batch: int) -> mx.array:
        n = self._text_seq_len
        base = mx.triu(mx.full((n, n), NEG_INF, dtype=mx.float32), k=1)
        return mx.broadcast_to(base[None, None, :, :], (batch, 1, n, n))

    def _rope(
        self,
        query: mx.array,
        key: mx.array,
        cos: mx.array,
        sin: mx.array,
        patch_mode: bool,
    ) -> tuple[mx.array, mx.array]:
        if patch_mode:
            cos = cos[self._text_seq_len + self._patch_start : self._text_seq_len + self._patch_end]
            sin = sin[self._text_seq_len + self._patch_start : self._text_seq_len + self._patch_end]
        else:
            cos = cos[self._text_seq_len :]
            sin = sin[self._text_seq_len :]
        return apply_rope_21(query, cos, sin), apply_rope_21(key, cos, sin)

    def _compute_qkv(
        self,
        hidden_states: mx.array,
        encoder_hidden_states: mx.array,
        text_embeddings: mx.array,
        rotary_embeddings: RotaryEmbeddings,
        patch_mode: bool = False,
    ) -> tuple[mx.array, mx.array, mx.array]:
        assert isinstance(rotary_embeddings, tuple)
        cos, sin = rotary_embeddings
        block = self.block

        # Per-token shared modulation: text_embeddings is [batch, 2, 4*hidden].
        img_mod = mx.broadcast_to(
            text_embeddings[:, 0:1, :],
            (text_embeddings.shape[0], hidden_states.shape[1], text_embeddings.shape[2]),
        )
        txt_mod = mx.broadcast_to(
            text_embeddings[:, 1:2, :],
            (
                text_embeddings.shape[0],
                encoder_hidden_states.shape[1],
                text_embeddings.shape[2],
            ),
        )
        img_scale1, img_gate1, img_scale2, img_gate2 = Qwen21TransformerBlock.split_mod(img_mod)
        txt_scale1, txt_gate1, txt_scale2, txt_gate2 = Qwen21TransformerBlock.split_mod(txt_mod)

        self._img_mod = (img_scale2, img_gate2)
        self._txt_mod = (txt_scale2, txt_gate2)

        img_normed = Qwen21TransformerBlock.modulate(block.norm1(hidden_states), img_scale1)
        txt_normed = Qwen21TransformerBlock.modulate(block.norm1(encoder_hidden_states), txt_scale1)

        img_q, img_k, img_v = block.attn.project(img_normed)
        txt_q, txt_k, txt_v = block.attn.project(txt_normed)

        img_q, img_k = self._rope(img_q, img_k, cos, sin, patch_mode)
        txt_q, txt_k = self._text_rope(txt_q, txt_k, cos, sin)

        query = mx.concatenate([txt_q, img_q], axis=2)
        key = mx.concatenate([txt_k, img_k], axis=2)
        value = mx.concatenate([txt_v, img_v], axis=2)
        return query, key, value

    def _text_rope(
        self, query: mx.array, key: mx.array, cos: mx.array, sin: mx.array
    ) -> tuple[mx.array, mx.array]:
        cos_t = cos[: self._text_seq_len]
        sin_t = sin[: self._text_seq_len]
        return apply_rope_21(query, cos_t, sin_t), apply_rope_21(key, cos_t, sin_t)

    def _compute_attention(
        self, query: mx.array, key: mx.array, value: mx.array
    ) -> mx.array:
        full_seq = key.shape[2]
        mask = self._text_additive_mask(full_seq)
        if mask is None and self._text_seq_len > 0:
            # segmented path: causal over text region, full attention for image
            batch = query.shape[0]
            txt_q = query[:, :, : self._text_seq_len, :]
            txt_k = key[:, :, : self._text_seq_len, :]
            txt_v = value[:, :, : self._text_seq_len, :]
            img_q = query[:, :, self._text_seq_len :, :]
            img_k = key[:, :, self._text_seq_len :, :]
            img_v = value[:, :, self._text_seq_len :, :]
            txt_out = (
                mx.fast.scaled_dot_product_attention(
                    txt_q, txt_k, txt_v, scale=self._head_dim**-0.5,
                    mask=self._text_causal_mask(batch),
                )
                if self._text_seq_len > 0
                else txt_q
            )
            img_out = mx.fast.scaled_dot_product_attention(
                img_q, img_k, img_v, scale=self._head_dim**-0.5
            )
            return mx.concatenate([txt_out, img_out], axis=2)

        return mx.fast.scaled_dot_product_attention(
            query, key, value, scale=self._head_dim**-0.5, mask=mask
        )

    def _apply_output(
        self,
        attn_out: mx.array,
        hidden_states: mx.array,
        encoder_hidden_states: mx.array,
        text_embeddings: mx.array,  # noqa: ARG002 (kept for interface parity)
    ) -> tuple[mx.array, mx.array]:
        block = self.block
        txt_attn_out = attn_out[:, :, : self._text_seq_len, :]
        img_attn_out = attn_out[:, :, self._text_seq_len :, :]

        txt_out = block.attn.unproject(txt_attn_out)
        img_out = block.attn.unproject(img_attn_out)

        img_scale2, img_gate2 = self._img_mod
        txt_scale2, txt_gate2 = self._txt_mod
        assert img_scale2 is not None and txt_scale2 is not None

        hidden_states = hidden_states + mx.tanh(img_gate2) * img_out
        encoder_hidden_states = encoder_hidden_states + mx.tanh(txt_gate2) * txt_out

        img_normed2 = Qwen21TransformerBlock.modulate(
            block.norm2(hidden_states), img_scale2
        )
        txt_normed2 = Qwen21TransformerBlock.modulate(
            block.norm2(encoder_hidden_states), txt_scale2
        )
        hidden_states = hidden_states + mx.tanh(img_gate2) * block.ff(img_normed2)
        encoder_hidden_states = encoder_hidden_states + txt_gate2 * block.ff(txt_normed2)

        return encoder_hidden_states, hidden_states