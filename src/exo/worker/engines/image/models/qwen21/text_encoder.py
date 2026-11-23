"""Pure-MLX Qwen3-VL-8B text encoder + Qwen-Image-2.1 prompt encoding (no mflux).

The checkpoint's text encoder is the Qwen3-VL 8B language tower (36 layers,
hidden 4096, GQA 32/8 heads, mrope with interleaved sections [24, 20, 20]).
We reuse the text-only path: position ids are a 2D ladder broadcast to three
identical axes, so the interleaved mrope selection collapses to the ladder.
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx import nn

from exo.worker.engines.image.models.qwen21.transformer import HIDDEN_DIM

NUM_LAYERS = 36
NUM_HEADS = 32
NUM_KV_HEADS = 8
HEAD_DIM = 128
MLP_HIDDEN = 12288
ROPE_THETA = 5000000.0
MROPE_SECTION = (24, 20, 20)
VOCAB_SIZE = 151936
MAX_LENGTH = 2048

SYSTEM_PREFIX = (
    "<|im_start|>system\nComprehend and analyze the provided prompt.<|im_end|>\n"
)

# mflux 0.24 PROMPT_TEMPLATE_T2I verbatim (raw form, not apply_chat_template).
PROMPT_TEMPLATE_T2I = (
    "<|im_start|>system\nComprehend and analyze the provided prompt.<|im_end|>\n"
    "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
)


class Qwen3VLRMSNorm(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.weight = mx.ones((dim,))

    def __call__(self, x: mx.array) -> mx.array:
        input_dtype = x.dtype
        x = x.astype(mx.float32)
        variance = mx.mean(x * x, axis=-1, keepdims=True)
        x = x * mx.rsqrt(variance + 1e-6)
        result = self.weight.astype(mx.float32) * x
        return result.astype(input_dtype)


class Qwen3VLMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(HIDDEN_DIM, MLP_HIDDEN, bias=False)
        self.up_proj = nn.Linear(HIDDEN_DIM, MLP_HIDDEN, bias=False)
        self.down_proj = nn.Linear(MLP_HIDDEN, HIDDEN_DIM, bias=False)

    def __call__(self, hidden: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(hidden)) * self.up_proj(hidden))


class Qwen3VLRotaryEmbedding:
    """Interleaved mrope (mflux 0.24 parity): dim=128 freqs over full head_dim,
    interleave h/w indices every 3rd within mrope_section*3 span, emb doubled.
    position_ids: [3, batch, seq] -> cos/sin [batch, seq, head_dim]."""

    def __init__(self) -> None:
        self.inv_freq = 1.0 / (
            ROPE_THETA ** (mx.arange(0, HEAD_DIM, 2, dtype=mx.float32) / HEAD_DIM)
        )

    def __call__(self, position_ids: mx.array) -> tuple[mx.array, mx.array]:
        batch = position_ids.shape[1]
        inv_exp = mx.broadcast_to(
            self.inv_freq[None, None, :, None].astype(mx.float32),
            (3, batch, self.inv_freq.shape[0], 1),
        )
        pos_exp = position_ids[:, :, None, :].astype(mx.float32)
        freqs = mx.transpose(mx.matmul(inv_exp, pos_exp), (0, 1, 3, 2))
        freqs_t = freqs[0]
        freqs_t_np = np.array(freqs_t, copy=True)
        for dim, offset in enumerate((1, 2), start=1):
            length = MROPE_SECTION[dim] * 3
            indices_np = np.arange(offset, length, 3)
            freqs_dim_np = np.array(freqs[dim], copy=False)
            freqs_t_np[..., indices_np] = freqs_dim_np[..., indices_np]
        emb = np.concatenate([freqs_t_np, freqs_t_np], axis=-1)
        emb_mx = mx.array(emb)
        cos = mx.cos(emb_mx)
        sin = mx.sin(emb_mx)
        return cos, sin


def _rotate_half_interleaved(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    """Interleaved-pair rotation for the Qwen3-VL text rope.

    cos/sin: [batch, seq, head_dim] (mflux convention). x: [batch, heads,
    seq, head_dim] — head axis inserted at unsqueeze_dim=1, matching
    mflux's _apply_rotary_pos_emb.
    """
    cos_b = mx.expand_dims(cos, axis=1)  # [batch, 1, seq, head_dim]
    sin_b = mx.expand_dims(sin, axis=1)
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    x_rot = mx.concatenate([-x2, x1], axis=-1)
    # bf16 arithmetic throughout — mflux rotates q/cos/sin in their native dtype
    return (x * cos_b) + (x_rot * sin_b)


class Qwen3VLAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # checkpoint (and mflux) carry no attention biases
        self.q_proj = nn.Linear(HIDDEN_DIM, NUM_HEADS * HEAD_DIM, bias=False)
        self.k_proj = nn.Linear(HIDDEN_DIM, NUM_KV_HEADS * HEAD_DIM, bias=False)
        self.v_proj = nn.Linear(HIDDEN_DIM, NUM_KV_HEADS * HEAD_DIM, bias=False)
        self.o_proj = nn.Linear(NUM_HEADS * HEAD_DIM, HIDDEN_DIM, bias=False)
        self.q_norm = Qwen3VLRMSNorm(HEAD_DIM)
        self.k_norm = Qwen3VLRMSNorm(HEAD_DIM)

    def __call__(
        self,
        hidden: mx.array,
        cos: mx.array,
        sin: mx.array,
        mask: mx.array | None,
    ) -> mx.array:
        batch_size, seq_len, _ = hidden.shape
        query = self.q_proj(hidden).reshape(batch_size, seq_len, NUM_HEADS, HEAD_DIM)
        key = self.k_proj(hidden).reshape(batch_size, seq_len, NUM_KV_HEADS, HEAD_DIM)
        value = self.v_proj(hidden).reshape(batch_size, seq_len, NUM_KV_HEADS, HEAD_DIM)
        query = self.q_norm(query)
        key = self.k_norm(key)
        query = mx.transpose(query, (0, 2, 1, 3))
        key = mx.transpose(key, (0, 2, 1, 3))
        value = mx.transpose(value, (0, 2, 1, 3))
        query = _rotate_half_interleaved(query, cos, sin)
        key = _rotate_half_interleaved(key, cos, sin)
        attn_dtype = query.dtype
        out = mx.fast.scaled_dot_product_attention(
            query.astype(mx.float32),
            key.astype(mx.float32),
            value.astype(mx.float32),
            scale=HEAD_DIM**-0.5,
            mask=mask,
        )
        out = out.astype(attn_dtype)
        out = mx.transpose(out, (0, 2, 1, 3)).reshape(
            batch_size, seq_len, NUM_HEADS * HEAD_DIM
        )
        return self.o_proj(out)


class Qwen3VLDecoderLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.input_layernorm = Qwen3VLRMSNorm(HIDDEN_DIM)
        self.self_attn = Qwen3VLAttention()
        self.post_attention_layernorm = Qwen3VLRMSNorm(HIDDEN_DIM)
        self.mlp = Qwen3VLMLP()

    def __call__(
        self,
        hidden: mx.array,
        cos: mx.array,
        sin: mx.array,
        mask: mx.array | None,
    ) -> mx.array:
        residual = hidden
        hidden = self.input_layernorm(hidden)
        hidden = self.self_attn(hidden, cos, sin, mask)
        hidden = residual + hidden
        residual = hidden
        hidden = self.post_attention_layernorm(hidden)
        hidden = self.mlp(hidden)
        return residual + hidden


class Qwen3VLLanguageModel(nn.Module):
    """Qwen3-VL text tower (embeddings + 36 decoder layers + final norm)."""

    def __init__(self) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(VOCAB_SIZE, HIDDEN_DIM)
        self.layers = [Qwen3VLDecoderLayer() for _ in range(NUM_LAYERS)]
        self.norm = Qwen3VLRMSNorm(HIDDEN_DIM)
        self.rope = Qwen3VLRotaryEmbedding()

    def _build_mask(
        self, attention_mask: mx.array | None, seq_len: int
    ) -> mx.array | None:
        """mflux 0.24 verbatim: causal + padding additive fp32 mask."""
        mask_dtype = mx.float32
        if attention_mask is None:
            attention_mask = mx.ones((1, seq_len), dtype=mx.int32)
        padding_mask = mx.where(
            attention_mask == 1,
            mx.zeros(attention_mask.shape, dtype=mask_dtype),
            mx.full(attention_mask.shape, -float("inf"), dtype=mask_dtype),
        )
        padding_mask = mx.expand_dims(mx.expand_dims(padding_mask, axis=1), axis=1)
        idx = mx.arange(seq_len, dtype=mx.int32)
        causal = mx.where(
            idx[None, :] > idx[:, None],
            mx.full((seq_len, seq_len), -float("inf"), dtype=mask_dtype),
            mx.zeros((seq_len, seq_len), dtype=mask_dtype),
        )
        return (
            mx.broadcast_to(
                causal[None, None, :, :], (attention_mask.shape[0], 1, seq_len, seq_len)
            )
            + padding_mask
        )

    def __call__(
        self,
        input_ids: mx.array,
        attention_mask: mx.array | None = None,
    ) -> mx.array:
        batch_size, seq_len = input_ids.shape
        hidden = self.embed_tokens(input_ids)
        ladder = mx.arange(seq_len, dtype=mx.float32)[None, :]
        position_ids = mx.stack([ladder, ladder, ladder], axis=0)  # [3, batch, seq]
        cos, sin = self.rope(position_ids)
        # mflux casts cos/sin to the hidden dtype (bf16) inside the rope call
        cos = cos.astype(hidden.dtype)
        sin = sin.astype(hidden.dtype)
        mask = self._build_mask(attention_mask, seq_len)
        for layer in self.layers:
            hidden = layer(hidden, cos, sin, mask)
        return self.norm(hidden)


class Qwen21TextEncoder(nn.Module):
    """Text encoder for 2.1 prompt embeddings: raw hidden states after the tower."""

    def __init__(self) -> None:
        super().__init__()
        self.model = Qwen3VLLanguageModel()

    def __call__(
        self,
        input_ids: mx.array,
        attention_mask: mx.array | None = None,
    ) -> mx.array:
        return self.model(input_ids, attention_mask)


class Qwen21Tokenizer:
    """Chat-template prompt tokenization via the HF tokenizer in the processor dir."""

    def __init__(self, processor_dir: str | Path) -> None:
        from transformers import AutoTokenizer  # noqa: PLC0415

        self._tokenizer = AutoTokenizer.from_pretrained(  # pyright: ignore[reportUnknownMemberType]
            str(processor_dir), local_files_only=True
        )
        self._prefix_len = self._prefix_token_count()

    def _prefix_token_count(self) -> int:
        encoded = self._tokenizer(SYSTEM_PREFIX, add_special_tokens=False)  # pyright: ignore[reportAny]
        input_ids = encoded["input_ids"]  # pyright: ignore[reportAny]
        if isinstance(input_ids[0], list):  # pyright: ignore[reportAny]
            input_ids = input_ids[0]  # pyright: ignore[reportAny]
        return len(input_ids)

    def tokenize(self, prompt: str) -> dict[str, np.ndarray]:
        rendered = PROMPT_TEMPLATE_T2I.format(prompt)
        encoded = self._tokenizer(  # pyright: ignore[reportAny]
            [rendered],
            padding="longest",
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="np",
            add_special_tokens=False,
        )
        input_ids = np.asarray(encoded["input_ids"], dtype=np.int64)  # pyright: ignore[reportAny]
        attention_mask = np.asarray(encoded["attention_mask"], dtype=np.int64)  # pyright: ignore[reportAny]
        # Drop the system prefix tokens (positions 0..prefix_len) as in mflux 0.24.
        input_ids = input_ids[:, self._prefix_len :]
        attention_mask = attention_mask[:, self._prefix_len :]
        return {"input_ids": input_ids, "attention_mask": attention_mask}


def encode_prompts(
    encoder: Qwen21TextEncoder,
    tokenizer: Qwen21Tokenizer,
    prompts: list[str],
    prompt_cache: dict[str, tuple[mx.array, mx.array]],
) -> list[tuple[mx.array, mx.array]]:
    """Tokenize (with cache) and run the text tower. Returns [(embeds, mask), ...]."""
    results: list[tuple[mx.array, mx.array]] = []
    for prompt in prompts:
        if prompt in prompt_cache:
            results.append(prompt_cache[prompt])
            continue
        effective = prompt if prompt != "" else " "
        tokens = tokenizer.tokenize(effective)
        input_ids = mx.array(tokens["input_ids"])
        attention_mask = mx.array(tokens["attention_mask"])
        hidden = encoder(input_ids, attention_mask)
        pair = (hidden, attention_mask)
        prompt_cache[prompt] = pair
        results.append(pair)
    return results
