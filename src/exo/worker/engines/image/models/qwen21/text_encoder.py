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

# mflux 0.24 Qwen21PromptEncoder verbatim.
SYSTEM_PROMPT = "Comprehend and analyze the provided prompt."
SYSTEM_PREFIX = f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
PROMPT_TEMPLATE_T2I = (
    f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
    "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
)


class Qwen3VLRMSNorm(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.weight = mx.ones((dim,))

    def __call__(self, x: mx.array) -> mx.array:
        dtype = x.dtype
        x = x.astype(mx.float32)
        out = x * mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + 1e-6)
        return (self.weight * out).astype(dtype)


class Qwen3VLMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(HIDDEN_DIM, MLP_HIDDEN, bias=False)
        self.up_proj = nn.Linear(HIDDEN_DIM, MLP_HIDDEN, bias=False)
        self.down_proj = nn.Linear(MLP_HIDDEN, HIDDEN_DIM, bias=False)

    def __call__(self, hidden: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(hidden)) * self.up_proj(hidden))


class Qwen3VLRotaryEmbedding:
    """Interleaved mrope. position_ids: [3, batch, seq] int -> cos/sin [batch, seq, head_dim]."""

    def __init__(self) -> None:
        half = HEAD_DIM // 2
        inv_freq = 1.0 / (
            ROPE_THETA ** (mx.arange(0, half, dtype=mx.float32) / half)
        )
        self.inv_freq = inv_freq

    def __call__(self, position_ids: mx.array) -> tuple[mx.array, mx.array]:
        inv_freq = self.inv_freq
        freqs = mx.outer(position_ids.reshape(-1).astype(mx.float32), inv_freq)
        freqs = freqs.reshape(3, position_ids.shape[1], position_ids.shape[2], -1)
        freqs_t = freqs[0]
        offsets = np.cumsum((1,) + MROPE_SECTION[:-1])
        freqs_h = freqs[1]
        freqs_w = freqs[2]
        freqs_int = np.array(freqs_t, copy=False)
        freqs_h_np = np.array(freqs_h, copy=False)
        freqs_w_np = np.array(freqs_w, copy=False)
        idx = np.arange(HEAD_DIM // 2)
        h_idx = idx[offsets[0] :: 3]
        w_idx = idx[offsets[1] :: 3]
        freqs_int[:, :, h_idx] = freqs_h_np[:, :, h_idx]
        freqs_int[:, :, w_idx] = freqs_w_np[:, :, w_idx]
        emb = np.concatenate([freqs_int, freqs_int], axis=-1)
        cos = mx.array(np.cos(emb))
        sin = mx.array(np.sin(emb))
        return cos, sin


def _rotate_half_interleaved(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    """Interleaved-pair rotation for the Qwen3-VL text rope."""
    x_float = x.astype(mx.float32)
    shape = (*x_float.shape[:-1], -1, 2)
    x_reshaped = mx.reshape(x_float, shape)
    x_real = x_reshaped[..., 0]
    x_imag = x_reshaped[..., 1]
    cos_b = cos[:, :, None, :]  # [batch, seq, 1, head/2]
    sin_b = sin[:, :, None, :]
    out_real = x_real * cos_b - x_imag * sin_b
    out_imag = x_real * sin_b + x_imag * cos_b
    out = mx.stack([out_real, out_imag], axis=-1)
    return mx.reshape(out, x_float.shape).astype(x.dtype)


class Qwen3VLAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = nn.Linear(HIDDEN_DIM, NUM_HEADS * HEAD_DIM, bias=True)
        self.k_proj = nn.Linear(HIDDEN_DIM, NUM_KV_HEADS * HEAD_DIM, bias=True)
        self.v_proj = nn.Linear(HIDDEN_DIM, NUM_KV_HEADS * HEAD_DIM, bias=True)
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
        query = self.q_proj(hidden).reshape(
            batch_size, seq_len, NUM_HEADS, HEAD_DIM
        )
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
        batch = 1
        causal = mx.triu(
            mx.full((seq_len, seq_len), -mx.inf, dtype=mx.float32), k=1
        )
        if attention_mask is None:
            return causal[None, None, :, :]
        valid = attention_mask.astype(mx.float32)  # [batch, seq]
        pad_additive = (1.0 - valid) * -mx.inf  # [batch, seq]
        key_mask = pad_additive[:, None, None, :]
        combined = mx.maximum(causal[None, None, :, :], key_mask)
        return mx.concatenate([combined] * batch, axis=0)

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
        """Tokenize the FULL rendered template — mflux encodes the whole
        sequence and drops the system-prefix positions from the hidden states
        and mask AFTER the tower (dropping before would shift the positions)."""
        rendered = PROMPT_TEMPLATE_T2I.format(prompt)
        encoded = self._tokenizer(  # pyright: ignore[reportAny]
            [rendered],
            padding="longest",
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="np",
        )
        input_ids = np.asarray(encoded["input_ids"], dtype=np.int64)  # pyright: ignore[reportAny]
        attention_mask = np.asarray(encoded["attention_mask"], dtype=np.int64)  # pyright: ignore[reportAny]
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    @property
    def prefix_len(self) -> int:
        return self._prefix_len


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
        if not prompt or not prompt.strip():
            prompt = " "
        tokens = tokenizer.tokenize(prompt)
        input_ids = mx.array(tokens["input_ids"])
        attention_mask = mx.array(tokens["attention_mask"])
        hidden = encoder(input_ids, attention_mask)
        # Drop the system-prefix positions from embeds AND mask after encoding
        # (mflux 0.24 Qwen21PromptEncoder.encode_prompt).
        drop = tokenizer.prefix_len
        pair = (hidden[:, drop:, :], attention_mask[:, drop:])
        prompt_cache[prompt] = pair
        results.append(pair)
    return results