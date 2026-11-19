"""Pure-MLX Qwen-Image-2.1 Turbo transformer (no mflux).

Reimplements the mflux-0.24 qwen21 transformer semantics against MLX primitives
only, shaped for EXO's two-stream joint-block integration:
- LayerNorm (affine=False) with per-token shared modulation chunks
- tanh gating on both attention and FFN branches
- plain RMSNorm q/k normalization (checkpoint weight used directly;
  only txt_in.text_norm is zero-centered, weight + 1)
- RoPE with theta 10000, axes (16, 56, 56) over (t, h, w)
- scale-only output norm + AdaLayerNormContinuous-style final projection
"""

import mlx.core as mx
import numpy as np
from mlx import nn

HIDDEN_DIM = 4096
NUM_HEADS = 32
HEAD_DIM = 128
NUM_LAYERS = 32
INNER_DIM = NUM_HEADS * HEAD_DIM  # 4096
MLP_HIDDEN = int(INNER_DIM * 3)  # 12288
ROPE_AXES = (16, 56, 56)
ROPE_THETA = 10000.0
EPS = 1e-6

MOD_PER_TOKEN = 4 * INNER_DIM  # modulation output per row (scale1, gate1, scale2, gate2)

TIME_DIM = 256  # sinusoidal timestep features
MODEL_PRECISION = mx.bfloat16  # mflux 0.24 ModelConfig.precision for qwen21


class Qwen21Timesteps(nn.Module):
    """Sinusoidal timestep features: cos in the first half, sin in the second.

    args = t * 1000 * freqs; freqs = exp(-log(10000) * arange(128)/128).
    Weightless: no parameters are loaded for this module.
    """

    def __init__(self) -> None:
        super().__init__()
        self.timestep_dim = TIME_DIM
        self.time_factor = 1000.0
        half = TIME_DIM // 2
        self._freqs = mx.array(
            np.exp(-np.log(10000.0) * np.arange(half, dtype=np.float32) / half)
        )

    def __call__(self, timestep: mx.array) -> mx.array:
        args = timestep.astype(mx.float32)[:, None] * self.time_factor * self._freqs[None, :]
        # mflux casts the sinusoid features to ModelConfig.precision (bf16)
        # before the embedder MLP; match it to keep the stream bf16.
        return mx.concatenate([mx.cos(args), mx.sin(args)], axis=-1).astype(
            MODEL_PRECISION
        )


class Qwen21TimestepEmbedder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(TIME_DIM, INNER_DIM, bias=False)
        self.linear_2 = nn.Linear(INNER_DIM, INNER_DIM, bias=False)

    def __call__(self, timesteps_proj: mx.array) -> mx.array:
        return self.linear_2(nn.silu(self.linear_1(timesteps_proj)))


class Qwen21TimeTextEmbed(nn.Module):
    """[t, 0] timestep-row embedder: row 0 = sampled t, row 1 = t=0 (text)."""

    def __init__(self) -> None:
        super().__init__()
        self.time_proj = Qwen21Timesteps()
        self.timestep_embedder = Qwen21TimestepEmbedder()

    def __call__(self, timestep: mx.array) -> mx.array:
        """timestep: [batch] float32 -> [batch, 2, inner_dim]."""
        timestep = timestep.astype(mx.float32)
        batch = timestep.shape[0]
        rows = mx.concatenate(
            [timestep, mx.zeros((batch,), dtype=mx.float32)], axis=0
        )
        emb = self.timestep_embedder(self.time_proj(rows))  # [2*batch, inner]
        return emb.reshape(batch, 2, INNER_DIM)


class Qwen21TextProjection(nn.Module):
    """ZeroCenterRMSNorm -> Linear -> GELU(approx) -> Linear (4096 throughout)."""

    def __init__(self) -> None:
        super().__init__()
        self.text_norm = Qwen21ZeroCenterRMSNorm(HIDDEN_DIM)
        self.in_layer = nn.Linear(HIDDEN_DIM, INNER_DIM, bias=False)
        self.out_layer = nn.Linear(INNER_DIM, INNER_DIM, bias=False)

    def __call__(self, text_hidden: mx.array) -> mx.array:
        hidden = self.text_norm(text_hidden)
        hidden = self.in_layer(hidden)
        hidden = nn.gelu_approx(hidden)
        return self.out_layer(hidden)


class Qwen21ZeroCenterRMSNorm(nn.Module):
    """weight + 1, reciprocal-RMS in fp32 (matches mflux Qwen21ZeroCenterRMSNorm)."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.weight = mx.ones((dim,))

    def __call__(self, x: mx.array) -> mx.array:
        dtype = x.dtype
        x = x.astype(mx.float32)
        rrms = mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + EPS)
        out = x * rrms * (self.weight + 1)
        return out.astype(dtype)


class Qwen21Rope:
    """2.1 RoPE over the joint sequence (mflux-0.24 QwenImage21Layout.create).

    Joint layout: [text tokens] [image tokens]. The rope angle stream is ONE
    continuous position cursor:
    - text tokens occupy positions 0..text_len-1 (all three axes advance together)
    - the first image token takes position text_len (the t frame position), then
      h/w centred on zero; the t cursor advances past the image block.
    Angles: axis0*theta^(-j/8), axis1/2*theta^(-j/28) concatenated -> 64 per pos.
    """

    def __init__(self, text_len: int, height: int, width: int) -> None:
        self._text_len = text_len
        self._height = height
        self._width = width
        self._cache: dict[int, tuple[mx.array, mx.array]] = {}

    @staticmethod
    def _build_tables(text_len: int, height: int, width: int) -> tuple[mx.array, mx.array]:
        n = text_len + height * width
        frame = np.zeros(n, dtype=np.float32)
        hax = np.zeros(n, dtype=np.float32)
        wax = np.zeros(n, dtype=np.float32)
        frame[:text_len] = np.arange(text_len)
        hax[:text_len] = np.arange(text_len)
        wax[:text_len] = np.arange(text_len)
        # image block starts at position text_len (frame frozen at text len)
        frame[text_len:] = text_len
        hax[text_len:] = np.repeat(np.arange(-(height - height // 2), height // 2), width)
        wax[text_len:] = np.tile(np.arange(-(width - width // 2), width // 2), height)
        angles = np.concatenate(
            [
                ax[:, None] * (10000.0 ** (-np.arange(0, dim, 2, dtype=np.float32) / dim))[None]
                for ax, dim in zip((frame, hax, wax), ROPE_AXES, strict=True)
            ],
            axis=-1,
        )
        cos = mx.array(np.cos(angles))
        sin = mx.array(np.sin(angles))
        return cos, sin

    def get(self) -> tuple[mx.array, mx.array]:
        if 0 not in self._cache:
            self._cache[0] = Qwen21Rope._build_tables(
                self._text_len, self._height, self._width
            )
        return self._cache[0]

    def image_slice(self, patch_start: int = 0, patch_end: int | None = None) -> slice:
        start = self._text_len + patch_start
        end = self._text_len + (self._height * self._width if patch_end is None else patch_end)
        return slice(start, end)


def apply_rope_21(
    x: mx.array, cos_vals: mx.array, sin_vals: mx.array
) -> mx.array:
    """x: [batch, heads, seq, head_dim]; cos/sin: [seq, head_dim // 2]."""
    x_float = x.astype(mx.float32)
    x_reshaped = mx.reshape(x_float, (*x.shape[:-1], -1, 2))
    x_real = x_reshaped[..., 0]
    x_imag = x_reshaped[..., 1]
    freqs_cos = cos_vals[None, None, :, :]
    freqs_sin = sin_vals[None, None, :, :]
    out_real = x_real * freqs_cos - x_imag * freqs_sin
    out_imag = x_real * freqs_sin + x_imag * freqs_cos
    out_pairs = mx.stack([out_real, out_imag], axis=-1)
    out = mx.reshape(out_pairs, (*x.shape[:-1], -1))
    return out.astype(x.dtype)


class Qwen21Attention(nn.Module):
    """Single-stream 2.1 attention: to_q/k/v, q/k RMSNorm, to_out."""

    def __init__(self) -> None:
        super().__init__()
        self.to_q = nn.Linear(INNER_DIM, INNER_DIM, bias=False)
        self.to_k = nn.Linear(INNER_DIM, INNER_DIM, bias=False)
        self.to_v = nn.Linear(INNER_DIM, INNER_DIM, bias=False)
        self.norm_q = nn.RMSNorm(HEAD_DIM, eps=EPS)
        self.norm_k = nn.RMSNorm(HEAD_DIM, eps=EPS)
        self.to_out = [nn.Linear(INNER_DIM, INNER_DIM, bias=False)]

    def project(self, hidden: mx.array) -> tuple[mx.array, mx.array, mx.array]:
        batch_size, seq_len, _ = hidden.shape
        query = self.to_q(hidden).reshape(batch_size, seq_len, NUM_HEADS, HEAD_DIM)
        key = self.to_k(hidden).reshape(batch_size, seq_len, NUM_HEADS, HEAD_DIM)
        value = self.to_v(hidden).reshape(batch_size, seq_len, NUM_HEADS, HEAD_DIM)
        query = self.norm_q(query)
        key = self.norm_k(key)
        return (
            mx.transpose(query, (0, 2, 1, 3)),
            mx.transpose(key, (0, 2, 1, 3)),
            mx.transpose(value, (0, 2, 1, 3)),
        )

    def unproject(self, attn_out: mx.array) -> mx.array:
        batch_size, _, seq_len, _ = attn_out.shape
        hidden = mx.transpose(attn_out, (0, 2, 1, 3)).reshape(
            batch_size, seq_len, INNER_DIM
        )
        return self.to_out[0](hidden)


class Qwen21SwiGLUFeedForward(nn.Module):
    """SwiGLU: out(silu(gate_layer(x)) * proj(x)) — checkpoint keys img_mlp.*."""

    def __init__(self) -> None:
        super().__init__()
        self.gate_layer = nn.Linear(INNER_DIM, MLP_HIDDEN, bias=False)
        self.proj = nn.Linear(INNER_DIM, MLP_HIDDEN, bias=False)
        self.out = nn.Linear(MLP_HIDDEN, INNER_DIM, bias=False)

    def __call__(self, hidden: mx.array) -> mx.array:
        return self.out(nn.silu(self.gate_layer(hidden)) * self.proj(hidden))


class Qwen21TransformerBlock(nn.Module):
    """One 2.1 joint block. norm_* are affine=False LayerNorms."""

    def __init__(self) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(INNER_DIM, affine=False, eps=EPS)
        self.attn = Qwen21Attention()
        self.norm2 = nn.LayerNorm(INNER_DIM, affine=False, eps=EPS)
        self.ff = Qwen21SwiGLUFeedForward()

    @staticmethod
    def modulate(normed: mx.array, scale: mx.array) -> mx.array:
        return normed * (1 + scale)

    @staticmethod
    def split_mod(mod: mx.array) -> tuple[mx.array, mx.array]:
        scale1, gate1, scale2, gate2 = mx.split(mod, 4, axis=-1)
        return scale1, gate1, scale2, gate2


class Qwen21AdaLayerNormContinuous(nn.Module):
    """Scale-only output conditioning from the timestep embedding (row 0)."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(INNER_DIM, INNER_DIM, bias=False)
        self.norm = nn.LayerNorm(INNER_DIM, affine=False, eps=EPS)

    def scale_for(self, temb: mx.array) -> mx.array:
        """temb: [batch, inner_dim] -> scale [batch, inner_dim]."""
        return self.linear(nn.silu(temb))

    def __call__(self, hidden_states: mx.array, scale: mx.array) -> mx.array:
        return self.norm(hidden_states) * (1 + scale)


def build_transformer() -> nn.Module:
    """Assemble the transformer container (weights loaded separately)."""
    return Qwen21Transformer()


class Qwen21Transformer(nn.Module):
    """Full 2.1 transformer. Owns blocks so EXO can shard by slicing the list."""

    def __init__(self) -> None:
        super().__init__()
        self.time_text_embed = Qwen21TimeTextEmbed()
        self.txt_in = Qwen21TextProjection()
        self.img_in = nn.Linear(64, INNER_DIM, bias=False)
        self.modulation = nn.Linear(INNER_DIM, MOD_PER_TOKEN, bias=False)
        self.transformer_blocks = [Qwen21TransformerBlock() for _ in range(NUM_LAYERS)]
        self.norm_out = Qwen21AdaLayerNormContinuous()
        self.proj_out = nn.Linear(INNER_DIM, 64, bias=False)

    def modulation_for(self, temb: mx.array) -> mx.array:
        """temb: [batch, 2, inner_dim] -> per-token mod [batch, 2, 4*inner]."""
        flat = nn.silu(temb.reshape(-1, INNER_DIM))
        return self.modulation(flat).reshape(temb.shape[0], temb.shape[1], MOD_PER_TOKEN)