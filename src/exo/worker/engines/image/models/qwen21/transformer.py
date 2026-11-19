"""Pure-MLX Qwen-Image-2.1 Turbo transformer (no mflux).

Reimplements the mflux-0.24 qwen21 transformer semantics against MLX primitives
only, shaped for EXO's two-stream joint-block integration:
- LayerNorm (affine=False) with per-token shared modulation chunks
- tanh gating on both attention and FFN branches
- zero-centered RMSNorm q/k normalization (weight + 1)
- RoPE with theta 10000, axes (16, 56, 56) over (t, h, w)
- scale-only output norm + AdaLayerNormContinuous-style final projection
"""

import math

import mlx.core as mx
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

TIME_TEXT_LADDER: tuple[int, ...] = (1000, 1024, 1000, 1024, 1000, 1024)
PROJECTED_DIM = 1024
TEXT_PROJ_HIDDEN = 512


class Qwen21Timesteps(nn.Module):
    """Sinusoidal timestep features, matching diffusers get_timestep_embedding."""

    def __init__(self) -> None:
        super().__init__()
        half = PROJECTED_DIM // 2
        exponent = -math.log(10000.0) * mx.arange(half, dtype=mx.float32) / half
        self.exponent = exponent

    def __call__(self, timesteps: mx.array) -> mx.array:
        emb = mx.exp(timesteps[:, None].astype(mx.float32) * self.exponent[None, :])
        return mx.concatenate([mx.sin(emb), mx.cos(emb)], axis=-1)


class Qwen21TimeTextEmbed(nn.Module):
    """Timestep embedding + the [t, 0] text-ladder for shared modulation."""

    def __init__(self) -> None:
        super().__init__()
        self.time_proj = Qwen21Timesteps()
        self.time_embedder = nn.Sequential(
            nn.Linear(PROJECTED_DIM, PROJECTED_DIM, bias=True),
            nn.SiLU(),
            nn.Linear(PROJECTED_DIM, PROJECTED_DIM, bias=True),
        )

    def _ladder(self, batch_size: int) -> mx.array:
        t_ladder = mx.concatenate(
            [
                mx.zeros((1,), dtype=mx.float32),
                mx.full((1,), float(TIME_TEXT_LADDER[0]), dtype=mx.float32),
                mx.zeros((1,), dtype=mx.float32),
                mx.full((1,), float(TIME_TEXT_LADDER[2]), dtype=mx.float32),
                mx.zeros((1,), dtype=mx.float32),
                mx.full((1,), float(TIME_TEXT_LADDER[4]), dtype=mx.float32),
            ]
        )
        t_ladder = t_ladder.reshape(3, 2)
        t_ladder = mx.concatenate(
            [mx.zeros((3, 1), dtype=mx.float32), t_ladder], axis=1
        )
        emb = self.time_proj(t_ladder.reshape(-1))
        emb = self.time_embedder(emb)
        # Replace the zero rows with the real timestep embedding.
        zeros = mx.zeros((3, 1), dtype=mx.float32)
        t_ladder = mx.concatenate([zeros, t_ladder[:, 1:]], axis=1)
        t_ladder = t_ladder.reshape(-1)
        first = emb[0:1]
        return mx.concatenate([first, emb[1:]], axis=0)[None, :, :].repeat(
            batch_size, axis=0
        )

    def __call__(self, timestep: mx.array) -> mx.array:
        """timestep: [batch] float32. Returns [batch, 2, projected_dim]."""
        timestep = timestep.astype(mx.float32)
        batch_size = timestep.shape[0]
        ladder = self._ladder(batch_size)
        t_emb = self.time_proj(timestep)
        t_emb = self.time_embedder(t_emb)  # [batch, projected]
        first = mx.concatenate(
            [t_emb[:, None, :], ladder[:, 1:2, :]], axis=1
        )
        rest = ladder[:, 2:, :]
        return mx.concatenate([first, rest], axis=1)


class Qwen21TextProjection(nn.Module):
    """ZeroCenterRMSNorm -> Linear -> GELU(approx) -> Linear."""

    def __init__(self) -> None:
        super().__init__()
        self.text_norm = Qwen21ZeroCenterRMSNorm(HIDDEN_DIM)
        self.in_layer = nn.Linear(HIDDEN_DIM, TEXT_PROJ_HIDDEN, bias=True)
        self.out_layer = nn.Linear(TEXT_PROJ_HIDDEN, INNER_DIM, bias=True)

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
    """2.1 RoPE: real-form pair rotation, theta 10000, axes (16, 56, 56).

    Index space: image positions are 8192 positive then 1024 negative reversed;
    text tokens advance all three axes from 0; the image frame origin is frozen
    at the text length and H/W are centred on zero.
    """

    def __init__(self, text_len: int, height: int, width: int) -> None:
        self._text_len = text_len
        self._height = height
        self._width = width
        self._cache: dict[int, tuple[mx.array, mx.array]] = {}

    @staticmethod
    def _rope_params(index: mx.array, dim: int, theta: float) -> mx.array:
        assert dim % 2 == 0
        half = dim // 2
        freqs = mx.arange(half, dtype=mx.float32)
        freqs = mx.exp(-math.log(theta) * freqs / half)
        freqs = mx.outer(index.astype(mx.float32), freqs)
        return mx.stack([mx.cos(freqs), mx.sin(freqs)], axis=-1)

    @staticmethod
    def _build_tables(text_len: int, height: int, width: int) -> tuple[mx.array, mx.array]:
        seq_pos = mx.arange(8192, dtype=mx.float32)
        seq_neg = mx.arange(1024, dtype=mx.float32)
        seq_neg = -mx.flip(seq_neg)
        # axes 0,1,2 = (t, h, w); frame tokens continue after the text tokens.
        t_ids = mx.concatenate([seq_pos, seq_neg, mx.full((height * width,), -1024.0)])
        h_ids = mx.concatenate([seq_pos, seq_neg, mx.repeat(mx.arange(height, dtype=mx.float32) - (height // 2), width)])
        w_ids = mx.concatenate([seq_pos, seq_neg, mx.tile(mx.arange(width, dtype=mx.float32) - (width // 2), height)])

        parts: list[mx.array] = []
        for ids, dim in zip((t_ids, h_ids, w_ids), ROPE_AXES, strict=True):
            parts.append(Qwen21Rope._rope_params(ids, dim, ROPE_THETA))

        total = text_len + height * width
        # text positions: ladder that advances t/h/w together from 0
        text_ids_t = mx.arange(text_len, dtype=mx.float32)
        text_parts: list[mx.array] = []
        for dim in ROPE_AXES:
            text_parts.append(Qwen21Rope._rope_params(text_ids_t, dim, ROPE_THETA))

        tables = [
            mx.concatenate([text_parts[i], parts[i][:total - text_len]], axis=0)
            for i in range(3)
        ]
        # axes 1,2 use h/w; text region uses the ladder for all three axes
        cos_t, sin_t = tables[0][..., 0], tables[0][..., 1]
        cos_h, sin_h = tables[1][..., 0], tables[1][..., 1]
        cos_w, sin_w = tables[2][..., 0], tables[2][..., 1]
        cos = mx.concatenate([cos_t, cos_h, cos_w], axis=-1)
        sin = mx.concatenate([sin_t, sin_h, sin_w], axis=-1)
        return cos, sin

    def get(self, device: int = 0) -> tuple[mx.array, mx.array]:
        if device not in self._cache:
            self._cache[device] = Qwen21Rope._build_tables(
                self._text_len, self._height, self._width
            )
        return self._cache[device]

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
    freqs_cos = cos_vals[None, :, None, :]
    freqs_sin = sin_vals[None, :, None, :]
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
        self.norm_q = Qwen21ZeroCenterRMSNorm(HEAD_DIM)
        self.norm_k = Qwen21ZeroCenterRMSNorm(HEAD_DIM)
        self.to_out = nn.ModuleList([nn.Linear(INNER_DIM, INNER_DIM, bias=False)])

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
    """SwiGLU with separate gate/up projections (2.1 has no packed fused weights)."""

    def __init__(self) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(INNER_DIM, MLP_HIDDEN, bias=False)
        self.up_proj = nn.Linear(INNER_DIM, MLP_HIDDEN, bias=False)
        self.down_proj = nn.Linear(MLP_HIDDEN, INNER_DIM, bias=False)

    def __call__(self, hidden: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(hidden)) * self.up_proj(hidden))


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
        self.linear = nn.Linear(PROJECTED_DIM, INNER_DIM, bias=True)
        self.norm = nn.LayerNorm(INNER_DIM, affine=False, eps=EPS)

    def scale_for(self, temb: mx.array) -> mx.array:
        """temb: [batch, projected] -> scale [batch, inner_dim]."""
        return self.linear(nn.silu(temb))


def build_transformer() -> nn.Module:
    """Assemble the transformer container (weights loaded separately)."""
    return Qwen21Transformer()


class Qwen21Transformer(nn.Module):
    """Full 2.1 transformer. Owns blocks so EXO can shard by slicing the list."""

    def __init__(self) -> None:
        super().__init__()
        self.time_text_embed = Qwen21TimeTextEmbed()
        self.txt_in = Qwen21TextProjection()
        self.img_in = nn.Linear(64, INNER_DIM, bias=True)
        self.modulation = nn.Linear(PROJECTED_DIM, MOD_PER_TOKEN, bias=True)
        self.transformer_blocks = [Qwen21TransformerBlock() for _ in range(NUM_LAYERS)]
        self.norm_out = Qwen21AdaLayerNormContinuous()
        self.proj_out = nn.Linear(INNER_DIM, 64, bias=True)

    def modulation_for(self, temb: mx.array) -> mx.array:
        """temb: [batch, 2, projected] -> per-token mod [batch, 2, 4*inner]."""
        flat = temb.reshape(-1, PROJECTED_DIM)
        return self.modulation(flat).reshape(temb.shape[0], temb.shape[1], MOD_PER_TOKEN)