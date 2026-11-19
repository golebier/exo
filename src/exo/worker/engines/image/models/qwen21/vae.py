"""Pure-MLX Qwen-Image-2.1 VAE decoder (no mflux).

Decoder-only port of the mflux-0.24 Qwen21VAE (text_mode stream: NCHW data flow
with channel-last MLX convs inside a transpose wrapper). Weights come from the
abenzerps flat checkpoint; see weights.py for the key remap.
"""

import mlx.core as mx
from mlx import nn

LATENT_CHANNELS = 64
SPATIAL_SCALE = 16


class ChannelNorm(nn.Module):
    """NCHW channel-axis L2 norm (text_mode variant, gamma kept 1-D)."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.gamma = mx.ones((channels,))
        self.scale = float(channels) ** 0.5

    def __call__(self, x: mx.array) -> mx.array:
        value = x.astype(mx.float32)
        norm = mx.maximum(mx.sqrt(mx.sum(value * value, axis=1, keepdims=True)), 1e-12)
        value = value / norm
        # NCHW stream: gamma broadcasts as [1, C, 1, 1] (mflux text_mode).
        return (value * self.scale * self.gamma[None, :, None, None]).astype(x.dtype)


def conv_nchw(layer: nn.Conv2d, x: mx.array) -> mx.array:
    """MLX conv on channel-last input with NCHW in/out."""
    return layer(x.transpose(0, 2, 3, 1)).transpose(0, 3, 1, 2)


class ResidualBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.norm1 = ChannelNorm(in_dim)
        self.conv1 = nn.Conv2d(in_dim, out_dim, 3, padding=1)
        self.norm2 = ChannelNorm(out_dim)
        self.conv2 = nn.Conv2d(out_dim, out_dim, 3, padding=1)
        self.conv_shortcut = (
            nn.Conv2d(in_dim, out_dim, 1) if in_dim != out_dim else nn.Identity()
        )

    def __call__(self, x: mx.array) -> mx.array:
        shortcut = conv_nchw(self.conv_shortcut, x) if isinstance(
            self.conv_shortcut, nn.Conv2d
        ) else x
        x = conv_nchw(self.conv1, nn.silu(self.norm1(x)))
        x = conv_nchw(self.conv2, nn.silu(self.norm2(x)))
        return shortcut + x


class AttentionBlock(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm = ChannelNorm(dim)
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

    def __call__(self, x: mx.array) -> mx.array:
        normalized = self.norm(x)
        b, c, h, w = normalized.shape
        qkv = self.to_qkv(normalized.transpose(0, 2, 3, 1))  # [b, h, w, 3c]
        q, k, v = mx.split(qkv.reshape(b, 1, h * w, 3 * c), 3, axis=-1)
        output = mx.fast.scaled_dot_product_attention(q, k, v, scale=c**-0.5)
        output = self.proj(output.reshape(b, h, w, c))
        return x + output.transpose(0, 3, 1, 2)


class MidBlock(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.resnets = [ResidualBlock(dim, dim), ResidualBlock(dim, dim)]
        self.attentions = [AttentionBlock(dim)]

    def __call__(self, x: mx.array) -> mx.array:
        return self.resnets[1](self.attentions[0](self.resnets[0](x)))


class Resample(nn.Module):
    """Nearest 2x upsample followed by a 3x3 conv (text_mode path: no time_conv)."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, 3, stride=1, padding=1)

    def __call__(self, x: mx.array) -> mx.array:
        x = mx.repeat(mx.repeat(x, 2, axis=2), 2, axis=3)
        return conv_nchw(self.conv, x)


class Qwen21DupUp(nn.Module):
    """Parameterless shortcut: channel repeat -> nearest (t, s) upsample."""

    def __init__(self, in_channels: int, out_channels: int, factor_t: int) -> None:
        super().__init__()
        self.factor_t = factor_t
        self.factor_s = 2
        self.factor = factor_t * self.factor_s * self.factor_s
        if out_channels * self.factor % in_channels != 0:
            raise ValueError(
                f"out_channels ({out_channels}) * factor ({self.factor}) must be"
                f" divisible by in_channels ({in_channels})"
            )
        self.repeats = out_channels * self.factor // in_channels

    def __call__(self, x: mx.array) -> mx.array:
        b, _, h, w = x.shape
        ft, fsh, fsw = self.factor_t, self.factor_s, self.factor_s
        x = mx.repeat(x, self.repeats, axis=1)
        x = x.reshape(b, x.shape[1] // (ft * fsh * fsw), ft, fsh, fsw, 1, h, w)
        x = mx.transpose(x, (0, 1, 5, 2, 6, 3, 7, 4))
        x = x.reshape(b, x.shape[1], ft, h * fsh, w * fsw)
        x = x[:, :, ft - 1 :, :, :]
        return mx.squeeze(x, 2)


class UpBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, num_blocks: int, up: bool, temporal: bool) -> None:
        super().__init__()
        self.resnets = [
            ResidualBlock(in_dim if i == 0 else out_dim, out_dim)
            for i in range(num_blocks + 1)
        ]
        self.upsampler = Resample(out_dim) if up else None
        self.avg_shortcut = Qwen21DupUp(in_dim, out_dim, 2 if temporal else 1) if up else None

    def __call__(self, x: mx.array) -> mx.array:
        original = x
        for block in self.resnets:
            x = block(x)
        if self.upsampler is not None:
            assert self.avg_shortcut is not None
            shortcut = self.avg_shortcut(original)
            x = self.upsampler(x) + shortcut
        return x


class Qwen21VaeDecoder(nn.Module):
    """2.1 VAE decoder. decode() takes NCHW latents [1,64,h,w] (de-normalized)
    and returns NCHW image [1,3,H,W] in [-1, 1] (ImageUtil-compatible)."""

    def __init__(
        self,
        decoder_base_dim: int = 144,
        dim_mult: tuple[int, ...] = (1, 2, 4, 8, 8),
        num_res_blocks: int = 2,
        z_dim: int = LATENT_CHANNELS,
        out_channels: int = 4,
        temporal_downsample: tuple[bool, ...] = (False, True, True, True),
    ) -> None:
        super().__init__()
        mult = list(dim_mult)
        dims = [decoder_base_dim * factor for factor in [mult[-1], *reversed(mult)]]
        self.conv_in = nn.Conv2d(z_dim, dims[0], 3, padding=1)
        self.mid_block = MidBlock(dims[0])
        temporal = list(reversed(temporal_downsample))
        self.up_blocks = [
            UpBlock(a, b, num_res_blocks, i < len(dims) - 2, temporal[i] if i < len(temporal) else False)
            for i, (a, b) in enumerate(zip(dims[:-1], dims[1:], strict=True))
        ]
        self.norm_out = ChannelNorm(dims[-1])
        self.conv_out = nn.Conv2d(dims[-1], out_channels, 3, padding=1)

    def __call__(self, latents: mx.array) -> mx.array:
        x = conv_nchw(self.conv_in, latents)
        x = self.mid_block(x)
        for block in self.up_blocks:
            x = block(x)
        return conv_nchw(self.conv_out, nn.silu(self.norm_out(x)))[:, :3]


LATENTS_MEAN: tuple[float, ...] = (
    0.5126, 0.7721, -0.0631, 1.3506, -0.7855, -2.1025, -0.3458, 1.3722,
    1.8873, -1.7177, -0.651, 0.2732, 0.7562, -0.6163, -1.0277, 3.8363,
    2.021, 0.0472, 0.932, 2.0087, 2.4954, -0.1391, -1.4249, 1.8464,
    -0.5236, 1.2826, 3.7046, -1.3035, 2.7286, -1.4518, -1.9036, -1.9955,
    -0.0342, -1.0265, -0.7636, 3.0555, 0.0746, -3.0751, -0.107, 1.7376,
    -1.0914, -1.9435, -0.2784, -1.368, 0.4809, -0.4433, 0.3764, 0.5729,
    -2.0595, 1.096, -1.326, -2.0211, -5.0179, 0.5275, 4.0162, 1.8505,
    0.3026, 1.9373, 1.4937, 0.2632, 0.5547, -1.7121, -0.1566, 0.0304,
)
LATENTS_STD: tuple[float, ...] = (
    3.2001, 3.2936, 3.4321, 3.0091, 3.106, 4.0379, 4.0705, 3.791,
    3.0785, 3.65, 3.9308, 3.0904, 2.8778, 3.7675, 3.732, 5.0756,
    3.2864, 4.0397, 3.1317, 4.0443, 2.9249, 3.9454, 3.0988, 4.2489,
    3.4896, 3.8513, 3.9323, 3.4719, 3.7498, 4.283, 3.5694, 4.2467,
    3.9037, 3.2947, 5.077, 3.5075, 3.27, 3.4767, 2.8063, 5.1125,
    3.532, 4.7833, 3.1284, 4.181, 3.8527, 3.8317, 3.5603, 4.3867,
    3.9624, 4.0168, 3.5643, 4.055, 5.5614, 4.2963, 4.44, 3.4957,
    3.8747, 3.7608, 3.5735, 3.149, 3.7662, 3.6746, 3.4563, 3.8161,
)


class Qwen21Vae:
    """Wraps the decoder with latent de-normalization (mean/std un-shift)."""

    def __init__(self) -> None:
        self.decoder = Qwen21VaeDecoder()
        # Checkpoint top-level `conv2` (64->64 1x1); mflux decode order is
        # unnormalize -> post_quant_conv -> decoder -> [:, :3].
        self.post_quant_conv = nn.Conv2d(LATENT_CHANNELS, LATENT_CHANNELS, 1)

    def decode(self, latents: mx.array) -> mx.array:
        if latents.ndim == 5:
            if latents.shape[2] != 1:
                raise ValueError("Qwen-Image-2.1 supports single-frame images only.")
            latents = latents[:, :, 0]
        mean = mx.array(LATENTS_MEAN, dtype=mx.float32).reshape(1, LATENT_CHANNELS, 1, 1)
        std = mx.array(LATENTS_STD, dtype=mx.float32).reshape(1, LATENT_CHANNELS, 1, 1)
        latents = latents * std + mean
        latents = conv_nchw(self.post_quant_conv, latents)
        return self.decoder(latents)