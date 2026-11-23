"""Weight loading for the pure-MLX qwen21 engine (no mflux).

- MLX-8bit transformer: mx.quantized uint32 weights -> mx.dequantize parity
  (verified bitwise: codes uint8, out = code*scale.repeat(64,1)+bias.repeat(64,1)).
- Text encoder: BF16 Qwen3-VL tower; drop model.visual.*/lm_head.*, strip the
  model.language_model. prefix.
- VAE: abenzerps flat naming -> decoder modules; conv weights are 5-D
  [out, T, kH, kW, in] (squeeze T); gamma tensors are 4-D (squeeze to 1-D).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import mlx.core as mx

from exo.worker.engines.image.models.qwen21 import vae as vae_mod
from exo.worker.engines.image.models.qwen21.text_encoder import Qwen21TextEncoder
from exo.worker.engines.image.models.qwen21.transformer import (
    HIDDEN_DIM,
    Qwen21Transformer,
)

# abenzerps flat filenames inside the model repo
TRANSFORMER_FILE = "qwen-image-2.1-turbo-MLX-8bit.safetensors"
TEXT_ENCODER_FILE = "text_encoders/qwen3vl_8b_bf16.safetensors"
VAE_FILE = "vae/qwen_image_2.1_vae_bf16.safetensors"


def _load_safetensors(path: Path) -> dict[str, mx.array]:
    return dict(mx.load(str(path)))


def _dequantize_8bit(weights: dict[str, mx.array], prefix: str) -> dict[str, mx.array]:
    """Expand '<prefix>.weight/scales/biases' triplets into dequantized weights."""
    out: dict[str, mx.array] = {}
    for key in [k for k in weights if k.startswith(prefix) and k.endswith(".weight")]:
        base = key[: -len(".weight")]
        scales = weights.get(f"{base}.scales")
        biases = weights.get(f"{base}.biases")
        if scales is None or biases is None:
            out[key] = weights[key]
            continue
        out[key] = mx.dequantize(weights[key], scales, biases, group_size=64, bits=8)
    return out


def load_transformer(model_dir: Path) -> Qwen21Transformer:
    """Load the 7.56 GB MLX-8bit transformer (key map verified vs checkpoint)."""
    data = _load_safetensors(model_dir / TRANSFORMER_FILE)
    data = _dequantize_8bit(data, "")
    model = Qwen21Transformer()
    params: dict[str, Any] = {}

    # MLX 0.32 load_weights(strict=True) matches FULL leaf keys ("img_in.weight"),
    # so every entry below is the complete parameter path.

    # time_text_embed: weightless sinusoid + bias-free embedder (256 -> 4096 -> 4096)
    params["time_text_embed.timestep_embedder.linear_1.weight"] = data[
        "time_text_embed.timestep_embedder.linear_1.weight"
    ]
    params["time_text_embed.timestep_embedder.linear_2.weight"] = data[
        "time_text_embed.timestep_embedder.linear_2.weight"
    ]
    # zero-centered checkpoint weight; the module adds +1 at runtime
    params["txt_in.text_norm.weight"] = data["txt_in.text_norm.weight"]
    params["txt_in.in_layer.weight"] = data["txt_in.in_layer.weight"]
    params["txt_in.out_layer.weight"] = data["txt_in.out_layer.weight"]
    params["img_in.weight"] = data["img_in.weight"]
    # checkpoint key modulation.1.* (Sequential(SiLU, Linear)) -> our modulation Linear;
    # SiLU is applied explicitly in modulation_for()
    params["modulation.weight"] = data["modulation.1.weight"]
    params["norm_out.linear.weight"] = data["norm_out.linear.weight"]
    params["proj_out.weight"] = data["proj_out.weight"]

    n_layers = len(model.transformer_blocks)
    for i in range(n_layers):
        p = f"transformer_blocks.{i}"
        params[f"{p}.attn.to_q.weight"] = data[f"{p}.attn.to_q.weight"]
        params[f"{p}.attn.to_k.weight"] = data[f"{p}.attn.to_k.weight"]
        params[f"{p}.attn.to_v.weight"] = data[f"{p}.attn.to_v.weight"]
        # Plain RMSNorm weight (mean ~1.14 in the checkpoint); mflux 0.24
        # maps it directly into nn.RMSNorm with no transform.
        params[f"{p}.attn.norm_q.weight"] = data[f"{p}.attn.norm_q.weight"]
        params[f"{p}.attn.norm_k.weight"] = data[f"{p}.attn.norm_k.weight"]
        params[f"{p}.attn.to_out.0.weight"] = data[f"{p}.attn.to_out.0.weight"]
        # checkpoint img_mlp.* -> engine ff.* (SwiGLU)
        params[f"{p}.ff.gate_layer.weight"] = data[f"{p}.img_mlp.gate_layer.weight"]
        params[f"{p}.ff.proj.weight"] = data[f"{p}.img_mlp.proj.weight"]
        params[f"{p}.ff.out.weight"] = data[f"{p}.img_mlp.out.weight"]
        # norm1/norm2 are affine=False LayerNorms: no parameters

    model.load_weights(list(params.items()), strict=True)
    mx.eval(model.parameters())
    return model


def load_text_encoder(model_dir: Path) -> Qwen21TextEncoder:
    """Load the BF16 Qwen3-VL text tower from the abenzerps TE file.

    The checkpoint's keys are bare ``model.*`` (no ``model.language_model.``
    prefix): language-tower layers, embed_tokens and norm load directly;
    ``model.visual.*`` (vision tower) and ``lm_head.weight`` are dropped —
    the engine needs raw hidden states only.
    """
    data = _load_safetensors(model_dir / TEXT_ENCODER_FILE)
    model = Qwen21TextEncoder()
    params: dict[str, mx.array] = {}

    params["model.embed_tokens.weight"] = data["model.embed_tokens.weight"]
    params["model.norm.weight"] = data["model.norm.weight"]
    for i in range(36):
        src = f"model.layers.{i}"
        params[f"{src}.input_layernorm.weight"] = data[f"{src}.input_layernorm.weight"]
        params[f"{src}.post_attention_layernorm.weight"] = data[
            f"{src}.post_attention_layernorm.weight"
        ]
        # checkpoint (and mflux) carry no attention biases
        params[f"{src}.self_attn.q_proj.weight"] = data[
            f"{src}.self_attn.q_proj.weight"
        ]
        params[f"{src}.self_attn.k_proj.weight"] = data[
            f"{src}.self_attn.k_proj.weight"
        ]
        params[f"{src}.self_attn.v_proj.weight"] = data[
            f"{src}.self_attn.v_proj.weight"
        ]
        params[f"{src}.self_attn.o_proj.weight"] = data[
            f"{src}.self_attn.o_proj.weight"
        ]
        params[f"{src}.self_attn.q_norm.weight"] = data[
            f"{src}.self_attn.q_norm.weight"
        ]
        params[f"{src}.self_attn.k_norm.weight"] = data[
            f"{src}.self_attn.k_norm.weight"
        ]
        params[f"{src}.mlp.gate_proj.weight"] = data[f"{src}.mlp.gate_proj.weight"]
        params[f"{src}.mlp.up_proj.weight"] = data[f"{src}.mlp.up_proj.weight"]
        params[f"{src}.mlp.down_proj.weight"] = data[f"{src}.mlp.down_proj.weight"]

    model.load_weights(list(params.items()), strict=True)
    mx.eval(model.parameters())
    return model


def _v() -> dict[str, Any]:  # view helper for flat vae entries
    return {}


def load_vae(model_dir: Path) -> vae_mod.Qwen21Vae:
    """Load the BF16 VAE decoder from the abenzerps flat file."""
    data = _load_safetensors(model_dir / VAE_FILE)
    wrapper = vae_mod.Qwen21Vae()
    decoder = wrapper.decoder
    params: dict[str, Any] = {}

    def flat(name: str) -> mx.array:
        return data[name]

    def gamma(name: str) -> mx.array:
        g = data[name]
        return g.reshape(-1)

    def conv_weight(name: str) -> mx.array:
        # Conv weights ship as [out, in, T, kH, kW] (torch conv3d layout, T=1)
        # or [out, in, kH, kW] (torch conv2d). Convert to the MLX Conv2d
        # layout [out, kH, kW, in].
        w = data[name]
        if w.ndim == 5:  # [out, in, T, kH, kW] -> [out, in, kH, kW]
            w = w.reshape(w.shape[0], w.shape[1], w.shape[3], w.shape[4])
        return w.transpose(0, 2, 3, 1)  # [out, kH, kW, in]

    # decoder.conv1 (flat) -> conv_in
    params["decoder.conv_in"] = {
        "weight": conv_weight("decoder.conv1.weight"),
        "bias": flat("decoder.conv1.bias"),
    }
    # mid block
    params["decoder.mid_block.resnets.0.norm1"] = {
        "gamma": gamma("decoder.middle.0.residual.0.gamma")
    }
    params["decoder.mid_block.resnets.0.conv1"] = {
        "weight": conv_weight("decoder.middle.0.residual.2.weight"),
        "bias": flat("decoder.middle.0.residual.2.bias"),
    }
    params["decoder.mid_block.resnets.0.norm2"] = {
        "gamma": gamma("decoder.middle.0.residual.3.gamma")
    }
    params["decoder.mid_block.resnets.0.conv2"] = {
        "weight": conv_weight("decoder.middle.0.residual.6.weight"),
        "bias": flat("decoder.middle.0.residual.6.bias"),
    }
    params["decoder.mid_block.attentions.0.norm"] = {
        "gamma": gamma("decoder.middle.1.norm.gamma")
    }
    params["decoder.mid_block.attentions.0.to_qkv"] = {
        "weight": conv_weight("decoder.middle.1.to_qkv.weight"),
        "bias": flat("decoder.middle.1.to_qkv.bias"),
    }
    params["decoder.mid_block.attentions.0.proj"] = {
        "weight": conv_weight("decoder.middle.1.proj.weight"),
        "bias": flat("decoder.middle.1.proj.bias"),
    }
    params["decoder.mid_block.resnets.1.norm1"] = {
        "gamma": gamma("decoder.middle.2.residual.0.gamma")
    }
    params["decoder.mid_block.resnets.1.conv1"] = {
        "weight": conv_weight("decoder.middle.2.residual.2.weight"),
        "bias": flat("decoder.middle.2.residual.2.bias"),
    }
    params["decoder.mid_block.resnets.1.norm2"] = {
        "gamma": gamma("decoder.middle.2.residual.3.gamma")
    }
    params["decoder.mid_block.resnets.1.conv2"] = {
        "weight": conv_weight("decoder.middle.2.residual.6.weight"),
        "bias": flat("decoder.middle.2.residual.6.bias"),
    }
    # up blocks: decoder.upsamples.{0..4}.upsamples.{0..3}
    n_up = len(decoder.up_blocks)
    for n in range(n_up):
        prefix = f"decoder.upsamples.{n}.upsamples"
        block = decoder.up_blocks[n]
        for m in range(len(block.resnets)):
            src = f"{prefix}.{m}"
            dst = f"decoder.up_blocks.{n}.resnets.{m}"
            params[f"{dst}.norm1"] = {"gamma": gamma(f"{src}.residual.0.gamma")}
            params[f"{dst}.conv1"] = {
                "weight": conv_weight(f"{src}.residual.2.weight"),
                "bias": flat(f"{src}.residual.2.bias"),
            }
            params[f"{dst}.norm2"] = {"gamma": gamma(f"{src}.residual.3.gamma")}
            params[f"{dst}.conv2"] = {
                "weight": conv_weight(f"{src}.residual.6.weight"),
                "bias": flat(f"{src}.residual.6.bias"),
            }
            shortcut_w = data.get(f"{prefix}.{m}.shortcut.weight")
            if shortcut_w is not None:
                params[f"{dst}.conv_shortcut"] = {
                    "weight": conv_weight(f"{prefix}.{m}.shortcut.weight"),
                    "bias": flat(f"{prefix}.{m}.shortcut.bias"),
                }
        if block.upsampler is not None:
            src = f"{prefix}.3"
            params[f"decoder.up_blocks.{n}.upsampler.resample.1"] = {
                "weight": conv_weight(f"{src}.resample.1.weight"),
                "bias": flat(f"{src}.resample.1.bias"),
            }
    # post_quant_conv: checkpoint ships it as top-level `conv2` (quant_conv is
    # the encoder-side `conv1`; the decoder-only engine never uses it).
    params["post_quant_conv"] = {
        "weight": conv_weight("conv2.weight"),
        "bias": flat("conv2.bias"),
    }
    params["decoder.norm_out"] = {"gamma": gamma("decoder.head.0.gamma")}
    params["decoder.conv_out"] = {
        "weight": conv_weight("decoder.head.2.weight"),
        "bias": flat("decoder.head.2.bias"),
    }

    # strict load_weights compares LEAF keys against the decoder's own scope
    # (no "decoder." prefix); post_quant_conv sits outside the decoder.
    decoder_params: list[tuple[str, mx.array]] = []
    post_params: dict[str, mx.array] = {}
    for key, value in params.items():
        if not key.startswith("decoder."):
            post_params.update(value)  # post_quant_conv
            continue
        prefix = key.removeprefix("decoder.")
        if isinstance(value, dict):
            decoder_params.extend(
                (f"{prefix}.{leaf}", leaf_value) for leaf, leaf_value in value.items()
            )
        else:
            decoder_params.append((prefix, value))
    decoder.load_weights(decoder_params, strict=True)
    wrapper.post_quant_conv.load_weights(list(post_params.items()), strict=True)
    mx.eval(decoder.parameters(), wrapper.post_quant_conv.parameters())
    return wrapper


def load_processor_dir(model_dir: Path) -> Path:
    """Resolve the processor dir (abenzerps ships it at <model>/processor)."""
    processor = model_dir / "processor"
    if processor.is_dir():
        return processor
    if (model_dir / "tokenizer_config.json").is_file():
        return model_dir
    raise FileNotFoundError(f"No processor directory under {model_dir}")


def load_tekken_config(model_dir: Path) -> dict[str, Any]:
    cfg = model_dir / "processor" / "tokenizer_config.json"
    if not cfg.is_file():
        cfg = model_dir / "tokenizer_config.json"
    with cfg.open() as f:
        return json.load(f)


def inner_dim() -> int:
    return HIDDEN_DIM
