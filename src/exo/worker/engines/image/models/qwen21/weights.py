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
from mlx import nn

from exo.worker.engines.image.models.qwen21 import vae as vae_mod
from exo.worker.engines.image.models.qwen21.transformer import (
    HIDDEN_DIM,
    MOD_PER_TOKEN,
    PROJECTED_DIM,
    Qwen21Transformer,
)
from exo.worker.engines.image.models.qwen21.text_encoder import Qwen21TextEncoder

# abenzerps flat filenames inside the model repo
TRANSFORMER_FILE = "qwen-image-2.1-turbo-MLX-8bit.safetensors"
TEXT_ENCODER_FILE = "text_encoders/qwen3vl_8b_bf16.safetensors"
VAE_FILE = "vae/qwen_image_2.1_vae_bf16.safetensors"


def _load_safetensors(path: Path) -> dict[str, mx.array]:
    return dict(mx.load(str(path)))


def _dequantize_8bit(
    weights: dict[str, mx.array], prefix: str
) -> dict[str, mx.array]:
    """Expand '<prefix>.weight/scales/biases' triplets into dequantized weights."""
    out: dict[str, mx.array] = {}
    for key in [k for k in weights if k.startswith(prefix) and k.endswith(".weight")]:
        base = key[: -len(".weight")]
        scales = weights.get(f"{base}.scales")
        biases = weights.get(f"{base}.biases")
        if scales is None or biases is None:
            out[key] = weights[key]
            continue
        out[key] = mx.dequantize(
            weights[key], scales, biases, group_size=64, bits=8
        )
    return out


def load_transformer(model_dir: Path) -> Qwen21Transformer:
    """Load the 7.56 GB MLX-8bit transformer with the modulation rename."""
    data = _load_safetensors(model_dir / TRANSFORMER_FILE)
    data = _dequantize_8bit(data, "")
    model = Qwen21Transformer()
    params: dict[str, Any] = {}

    params["time_text_embed.time_embedder.layers.0"] = {
        "weight": data["time_text_embed.timestep_embedder.linear_1.weight"],
        "bias": data["time_text_embed.timestep_embedder.linear_1.bias"],
    }
    params["time_text_embed.time_embedder.layers.2"] = {
        "weight": data["time_text_embed.timestep_embedder.linear_2.weight"],
        "bias": data["time_text_embed.timestep_embedder.linear_2.bias"],
    }
    params["time_text_embed.time_proj"] = {
        "exponent": model.time_text_embed.time_proj.exponent
    }
    params["txt_in.text_norm"] = {
        "weight": data["txt_in.text_norm.weight"] + 1
    }
    params["txt_in.in_layer"] = {
        "weight": data["txt_in.in_layer.weight"],
        "bias": data["txt_in.in_layer.bias"],
    }
    params["txt_in.out_layer"] = {
        "weight": data["txt_in.out_layer.weight"],
        "bias": data["txt_in.out_layer.bias"],
    }
    params["img_in"] = {
        "weight": data["img_in.weight"],
        "bias": data["img_in.bias"],
    }
    params["modulation"] = {
        "weight": data["modulation.layers.1.weight"].reshape(MOD_PER_TOKEN, PROJECTED_DIM),
        "bias": data["modulation.layers.1.bias"].reshape(MOD_PER_TOKEN),
    }
    params["norm_out.linear"] = {
        "weight": data["norm_out.linear.weight"],
        "bias": data["norm_out.linear.bias"],
    }
    params["proj_out"] = {
        "weight": data["proj_out.weight"],
        "bias": data["proj_out.bias"],
    }

    n_layers = len(model.transformer_blocks)
    for i in range(n_layers):
        p = f"transformer_blocks.{i}"
        f = f"transformer_blocks.{i}"
        params[f"{p}.norm1"] = {}  # affine=False LayerNorm: no params
        params[f"{p}.attn.to_q"] = {
            "weight": data[f"{f}.attn.to_q.weight"]
        }
        params[f"{p}.attn.to_k"] = {
            "weight": data[f"{f}.attn.to_k.weight"]
        }
        params[f"{p}.attn.to_v"] = {
            "weight": data[f"{f}.attn.to_v.weight"]
        }
        params[f"{p}.attn.norm_q"] = {
            "weight": data[f"{f}.attn.norm_q.weight"] + 1
        }
        params[f"{p}.attn.norm_k"] = {
            "weight": data[f"{f}.attn.norm_k.weight"] + 1
        }
        params[f"{p}.attn.to_out.0"] = {
            "weight": data[f"{f}.attn.to_out.0.weight"]
        }
        params[f"{p}.ff.gate_proj"] = {
            "weight": data[f"{f}.ff.gate_proj.weight"]
        }
        params[f"{p}.ff.up_proj"] = {
            "weight": data[f"{f}.ff.up_proj.weight"]
        }
        params[f"{p}.ff.down_proj"] = {
            "weight": data[f"{f}.ff.down_proj.weight"]
        }
        params[f"{p}.norm2"] = {}

    model.load_weights(list(params.items()), strict=True)
    mx.eval(model.parameters())
    return model


def load_text_encoder(model_dir: Path) -> Qwen21TextEncoder:
    """Load the BF16 Qwen3-VL text tower from the abenzerps TE file."""
    data = _load_safetensors(model_dir / TEXT_ENCODER_FILE)
    model = Qwen21TextEncoder()
    params: dict[str, Any] = {}

    params["model.embed_tokens"] = {"weight": data["model.language_model.embed_tokens.weight"]}
    params["model.norm"] = {"weight": data["model.language_model.norm.weight"]}
    for i in range(36):
        src = f"model.language_model.layers.{i}"
        dst = f"model.layers.{i}"
        params[f"{dst}.input_layernorm"] = {"weight": data[f"{src}.input_layernorm.weight"]}
        params[f"{dst}.post_attention_layernorm"] = {
            "weight": data[f"{src}.post_attention_layernorm.weight"]
        }
        params[f"{dst}.self_attn.q_proj"] = {
            "weight": data[f"{src}.self_attn.q_proj.weight"],
            "bias": data[f"{src}.self_attn.q_proj.bias"],
        }
        params[f"{dst}.self_attn.k_proj"] = {
            "weight": data[f"{src}.self_attn.k_proj.weight"],
            "bias": data[f"{src}.self_attn.k_proj.bias"],
        }
        params[f"{dst}.self_attn.v_proj"] = {
            "weight": data[f"{src}.self_attn.v_proj.weight"],
            "bias": data[f"{src}.self_attn.v_proj.bias"],
        }
        params[f"{dst}.self_attn.o_proj"] = {
            "weight": data[f"{src}.self_attn.o_proj.weight"]
        }
        params[f"{dst}.self_attn.q_norm"] = {"weight": data[f"{src}.self_attn.q_norm.weight"]}
        params[f"{dst}.self_attn.k_norm"] = {"weight": data[f"{src}.self_attn.k_norm.weight"]}
        params[f"{dst}.mlp.gate_proj"] = {"weight": data[f"{src}.mlp.gate_proj.weight"]}
        params[f"{dst}.mlp.up_proj"] = {"weight": data[f"{src}.mlp.up_proj.weight"]}
        params[f"{dst}.mlp.down_proj"] = {"weight": data[f"{src}.mlp.down_proj.weight"]}

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
        w = data[name]
        if w.ndim == 5:  # [out, T, kH, kW, in]
            w = w.reshape(w.shape[0], w.shape[2], w.shape[3], w.shape[4])
        return w

    # decoder.conv1 (flat) -> conv_in
    params["decoder.conv_in"] = {
        "weight": conv_weight("decoder.conv1.weight"),
        "bias": flat("decoder.conv1.bias"),
    }
    # mid block
    params["decoder.mid_block.resnets.0.norm1"] = {"gamma": gamma("decoder.middle.0.residual.0.gamma")}
    params["decoder.mid_block.resnets.0.conv1"] = {
        "weight": conv_weight("decoder.middle.0.residual.2.weight"),
        "bias": flat("decoder.middle.0.residual.2.bias"),
    }
    params["decoder.mid_block.resnets.0.norm2"] = {"gamma": gamma("decoder.middle.0.residual.3.gamma")}
    params["decoder.mid_block.resnets.0.conv2"] = {
        "weight": conv_weight("decoder.middle.0.residual.6.weight"),
        "bias": flat("decoder.middle.0.residual.6.bias"),
    }
    params["decoder.mid_block.attentions.0.norm"] = {"gamma": gamma("decoder.middle.1.norm.gamma")}
    params["decoder.mid_block.attentions.0.to_qkv"] = {
        "weight": conv_weight("decoder.middle.1.to_qkv.weight"),
        "bias": flat("decoder.middle.1.to_qkv.bias"),
    }
    params["decoder.mid_block.attentions.0.proj"] = {
        "weight": conv_weight("decoder.middle.1.proj.weight"),
        "bias": flat("decoder.middle.1.proj.bias"),
    }
    params["decoder.mid_block.resnets.1.norm1"] = {"gamma": gamma("decoder.middle.2.residual.0.gamma")}
    params["decoder.mid_block.resnets.1.conv1"] = {
        "weight": conv_weight("decoder.middle.2.residual.2.weight"),
        "bias": flat("decoder.middle.2.residual.2.bias"),
    }
    params["decoder.mid_block.resnets.1.norm2"] = {"gamma": gamma("decoder.middle.2.residual.3.gamma")}
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
            params[f"decoder.up_blocks.{n}.upsampler.conv"] = {
                "weight": conv_weight(f"{src}.resample.1.weight"),
                "bias": flat(f"{src}.resample.1.bias"),
            }
    params["decoder.norm_out"] = {"gamma": gamma("decoder.head.0.gamma")}
    params["decoder.conv_out"] = {
        "weight": conv_weight("decoder.head.2.weight"),
        "bias": flat("decoder.head.2.bias"),
    }

    decoder.load_weights(list(params.items()), strict=True)
    mx.eval(decoder.parameters())
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


def zero_center_norm_weight(weight: mx.array) -> mx.array:
    """Checkpoint stores weight-1 (zero-centered); runtime uses weight+1."""
    return weight + 1


def inner_dim() -> int:
    return HIDDEN_DIM