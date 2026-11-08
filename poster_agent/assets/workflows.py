"""ComfyUI API graphs for the two models on the GPU PC (24 GB: they run one at a time, never together).

Ideogram 4.0: structured-JSON prompt -> dual-model guidance (conditional + unconditional UNets), Qwen3-VL text encoder.
Qwen-Image 2.1: plain text-to-image; cutouts are produced by generating the subject on a flat ground and matting it locally.
"""
from __future__ import annotations

import json

from .prompts import caption_text

IDEOGRAM_PRESETS = {  # ComfyUI's Ideogram 4 template presets
    "quality": {"steps": 48, "mu": 0.0, "std": 1.5},
    "default": {"steps": 20, "mu": 0.0, "std": 1.75},
    "turbo": {"steps": 12, "mu": 0.5, "std": 1.75},
}

QWEN_BG_NEGATIVE = "text, letters, words, numbers, watermark, logo, signature, people, faces, blurry, low quality"  # a scene, not a cutout
QWEN_NEGATIVE = ("blurry, low quality, noise, text, letters, watermark, logo, frame, border, cropped, cut off, "
                 "multiple subjects, busy background, floor, shadow on the ground, hands holding it")


def round16(v: float) -> int:
    return max(256, int(round(v / 16.0)) * 16)


def ideogram_workflow(prompt: dict | str, width: int, height: int, seed: int, preset: str = "default",
                      prefix: str = "poster_bg") -> tuple[dict, str]:
    """Returns (workflow, save_node_id)."""
    p = IDEOGRAM_PRESETS[preset]
    w, h = round16(width), round16(height)
    text = prompt if isinstance(prompt, str) else caption_text(prompt)  # compact, literal unicode: the format it was trained on
    wf = {
        "unet": {"class_type": "UNETLoader", "inputs": {"unet_name": "ideogram4_fp8_scaled.safetensors", "weight_dtype": "default"}},
        "unet_neg": {"class_type": "UNETLoader",
                     "inputs": {"unet_name": "ideogram4_unconditional_fp8_scaled.safetensors", "weight_dtype": "default"}},
        "clip": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen3vl_8b_fp8_scaled.safetensors", "type": "ideogram4",
                                                         "device": "default"}},
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": "flux2-vae.safetensors"}},
        "pos": {"class_type": "CLIPTextEncode", "inputs": {"text": text, "clip": ["clip", 0]}},
        "neg": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["pos", 0]}},
        "latent": {"class_type": "EmptyFlux2LatentImage", "inputs": {"width": w, "height": h, "batch_size": 1}},
        "sigmas": {"class_type": "Ideogram4Scheduler",
                   "inputs": {"steps": p["steps"], "width": w, "height": h, "mu": p["mu"], "std": p["std"]}},
        "cfg_override": {"class_type": "CFGOverride",
                         "inputs": {"cfg": 3.0, "start_percent": 0.7, "end_percent": 1.0, "model": ["unet", 0]}},
        "guider": {"class_type": "DualModelGuider",
                   "inputs": {"cfg": 7.0, "model": ["cfg_override", 0], "positive": ["pos", 0],
                              "model_negative": ["unet_neg", 0], "negative": ["neg", 0]}},
        "sampler": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
        "noise": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "sample": {"class_type": "SamplerCustomAdvanced",
                   "inputs": {"noise": ["noise", 0], "guider": ["guider", 0], "sampler": ["sampler", 0],
                              "sigmas": ["sigmas", 0], "latent_image": ["latent", 0]}},
        "decode": {"class_type": "VAEDecode", "inputs": {"samples": ["sample", 0], "vae": ["vae", 0]}},
        "save": {"class_type": "SaveImage", "inputs": {"filename_prefix": prefix, "images": ["decode", 0]}},
    }
    return wf, "save"


def flux_workflow(prompt: str, width: int, height: int, seed: int, steps: int = 15, prefix: str = "poster_bg_flux") -> tuple[dict, str]:
    """FLUX 3 Klein text-to-image (graph from ComfyUI's own `image_flux3_klein_distilled_t2i` template). Plain-prose prompt;
    cfg 1 (distilled), so there is no negative prompt: describe what you want, never what you don't."""
    wf = {
        "unet": {"class_type": "UNETLoader", "inputs": {"unet_name": "flux3-image-klein-int8-convrot.safetensors", "weight_dtype": "default"}},
        "clip": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen3vl_4b_int8_convrot.safetensors", "type": "flux3", "device": "default"}},
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": "flux2-vae.safetensors"}},
        "pos": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["clip", 0]}},
        "neg": {"class_type": "CLIPTextEncode", "inputs": {"text": "", "clip": ["clip", 0]}},
        "latent": {"class_type": "EmptyFlux2LatentImage", "inputs": {"width": round16(width), "height": round16(height), "batch_size": 1}},
        "sample": {"class_type": "KSampler", "inputs": {"seed": seed, "steps": steps, "cfg": 1.0, "sampler_name": "er_sde",
                                                       "scheduler": "simple", "denoise": 1.0, "model": ["unet", 0],
                                                       "positive": ["pos", 0], "negative": ["neg", 0], "latent_image": ["latent", 0]}},
        "decode": {"class_type": "VAEDecode", "inputs": {"samples": ["sample", 0], "vae": ["vae", 0]}},
        "save": {"class_type": "SaveImage", "inputs": {"filename_prefix": prefix, "images": ["decode", 0]}},
    }
    return wf, "save"


def qwen_workflow(prompt: str, width: int, height: int, seed: int, steps: int = 20, cfg: float = 4.5,
                  negative: str = QWEN_NEGATIVE, prefix: str = "poster_hero") -> tuple[dict, str]:
    wf = {
        "unet": {"class_type": "UNETLoader", "inputs": {"unet_name": "qwen_image_2.1_int8_convrot.safetensors", "weight_dtype": "default"}},
        "clip": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen3vl_8b_fp8_scaled.safetensors", "type": "qwen_image"}},
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": "qwen_image_2.1_vae_bf16.safetensors"}},
        "pos": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["clip", 0]}},
        "neg": {"class_type": "CLIPTextEncode", "inputs": {"text": negative, "clip": ["clip", 0]}},
        "latent": {"class_type": "EmptySD3LatentImage", "inputs": {"width": round16(width), "height": round16(height), "batch_size": 1}},
        "sample": {"class_type": "KSampler",
                   "inputs": {"seed": seed, "steps": steps, "cfg": cfg, "sampler_name": "euler", "scheduler": "simple",
                              "denoise": 1.0, "model": ["unet", 0], "positive": ["pos", 0], "negative": ["neg", 0],
                              "latent_image": ["latent", 0]}},
        "decode": {"class_type": "VAEDecode", "inputs": {"samples": ["sample", 0], "vae": ["vae", 0]}},
        "save": {"class_type": "SaveImage", "inputs": {"filename_prefix": prefix, "images": ["decode", 0]}},
    }
    return wf, "save"
