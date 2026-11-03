# ComfyUI environment (GPU PC)

Verified 2026-11-03 against `http://<gpu-pc-ip>:8188` (reachable from the dev PC; started with `--listen 0.0.0.0 --enable-cors-header *`).

| | |
|---|---|
| ComfyUI | 0.43.0, PyTorch 2.11 + cu128, Python 3.13 |
| GPU | NVIDIA RTX 4090, **24 GB VRAM** (~22.5 GB free at idle), 32 GB system RAM |

## Models available (via `/object_info`)
- **UNETLoader:** `ideogram4_fp8_scaled.safetensors`, `ideogram4_unconditional_fp8_scaled.safetensors`,
  `qwen_image_2.1_int8_convrot.safetensors`, `flux3-image-klein-int8-convrot.safetensors`
- **CLIPLoader:** `qwen3vl_8b_fp8_scaled`, `qwen3vl_8b_int8_convrot`, `qwen3vl_4b_int8_convrot`,
  `qwen3.5_9b_qwen_image_2.1_pe_t2i.int8_convrot` (CLIP types include `ideogram4`, `qwen_image`, `flux3`)
- **VAELoader:** `qwen_image_2.1_vae_bf16`, `flux2-vae`
- **Local background removal nodes:** `LoadBackgroundRemovalModel`, `RemoveBackground` (the cutout fallback needs no cloud call),
  plus `SplitImageWithAlpha` / `JoinImageWithAlpha`
- Ideogram/Qwen API-node variants (`IdeogramV4`, `QwenImageTextToImageApi`, ...) exist but call cloud services — not used.

## Constraints for the pipeline
- 24 GB fits **one** diffusion model at a time: run Ideogram background and Qwen hero **sequentially**
  (ComfyUI offloads between them); never queue them as parallel jobs expecting both resident.
- Generate at modest resolution and upscale/crop in the Assets stage (e.g. Ideogram ~896x1120 for 4:5, Qwen ~768-1024 square).
- Persist every seed + prompt on the `AssetRef` so a run is reproducible.
