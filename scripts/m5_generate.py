"""M5 live check on the GPU PC: one Ideogram background + one Qwen cutout. Usage: python scripts/m5_generate.py [bg|hero|both]"""
from __future__ import annotations

import sys

from poster_agent.assets import ArtDirection, AssetGenerator, ComfyClient
from poster_agent.compile.vc_client import ROOT
from poster_agent.spec import CanvasSpec

OUT = ROOT / "runs/m5_assets"


def main(which: str) -> None:
    comfy = ComfyClient()
    print("ComfyUI up:", comfy.stats()["devices"][0]["name"])
    gen = AssetGenerator(comfy, OUT)
    if which in ("bg", "both"):
        art = ArtDirection(
            description="Soft abstract architectural light, translucent navy and teal glass panels with gentle gradients "
                        "and a faint dotted grid, evoking a modern engineering campus at dawn",
            style="clean minimal geometric, refined, premium institutional", lighting="soft diffused morning light",
            medium="digital illustration", palette=["#1e2842", "#85cebd", "#dde3ed"])
        calm = ("Composition: keep the upper, upper middle, lower middle part of the image calm — smooth, low-detail and "
                "low-contrast with no focal objects — and concentrate the main visual interest in the very bottom part.")
        ref, info = gen.background("bg_demo", art, CanvasSpec.from_format("fourfive"), calm)
        print(f"background: {info.width}x{info.height} seed={info.seed} {info.seconds:.0f}s cached={info.cached} -> {info.path}")
    if which in ("hero", "both"):
        ref, info = gen.hero_cutout("hero_demo", "a sleek modern industrial robotic arm in white and navy with teal accents, "
                                    "three-quarter view", aspect="1:1")
        print(f"hero: {info.width}x{info.height} seed={info.seed} {info.seconds:.0f}s cached={info.cached} -> {info.path}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "both")
