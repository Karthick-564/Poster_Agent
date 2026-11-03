"""Render a poster built from the structural components (BigTitle, TabBanner, Columns, ImageBlock, PhotoFade, footer band, border)
to see how structure - not ornament - fills a poster. Uses a catalog poster crop as a stand-in photo and a placeholder QR.
Usage: python scripts/structure_demo.py     Output: runs/structure_demo/"""
from __future__ import annotations

import asyncio
import random

from PIL import Image, ImageDraw

from poster_agent.checks import run_checks
from poster_agent.compile.compiler import compile_spec
from poster_agent.compile.vc_client import ROOT, VectorCraftSession
from poster_agent.enrich import default_preset, is_dark_ground
from poster_agent.layout.solver import apply_layout
from poster_agent.spec import AssetKind, AssetRef, BackgroundSpec, CanvasSpec, ComponentSpec, GradientStop, PosterSpec
from poster_agent.spec.brand import load_brand

OUT = ROOT / "runs/structure_demo"
C = ComponentSpec


def make_assets() -> dict[str, AssetRef]:
    OUT.mkdir(parents=True, exist_ok=True)
    photo = OUT / "photo.jpg"  # a text-free stand-in photo: a sunset-lit track with simple silhouettes
    im = Image.new("RGB", (1600, 700))
    d = ImageDraw.Draw(im)
    for yy in range(700):
        k = yy / 700
        d.line([(0, yy), (1600, yy)], fill=(int(60 + 150 * k), int(90 + 70 * k), int(150 - 60 * k)))
    d.rectangle((0, 470, 1600, 700), fill=(48, 52, 60))
    for i, x0 in enumerate((180, 620, 1050)):
        d.polygon([(x0, 470), (x0 + 120, 380), (x0 + 380, 380), (x0 + 470, 470)], fill=(20, 40, 90))
        d.ellipse((x0 + 20, 450, x0 + 120, 550), fill=(15, 15, 18))
        d.ellipse((x0 + 340, 450, x0 + 440, 550), fill=(15, 15, 18))
    im.save(photo, quality=90)
    qr = OUT / "qr.png"
    rnd = random.Random(7)
    n = 25
    img = Image.new("RGB", (n * 8, n * 8), "white")
    for r in range(n):
        for c in range(n):
            if rnd.random() < 0.5:
                img.paste((20, 30, 60), (c * 8, r * 8, c * 8 + 8, r * 8 + 8))
    img.save(qr)
    rel = lambda p: p.relative_to(ROOT).as_posix()
    return {"photo": AssetRef(id="photo", kind=AssetKind.photo, path=rel(photo)),
            "qr": AssetRef(id="qr", kind=AssetKind.photo, path=rel(qr))}


def build_spec() -> PosterSpec:
    return PosterSpec(
        brief_id="structure_demo", brand="kct", canvas=CanvasSpec.from_format("fourfive"), assets=make_assets(),
        background=BackgroundSpec(kind="gradient", angle=90, gradient=[GradientStop(offset=0.0, color="#ffffff"),
                                                                       GradientStop(offset=1.0, color="#eef2f8")]),
        components=[
            C(id="tab", component="TabBanner", anchor="top", content={"text": "Join the force behind the race!"}),
            C(id="logo", component="LogoBlock", content={"logo": "rectangle", "width_pct": 20}),
            C(id="title", component="BigTitle", emphasis="hero", content={
                "lines": ["RECRUITMENT", "DRIVE"], "colors": ["primary", "text"], "pill": "Management", "cap_pct": 9}),
            C(id="intro", component="TextBlock", content={
                "text": "The student team welcomes all interested students to join and pursue excellence, teamwork and growth.",
                "role": "body", "max_lines": 3}),
            C(id="cols", component="Columns", content={"left_pct": 58}, children=[
                C(id="domains_h", component="TextBlock", content={"text": "DOMAINS", "role": "heading", "align": "left", "color": "primary",
                                                                   "font_role": "display", "column": "left"}),
                C(id="domains", component="TextBlock", content={"lines": [
                    "Team/Project Management", "Finance & Sponsorship", "Marketing & Public Relations", "Logistics & Operations",
                    "Human Resources / Team Affairs", "Technical Documentation"], "role": "body", "align": "left", "column": "left"}),
                C(id="scan", component="TextBlock", content={"text": "SCAN TO REGISTER", "role": "heading", "color": "primary",
                                                              "font_role": "display", "column": "right", "max_lines": 2}),
                C(id="qr", component="ImageBlock", content={"asset": "qr", "width_pct": 55, "column": "right"})]),
            C(id="photo", component="PhotoFade", anchor="bottom", content={"asset": "photo", "height_pct": 20}),
            C(id="footer", component="InfoBar", variant="band", anchor="bottom", content={
                "items": ["Last date to register: 20 March 2026 | Queries: +00 00000 00000"], "fill": "primary"}),
        ])


async def main() -> None:
    spec = build_spec()
    brand = load_brand(spec.brand)
    apply_layout(spec, brand)
    spec.decorations = default_preset(spec, brand, is_dark_ground(spec, brand))
    res0 = apply_layout(spec, brand)
    print("fits:", res0.fits, "scale:", res0.scale, [i.evidence for i in res0.issues][:4])
    async with VectorCraftSession() as vc:
        res = await compile_spec(spec, vc, OUT, name="structure_demo")
    rep = run_checks(spec, brand, res, use_ocr=False)
    emp = next(r for r in rep.results if r.name == "emptiness")
    print(f"empty={emp.measured['empty_fraction']:.0%}", "->", res.png)
    for i in rep.issues:
        print(" ", i.severity, i.category.value, i.evidence)


if __name__ == "__main__":
    asyncio.run(main())
