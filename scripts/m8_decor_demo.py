"""Before/after of the default decorative layer on the plain component posters. Usage: python scripts/m8_decor_demo.py"""
from __future__ import annotations

import asyncio

from PIL import Image

from poster_agent.checks import run_checks
from poster_agent.compile.compiler import compile_spec
from poster_agent.compile.vc_client import ROOT, VectorCraftSession
from poster_agent.enrich import default_preset, is_dark_ground
from poster_agent.fixtures import engineers_day, kciiri_components
from poster_agent.layout.solver import apply_layout
from poster_agent.spec.brand import load_brand

OUT = ROOT / "runs/m8_decor"


async def render(spec, brand, name, vc):
    res = await compile_spec(spec, vc, OUT, name=name)
    rep = run_checks(spec, brand, res, use_ocr=False)
    emp = next(r for r in rep.results if r.name == "emptiness")
    print(f"  {name:28s} scale={spec.canvas.export_scale} empty={emp.measured['empty_fraction']:.0%} "
          f"largest={emp.measured['largest_fraction']:.0%} band={emp.measured['longest_band_fraction']:.1%} "
          f"{'PASS' if emp.passed else 'FAIL'}  ({len(spec.decorations)} decorations, {len(spec.placed)} elements)")
    return res.png


async def main() -> None:
    sheets = []
    async with VectorCraftSession() as vc:
        for label, mod in (("kciiri", kciiri_components), ("engineers", engineers_day)):
            spec = mod.build_spec()
            brand = load_brand(spec.brand)
            apply_layout(spec, brand)
            before = await render(spec, brand, f"{label}_before", vc)
            spec.decorations = default_preset(spec, brand, is_dark_ground(spec, brand))
            apply_layout(spec, brand)
            after = await render(spec, brand, f"{label}_after", vc)
            sheets += [before, after]
    ims = [Image.open(p).convert("RGB") for p in sheets]
    h = 900
    ims = [im.resize((round(im.width * h / im.height), h), Image.LANCZOS) for im in ims]
    sheet = Image.new("RGB", (sum(i.width for i in ims) + 10 * (len(ims) + 1), h + 20), (235, 235, 235))
    x = 10
    for im in ims:
        sheet.paste(im, (x, 10))
        x += im.width + 10
    sheet.save(OUT / "before_after.png")
    print("sheet ->", OUT / "before_after.png", sheet.size)


if __name__ == "__main__":
    asyncio.run(main())
