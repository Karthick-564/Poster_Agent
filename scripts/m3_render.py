"""Render a component-based fixture: layout solve -> compile -> PNG. Usage: python scripts/m3_render.py <fixture>"""
from __future__ import annotations

import asyncio
import importlib
import sys

from poster_agent.compile.compiler import compile_spec
from poster_agent.compile.vc_client import ROOT, VectorCraftSession
from poster_agent.layout.solver import apply_layout
from poster_agent.spec.brand import load_brand


async def main(name: str) -> None:
    mod = importlib.import_module(f"poster_agent.fixtures.{name}")
    spec = mod.build_spec()
    brand = load_brand(spec.brand)
    res = apply_layout(spec, brand)
    print(f"layout: scale={res.scale} fits={res.fits} elements={len(spec.placed)} issues={len(res.issues)}")
    for i in res.issues:
        print("  ISSUE", i.category.value, "|", i.evidence)
    out = ROOT / "runs" / f"m3_{name}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "spec.json").write_text(spec.model_dump_json(indent=1), encoding="utf-8")
    async with VectorCraftSession() as vc:
        r = await compile_spec(spec, vc, out, name=name)
    print("png:", r.png, "| overflow:", r.overflow, "| warnings:", r.warnings)


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1]))
