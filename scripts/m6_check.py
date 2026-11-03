"""Render fixture posters and run the pixel checks on them. Usage: python scripts/m6_check.py [fixture ...]"""
from __future__ import annotations

import asyncio
import sys
import time

from poster_agent.checks import run_checks
from poster_agent.compile.compiler import compile_spec
from poster_agent.compile.vc_client import ROOT, VectorCraftSession
from poster_agent.fixtures import engineers_day, industry_conclave, kciiri, kciiri_components
from poster_agent.layout.solver import apply_layout
from poster_agent.spec.brand import load_brand

FIXTURES = {"kciiri_components": kciiri_components, "industry_conclave": industry_conclave, "engineers_day": engineers_day,
            "kciiri_golden": kciiri}


async def main(names: list[str]) -> None:
    for name in names or list(FIXTURES):
        spec = FIXTURES[name].build_spec()
        brand = load_brand(spec.brand)
        if name != "kciiri_golden":
            apply_layout(spec, brand)
        out = ROOT / "runs" / f"m6_{name}"
        async with VectorCraftSession() as vc:
            compiled = await compile_spec(spec, vc, out, name=name)
        t = time.time()
        rep = run_checks(spec, brand, compiled)
        print(f"\n=== {name}  (checks {time.time() - t:.0f}s)  ok={rep.ok}")
        for r in rep.results:
            print(f"  [{'PASS' if r.passed else 'FAIL'}] {r.name}: {r.detail}")
        for i in rep.issues:
            print(f"     - {i.severity.upper()} {i.category.value}: {i.evidence[:210]}")


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1:]))
