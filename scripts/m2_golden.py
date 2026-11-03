"""M2 golden: compile the hand-written KCIIRI spec through native VectorCraft MCP."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

from poster_agent.compile.compiler import compile_spec
from poster_agent.compile.vc_client import ROOT, VectorCraftSession
from poster_agent.fixtures.kciiri import allowed_text, build_spec

OUT = ROOT / "runs/m2_kciiri"


async def main() -> None:
    spec = build_spec()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "spec.json").write_text(spec.model_dump_json(indent=1), encoding="utf-8")
    async with VectorCraftSession() as vc:
        res = await compile_spec(spec, vc, OUT, allowed_text=allowed_text(), name="kciiri")
    print("steps:", res.step_count, "| png:", res.png, "| overflow:", res.overflow)
    print("warnings:", res.warnings)
    for k, b in res.bounds.items():
        print(f"  {k:14s} x={b.x:7.1f} y={b.y:7.1f} w={b.w:7.1f} h={b.h:7.1f}")
    (OUT / "bounds.json").write_text(json.dumps({k: b.model_dump() for k, b in res.bounds.items()}, indent=1))


if __name__ == "__main__":
    asyncio.run(main())
