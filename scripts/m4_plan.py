"""M4: brief -> LLM plan -> layout -> render. Usage: python scripts/m4_plan.py <briefs/dir> [example names...]"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

from poster_agent.compile.compiler import compile_spec
from poster_agent.compile.vc_client import ROOT, VectorCraftSession
from poster_agent.ingest import load_brief, scan_assets
from poster_agent.llm import default_llm
from poster_agent.plan import plan_poster
from poster_agent.plan.examples import example_plans
from poster_agent.spec.brand import load_brand


async def main(folder: Path, example_names: list[str]) -> None:
    brief = load_brief(folder / "brief.yaml")
    brand = load_brand(brief.brand)
    inventory = scan_assets(folder / "assets")
    llm = default_llm()
    res = plan_poster(brief, brand, inventory, llm, examples=example_plans(example_names))
    out = ROOT / "runs" / f"m4_{folder.name}"
    out.mkdir(parents=True, exist_ok=True)
    for r in res.rounds:
        print(f"round {r.round}: {r.stage} ({r.llm})", *[f"\n   - {p}" for p in r.problems])
    (out / "plan.json").write_text(json.dumps(res.plan, indent=1, ensure_ascii=False), encoding="utf-8")
    if not res.ok:
        print("PLAN FAILED:", *[f"\n   - {p}" for p in res.problems])
        return
    (out / "spec.json").write_text(res.spec.model_dump_json(indent=1), encoding="utf-8")
    async with VectorCraftSession() as vc:
        r = await compile_spec(res.spec, vc, out, name=folder.name)
    print("OK  scale=%.2f  png=%s  overflow=%s  warnings=%s" % (res.layout.scale, r.png, r.overflow, r.warnings))


if __name__ == "__main__":
    asyncio.run(main(Path(sys.argv[1]).resolve(), sys.argv[2:] or ["engineers_day", "industry_conclave"]))
