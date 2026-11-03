"""Manual smoke test of watch mode against a running VectorCraft app (start it with `python scripts/watch_smoke.py --launch`, or
yourself with `vectorcraft --control 7979 --automation-read-root <proj> --automation-write-root <proj>`).

Draws the decorated KCIIRI fixture step by step in the app window, then captures the *window* (not just the artboard) so you
can confirm the drawing really happened where a person can see it. Output: runs/watch_smoke/."""
from __future__ import annotations

import argparse
import asyncio
import base64

from poster_agent.compile.compiler import compile_spec
from poster_agent.compile.vc_client import ROOT, VectorCraftSession, WatchConfig
from poster_agent.enrich import default_preset, is_dark_ground
from poster_agent.fixtures import kciiri_components
from poster_agent.layout.solver import apply_layout
from poster_agent.spec.brand import load_brand

OUT = ROOT / "runs/watch_smoke"


async def main(port: int, pace: float, launch: bool) -> None:
    spec = kciiri_components.build_spec()
    brand = load_brand(spec.brand)
    apply_layout(spec, brand)
    spec.decorations = default_preset(spec, brand, is_dark_ground(spec, brand))
    apply_layout(spec, brand)
    OUT.mkdir(parents=True, exist_ok=True)
    async with VectorCraftSession(watch=WatchConfig(port=port, pace=pace, launch=launch)) as vc:
        print("attached to the app; drawing step by step ...", flush=True)
        res = await compile_spec(spec, vc, OUT, name="watch_smoke", watched=True, pace=pace, doc_title="watch_smoke - round 0")
        print(f"done: {res.step_count} steps, warnings={res.warnings}", flush=True)
        shot = await vc.session.call_tool("screenshot", {"window": True, "max_side": 1600, "path": str(OUT / "window.png")})
        for part in shot.content:
            if getattr(part, "type", "") == "image":
                (OUT / "window_inline.png").write_bytes(base64.b64decode(part.data))
        print("window captured ->", OUT / "window.png", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=7979)
    ap.add_argument("--pace", type=float, default=0.12)
    ap.add_argument("--launch", action="store_true", help="start the app if nothing is listening")
    a = ap.parse_args()
    asyncio.run(main(a.port, a.pace, a.launch))
