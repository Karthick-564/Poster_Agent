"""Smoke test: drive native vectorcraft-cli MCP (as a client) to make a tiny poster and export PNG/PDF."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "tools/vectorcraft-0.21.0/vectorcraft-0.21.0-windows-x64-portable/vectorcraft-cli.exe"
OUT = ROOT / "runs/smoke"


def _text(result) -> str:
    return "".join(getattr(c, "text", "") for c in result.content)


async def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    params = StdioServerParameters(
        command=str(CLI),
        args=["mcp", "--headless", "--automation-read-root", str(ROOT), "--automation-write-root", str(ROOT)],
    )
    async with stdio_client(params) as (r, w):
        async with ClientSession(r, w) as s:
            await s.initialize()
            res = await s.call_tool("run_command", {"command": "file.new", "params": {"width": 1080, "height": 1350}})
            print("file.new ->", _text(res)[:200])
            res = await s.call_tool("run_command", {"command": "text.create", "params": {
                "text": "ANNUAL DAY 2026", "x": 90, "y": 300,
                "area": {"width": 900, "height": 200, "fit": "shrinkText"},
                "size": 96, "font": "Montserrat", "style": "Bold", "color": "#1E2842", "justify": "center",
            }})
            print("text.create ->", _text(res)[:400])
            for ext in ("png", "pdf"):
                res = await s.call_tool("export", {"path": str(OUT / f"smoke.{ext}")})
                print(f"export {ext} ->", _text(res)[:200])


if __name__ == "__main__":
    asyncio.run(main())
