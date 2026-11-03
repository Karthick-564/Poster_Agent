"""Guardrail 1: this repo is an MCP *client* of VectorCraft's native server, never a server itself."""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCAN_DIRS = ["poster_agent", "scripts"]
FORBIDDEN = [
    re.compile(r"\bmcp\.server\b"),
    re.compile(r"\bFastMCP\b"),
    re.compile(r"""["']jsonrpc["']\s*:\s*["']2\.0["']"""),
    re.compile(r"\bhandle_message\b"),
    re.compile(r"\bserver\.run\("),
]


def _python_files():
    for d in SCAN_DIRS:
        base = ROOT / d
        if base.exists():
            yield from base.rglob("*.py")


def test_no_mcp_server_code():
    offenders = []
    for path in _python_files():
        text = path.read_text(encoding="utf-8", errors="ignore")
        for pat in FORBIDDEN:
            if pat.search(text):
                offenders.append(f"{path.relative_to(ROOT)}: {pat.pattern}")
    assert not offenders, "MCP server code is forbidden (use native vectorcraft-cli mcp):\n" + "\n".join(offenders)


def test_no_server_named_modules():
    bad = [p for p in _python_files() if "server" in p.stem.lower()]
    assert not bad, f"Modules named *server* are forbidden: {[str(p.relative_to(ROOT)) for p in bad]}"
