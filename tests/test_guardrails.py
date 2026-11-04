"""Guardrail 1: VectorCraft is driven only through its NATIVE MCP server; we never re-implement its tools. The single exception (the user
allowed it on 2026-11-04) is a small helper server of MEASURING / CHECKING tools for a model that designs in VectorCraft
(poster_agent/agent_tools/poster_tools.py): it must never expose drawing or document tools."""
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


HELPER_SERVER = "poster_agent/agent_tools/poster_tools.py"  # the one allowed MCP server module
DRAWING_TOOL_NAMES = ["add_text", "create_path", "add_shape", "run_command", "command_batch", "apply_effect", "export_document",
                      "document.", "text.create", "object.", "paint.setFill", "file.place"]


def _python_files():
    for d in SCAN_DIRS:
        base = ROOT / d
        if base.exists():
            yield from base.rglob("*.py")


def test_no_mcp_server_code():
    offenders = []
    for path in _python_files():
        if path.relative_to(ROOT).as_posix() == HELPER_SERVER:
            continue  # checked by test_helper_server_only_measures_and_checks
        text = path.read_text(encoding="utf-8", errors="ignore")
        for pat in FORBIDDEN:
            if pat.search(text):
                offenders.append(f"{path.relative_to(ROOT)}: {pat.pattern}")
    assert not offenders, "MCP server code is forbidden (use native vectorcraft-cli mcp):\n" + "\n".join(offenders)


def test_no_server_named_modules():
    bad = [p for p in _python_files() if "server" in p.stem.lower()]
    assert not bad, f"Modules named *server* are forbidden: {[str(p.relative_to(ROOT)) for p in bad]}"


def test_helper_server_only_measures_and_checks():
    """The allowed MCP server offers no drawing / document tools (VectorCraft's native MCP does the drawing)."""
    text = (ROOT / HELPER_SERVER).read_text(encoding="utf-8")
    tools = re.findall(r"@mcp\.tool\(\)\s+def (\w+)", text)
    assert tools, "helper server defines no tools"
    assert not [t for t in tools if any(t.startswith(n.rstrip(".")) for n in DRAWING_TOOL_NAMES)], tools
    body = chr(10).join(re.findall(r"@mcp\.tool\(\).*?(?=@mcp\.tool|if __name__)", text, flags=re.S))
    assert not [n for n in DRAWING_TOOL_NAMES if n in body], "the helper server must not call or expose drawing tools"
    assert "Image.save" not in text and ".write_text(" not in text, "helper tools must not write files"
