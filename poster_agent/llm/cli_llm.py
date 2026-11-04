"""Planner adapter that runs each request as a FRESH, tool-less session of a local command-line LLM agent.

Every call starts a new session with only the system and user prompt we pass: no tools, no MCP servers, no setting sources, no saved
session, and an empty working directory, so no project notes or memory reach it. It is an LLM like any other here (a subprocess client;
guardrail 1 is untouched). The executable comes from PLANNER_CLI and the model from PLANNER_MODEL (.env or environment)."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from dotenv import dotenv_values

from .base import ROOT, LLMError, parse_json_object

NEUTRAL_CWD = ROOT / "data" / "cache" / "planner_cwd"  # empty on purpose: the session must see nothing of the project


def _setting(name: str, default: str = "") -> str:
    return os.environ.get(name) or dotenv_values(ROOT / ".env").get(name) or default


def cli_path() -> str | None:
    """The planner CLI executable: the PLANNER_CLI path (or a name found on PATH); None when it is not configured."""
    p = _setting("PLANNER_CLI")
    if p and Path(p).exists():
        return p
    return shutil.which(p) if p else None


class CLIAgentLLM:
    """`generate_json` through one CLI call per request. Text only: requests with images raise LLMError (the critic uses a VLM)."""

    def __init__(self, model: str | None = None, timeout: float = 300.0, retries: int = 2, run=subprocess.run):
        self.model = model or _setting("PLANNER_MODEL", "sonnet")
        self.name = f"cli:{self.model}"
        self.timeout, self.retries, self._run = timeout, retries, run

    def available(self) -> bool:
        return cli_path() is not None

    def command(self, system_file: Path) -> list[str]:
        exe = cli_path()
        if not exe:
            raise LLMError("planner CLI not found (set PLANNER_CLI in .env)")
        return [exe, "-p", "--system-prompt-file", str(system_file), "--tools", "", "--output-format", "json",
                "--no-session-persistence", "--strict-mcp-config", "--setting-sources", "", "--model", self.model]

    def generate_json(self, system: str, user: str, *, images: list[Path] | None = None, temperature: float = 0.2) -> dict:
        if images:
            raise LLMError(f"{self.name} does not take images")
        NEUTRAL_CWD.mkdir(parents=True, exist_ok=True)
        last = ""
        with tempfile.TemporaryDirectory(dir=NEUTRAL_CWD.parent) as tmp:
            sys_file = Path(tmp) / "system.txt"
            sys_file.write_text(system + "\n\nReply with exactly one JSON object and nothing else.", encoding="utf-8")
            for _ in range(self.retries):
                try:
                    proc = self._run(self.command(sys_file), input=user, capture_output=True, text=True, encoding="utf-8",
                                     timeout=self.timeout, cwd=NEUTRAL_CWD)
                except subprocess.TimeoutExpired:
                    last = f"timed out after {self.timeout:.0f}s"
                    continue
                except OSError as e:
                    raise LLMError(f"{self.name} could not start: {e}") from None
                try:
                    envelope = json.loads(proc.stdout or "")
                except json.JSONDecodeError:
                    last = f"exit {proc.returncode}: {(proc.stdout or proc.stderr or '')[:300]}"
                    continue
                if envelope.get("is_error") or envelope.get("subtype") != "success":
                    last = f"session error: {str(envelope.get('result') or envelope.get('subtype'))[:300]}"
                    continue
                try:
                    return parse_json_object(envelope.get("result") or "")
                except LLMError as e:
                    last = str(e)
        raise LLMError(f"{self.name} failed: {last}")
