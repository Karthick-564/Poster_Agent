"""Thin async MCP *client* for VectorCraft's native `vectorcraft-cli mcp` server (guardrail 1: client only).

Two ways to reach VectorCraft:
  headless (default)  — a private in-process engine session; file access confined to `roots`.
  watch               — attach to the VectorCraft desktop app (`vectorcraft --control PORT`) so every tool call happens in a window you
                        can watch, and the finished document is left open for hand editing. Opt-in; used rarely (see run_poster --watch).
In watch mode the *app* confines itself (it is started with the same --automation-*-root flags); the CLI server writes nothing itself.
"""
from __future__ import annotations

import asyncio
import json
import os
import socket
import subprocess
import sys
import time
from contextlib import AsyncExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CLI = ROOT / "tools/vectorcraft-0.21.0/vectorcraft-0.21.0-windows-x64-portable/vectorcraft-cli.exe"
APP_START_TIMEOUT_S = 45.0


class VectorCraftError(RuntimeError):
    pass


@dataclass
class WatchConfig:
    port: int = 7979
    pace: float = 0.35  # seconds between visible steps, so a human can follow along
    launch: bool = True  # start the desktop app if nothing is listening on the control port


def cli_path() -> Path:
    return Path(os.environ.get("VECTORCRAFT_CLI") or DEFAULT_CLI)


def app_path(cli: Path | None = None) -> Path:
    """The desktop app shipped next to the CLI."""
    exe = "vectorcraft.exe" if sys.platform == "win32" else "vectorcraft"
    return (cli or cli_path()).with_name(exe)


def session_args(roots: list[Path], watch: WatchConfig | None) -> list[str]:
    """Arguments for `vectorcraft-cli`: attach to the running app (watch) or run a private confined headless engine."""
    if watch:
        return ["mcp", "--connect", f"127.0.0.1:{watch.port}"]
    args = ["mcp", "--headless"]
    for r in roots:
        args += ["--automation-read-root", str(r), "--automation-write-root", str(r)]
    return args


def app_args(roots: list[Path], watch: WatchConfig) -> list[str]:
    """Arguments for the desktop app: open the control port and confine its file access to the project."""
    args = ["--control", str(watch.port)]
    for r in roots:
        args += ["--automation-read-root", str(r), "--automation-write-root", str(r)]
    return args


def port_open(port: int, host: str = "127.0.0.1", timeout: float = 0.5) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def ensure_app(watch: WatchConfig, roots: list[Path], cli: Path | None = None) -> subprocess.Popen | None:
    """Make sure the desktop app is listening on the control port. Returns the process we started, or None if one was already
    running (then its own file-access limits apply, whatever they are). The app is deliberately never stopped by us."""
    if port_open(watch.port):
        return None
    if not watch.launch:
        raise VectorCraftError(f"no VectorCraft app is listening on 127.0.0.1:{watch.port}; start it with `vectorcraft --control {watch.port}`")
    exe = app_path(cli)
    if not exe.exists():
        raise VectorCraftError(f"VectorCraft desktop app not found at {exe}")
    flags = 0
    if sys.platform == "win32":  # keep the window alive after this script ends
        flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
    proc = subprocess.Popen([str(exe), *app_args(roots, watch)], cwd=str(exe.parent), creationflags=flags, close_fds=True,
                            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    deadline = time.time() + APP_START_TIMEOUT_S
    while time.time() < deadline:
        if port_open(watch.port):
            return proc
        if proc.poll() is not None:
            raise VectorCraftError(f"VectorCraft app exited immediately (code {proc.returncode})")
        time.sleep(0.5)
    raise VectorCraftError(
        f"VectorCraft app did not open its control port {watch.port} within {APP_START_TIMEOUT_S:.0f}s. Look at the VectorCraft window: "
        "a start-up dialog (for example 'recover unsaved documents') can hold the app back until you answer it. Close that "
        "window and try again.")


def _parse(result: Any) -> Any:
    text = "".join(getattr(c, "text", "") for c in result.content)
    if getattr(result, "is_error", False) or getattr(result, "isError", False):
        raise VectorCraftError(text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


class VectorCraftSession:
    """One VectorCraft session: headless (default, confined to `roots`) or attached to the desktop app (`watch`)."""

    def __init__(self, roots: list[Path] | None = None, cli: Path | None = None, watch: WatchConfig | None = None):
        self.cli = Path(cli) if cli else cli_path()
        self.roots = [Path(r) for r in (roots or [ROOT])]
        self.watch = watch
        self._stack = AsyncExitStack()
        self.session: ClientSession | None = None

    async def __aenter__(self) -> "VectorCraftSession":
        if self.watch:
            await asyncio.get_running_loop().run_in_executor(None, ensure_app, self.watch, self.roots, self.cli)
        params = StdioServerParameters(command=str(self.cli), args=session_args(self.roots, self.watch))
        read, write = await self._stack.enter_async_context(stdio_client(params))
        self.session = await self._stack.enter_async_context(ClientSession(read, write))
        await self.session.initialize()
        return self

    async def __aexit__(self, *exc) -> None:
        await self._stack.aclose()  # closes our MCP connection only; the desktop app (watch mode) stays open on purpose

    async def call(self, tool: str, args: dict[str, Any] | None = None) -> Any:
        assert self.session is not None, "use `async with VectorCraftSession()`"
        return _parse(await self.session.call_tool(tool, args or {}))

    async def command(self, command_id: str, **params: Any) -> Any:
        return await self.call("run_command", {"command": command_id, "params": params})

    async def batch(self, steps: list[dict[str, Any]], atomic: bool = True) -> dict[str, Any]:
        """Run steps ({"id": cmd, "params": {...}}); "$N.id" in params refers to step N's result."""
        out = await self.call("command_batch", {"steps": steps, "atomic": atomic})
        if isinstance(out, dict) and out.get("failed"):
            raise VectorCraftError(f"batch failed: {json.dumps(out)[:1500]}")
        return out

    async def inspect(self, ids: list[Any] | None = None) -> Any:
        return await self.call("doc_inspect", {"ids": ids} if ids else {})

    async def export(self, path: Path, **options: Any) -> Any:
        return await self.call("export", {"path": str(path), **options})
