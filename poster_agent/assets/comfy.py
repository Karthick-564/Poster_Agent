"""Minimal ComfyUI HTTP client (queue a workflow, wait, download the result). The GPU PC runs ComfyUI; we only talk to its API."""
from __future__ import annotations

import time
import uuid

import httpx

from poster_agent.llm.base import env_value


class ComfyError(RuntimeError):
    pass


class ComfyClient:
    def __init__(self, base_url: str | None = None, timeout: float = 900.0):
        self.base = (base_url or env_value("COMFYUI_URL")).rstrip("/")
        self.timeout = timeout
        self.client_id = uuid.uuid4().hex
        self.http = httpx.Client(timeout=60.0)

    def stats(self) -> dict:
        r = self.http.get(f"{self.base}/system_stats")
        r.raise_for_status()
        return r.json()

    def free_vram(self) -> None:
        """Unload models so the next (different) model gets the whole 24 GB."""
        self.http.post(f"{self.base}/free", json={"unload_models": True, "free_memory": True})

    def queue(self, workflow: dict) -> str:
        r = self.http.post(f"{self.base}/prompt", json={"prompt": workflow, "client_id": self.client_id})
        if r.status_code != 200:
            raise ComfyError(f"ComfyUI rejected the workflow ({r.status_code}): {r.text[:1200]}")
        return r.json()["prompt_id"]

    def wait(self, prompt_id: str, poll: float = 1.0) -> dict:
        deadline = time.time() + self.timeout
        while time.time() < deadline:
            r = self.http.get(f"{self.base}/history/{prompt_id}")
            r.raise_for_status()
            hist = r.json().get(prompt_id)
            if hist:
                status = hist.get("status", {})
                if status.get("status_str") == "error":
                    msgs = [m for m in status.get("messages", []) if m and m[0] == "execution_error"]
                    raise ComfyError(f"ComfyUI execution failed: {msgs[-1][1] if msgs else status}")
                if hist.get("outputs") or status.get("completed"):
                    return hist
            time.sleep(poll)
        raise ComfyError(f"timed out after {self.timeout:.0f}s waiting for {prompt_id}")

    def fetch(self, filename: str, subfolder: str = "", folder_type: str = "output") -> bytes:
        r = self.http.get(f"{self.base}/view", params={"filename": filename, "subfolder": subfolder, "type": folder_type})
        r.raise_for_status()
        return r.content

    def run(self, workflow: dict, save_node: str) -> bytes:
        """Queue, wait, and return the first image saved by `save_node`."""
        hist = self.wait(self.queue(workflow))
        images = hist.get("outputs", {}).get(save_node, {}).get("images", [])
        if not images:
            raise ComfyError(f"workflow produced no image at node {save_node}: {str(hist.get('outputs'))[:300]}")
        i = images[0]
        return self.fetch(i["filename"], i.get("subfolder", ""), i.get("type", "output"))
