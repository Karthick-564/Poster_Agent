"""Client for a vision-language model served through an OpenAI-compatible `/v1/chat/completions` endpoint (guardrail 1: client only).

Used for the local critic: LiquidAI LFM3-VL-2B running on the GPU PC (default `http://<comfyui-host>:8080/v1`, model id `LFM3-VL-2B`).
The server is not part of this repository. A 2 B model is good at short, closed questions and poor at long JSON, so the critic asks it
narrow yes/no probes (see critic/probes.py); this client therefore tolerates a plain "yes, because ..." reply as well as JSON.
Config (.env): VLM_URL (optional; default derived from COMFYUI_URL on port 8080), VLM_MODEL (default LFM3-VL-2B).
"""
from __future__ import annotations

import base64
import re
import time
from pathlib import Path
from urllib.parse import urlparse

import httpx
from dotenv import dotenv_values
import os

from .base import LLMError, ROOT, parse_json_object

DEFAULT_MODEL = "LFM3-VL-2B"
DEFAULT_PORT = 8080
RETRY_STATUS = {429, 500, 502, 503, 504}
_YES_NO = re.compile(r"\s*\**(yes|no)\b", re.I)


def _env(name: str) -> str:
    return os.environ.get(name) or dotenv_values(ROOT / ".env").get(name) or ""


def resolve_url() -> str | None:
    """VLM_URL if set, else the ComfyUI host on port 8080 (where the local VLM runs), else None."""
    url = _env("VLM_URL")
    if url:
        return url.rstrip("/")
    comfy = _env("COMFYUI_URL")
    host = urlparse(comfy).hostname if comfy else None
    return f"http://{host}:{DEFAULT_PORT}/v1" if host else None


MAX_SIDE = 1280  # a poster is 2160 x 3840 px; a vision model reads it fine at this size and spends a fraction of the image tokens


def _data_url(path: Path, max_side: int = MAX_SIDE) -> str:
    """The image as a data URL, downscaled so its longest side is at most `max_side` px (JPEG; PNG kept when already small)."""
    import io
    from PIL import Image
    with Image.open(path) as im:
        im = im.convert("RGB")
        if max(im.size) <= max_side:
            return "data:image/png;base64," + base64.b64encode(Path(path).read_bytes()).decode("ascii")
        im.thumbnail((max_side, max_side))
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def unwrap_server_placeholder(obj: dict) -> str | None:
    """The reviewer server wraps any NON-JSON model reply in a canned rubric ({"hierarchy": 3, "balance": 3, ..., "raw_critique": <the real
    text>}) and, on an internal error, in a fixed 3/3/3/4/3 rubric. Those numbers are placeholders, not the model's opinion. Returns the
    real model text when it is there, '' for a pure error placeholder, None when `obj` is not such a wrapper."""
    if "raw_critique" in obj:
        return str(obj["raw_critique"])
    fixes = obj.get("fixes")
    if "three_second_test" in obj and isinstance(fixes, list) and any("Review processed with warning" in str(f) for f in fixes):
        return ""
    return None


def coerce_reply(text: str) -> dict:
    """The model's reply as a dict: a JSON object if it gave one, else a leading 'yes'/'no' as {"answer", "reason"}. Server placeholders
    (see unwrap_server_placeholder) are never returned as if they were an answer."""
    try:
        obj = parse_json_object(text)
    except LLMError:
        obj = None
    if obj is not None:
        inner = unwrap_server_placeholder(obj)
        if inner is None:
            return obj
        if not inner:
            raise LLMError("the VLM server reported an internal error (placeholder reply)")
        text = inner  # the model's real, non-JSON words
        try:
            return parse_json_object(text)
        except LLMError:
            pass
    m = _YES_NO.match(text or "")
    if m:
        return {"answer": m.group(1).lower(), "reason": text.strip()[:160]}
    raise LLMError(f"reply is not JSON or yes/no: {(text or '')[:160]!r}")


def free_comfy_gpu() -> bool:
    """Ask ComfyUI (COMFYUI_URL) to unload its models so a local vision model fits on the same GPU; they reload on the next generation."""
    url = _env("COMFYUI_URL")
    if not url:
        return False
    try:
        return httpx.post(f"{url.rstrip('/')}/free", json={"unload_models": True, "free_memory": True}, timeout=20).is_success
    except httpx.HTTPError:
        return False


class OpenAICompatVLM:
    probe_critic = True  # the critic asks this model narrow yes/no probes instead of one open-ended review

    def __init__(self, base_url: str | None = None, model: str | None = None, timeout: float = 120.0, retries: int = 2,
                 max_tokens: int = 256):
        self.base_url = (base_url or resolve_url() or "").rstrip("/")
        self.model = model or _env("VLM_MODEL") or DEFAULT_MODEL
        self.timeout, self.retries, self.max_tokens = timeout, retries, max_tokens
        self.name = f"vlm:{self.model}"

    def available(self, timeout: float = 3.0) -> bool:
        """Is the endpoint answering? (a cheap GET; used to choose the local critic over a cloud one)"""
        if not self.base_url:
            return False
        try:
            return httpx.get(f"{self.base_url}/models", timeout=timeout).is_success
        except httpx.HTTPError:
            return False

    def chat(self, system: str, user: str, *, images: list[Path] | None = None, temperature: float = 0.0) -> str:
        if not self.base_url:
            raise LLMError("no local VLM endpoint: set VLM_URL (or COMFYUI_URL, whose host serves the VLM on port 8080) in .env")
        if self.is_ollama:
            return self._ollama_chat(system, user, images or [], temperature)
        content: list[dict] = [{"type": "image_url", "image_url": {"url": _data_url(p)}} for p in (images or [])]
        content.append({"type": "text", "text": user})
        body = {"model": self.model, "temperature": temperature, "max_tokens": self.max_tokens, "thinking": False,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": content}]}
        last = ""
        for attempt in range(self.retries + 1):
            try:
                r = httpx.post(f"{self.base_url}/chat/completions", json=body, timeout=self.timeout)
            except httpx.HTTPError as e:
                last = f"network error: {type(e).__name__}: {str(e)[:120]}"
            else:
                if r.is_success:
                    return r.json()["choices"][0]["message"]["content"] or ""
                last = f"HTTP {r.status_code}: {r.text[:300]}"
                if r.status_code not in RETRY_STATUS:
                    break
            time.sleep(2 ** attempt)
        raise LLMError(f"{self.name} failed: {last}")

    @property
    def is_ollama(self) -> bool:
        """Ollama (port 11434 or VLM_API=ollama): its native API is used, because thinking models answer there after their reasoning
        (the OpenAI-compatible endpoint spends the whole token budget on hidden reasoning and returns an empty answer)."""
        return _env("VLM_API").lower() == "ollama" or ":11434" in self.base_url

    def _ollama_chat(self, system: str, user: str, images: list[Path], temperature: float) -> str:
        free_comfy_gpu()  # the critic and ComfyUI share one 24 GB card: release ComfyUI's models first
        root = self.base_url[:-3] if self.base_url.endswith("/v1") else self.base_url
        imgs = [_data_url(p).split(",", 1)[1] for p in images]
        body = {"model": self.model, "stream": False, "think": False, "keep_alive": "5m",
                "options": {"temperature": temperature, "num_ctx": 8192, "num_predict": self.max_tokens + 1500},  # room for its reasoning
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user, "images": imgs}]}
        last = ""
        for attempt in range(self.retries + 1):
            try:
                r = httpx.post(f"{root}/api/chat", json=body, timeout=max(self.timeout, 600.0))
            except httpx.HTTPError as e:
                last = f"network error: {type(e).__name__}: {str(e)[:120]}"
            else:
                if r.is_success:
                    return (r.json().get("message") or {}).get("content") or ""
                last = f"HTTP {r.status_code}: {r.text[:300]}"
                if r.status_code not in RETRY_STATUS:
                    break
            time.sleep(2 ** attempt)
        raise LLMError(f"{self.name} failed: {last}")

    def generate_json(self, system: str, user: str, *, images: list[Path] | None = None, temperature: float = 0.0) -> dict:
        return coerce_reply(self.chat(system, user, images=images, temperature=temperature))

    def ask_text(self, system: str, user: str, *, images: list[Path] | None = None, temperature: float = 0.0) -> str:
        """The model's own words for an open question (server placeholders removed)."""
        raw = self.chat(system, user, images=images, temperature=temperature)
        try:
            obj = parse_json_object(raw)
        except LLMError:
            return raw.strip()
        inner = unwrap_server_placeholder(obj)
        if inner is None:
            return raw.strip()
        if not inner:
            raise LLMError("the VLM server reported an internal error (placeholder reply)")
        return inner.strip()
