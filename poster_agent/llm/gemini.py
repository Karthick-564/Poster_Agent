"""Gemini adapter over the REST API (text + optional images). Used for planning and, later, the vision critic."""
from __future__ import annotations

import base64
import mimetypes
import time
from pathlib import Path

import httpx

from .base import LLMError, env_value, parse_json_object, retry_delay

API = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
RETRY_STATUS = {429, 500, 502, 503, 504}


class GeminiLLM:
    def __init__(self, model: str = "gemini-flash-latest", timeout: float = 120.0, retries: int = 3):
        self.model, self.timeout, self.retries = model, timeout, retries
        self.name = f"gemini:{model}"

    def generate_json(self, system: str, user: str, *, images: list[Path] | None = None,
                      temperature: float = 0.2) -> dict:
        parts: list[dict] = [{"text": user}]
        for img in images or []:
            mime = mimetypes.guess_type(str(img))[0] or "image/png"
            parts.append({"inlineData": {"mimeType": mime, "data": base64.b64encode(Path(img).read_bytes()).decode()}})
        body = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {"temperature": temperature, "responseMimeType": "application/json"},
        }
        last = ""
        for attempt in range(self.retries):
            try:
                r = httpx.post(API.format(model=self.model), params={"key": env_value("GEMINI_API_KEY")}, json=body,
                               timeout=self.timeout)
            except httpx.HTTPError as e:
                last = f"network error: {type(e).__name__}"
            else:
                if r.is_success:
                    return parse_json_object(_reply_text(r.json()))
                last = f"HTTP {r.status_code}: {r.text[:700]}"
                if r.status_code not in RETRY_STATUS:
                    break
            time.sleep(retry_delay(last) or 2 ** attempt * 2)
        raise LLMError(f"{self.name} failed: {last}")


def _reply_text(data: dict) -> str:
    try:
        parts = data["candidates"][0]["content"]["parts"]
    except (KeyError, IndexError):
        raise LLMError(f"no candidates in reply: {str(data)[:300]}") from None
    text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
    if not text:
        raise LLMError("empty reply")
    return text
