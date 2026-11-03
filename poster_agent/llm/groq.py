"""Groq adapter (OpenAI-compatible chat API). Text only — a fast fallback planner, not for the vision critic."""
from __future__ import annotations

import time
from pathlib import Path

import httpx

from .base import LLMError, env_value, parse_json_object, retry_delay

API = "https://api.groq.com/openai/v1/chat/completions"
RETRY_STATUS = {429, 500, 502, 503, 504}


class GroqLLM:
    def __init__(self, model: str = "openai/gpt-oss-120b", timeout: float = 120.0, retries: int = 3):
        self.model, self.timeout, self.retries = model, timeout, retries
        self.name = f"groq:{model}"

    def generate_json(self, system: str, user: str, *, images: list[Path] | None = None,
                      temperature: float = 0.2) -> dict:
        if images:
            raise LLMError("GroqLLM does not support images")
        body = {"model": self.model, "temperature": temperature, "response_format": {"type": "json_object"},
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        last = ""
        for attempt in range(self.retries):
            try:
                r = httpx.post(API, headers={"Authorization": f"Bearer {env_value('GROQ_API_KEY')}"}, json=body,
                               timeout=self.timeout)
            except httpx.HTTPError as e:
                last = f"network error: {type(e).__name__}"
            else:
                if r.is_success:
                    return parse_json_object(r.json()["choices"][0]["message"]["content"] or "")
                last = f"HTTP {r.status_code}: {r.text[:700]}"
                if r.status_code not in RETRY_STATUS:
                    break
            time.sleep(retry_delay(last) or 2 ** attempt * 2)
        raise LLMError(f"{self.name} failed: {last}")
