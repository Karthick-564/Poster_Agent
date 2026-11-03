"""Provider-neutral LLM interface. Planner, critic and polish agent each take an `LLM`; adapters live beside this file."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Protocol

from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[2]


class LLMError(RuntimeError):
    pass


class LLM(Protocol):
    name: str

    def generate_json(self, system: str, user: str, *, images: list[Path] | None = None,
                      temperature: float = 0.2) -> dict:
        """Return one parsed JSON object. Raises LLMError after retries are exhausted."""


def env_value(name: str) -> str:
    """Read a secret from the process env or the project's .env (never logged)."""
    import os
    val = os.environ.get(name) or dotenv_values(ROOT / ".env").get(name) or ""
    if not val:
        raise LLMError(f"{name} is not set (add it to .env)")
    return val


_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.S)


def parse_json_object(text: str) -> dict:
    """Parse a model reply as a JSON object, tolerating code fences or leading/trailing prose."""
    text = _FENCE.sub("", text.strip())
    try:
        out = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise LLMError(f"reply is not JSON: {text[:200]!r}") from None
        try:
            out = json.loads(text[start:end + 1])
        except json.JSONDecodeError as e:
            raise LLMError(f"reply is not valid JSON ({e}): {text[:200]!r}") from None
    if not isinstance(out, dict):
        raise LLMError(f"expected a JSON object, got {type(out).__name__}")
    return out


_RETRY_HINT = re.compile(r"(?:try again|retry) in\s+(?:(\d+(?:\.\d+)?)m)?\s*(\d+(?:\.\d+)?)(ms|s)", re.I)


def retry_delay(text: str, cap: float = 30.0) -> float | None:
    """Seconds a provider asks us to wait ('Please try again in 7.5s', 'retry in 1m2s'), capped; None if no hint."""
    m = _RETRY_HINT.search(text)
    if not m:
        return None
    secs = float(m.group(2)) / (1000.0 if m.group(3).lower() == "ms" else 1.0) + 60.0 * float(m.group(1) or 0)
    return min(secs + 0.5, cap)
