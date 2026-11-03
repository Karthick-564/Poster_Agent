"""Deterministic fake LLM for tests: replays queued replies and records every prompt it was given."""
from __future__ import annotations

from pathlib import Path

from .base import LLMError


class ScriptedLLM:
    name = "scripted"

    def __init__(self, replies: list[dict]):
        self.replies = list(replies)
        self.calls: list[dict] = []

    def generate_json(self, system: str, user: str, *, images: list[Path] | None = None,
                      temperature: float = 0.2) -> dict:
        self.calls.append({"system": system, "user": user, "images": images})
        if not self.replies:
            raise LLMError("ScriptedLLM: no replies left")
        return self.replies.pop(0)
