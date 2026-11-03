"""Try providers in order; the first that answers wins. Keeps the pipeline alive through provider overload (e.g. 503)."""
from __future__ import annotations

from pathlib import Path

from .base import LLM, LLMError


class FallbackLLM:
    def __init__(self, chain: list[LLM]):
        if not chain:
            raise ValueError("FallbackLLM needs at least one provider")
        self.chain = chain
        self.name = "fallback[" + ",".join(c.name for c in chain) + "]"
        self.last_used: str | None = None

    def generate_json(self, system: str, user: str, *, images: list[Path] | None = None,
                      temperature: float = 0.2) -> dict:
        errors: list[str] = []
        for llm in self.chain:
            if images and llm.name.startswith("groq"):
                continue  # text-only provider
            try:
                out = llm.generate_json(system, user, images=images, temperature=temperature)
                self.last_used = llm.name
                return out
            except LLMError as e:
                errors.append(str(e)[:200])
        raise LLMError("all providers failed: " + " | ".join(errors))
