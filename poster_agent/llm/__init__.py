from .base import LLM, LLMError, parse_json_object
from .fallback import FallbackLLM
from .gemini import GeminiLLM
from .groq import GroqLLM
from .scripted import ScriptedLLM


def default_llm() -> FallbackLLM:
    """Gemini 3.5 Flash -> Gemini 2.5 Flash -> Groq gpt-oss-120b -> flash-latest (providers overload; the chain keeps us running)."""
    return FallbackLLM([GeminiLLM("gemini-3.5-flash", retries=2), GeminiLLM("gemini-2.5-flash"), GroqLLM(),
                        GeminiLLM("gemini-flash-latest", retries=1)])


def default_vlm() -> FallbackLLM:
    """Vision-capable chain for the critic (Groq is text-only and is skipped automatically)."""
    # newer models give better reviews but are often overloaded: fail fast on them, fall back to the reliable 2.5-flash
    return FallbackLLM([GeminiLLM("gemini-3.5-flash", retries=1, timeout=90), GeminiLLM("gemini-2.5-flash"),
                        GeminiLLM("gemini-3.6-flash", retries=1, timeout=90)])


__all__ = ["default_vlm", "default_llm", "FallbackLLM", "LLM", "LLMError", "parse_json_object", "GeminiLLM", "GroqLLM", "ScriptedLLM"]
