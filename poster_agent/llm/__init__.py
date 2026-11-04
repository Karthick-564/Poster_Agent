from .base import LLM, LLMError, parse_json_object
from .cli_llm import CLIAgentLLM
from .fallback import FallbackLLM
from .gemini import GeminiLLM
from .groq import GroqLLM
from .openai_vlm import OpenAICompatVLM
from .scripted import ScriptedLLM


def default_llm() -> FallbackLLM:
    """A fresh tool-less CLI planner session per request (when PLANNER_CLI is set) -> Gemini 3.5 Flash -> Groq gpt-oss-120b ->
    flash-latest. The free tiers overload often, so they are the fallback, not the main planner."""
    chain = [CLIAgentLLM()] if CLIAgentLLM().available() else []
    return FallbackLLM(chain + [GeminiLLM("gemini-3.5-flash", retries=2), GroqLLM(), GeminiLLM("gemini-flash-latest", retries=1)])


def default_vlm(prefer: str = "auto"):
    """The critic's vision model. 'auto' = the local LFM3-VL-2B when its endpoint answers, else the Gemini chain; 'lfm' / 'gemini' force one.
    The local model is asked closed reading questions (critic/probes.py); Gemini does the open-ended review."""
    if prefer in ("auto", "lfm"):
        local = OpenAICompatVLM()
        if prefer == "lfm" or local.available():
            return local
    return _gemini_vlm()


def _gemini_vlm() -> FallbackLLM:
    """Vision-capable cloud chain (Groq is text-only and is skipped automatically)."""
    # fail fast on overloaded models (2.5-flash is gone for new keys: HTTP 404)
    return FallbackLLM([GeminiLLM("gemini-3.5-flash", retries=1, timeout=90),
                        GeminiLLM("gemini-3.6-flash", retries=1, timeout=90)])


__all__ = ["OpenAICompatVLM", "default_vlm", "default_llm", "FallbackLLM", "LLM", "LLMError", "parse_json_object", "GeminiLLM", "GroqLLM", "ScriptedLLM", "CLIAgentLLM"]
