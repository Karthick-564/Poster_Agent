"""Guardrail 3: every piece of poster text must trace back to the brief or the brand profile."""
from __future__ import annotations

import re
import unicodedata

from .models import ElementKind, PlacedElement

_WS = re.compile(r"\s+")


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).casefold()
    return _WS.sub(" ", text).strip()


def find_unsourced_text(placed: list[PlacedElement], allowed_sources: list[str]) -> list[PlacedElement]:
    """Return text elements whose text is not a substring of any allowed source string.

    Substring (not equality) so a component may show one field, or one line/part of a longer field
    (e.g. one name out of a speakers list), but may never paraphrase or invent copy.
    """
    corpus = [normalize(s) for s in allowed_sources if s]

    def sourced(text: str) -> bool:
        # normalize() folds newlines to spaces, so layout-inserted line breaks still match the joined text;
        # failing that, every individual line must be sourced (e.g. one name per line from a list).
        if any(normalize(text) in src for src in corpus):
            return True
        lines = [normalize(line) for line in text.splitlines() if line.strip()]
        return bool(lines) and all(any(line in src for src in corpus) for line in lines)

    return [el for el in placed if el.kind == ElementKind.text and el.text and not sourced(el.text)]
