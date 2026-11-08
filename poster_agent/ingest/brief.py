"""Brief: the event requirements as structured, *verbatim* fields (guardrail 3: poster text comes only from here).

`extract_brief` turns free text (an email, a chat message) into a Brief with an LLM, then verifies that every extracted
string literally occurs in the source text — anything the model paraphrased or invented is dropped and reported.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from poster_agent.llm import LLM
from poster_agent.spec.verbatim import normalize


class Person(BaseModel):
    name: str
    lines: list[str] = Field(default_factory=list)  # designation / affiliation lines, verbatim
    asset: str | None = None  # inventory asset id of their photo


class Brief(BaseModel):
    brand: str  # brand key: kct, kclas, kctbs
    format: str | None = None  # story | fourfive | square | a3 | a4 (planner picks if None)
    event_title: str
    organizer: str | None = None
    subtitle: str | None = None
    tagline: str | None = None
    theme_lines: list[str] = Field(default_factory=list)
    date: str | None = None
    time: str | None = None
    venue: str | None = None
    chief_guest: Person | None = None
    speakers: list[Person] = Field(default_factory=list)
    partner_assets: list[str] = Field(default_factory=list)  # inventory ids of partner/sponsor logos
    photo_assets: list[str] = Field(default_factory=list)  # inventory ids of event photos
    contact_lines: list[str] = Field(default_factory=list)
    look: str | None = None  # optional: force a look (plan/looks.py: clean_light | brand_blue | tint_soft | dark_art)
    icons: bool | None = None  # optional: force an icon row on (true) or off (false); default alternates by title
    layout_style: str | None = None  # optional: force a composition recipe (see plan/prompt.py RECIPES)
    cta: str | None = None
    extra_lines: list[str] = Field(default_factory=list)
    style_notes: str | None = None  # free-form mood/audience hints for the planner (never printed)

    def text_fields(self) -> list[tuple[str, str]]:
        """(field path, text) for every printable string in the brief."""
        out: list[tuple[str, str]] = []
        for k in ("event_title", "organizer", "subtitle", "tagline", "date", "time", "venue", "cta"):
            if getattr(self, k):
                out.append((k, getattr(self, k)))
        for k in ("theme_lines", "contact_lines", "extra_lines"):
            out += [(f"{k}[{i}]", s) for i, s in enumerate(getattr(self, k))]
        people = ([("chief_guest", self.chief_guest)] if self.chief_guest else []) + \
                 [(f"speakers[{i}]", p) for i, p in enumerate(self.speakers)]
        for path, p in people:
            out.append((f"{path}.name", p.name))
            out += [(f"{path}.lines[{i}]", s) for i, s in enumerate(p.lines)]
        return out

    def corpus(self) -> list[str]:
        return [t for _, t in self.text_fields()]

    def required_fields(self) -> list[tuple[str, str]]:
        """Fields that must appear on the poster (everything the requester gave us)."""
        return [(p, t) for p, t in self.text_fields() if not p.startswith("style")]

    def asset_ids(self) -> set[str]:
        ids = set(self.partner_assets) | set(self.photo_assets)
        ids |= {p.asset for p in ([self.chief_guest] if self.chief_guest else []) + self.speakers if p.asset}
        return ids


def load_brief(path: Path) -> Brief:
    text = Path(path).read_text(encoding="utf-8")
    data = json.loads(text) if str(path).endswith(".json") else yaml.safe_load(text)
    return Brief.model_validate(data)


EXTRACT_SYSTEM = """You extract event-poster requirements from a message into JSON.
Rules:
- COPY every text value VERBATIM from the message: same words, spelling, punctuation and capitalisation. Never paraphrase,
  summarise, translate, correct or add words. If a value is not in the message, leave it null / empty.
- Split into the fields of the schema. Put each speaker/guest's name in `name` and their designation lines in `lines`.
- `asset` / `partner_assets` / `photo_assets` hold file ids ONLY when the message names the file; otherwise leave empty.
- `style_notes` may hold the requester's mood/style wishes (it is never printed on the poster).
Return one JSON object matching the schema."""


def _leaf_strings(obj: Any, path: str = "") -> list[tuple[str, str]]:
    if isinstance(obj, str):
        return [(path, obj)]
    if isinstance(obj, list):
        return [x for i, v in enumerate(obj) for x in _leaf_strings(v, f"{path}[{i}]")]
    if isinstance(obj, dict):
        return [x for k, v in obj.items() for x in _leaf_strings(v, f"{path}.{k}" if path else k)]
    return []


NON_PRINTED = {"brand", "format", "style_notes", "asset", "partner_assets", "photo_assets"}


def extract_brief(text: str, llm: LLM, brand: str, fmt: str | None = None) -> tuple[Brief, list[str]]:
    """Free text -> (Brief, problems). Strings not found verbatim in `text` are removed and listed in `problems`."""
    user = (f"SCHEMA:\n{json.dumps(Brief.model_json_schema())}\n\nBRAND: {brand}\nFORMAT: {fmt or 'unspecified'}\n\n"
            f"MESSAGE:\n{text}")
    raw = llm.generate_json(EXTRACT_SYSTEM, user, temperature=0.0)
    raw["brand"] = brand
    if fmt:
        raw["format"] = fmt
    source = normalize(text)
    problems: list[str] = []

    def prune(obj: Any, path: str = "") -> Any:
        if isinstance(obj, str):
            leaf = path.split(".")[-1].split("[")[0]
            if leaf in NON_PRINTED or not obj.strip() or normalize(obj) in source:
                return obj
            problems.append(f"{path}: {obj!r} is not verbatim in the message — dropped")
            return None
        if isinstance(obj, list):
            return [v for v in (prune(v, f"{path}[{i}]") for i, v in enumerate(obj)) if v not in (None, "")]
        if isinstance(obj, dict):
            return {k: prune(v, f"{path}.{k}" if path else k) for k, v in obj.items()}
        return obj

    cleaned = prune(raw)
    # a person without a verbatim name is unusable
    for key in ("speakers",):
        cleaned[key] = [p for p in cleaned.get(key, []) if p.get("name")]
    if cleaned.get("chief_guest") and not cleaned["chief_guest"].get("name"):
        cleaned["chief_guest"] = None
    if not cleaned.get("event_title"):
        raise ValueError("could not find an event title verbatim in the message; " + "; ".join(problems))
    return Brief.model_validate(cleaned), problems
