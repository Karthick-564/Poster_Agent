"""Design skills: plain-data guidance (skills/<name>/SKILL.md) that the planner and the checks draw on.

A skill file has YAML frontmatter (name, description, priority) and `##` sections. A section may carry a comment on its heading line,
`<!-- genres: workshop,talk ; formats: story ; core -->`, saying when it applies. `select()` returns only the sections that fit a brief
(its genre and format), most relevant first, within a character budget, so the planner prompt stays small. Skills are data: nothing is
executed, and no copy for the poster ever comes from them (poster text stays verbatim from the brief, guardrail 3).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import yaml

from poster_agent.spec.brand import ROOT

SKILLS_DIR = ROOT / "skills"
_HEADING = re.compile(r"^##\s+(.*?)\s*$")
_META = re.compile(r"<!--(.*?)-->")

# event genre -> keywords found in the brief's title / subtitle / tagline (checked in this order: most specific first)
GENRE_KEYWORDS: dict[str, tuple[str, ...]] = {
    "placement": ("placement", "career fair", "job fair"),
    "recruitment": ("recruitment", "recruiting", "hiring", "join the team", "join team", "club drive"),
    "admission": ("admission", "admissions", "enrol", "apply now", "programme", "program"),
    "workshop": ("workshop", "training", "bootcamp", "hands-on", "course"),
    "conference": ("conference", "symposium", "summit", "conclave", "congress"),
    "talk": ("talk", "seminar", "lecture", "webinar", "session"),
    "sports": ("sports", "athletic", "tournament", "league", "marathon"),
    "cultural": ("cultural", "festival", "fest", "concert", "music", "dance", "drama"),
    "commemoration": ("reunion", "anniversary", "convocation", "commemorat", "alumni", "founder", "award", "ceremony", "annual day"),
    "launch": ("launch", "inauguration", "unveil", "announcement"),
}


@dataclass
class Section:
    skill: str
    title: str
    text: str
    genres: frozenset[str] = frozenset()
    formats: frozenset[str] = frozenset()
    core: bool = False
    priority: int = 5


@dataclass
class Skill:
    name: str
    description: str
    priority: int
    use: tuple[str, ...] = ("plan",)  # which stage reads it: plan | critic | ingest
    digest: str = ""  # a few sentences: what the planner prompt carries when the full sections do not fit
    sections: list[Section] = field(default_factory=list)


def _parse_meta(comment: str) -> tuple[frozenset[str], frozenset[str], bool]:
    genres, formats, core = set(), set(), False
    for part in re.split(r"[;|]", comment):
        part = part.strip()
        if part == "core":
            core = True
        elif part.startswith("genres:"):
            genres = {g.strip().lower() for g in part[7:].split(",") if g.strip()}
        elif part.startswith("formats:"):
            formats = {f.strip().lower() for f in part[8:].split(",") if f.strip()}
    return frozenset(genres), frozenset(formats), core


def parse_skill(path: Path) -> Skill:
    raw = path.read_text(encoding="utf-8")
    m = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)$", raw, re.S)
    if not m:
        raise ValueError(f"{path}: missing YAML frontmatter")
    meta, body = yaml.safe_load(m.group(1)) or {}, m.group(2)
    skill = Skill(name=meta.get("name") or path.parent.name, description=str(meta.get("description", "")),
                  priority=int(meta.get("priority", 5)), use=tuple(meta.get("use") or ["plan"]),
                  digest=str(meta.get("digest", "")).strip())
    title, lines, comment = None, [], ""
    for line in body.splitlines() + ["## <end>"]:
        h = _HEADING.match(line)
        if h:
            if title is not None and "\n".join(lines).strip():
                genres, formats, core = _parse_meta(comment)
                skill.sections.append(Section(skill.name, title, "\n".join(lines).strip(), genres, formats, core, skill.priority))
            raw_title = h.group(1)
            cm = _META.search(raw_title)
            comment = cm.group(1) if cm else ""
            title, lines = _META.sub("", raw_title).strip(), []
        elif title is not None:
            lines.append(line)
    return skill


@lru_cache(maxsize=4)
def load_skills(directory: Path = SKILLS_DIR) -> dict[str, Skill]:
    """All skills under `directory` (empty when the folder is missing, so the agent still runs without them)."""
    out: dict[str, Skill] = {}
    for f in sorted(Path(directory).glob("*/SKILL.md")):
        s = parse_skill(f)
        out[s.name] = s
    return out


def genre_of(brief) -> str:
    """The event's genre from the brief's own words ('event' when nothing matches)."""
    text = " ".join(filter(None, [brief.event_title, getattr(brief, "subtitle", None), getattr(brief, "tagline", None)])).lower()
    for genre, words in GENRE_KEYWORDS.items():
        if any(w in text for w in words):
            return genre
    return "event"


def _block(sec: Section) -> tuple[str, str]:
    label = f"{sec.skill}: {sec.title}"
    return label, f"[{label}]" + NL + sec.text


NL = chr(10)


def select(brief, *, budget_chars: int = 3200, fmt: str | None = None, purpose: str = "plan",
           directory: Path = SKILLS_DIR) -> tuple[str, list[str]]:
    """(guidance text, names used) for this brief and stage. Order: the genre's own playbook sections in full, then every skill's digest (by
    priority), then further core sections while the budget lasts. Never exceeds `budget_chars`."""
    genre = genre_of(brief)
    fmt = (fmt or getattr(brief, "format", None) or "story").lower()
    skills = sorted((s for s in load_skills(directory).values() if purpose in s.use), key=lambda s: s.priority)

    def eligible(sec: Section) -> bool:
        return (not sec.genres or genre in sec.genres) and (not sec.formats or fmt in sec.formats)

    blocks: list[tuple[str, str]] = []  # (label, text) in priority order
    for sk in skills:  # 1. the genre's own sections, in full
        blocks += [_block(sec) for sec in sk.sections if genre in sec.genres and eligible(sec)]
    for sk in skills:  # 2. each skill's digest
        if sk.digest:
            blocks.append((f"{sk.name} (digest)", f"[{sk.name}]" + NL + sk.digest))
    for sk in skills:  # 3. format-specific rules (e.g. print preflight) and core sections of skills that have no digest
        blocks += [_block(sec) for sec in sk.sections
                   if eligible(sec) and genre not in sec.genres and ((sec.core and not sk.digest) or sec.formats)]
    for sk in skills:  # 4. remaining core sections of skills with a digest, while budget lasts
        blocks += [_block(sec) for sec in sk.sections if eligible(sec) and sec.core and genre not in sec.genres and sk.digest]
    out, used, size = [], [], 0
    for label, text in blocks:
        if size + len(text) + 2 > budget_chars:
            continue
        out.append(text)
        used.append(label)
        size += len(text) + 2
    return (NL + NL).join(out), used
