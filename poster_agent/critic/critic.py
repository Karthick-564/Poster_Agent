"""Vision critic. Judges what pixel checks cannot: hierarchy, balance, density, art/layout harmony, "does it look professional".

Honest by design: if the vision model is unavailable the report says so (`available=False`, score None) — a score is never
invented. Issues cite element ids from the spec so the router can send them to the stage that owns them.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image
from pydantic import BaseModel, Field, ValidationError

from poster_agent.checks import CheckReport
from poster_agent.ingest import Brief
from poster_agent.llm import LLM, LLMError
from poster_agent.spec import Issue, IssueCategory, PosterSpec
from poster_agent.spec.brand import BrandProfile

PASS_SCORE = 8
MAX_SIDE = 1600
CRITIC_CATEGORIES = {"hierarchy", "density", "component_choice", "spacing", "alignment", "bg_clash", "hero_quality",
                     "photo_treatment", "render_artifact"}

SYSTEM = f"""You are a senior print designer reviewing a finished institutional event poster against its brand and brief.
You receive the rendered image, the list of its elements (id, role, text, position), the automated pixel-check results, and the brief.

Judge ONLY what you can see. Score 1-10: 9-10 print-ready and polished; 8 good, minor polish only; 6-7 noticeable weaknesses;
<=5 needs rework. Be strict and do not default to high scores.

Look at: visual hierarchy (is there one clear focal point, then supporting info, then logistics?), balance and use of space (dead
zones, crowding, alignment), type scale and rhythm, harmony between background art / imagery and text, quality of any generated
artwork or cutouts (artefacts, odd edges, wrong subject), consistency with the brand tone.

Do NOT report: missing/illegible text, contrast ratios, text overlap, logo clear space/size, or text outside margins — the
automated checks cover those (shown to you). Report them only if the image plainly shows one the checks did not list.

Return one JSON object:
{{"score": integer 1-10, "summary": "two sentences", "strengths": ["..."],
  "issues": [{{"category": one of {sorted(CRITIC_CATEGORIES)}, "severity": "error|warning", "element_ids": ["ids from the element list"],
              "evidence": "what you see, specifically", "suggested_fix": "a concrete change (e.g. make X the hero, reduce Y, calmer background)"}}]}}
Rules: at most 6 issues, most important first; "error" only for problems that would embarrass the institution if printed; every
issue needs concrete visual evidence; if the poster is good, return an empty issues list."""


class _VlmIssue(BaseModel):
    category: str
    severity: str = "warning"
    element_ids: list[str] = Field(default_factory=list)
    evidence: str
    suggested_fix: str | None = None


class _VlmReply(BaseModel):
    score: int
    summary: str = ""
    strengths: list[str] = Field(default_factory=list)
    issues: list[_VlmIssue] = Field(default_factory=list)


@dataclass
class CriticReport:
    available: bool
    score: int | None = None
    summary: str = ""
    strengths: list[str] = field(default_factory=list)
    issues: list[Issue] = field(default_factory=list)
    dropped: list[str] = field(default_factory=list)  # reply parts we could not use (unknown category / id)
    note: str = ""
    method: str = "review"  # 'review' (open-ended, large model) | 'glance' (thumbnail reading test, small local model)
    probes: list[dict] = field(default_factory=list)  # glance-test readings, when method == 'glance'

    @property
    def passed(self) -> bool:
        return self.available and self.score is not None and self.score >= PASS_SCORE and \
            not any(i.severity == "error" for i in self.issues)

    def to_dict(self) -> dict:
        return {"available": self.available, "score": self.score, "summary": self.summary, "strengths": self.strengths,
                "issues": [i.model_dump(mode="json") for i in self.issues], "dropped": self.dropped, "note": self.note,
                "pass_score": PASS_SCORE, "method": self.method, "probes": self.probes}


def downscale(png: Path, out: Path, max_side: int = MAX_SIDE) -> Path:
    img = Image.open(png).convert("RGB")
    if max(img.size) > max_side:
        k = max_side / max(img.size)
        img = img.resize((round(img.width * k), round(img.height * k)), Image.LANCZOS)
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out, "PNG")
    return out


def element_summary(spec: PosterSpec) -> list[dict]:
    out = []
    for e in spec.placed:
        row = {"id": e.id, "kind": e.kind.value, "role": e.role,
               "box": [round(e.box.x), round(e.box.y), round(e.box.w), round(e.box.h)]}
        if e.text:
            row["text"] = e.text.replace("\n", " ")[:60]
        out.append(row)
    return out


def glance_critique(png: Path, brief: Brief, vlm, work_dir: Path) -> CriticReport:
    """Critique by the glance test: a small local VLM must read the title (and date) from a 300 px thumbnail. Deterministic verdicts;
    the 'score' is only the share of readings that matched the brief, scaled to 10 (and labelled as such in the summary)."""
    from .probes import THUMB_SIDE, glance_test
    results = glance_test(vlm, png, brief.event_title, brief.date, work_dir)
    answered = [r for r in results if r.passed is not None]
    if not answered:
        return CriticReport(available=False, method="glance", note="the local VLM gave no usable reading of the thumbnail")
    issues: list[Issue] = []
    for r in answered:
        if r.passed:
            continue
        what = "event title" if r.id == "title" else "date"
        issues.append(Issue(
            category=IssueCategory.hierarchy, severity="error" if r.id == "title" else "warning", element_ids=[], source="critic",
            evidence=f"glance test: from a {THUMB_SIDE}px thumbnail the vision model read the {what} as {r.read!r}, expected {r.expected!r}",
            suggested_fix=f"make the {what} larger / higher contrast so it reads at thumbnail size"))
    ok = sum(1 for r in answered if r.passed)
    return CriticReport(available=True, method="glance", score=max(1, round(10 * ok / len(answered))),
                        summary=f"glance test ({vlm.name}): {ok} of {len(answered)} thumbnail readings matched the brief",
                        strengths=[f"read the {r.id} correctly at thumbnail size" for r in answered if r.passed], issues=issues,
                        probes=[r.to_dict() for r in results])


def critique(png: Path, spec: PosterSpec, brand: BrandProfile, brief: Brief, checks: CheckReport | None, vlm: LLM,
             work_dir: Path) -> CriticReport:
    if getattr(vlm, "probe_critic", False):  # small local model: closed, checkable questions only
        return glance_critique(png, brief, vlm, work_dir)
    small = downscale(png, work_dir / "critic_input.png")
    ids = {e.id for e in spec.placed} | {"background"}
    check_lines = [f"[{'PASS' if r.passed else 'FAIL'}] {r.name}: {r.detail}" for r in (checks.results if checks else [])]
    check_lines += [f"- {i.severity} {i.category.value}: {i.evidence[:160]}" for i in (checks.issues if checks else [])]
    user = "\n".join([
        f"CANVAS: {spec.canvas.width:.0f}x{spec.canvas.height:.0f}pt ({spec.canvas.format}); the image is that canvas downscaled.",
        f"BRAND: {brand.name}. Palette: {json.dumps(brand.palette)}. Rules: " + " ".join(brand.background_rules),
        f"BRIEF STYLE NOTES: {brief.style_notes or 'none'}",
        "BRIEF EVENT: " + brief.event_title,
        "ELEMENTS: " + json.dumps(element_summary(spec), ensure_ascii=False, separators=(",", ":")),
        "PIXEL CHECKS (already known):\n" + ("\n".join(check_lines) or "none run"),
        "Review the poster image now and return the JSON.",
    ])
    try:
        raw = vlm.generate_json(SYSTEM, user, images=[small], temperature=0.2)
    except LLMError as e:
        return CriticReport(available=False, note=f"vision review unavailable: {str(e)[:200]}")
    try:
        reply = _VlmReply.model_validate(raw)
    except ValidationError as e:
        return CriticReport(available=False, note=f"vision reply did not match the schema: {str(e)[:200]}")

    issues, dropped = [], []
    for vi in reply.issues[:6]:
        if vi.category not in CRITIC_CATEGORIES:
            dropped.append(f"unknown category {vi.category!r}: {vi.evidence[:80]}")
            continue
        bad = [i for i in vi.element_ids if i not in ids]
        if bad:
            dropped.append(f"unknown element ids {bad} in: {vi.evidence[:80]}")
        issues.append(Issue(category=IssueCategory(vi.category), severity="error" if vi.severity == "error" else "warning",
                            element_ids=[i for i in vi.element_ids if i in ids], evidence=vi.evidence,
                            suggested_fix=vi.suggested_fix, source="critic"))
    return CriticReport(available=True, score=max(1, min(10, reply.score)), summary=reply.summary, strengths=reply.strengths,
                        issues=issues, dropped=dropped)
