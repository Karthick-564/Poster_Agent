"""What a small local vision model (LFM3-VL-2B) is reliably good for, found by testing it on real posters:

  * READING at thumbnail size  -> the glance test. Ask it to read the event title and the date from a 300 px thumbnail and compare its
    answer with the brief (fuzzy text match). Verdicts are deterministic: it either read the right title or it did not.
  * PAIRWISE preference        -> ranking design variants. It prefers the clearly better poster in BOTH orders; when the two orders
    disagree (position bias, similar quality) the result is a tie, never a guess.

It is NOT used for open yes/no "is this good?" judgements: tested on plain and designed posters alike it answered "healthy" to everything.
Nothing here invents a score; a failed read is reported as exactly that.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from poster_agent.checks.ocr import match_score
from poster_agent.llm import LLMError

READ_SYSTEM = "You read posters. Reply with plain text only: just the text asked for, nothing else."
PAIR_SYSTEM = "You compare two posters shown side by side: A on the left, B on the right. Reply with the single letter A or B only."
PAIR_QUESTION = ("Which poster looks more professionally designed (clear hierarchy, good structure, good use of space and colour)? "
                 "Answer A or B.")
THUMB_SIDE = 300
TITLE_MATCH_MIN = 70.0  # fuzzy match (0-100) between what it read and the brief's title
DATE_MATCH_MIN = 70.0


@dataclass
class ProbeResult:
    id: str
    expected: str
    read: str | None  # what the model read (None = no usable reply)
    score: float | None  # fuzzy match 0-100
    passed: bool | None

    def to_dict(self) -> dict:
        return {"id": self.id, "expected": self.expected, "read": self.read, "match": self.score, "passed": self.passed}


def thumbnail(png: Path, out: Path, side: int = THUMB_SIDE) -> Path:
    with Image.open(png) as im:
        im = im.convert("RGB")
        k = side / max(im.size)
        out.parent.mkdir(parents=True, exist_ok=True)
        im.resize((max(round(im.width * k), 1), max(round(im.height * k), 1)), Image.LANCZOS).save(out, "PNG")
    return out


def _read(vlm, image: Path, question: str) -> str | None:
    ask = getattr(vlm, "ask_text", None)
    if ask is None:
        return None
    try:
        return ask(READ_SYSTEM, question, images=[image], temperature=0.0).strip() or None
    except LLMError:
        return None


def glance_test(vlm, png: Path, expected_title: str, expected_date: str | None, work_dir: Path) -> list[ProbeResult]:
    """Can the title (and date) be read from a 300 px thumbnail? Each result compares the model's reading with the brief."""
    thumb = thumbnail(png, work_dir / "glance_300.png")
    out: list[ProbeResult] = []
    probes = [("title", expected_title, "What is the main event title written on this poster? Answer with only the title text.",
               TITLE_MATCH_MIN)]
    if expected_date:
        probes.append(("date", expected_date, "What date is shown on this poster? Answer with only the date text, or the word none.",
                       DATE_MATCH_MIN))
    for pid, expected, question, need in probes:
        read = _read(vlm, thumb, question)
        score = None if read is None else float(match_score(expected, read))
        out.append(ProbeResult(pid, expected, read, score, None if score is None else score >= need))
    return out


def _stitch(a: Path, b: Path, out: Path, height: int = 800) -> Path:
    ia, ib = Image.open(a).convert("RGB"), Image.open(b).convert("RGB")
    ia = ia.resize((max(round(ia.width * height / ia.height), 1), height))
    ib = ib.resize((max(round(ib.width * height / ib.height), 1), height))
    canvas = Image.new("RGB", (ia.width + ib.width + 20, height), "white")
    canvas.paste(ia, (0, 0))
    canvas.paste(ib, (ia.width + 20, 0))
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out, "PNG")
    return out


def _pick(vlm, pair: Path) -> str | None:
    text = _read_pair(vlm, pair)
    if text is None:
        return None
    t = text.strip().upper()[:1]
    return t if t in ("A", "B") else None


def _read_pair(vlm, pair: Path) -> str | None:
    ask = getattr(vlm, "ask_text", None)
    if ask is None:
        return None
    try:
        return ask(PAIR_SYSTEM, PAIR_QUESTION, images=[pair], temperature=0.0)
    except LLMError:
        return None


def prefer(vlm, a: Path, b: Path, work_dir: Path) -> str:
    """'a', 'b' or 'tie'. The pair is shown in both orders; only an answer that survives the swap counts (position bias -> tie)."""
    first = _pick(vlm, _stitch(a, b, work_dir / "pair_ab.png"))   # a is A
    second = _pick(vlm, _stitch(b, a, work_dir / "pair_ba.png"))  # b is A
    if first is None or second is None:
        return "tie"
    if first == "A" and second == "B":
        return "a"
    if first == "B" and second == "A":
        return "b"
    return "tie"
