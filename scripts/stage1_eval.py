"""Stage 1 scoreboard: free-text request -> structured, verbatim Brief.

For every briefs/*/brief.yaml we write the request the way a person might send it (alternating prose and labelled-bullet styles, fields
in a shuffled order), run `extract_brief` on it, and compare with the structured brief we started from:

  recall_any    share of the original printable strings found anywhere in the extracted brief
  recall_field  share found in the SAME kind of field (title / date / people / lists ...)
  dropped       strings the model produced that were NOT in the message (the verbatim guard removed them)
  extra         strings in the extracted brief that were not in the original (usually a split or merge of a line)

Usage:  PYTHONPATH=. python scripts/stage1_eval.py [--only NAME ...]      -> prints a table, writes eval/stage1/results.json
No GPU needed; uses the planner's language model (poster_agent.llm.default_llm)."""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

from poster_agent.ingest.brief import Brief, extract_brief, load_brief
from poster_agent.llm import default_llm
from poster_agent.spec.verbatim import normalize

ROOT = Path(__file__).resolve().parent.parent
LIST_FIELDS = {"theme_lines", "extra_lines", "contact_lines"}  # which of these a line lands in is a judgement call: one bucket
SCALARS = ["event_title", "organizer", "subtitle", "tagline", "date", "time", "venue", "cta"]


def bucket(path: str) -> str:
    """The kind of field a text field path belongs to."""
    leaf = path.split("[")[0]
    if leaf in LIST_FIELDS:
        return "lines"
    if path.startswith("chief_guest.name") or (path.startswith("speakers[") and path.endswith(".name")):
        return "person_name"
    if ".lines[" in path:
        return "person_lines"
    return leaf


def render_message(b: Brief, style: int, rng: random.Random) -> str:
    """The brief as a requester would write it (never labelled with our field names)."""
    parts: list[tuple[str, str]] = []
    if b.event_title:
        parts.append(("title", b.event_title))
    for k, label in (("organizer", "organised by"), ("subtitle", ""), ("tagline", "tagline"), ("date", "date"), ("time", "time"),
                     ("venue", "venue"), ("cta", "")):
        v = getattr(b, k)
        if v:
            parts.append((label, v))
    for t in b.theme_lines + b.extra_lines:
        parts.append(("", t))
    people = ([("chief guest", b.chief_guest)] if b.chief_guest else []) + [("speaker", s) for s in b.speakers]
    for label, p in people:
        parts.append((label, p.name + (" - " + "; ".join(p.lines) if p.lines else "")))
    for t in b.contact_lines:
        parts.append(("contact", t))
    body = parts[:1] + rng.sample(parts[1:], len(parts) - 1)  # the title first, the rest in whatever order the requester thought of it
    if style == 0:  # prose
        sentences = [(f"{lab.capitalize()}: " if lab else "") + txt for lab, txt in body]
        return "Hi, we need a poster for our event. " + " ".join(s if s.endswith((".", "!", "?")) else s + "." for s in sentences) + \
               " Please keep it formal. Thanks!"
    lines = ["Poster request", ""] + [f"- {lab + ': ' if lab else ''}{txt}" for lab, txt in body]
    return "\n".join(lines)


def truth_fields(b: Brief) -> list[tuple[str, str]]:
    return [(p, t) for p, t in b.text_fields()]


def score(orig: Brief, got: Brief, problems: list[str]) -> dict:
    ot, gt = truth_fields(orig), truth_fields(got)
    got_all = [normalize(t) for _, t in gt]
    got_by_bucket: dict[str, list[str]] = {}
    for p, t in gt:
        got_by_bucket.setdefault(bucket(p), []).append(normalize(t))
    any_hit = field_hit = 0
    missing: list[str] = []
    misfiled: list[str] = []
    for p, t in ot:
        n = normalize(t)
        a = any(n == g or n in g for g in got_all)
        f = any(n == g or n in g for g in got_by_bucket.get(bucket(p), []))
        any_hit += a
        field_hit += f
        if not a:
            missing.append(f"{p}: {t}")
        elif not f:
            misfiled.append(f"{p}: {t}")
    orig_all = [normalize(t) for _, t in ot]
    extra = [t for _, t in gt if not any(normalize(t) == o or normalize(t) in o or o in normalize(t) for o in orig_all)]
    n = max(len(ot), 1)
    return {"strings": len(ot), "recall_any": any_hit / n, "recall_field": field_hit / n, "dropped": len(problems), "extra": len(extra),
            "missing": missing, "misfiled": misfiled, "extra_strings": extra, "problems": problems}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()
    llm = default_llm()
    rows = []
    for i, d in enumerate(sorted((ROOT / "briefs").iterdir())):
        f = d / "brief.yaml"
        if not f.exists() or d.name.startswith("_") or (args.only and d.name not in args.only):
            continue
        orig = load_brief(f)
        msg = render_message(orig, i % 2, random.Random(i))
        t0 = time.time()
        try:
            got, problems = extract_brief(msg, llm, orig.brand, orig.format)
            r = score(orig, got, problems)
        except Exception as e:  # noqa: BLE001  (report, do not hide)
            r = {"strings": len(truth_fields(orig)), "recall_any": 0.0, "recall_field": 0.0, "dropped": 0, "extra": 0,
                 "error": str(e)[:200], "missing": [], "misfiled": [], "extra_strings": [], "problems": []}
        r.update(name=d.name, style="prose" if i % 2 == 0 else "bullets", seconds=round(time.time() - t0, 1), message=msg)
        rows.append(r)
        print(f"{d.name:22s} {r['style']:8s} strings={r['strings']:2d} recall_any={r['recall_any']:.2f} recall_field={r['recall_field']:.2f} "
              f"dropped={r['dropped']} extra={r['extra']} {r['seconds']}s" + (f"  ERROR {r['error']}" if r.get('error') else ""), flush=True)
    if rows:
        tot = sum(r["strings"] for r in rows)
        print(f"\nOVERALL ({len(rows)} briefs, {tot} strings): recall_any={sum(r['recall_any'] * r['strings'] for r in rows) / tot:.3f}  "
              f"recall_field={sum(r['recall_field'] * r['strings'] for r in rows) / tot:.3f}  dropped={sum(r['dropped'] for r in rows)}  "
              f"extra={sum(r['extra'] for r in rows)}")
    out = ROOT / "eval" / "stage1"
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
