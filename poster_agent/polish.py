"""Polish agent (M8): a vision LLM proposes a few fine touches; they are applied to the saved poster with native VectorCraft commands
and kept only if the same checks do not get worse.

The LLM never gets free tool access. It may only request four operations from an allow-list - move, resize text, recolour text
from the brand palette, set opacity - on named, non-brand elements, with bounded magnitudes. It cannot change copy, touch logos,
partner logos, headshots or decorations, add or delete objects. Everything it asks for is validated first (rejections are reported),
applied to a COPY of the .vectorcraft file (`polish/`), re-exported and re-checked by `finalize`. The original poster is never
overwritten; the polished one is offered only when the checks say it is no worse, and the report says exactly that.
"""
from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image

from poster_agent.finalize import FinalizeResult, finalize, resolve_target, top_level_nodes
from poster_agent.layout.base import resolve_color, valid_color_names
from poster_agent.llm import LLM, LLMError
from poster_agent.spec import ElementKind, PlacedElement, PosterSpec
from poster_agent.spec.brand import BrandProfile, ROOT, load_brand

ALLOWED_TOOLS = ("object.move", "text.resize", "text.recolour", "object.opacity")
PROTECTED_ROLES = {"logo", "logo_plate", "partner_logo", "headshot", "headshot_ring"}
MAX_OPS = 8
MAX_MOVE_FRACTION = 0.04  # of the shorter canvas side
MAX_SIZE_PCT = 15.0
MAX_SIDE = 1600

SYSTEM = f"""You are a senior print designer doing the FINAL POLISH pass on an institutional event poster that already passes its
brand checks. You see the render, its elements (id, role, text, box, size, colour), the palette and the check results.

You may request ONLY these operations (anything else is ignored), at most {MAX_OPS}, each a small refinement:
  {{"tool": "object.move", "id": <element id>, "dx": number pt, "dy": number pt}}      nudge to fix alignment / spacing / rhythm
  {{"tool": "text.resize", "id": <text element id>, "size_pct": -{MAX_SIZE_PCT:.0f}..{MAX_SIZE_PCT:.0f}}}      adjust hierarchy
  {{"tool": "text.recolour", "id": <text element id>, "color": <palette name>}}            improve emphasis (palette names only)
  {{"tool": "object.opacity", "id": <non-text element id>, "opacity": 30..100}}

Never change wording, never touch logos / partner logos / headshots, never move anything outside the margins. Moves are at most
{MAX_MOVE_FRACTION:.0%} of the poster's shorter side. If the poster is already well balanced, return no ops - a poster that needs
nothing is a valid answer.
Return one JSON object: {{"ops": [...], "rationale": "one or two sentences"}}"""


@dataclass
class PolishResult:
    ops_requested: list[dict] = field(default_factory=list)
    ops_applied: list[dict] = field(default_factory=list)
    rejected: list[str] = field(default_factory=list)
    before_errors: int | None = None
    after_errors: int | None = None
    accepted: bool = False
    reason: str = ""
    rationale: str = ""
    png: Path | None = None
    vectorcraft: Path | None = None

    def to_dict(self) -> dict:
        return {"accepted": self.accepted, "reason": self.reason, "rationale": self.rationale, "before_errors": self.before_errors,
                "after_errors": self.after_errors, "ops_requested": self.ops_requested, "ops_applied": self.ops_applied,
                "rejected": self.rejected, "png": str(self.png) if self.png else None,
                "vectorcraft": str(self.vectorcraft) if self.vectorcraft else None}


def validate_ops(raw: Any, spec: PosterSpec, brand: BrandProfile) -> tuple[list[dict], list[str]]:
    """Keep the ops that satisfy every limit; explain each one that does not."""
    good: list[dict] = []
    bad: list[str] = []
    if not isinstance(raw, list):
        return [], ["ops is not a list"]
    by_id = {e.id: e for e in spec.placed}
    c, safe = spec.canvas, spec.canvas.safe_box
    limit = MAX_MOVE_FRACTION * min(c.width, c.height)
    colors = valid_color_names(brand)
    for i, op in enumerate(raw[:MAX_OPS]):
        tag = f"op {i} {op.get('tool') if isinstance(op, dict) else op!r}"
        if not isinstance(op, dict) or op.get("tool") not in ALLOWED_TOOLS:
            bad.append(f"{tag}: tool is not on the allow-list {list(ALLOWED_TOOLS)}")
            continue
        el: PlacedElement | None = by_id.get(op.get("id"))
        if el is None:
            bad.append(f"{tag}: unknown element id {op.get('id')!r}")
        elif el.props.get("decorative") or el.role in PROTECTED_ROLES or el.locked:
            bad.append(f"{tag}: {el.id} ({el.role}) is protected (brand logo, headshot or decoration)")
        elif op["tool"] == "object.move":
            try:
                dx, dy = float(op.get("dx", 0)), float(op.get("dy", 0))
            except (TypeError, ValueError):
                bad.append(f"{tag}: dx/dy must be numbers")
                continue
            moved = el.box.model_copy(update={"x": el.box.x + dx, "y": el.box.y + dy})
            if abs(dx) > limit or abs(dy) > limit:
                bad.append(f"{tag}: move of ({dx:.0f},{dy:.0f})pt exceeds {limit:.0f}pt")
            elif el.kind == ElementKind.text and not safe.contains(moved, tol=0.5):
                bad.append(f"{tag}: {el.id} would leave the safe margins")
            else:
                good.append({"tool": "object.move", "id": el.id, "dx": round(dx, 1), "dy": round(dy, 1)})
        elif op["tool"] in ("text.resize", "text.recolour"):
            if el.kind != ElementKind.text:
                bad.append(f"{tag}: {el.id} is not text")
            elif op["tool"] == "text.resize":
                try:
                    pct = float(op.get("size_pct", 0))
                except (TypeError, ValueError):
                    bad.append(f"{tag}: size_pct must be a number")
                    continue
                if abs(pct) > MAX_SIZE_PCT:
                    bad.append(f"{tag}: size change {pct:.0f}% exceeds ±{MAX_SIZE_PCT:.0f}%")
                else:
                    good.append({"tool": "text.resize", "id": el.id, "size_pct": round(pct, 1)})
            elif op.get("color") not in colors:
                bad.append(f"{tag}: colour {op.get('color')!r} is not a palette name {sorted(colors)}")
            else:
                good.append({"tool": "text.recolour", "id": el.id, "color": op["color"]})
        else:  # object.opacity
            try:
                pct = float(op.get("opacity"))
            except (TypeError, ValueError):
                bad.append(f"{tag}: opacity must be a number")
                continue
            if el.kind == ElementKind.text or not 30 <= pct <= 100:
                bad.append(f"{tag}: opacity applies to non-text elements, 30..100")
            else:
                good.append({"tool": "object.opacity", "id": el.id, "opacity": round(pct, 1)})
    if isinstance(raw, list) and len(raw) > MAX_OPS:
        bad.append(f"{len(raw) - MAX_OPS} op(s) beyond the limit of {MAX_OPS} were ignored")
    return good, bad


def _element_rows(spec: PosterSpec) -> list[dict]:
    return [{"id": e.id, "role": e.role, "text": (e.text or "")[:60] or None, "box": [round(e.box.x), round(e.box.y), round(e.box.w), round(e.box.h)],
             "size": e.size, "color": e.color} for e in spec.placed if not e.props.get("decorative")]


def build_prompt(spec: PosterSpec, brand: BrandProfile, baseline: FinalizeResult) -> str:
    c = spec.canvas
    return json.dumps({
        "canvas": {"width": c.width, "height": c.height, "safe_box": c.safe_box.model_dump()},
        "palette": brand.palette, "palette_names": sorted(valid_color_names(brand)),
        "elements": _element_rows(spec),
        "checks": [{"name": r.name, "passed": r.passed, "detail": r.detail} for r in baseline.report.results],
    }, separators=(",", ":"), ensure_ascii=False)


def _downscale(png: Path, out: Path) -> Path:
    with Image.open(png) as im:
        k = min(1.0, MAX_SIDE / max(im.size))
        im.convert("RGB").resize((round(im.width * k), round(im.height * k))).save(out)
    return out


async def _apply(spec: PosterSpec, brand: BrandProfile, ops: list[dict], src: Path, dest: Path) -> list[str]:
    """Open the poster, run the validated ops as native VectorCraft commands, save a copy. Returns ops that could not be applied."""
    from poster_agent.compile.vc_client import VectorCraftSession
    by_id = {e.id: e for e in spec.placed}
    failed: list[str] = []
    async with VectorCraftSession(roots=[ROOT]) as vc:
        await vc.command("file.open", path=str(src))
        nodes = {n.get("name"): n["id"] for n in top_level_nodes(await vc.call("doc_inspect", {})) if n.get("name")}
        for op in ops:
            nid = nodes.get(op["id"])
            if nid is None:
                failed.append(f"{op['tool']} {op['id']}: no object with that name in the document")
                continue
            try:
                if op["tool"] == "object.move":
                    await vc.command("object.move", ids=[nid], dx=op["dx"], dy=op["dy"])
                elif op["tool"] == "text.resize":
                    await vc.command("text.setStyle", ids=[nid], size=round(by_id[op["id"]].size * (1 + op["size_pct"] / 100.0), 2))
                elif op["tool"] == "text.recolour":
                    await vc.command("text.setStyle", ids=[nid], fill=resolve_color(brand, op["color"]))
                else:
                    await vc.command("object.setProps", ids=[nid], opacity=op["opacity"])
            except Exception as e:  # a native command refusing an op is reported, not fatal
                failed.append(f"{op['tool']} {op['id']}: {str(e)[:160]}")
        await vc.command("file.saveAs", path=str(dest))
    return failed


async def polish(target: Path, llm: LLM, *, use_ocr: bool = True, brand: BrandProfile | None = None) -> PolishResult:
    run_dir, vcf = resolve_target(target)
    spec = PosterSpec.model_validate_json((run_dir / "spec.json").read_text(encoding="utf-8"))
    brand = brand or load_brand(spec.brand)
    out = PolishResult()
    work = run_dir / "polish"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir()
    shutil.copy(run_dir / "spec.json", work / "spec.json")
    baseline = await finalize(vcf, use_ocr=use_ocr, brand=brand)  # the checks on the poster exactly as it is now
    out.before_errors = sum(1 for i in baseline.report.issues if i.severity == "error") + sum(1 for i in baseline.extra_issues if i.severity == "error")
    png = baseline.png
    try:
        reply = llm.generate_json(SYSTEM, build_prompt(spec, brand, baseline), images=[_downscale(png, work / "polish_input.png")], temperature=0.2)
    except LLMError as e:
        out.reason = f"polish skipped: vision model unavailable ({str(e)[:120]})"
        (work / "polish_report.json").write_text(json.dumps(out.to_dict(), indent=1), encoding="utf-8")
        return out
    out.rationale = str(reply.get("rationale", ""))[:400]
    out.ops_requested = reply.get("ops") if isinstance(reply.get("ops"), list) else []
    ops, out.rejected = validate_ops(reply.get("ops", []), spec, brand)
    if not ops:
        out.reason = "no valid polish ops requested: the poster was left exactly as it is"
        (work / "polish_report.json").write_text(json.dumps(out.to_dict(), indent=1), encoding="utf-8")
        return out
    dest = work / f"{vcf.stem}.vectorcraft"
    out.rejected += await _apply(spec, brand, ops, vcf, dest)
    after = await finalize(dest, use_ocr=use_ocr, brand=brand)
    out.after_errors = sum(1 for i in after.report.issues if i.severity == "error") + sum(1 for i in after.extra_issues if i.severity == "error")
    out.ops_applied = ops
    out.png, out.vectorcraft = after.png, dest
    if out.after_errors > out.before_errors:
        out.reason = f"rejected: the polished version has more check errors ({out.after_errors}) than the original ({out.before_errors})"
    elif any(c.kind in ("deleted", "hidden", "added", "text_edited") for c in after.changes):
        out.reason = "rejected: the polish changed content (deleted/hidden/added/edited objects)"
    else:
        out.accepted = True
        out.reason = f"accepted: {len(ops)} op(s); check errors {out.before_errors} -> {out.after_errors}"
    (work / "polish_report.json").write_text(json.dumps(out.to_dict(), indent=1), encoding="utf-8")
    return out
