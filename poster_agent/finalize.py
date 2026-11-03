"""Finalize a hand-edited poster: re-export it and re-run the same checks on what you actually ended up with.

Every object the compiler draws is named after its spec element id, so a document edited by hand in VectorCraft can be mapped back
to the spec: moved/resized/recoloured/edited/hidden/added objects are detected, the spec is updated to match the document, and
the pixel checks (contrast, mandatory text, logo rules, margins, dead space ...) run on the new render. The point: you can edit
freely, and one command tells you whether you broke a brand rule.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from poster_agent.checks import CheckReport, run_checks
from poster_agent.compile.compiler import CompileResult
from poster_agent.compile.vc_client import ROOT, VectorCraftSession
from poster_agent.spec import Box, ElementKind, Issue, IssueCategory, PlacedElement, PosterSpec, normalize
from poster_agent.spec.brand import BrandProfile, load_brand

TOL_PT = 2.0  # position/size differences below this are not "edits"
HEX = re.compile(r"^#[0-9a-fA-F]{6}$")
SKIP_PREFIXES = ("decoration:", "background", "scrim ")


@dataclass
class Change:
    kind: str  # moved | resized | text_edited | recoloured | hidden | deleted | added
    id: str
    detail: str


@dataclass
class DocumentDiff:
    spec: PosterSpec  # the original spec updated to match the document
    bounds: dict[str, Box]
    overflow: list[str]
    changes: list[Change] = field(default_factory=list)


def top_level_nodes(info: dict[str, Any]) -> list[dict[str, Any]]:
    """The objects directly on the document's layers (groups are not opened: a decoration or photo crop is one object)."""
    return [n for layer in info.get("layers", []) for n in layer.get("children", [])]


def _box(n: dict[str, Any]) -> Box:
    b = n["bounds"]
    return Box(x=b["x"], y=b["y"], w=b["width"], h=b["height"])


def _differs(a: Box, b: Box) -> tuple[bool, bool]:
    return (abs(a.x - b.x) > TOL_PT or abs(a.y - b.y) > TOL_PT), (abs(a.w - b.w) > TOL_PT or abs(a.h - b.h) > TOL_PT)


def apply_document(spec: PosterSpec, nodes: list[dict[str, Any]]) -> DocumentDiff:
    """Compare the document's objects (matched by name = element id) with the spec."""
    by_name = {n.get("name"): n for n in nodes if n.get("name")}
    changes: list[Change] = []
    placed: list[PlacedElement] = []
    bounds: dict[str, Box] = {}
    overflow: list[str] = []
    known = {e.id for e in spec.placed}

    for el in spec.placed:
        if el.props.get("decorative"):
            placed.append(el)
            continue
        n = by_name.get(el.id)
        if n is None:
            changes.append(Change("deleted", el.id, f"{el.role} removed from the document"))
            continue
        if n.get("visible") is False:
            changes.append(Change("hidden", el.id, f"{el.role} is hidden"))
            continue
        new_box = _box(n)
        moved, resized = _differs(el.box, new_box)
        if moved:
            changes.append(Change("moved", el.id, f"({el.box.x:.0f},{el.box.y:.0f}) -> ({new_box.x:.0f},{new_box.y:.0f})"))
        if resized:
            changes.append(Change("resized", el.id, f"{el.box.w:.0f}x{el.box.h:.0f} -> {new_box.w:.0f}x{new_box.h:.0f}"))
        upd: dict[str, Any] = {"box": new_box}
        if el.kind == ElementKind.text:
            text = n.get("text")
            if isinstance(text, str) and normalize(text) != normalize(el.text or ""):
                changes.append(Change("text_edited", el.id, f"{(el.text or '')[:40]!r} -> {text[:40]!r}"))
                upd["text"] = text
            fill = n.get("fill")
            if isinstance(fill, str) and HEX.match(fill) and fill.lower() != (el.color or "").lower():
                changes.append(Change("recoloured", el.id, f"{el.color} -> {fill}"))
                upd["color"] = fill
        if n.get("overflow"):
            overflow.append(el.id)
        placed.append(el.model_copy(update=upd))
        bounds[el.id] = new_box

    for n in nodes:  # objects that are not in the spec: added by hand
        name = n.get("name") or ""
        if name in known or name.startswith(SKIP_PREFIXES) or name.startswith("<") or n.get("visible") is False:
            continue
        if n.get("kind") == "Type" and isinstance(n.get("text"), str) and n["text"].strip():
            eid = f"added.{n['id']}"
            color = n.get("fill") if isinstance(n.get("fill"), str) and HEX.match(n["fill"]) else "#000000"
            box = _box(n)
            placed.append(PlacedElement(id=eid, kind=ElementKind.text, role="added", text=n["text"], color=color, size=max(box.h * 0.6, 8.0),
                                        box=box, z=50))
            bounds[eid] = box
            changes.append(Change("added", eid, f"new text {n['text'][:40]!r}"))
        else:
            changes.append(Change("added", f"node {n.get('id')}", f"new {n.get('kind')} object {name!r}"))
    return DocumentDiff(spec=spec.model_copy(update={"placed": placed}), bounds=bounds, overflow=overflow, changes=changes)


def hand_edit_issues(original: PosterSpec, diff: DocumentDiff, brand: BrandProfile | None = None) -> list[Issue]:
    """Things the pixel checks cannot see but that matter: a removed brand logo, mandatory copy that no longer matches the brief."""
    issues: list[Issue] = []
    needs_logo = any(m.kind == "logo" for m in original.mandatory) or (brand is not None and "logo" in brand.mandatory_elements)
    if needs_logo and not any(e.role == "logo" for e in diff.spec.placed):
        issues.append(Issue(category=IssueCategory.logo_violation, severity="error", source="check",
                            evidence="the brand logo was removed or hidden: every poster must carry the institution logo",
                            suggested_fix="restore the logo object"))
    doc_text = normalize(" ".join(e.text or "" for e in diff.spec.placed if e.kind == ElementKind.text))
    for m in original.mandatory:
        if m.kind == "text" and m.text and normalize(m.text) not in doc_text:
            issues.append(Issue(category=IssueCategory.mandatory_missing, severity="warning", source="check",
                                evidence=f"mandatory text {m.text!r} no longer appears verbatim in the document (edited by hand?)"))
    return issues


@dataclass
class FinalizeResult:
    out_dir: Path
    png: Path
    pdf: Path
    changes: list[Change]
    report: CheckReport
    extra_issues: list[Issue]

    @property
    def ok(self) -> bool:
        return self.report.ok and not any(i.severity == "error" for i in self.extra_issues)

    def to_dict(self) -> dict:
        return {"ok": self.ok, "changes": [c.__dict__ for c in self.changes], "checks": self.report.to_dict(),
                "extra_issues": [i.model_dump(mode="json") for i in self.extra_issues], "png": str(self.png), "pdf": str(self.pdf)}


def resolve_target(target: Path) -> tuple[Path, Path]:
    """(run dir containing spec.json, the .vectorcraft file) from a run dir or a .vectorcraft path."""
    target = Path(target).resolve()
    run_dir = target if target.is_dir() else target.parent
    if not (run_dir / "spec.json").exists():
        raise FileNotFoundError(f"{run_dir}/spec.json not found: finalize needs the spec the poster was made from")
    if target.is_file():
        return run_dir, target
    files = sorted(run_dir.glob("*.vectorcraft"))
    if not files:
        raise FileNotFoundError(f"no .vectorcraft file in {run_dir}")
    return run_dir, files[0]


async def finalize(target: Path, *, use_ocr: bool = True, brand: BrandProfile | None = None) -> FinalizeResult:
    run_dir, vcf = resolve_target(target)
    spec = PosterSpec.model_validate_json((run_dir / "spec.json").read_text(encoding="utf-8"))
    brand = brand or load_brand(spec.brand)
    out_dir = run_dir / "final"
    out_dir.mkdir(exist_ok=True)
    png, pdf = out_dir / f"{vcf.stem}.png", out_dir / f"{vcf.stem}.pdf"
    async with VectorCraftSession(roots=[ROOT]) as vc:
        await vc.command("file.open", path=str(vcf))
        info = await vc.call("doc_inspect", {})
        await vc.export(png, scale=spec.canvas.export_scale)
        await vc.export(pdf)
    diff = apply_document(spec, top_level_nodes(info))
    compiled = CompileResult(png=png, pdf=pdf, vectorcraft=vcf, bounds=diff.bounds, overflow=diff.overflow)
    report = run_checks(diff.spec, brand, compiled, use_ocr=use_ocr)
    extra = hand_edit_issues(spec, diff, brand)
    result = FinalizeResult(out_dir, png, pdf, diff.changes, report, extra)
    (out_dir / "final_report.json").write_text(json.dumps(result.to_dict(), indent=1, ensure_ascii=False), encoding="utf-8")
    return result
