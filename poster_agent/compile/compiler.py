"""PosterSpec -> native VectorCraft MCP calls (guardrails 1 & 3).

`build_steps` is pure (spec -> command steps) so it is unit-testable without VectorCraft; `compile_spec`
executes the steps through a VectorCraftSession, exports PNG/PDF/.vectorcraft, and reads back real bounds.
Only `spec.placed` and `spec.background` are drawn — the compiler makes no layout decisions.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from poster_agent.spec import (BackgroundSpec, Box, ElementKind, PlacedElement, PosterSpec, ScrimSpec,
                               find_unsourced_text)

from .vc_client import ROOT, VectorCraftError, VectorCraftSession

MAX_BATCH_STEPS = 256
SCRIM_OVERSHOOT = 0.03  # extend gradient rectangles past the canvas so no 1px edge shows


class CompileError(RuntimeError):
    pass


@dataclass
class Unit:
    """Steps that draw one element/background layer. Refs are written as $<base+k> by `emit(base)`."""
    key: str  # element id, or "__bg__"/"__scrim_N__"
    emit: Any  # callable(base:int) -> (steps, index_of_main_node_step)
    main_ref: str = "id"  # how the main node is referenced in its result: "$N.id" or "$N.ids.0" (file.place)
    locked: bool = False  # brand-locked (logos): the object is locked in the document so a stray click cannot move it
    decorative: bool = False  # decoration: grouped as one object per decoration, not named one by one


def _fill_params(fill: Any) -> dict[str, Any]:
    if fill in (None, "none"):
        return {"none": True}
    if isinstance(fill, str):
        return {"color": fill}
    return dict(fill)  # a full paint.setFill params object, e.g. {"gradient": {...}}


def _scrim_gradient(s: ScrimSpec, w: float, h: float) -> tuple[dict[str, float], dict[str, Any]]:
    o = SCRIM_OVERSHOOT * min(w, h)
    ext = s.extent
    if s.edge in ("top", "bottom"):
        band = h * ext
        y = -o if s.edge == "top" else h - band
        rect = {"x": -o, "y": y, "width": w + 2 * o, "height": band + o}
        dense_y, clear_y = (-o, band) if s.edge == "top" else (h + o, h - band)
        start, end = [w / 2, dense_y], [w / 2, clear_y]
    else:
        band = w * ext
        x = -o if s.edge == "left" else w - band
        rect = {"x": x, "y": -o, "width": band + o, "height": h + 2 * o}
        dense_x, clear_x = (-o, band) if s.edge == "left" else (w + o, w - band)
        start, end = [dense_x, h / 2], [clear_x, h / 2]
    grad = {"gradient": {"kind": "linear", "start": start, "end": end, "stops": [
        {"offset": 0.0, "color": s.color, "opacity": s.opacity},
        {"offset": 1.0, "color": s.color, "opacity": 0.0}]}}
    return rect, grad


def _rect_unit(key: str, rect: dict[str, float], fill: Any, stroke: str | None = None,
               stroke_width: float | None = None, radius: float | None = None, opacity: float | None = None,
               ellipse: bool = False, rotation: float | None = None) -> Unit:
    def emit(base: int):
        shape = {"id": "shape.ellipse" if ellipse else "shape.rectangle", "params": dict(rect)}
        if radius and not ellipse:
            shape["params"]["radius"] = radius
        if rotation:
            shape["params"]["rotation"] = rotation  # degrees about the centre
        steps = [shape, {"id": "paint.setFill", "params": {"ids": [f"${base}.id"], **_fill_params(fill)}}]
        if stroke:
            steps.append({"id": "paint.setStroke", "params": {"ids": [f"${base}.id"], "color": stroke}})
            steps.append({"id": "stroke.set", "params": {"ids": [f"${base}.id"], "weight": stroke_width or 1.0}})
        else:
            steps.append({"id": "paint.setStroke", "params": {"ids": [f"${base}.id"], "none": True}})
        if opacity is not None:
            steps.append({"id": "object.setProps", "params": {"ids": [f"${base}.id"], "opacity": opacity * 100}})
        return steps, 0
    return Unit(key, emit)


def _cover_rect(path: Path, box: Box, focus: tuple[float, float] = (0.5, 0.5)) -> list[float]:
    """Rect (x, y, w, h) that scales the image to cover `box`, shifted so `focus` (0..1) stays in view."""
    from PIL import Image
    with Image.open(path) as im:
        aspect = im.width / im.height
    if aspect >= box.w / box.h:  # image is wider than the box: match height, crop the sides
        h, w = box.h, box.h * aspect
    else:
        w, h = box.w, box.w / aspect
    x = box.x - (w - box.w) * focus[0]
    y = box.y - (h - box.h) * focus[1]
    return [x, y, w, h]


def _image_unit(key: str, path: Path, box: Box, crop: str | None = None, radius: float | None = None,
                opacity: float | None = None, focus: tuple[float, float] = (0.5, 0.5), recolor: str | None = None) -> Unit:
    # circle/rounded/cover fill the box (cropping the overflow); no crop = fit inside, aspect kept
    rect = _cover_rect(path, box, focus) if crop in ("circle", "rounded", "cover") else [box.x, box.y, box.w, box.h]

    def emit(base: int):
        steps: list[dict[str, Any]] = [{"id": "file.place", "params": {"path": str(path), "rect": rect}}]
        main = 0
        if recolor and path.suffix.lower() == ".svg":  # an icon: editable vector strokes in the brand colour
            steps.append({"id": "paint.setStroke", "params": {"ids": [f"${base}.ids.0"], "color": recolor}})
        if crop in ("circle", "rounded", "cover"):
            mask = {"x": box.x, "y": box.y, "width": box.w, "height": box.h}
            if crop == "rounded" and radius:
                mask["radius"] = radius
            # the clipping path must be the topmost object, so create it after the image
            steps.append({"id": "shape.ellipse" if crop == "circle" else "shape.rectangle", "params": mask})
            steps.append({"id": "object.clippingMask.make", "params": {"ids": [f"${base}.ids.0", f"${base + 1}.id"]}})
            main = 2  # the clip group
        if opacity is not None:
            target = f"${base + main}.id" if main else f"${base}.ids.0"
            steps.append({"id": "object.setProps", "params": {"ids": [target], "opacity": opacity * 100}})
        return steps, main
    return Unit(key, emit, main_ref="id" if crop in ("circle", "rounded", "cover") else "ids.0")


def _text_unit(el: PlacedElement) -> Unit:
    box = el.box

    def emit(base: int):
        params: dict[str, Any] = {
            "text": el.text, "x": box.x, "y": box.y,
            "area": {"width": box.w, "height": box.h, "fit": el.props.get("fit", "none")},
            "size": el.size, "justify": el.align,
        }
        if el.font:
            params["font"] = el.font
        if el.style:
            params["style"] = el.style
        if el.color:
            params["color"] = el.color
        steps = [{"id": "text.create", "params": params}]
        style = {k: el.props[k] for k in ("leading", "tracking") if k in el.props}
        if style:
            steps.append({"id": "text.setStyle", "params": {"ids": [f"${base}.id"], **style}})
        if "opacity" in el.props:
            steps.append({"id": "object.setProps", "params": {"ids": [f"${base}.id"], "opacity": el.props["opacity"] * 100}})
        return steps, 0
    return Unit(el.id, emit)


def background_units(spec: PosterSpec, root: Path) -> list[Unit]:
    c, bg = spec.canvas, spec.background
    full = {"x": 0, "y": 0, "width": c.width, "height": c.height}
    units: list[Unit] = []
    if bg.kind == "solid":
        units.append(_rect_unit("__bg__", full, bg.color))
    elif bg.kind == "gradient":
        stops = [s.model_dump() for s in bg.gradient]
        units.append(_rect_unit("__bg__", full, {"gradient": {"kind": "linear", "angle": bg.angle, "stops": stops}}))
    else:  # texture / generated / photo: a pre-cropped asset covering the canvas
        if not bg.asset_id:
            raise CompileError(f"background kind {bg.kind!r} needs asset_id")
        units.append(_rect_unit("__bg_base__", full, bg.color))
        units.append(_image_unit("__bg__", root / spec.assets[bg.asset_id].path, Box(x=0, y=0, w=c.width, h=c.height)))
    for i, s in enumerate(bg.scrims):
        rect, grad = _scrim_gradient(s, c.width, c.height)
        units.append(_rect_unit(f"__scrim_{i}__", rect, grad))
    return units


def element_unit(el: PlacedElement, spec: PosterSpec, root: Path) -> Unit:
    p = el.props
    if el.kind == ElementKind.text:
        return _text_unit(el)
    if el.kind in (ElementKind.image, ElementKind.qr):
        asset = spec.assets[el.asset_id]
        return _image_unit(el.id, root / asset.path, el.box, p.get("crop"), p.get("radius"), p.get("opacity"),
                           tuple(p.get("focus", (0.5, 0.5))), p.get("recolor"))
    rect = {"x": el.box.x, "y": el.box.y, "width": el.box.w, "height": el.box.h}
    return _rect_unit(el.id, rect, p.get("fill"), p.get("stroke"), p.get("stroke_width"), p.get("radius"),
                      p.get("opacity"), ellipse=p.get("shape") == "ellipse", rotation=p.get("rotation"))


def build_units(spec: PosterSpec, root: Path = ROOT) -> list[Unit]:
    units = background_units(spec, root)
    for el in sorted(spec.placed, key=lambda e: e.z):
        u = element_unit(el, spec, root)
        u.locked, u.decorative = el.locked, bool(el.props.get("decorative"))
        units.append(u)
    return units


DISPLAY_NAMES = {"__bg__": "background", "__bg_base__": "background base"}


def display_name(key: str) -> str:
    """The layer name shown in VectorCraft: the element id (so a hand-edited file can be mapped back to the spec)."""
    if key.startswith("__scrim_"):
        return f"scrim {int(key[8:-2]) + 1}"
    return DISPLAY_NAMES.get(key, key)


def build_steps(units: list[Unit], first_base: int = 0) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Flatten units into one step list; returns steps and {unit key: index of its main node's step}.
    Every non-decorative object is named after its element id. Logos are NOT locked in the document: a person may move or resize
    one freely, and `finalize` re-checks the brand rules (size, clear space, field) on whatever they leave."""
    steps: list[dict[str, Any]] = []
    main_index: dict[str, int] = {}
    for u in units:
        base = first_base + len(steps)
        s, main = u.emit(base)
        main_index[u.key] = base + main
        steps.extend(s)
        if not u.decorative:
            props: dict[str, Any] = {"ids": [f"${base + main}.{u.main_ref}"], "name": display_name(u.key)}
            steps.append({"id": "object.setProps", "params": props})
    return steps, main_index


def watch_groups(units: list[Unit]) -> list[list[Unit]]:
    """What a human sees appear together: each decoration as one go, everything else one object at a time."""
    groups: list[list[Unit]] = []
    cur: list[Unit] = []
    cur_key: tuple[str, str] | None = None
    for u in units:
        k = ("decor", u.key.split(".")[1]) if u.decorative else ("one", u.key)
        if cur and k == cur_key and k[0] == "decor":
            cur.append(u)
            continue
        if cur:
            groups.append(cur)
        cur, cur_key = [u], k
    if cur:
        groups.append(cur)
    return groups


async def _run_units(vc: VectorCraftSession, units: list[Unit]) -> tuple[dict[str, dict], int]:
    """Execute units in as few batches as possible (refs are batch-local, so split at unit boundaries).
    Returns ({unit key: result of its main node's step}, steps executed)."""
    steps, main_index = build_steps(units)
    if len(steps) <= MAX_BATCH_STEPS:
        res = await vc.batch(steps)
        return {k: res["results"][i] for k, i in main_index.items()}, len(steps)
    out: dict[str, dict] = {}
    total, chunk, n = 0, [], 0

    async def flush() -> None:
        nonlocal chunk, total
        if not chunk:
            return
        s, mi = build_steps(chunk)
        res = await vc.batch(s)
        out.update({k: res["results"][i] for k, i in mi.items()})
        total += len(s)
        chunk = []

    for u in units:
        est = len(build_steps([u])[0])
        if n + est > MAX_BATCH_STEPS:
            await flush()
            n = 0
        chunk.append(u)
        n += est
    await flush()
    return out, total


async def _group_decorations(vc: VectorCraftSession, units: list[Unit], results: dict[str, dict], warnings: list[str]) -> None:
    """One selectable object per decoration (a dot grid is hundreds of dots: nobody wants to select them one by one)."""
    by_dec: dict[str, list[int]] = {}
    for u in units:
        if u.decorative and u.key in results:
            r = results[u.key].get("result", {})
            nid = r.get("id") if r.get("id") is not None else (r.get("ids") or [None])[0]
            if nid is not None:
                by_dec.setdefault(u.key.split(".")[1], []).append(nid)
    for dec_id, ids in by_dec.items():
        try:
            if len(ids) == 1:  # a single shape (e.g. a stripe): just name it
                await vc.command("object.setProps", ids=ids, name=f"decoration: {dec_id}")
                continue
            g = await vc.command("object.group", ids=ids)
            await vc.command("object.setProps", ids=[g["id"]], name=f"decoration: {dec_id}")
        except VectorCraftError as e:  # cosmetic: never fail a poster over it
            warnings.append(f"could not group decoration {dec_id}: {str(e)[:120]}")


@dataclass
class CompileResult:
    png: Path
    pdf: Path | None
    vectorcraft: Path | None
    node_ids: dict[str, int] = field(default_factory=dict)
    bounds: dict[str, Box] = field(default_factory=dict)  # real geometry read back from VectorCraft
    overflow: list[str] = field(default_factory=list)  # text elements whose text does not fit its frame
    step_count: int = 0
    warnings: list[str] = field(default_factory=list)


async def compile_spec(spec: PosterSpec, vc: VectorCraftSession, out_dir: Path, *, root: Path = ROOT,
                       allowed_text: list[str] | None = None, name: str = "poster", watched: bool = False, pace: float = 0.0,
                       doc_title: str | None = None) -> CompileResult:
    """Draw the spec. `watched`: execute step by step (each decoration as one go, everything else one object at a time) with
    `pace` seconds between steps, so a human watching the VectorCraft window can follow. The result is identical either way."""
    if allowed_text is not None:
        bad = find_unsourced_text(spec.placed, allowed_text)
        if bad:  # guardrail 3
            raise CompileError("text not traceable to brief/brand: " + "; ".join(f"{e.id}={e.text!r}" for e in bad))

    out_dir.mkdir(parents=True, exist_ok=True)
    c = spec.canvas
    await vc.command("file.new", width=c.width, height=c.height, name=doc_title or name)

    units = build_units(spec, root)
    results: dict[str, dict] = {}
    n_steps = 0
    if watched:
        for group in watch_groups(units):
            r, n = await _run_units(vc, group)
            results.update(r)
            n_steps += n
            if pace > 0:
                await asyncio.sleep(pace)
    else:
        results, n_steps = await _run_units(vc, units)
    group_warnings: list[str] = []
    await _group_decorations(vc, units, results, group_warnings)
    if watched:  # leave the window clean for the person: nothing selected
        try:
            await vc.command("select.none")
        except VectorCraftError:
            pass
    result_for = lambda key: results[key]  # noqa: E731
    steps = range(n_steps)

    node_ids: dict[str, int] = {}
    for u in units:
        r = result_for(u.key).get("result", {})
        nid = r.get("id") if r.get("id") is not None else (r.get("ids") or [None])[0]
        if nid is not None:
            node_ids[u.key] = nid

    stem = out_dir / name
    png = stem.with_suffix(".png")
    pdf = stem.with_suffix(".pdf")
    vcf = stem.with_suffix(".vectorcraft")
    warnings: list[str] = []
    for path, opts in ((png, {"scale": c.export_scale}), (pdf, {}), (vcf, {})):
        out = await vc.export(path, **opts)
        for w in (out.get("warnings") if isinstance(out, dict) else None) or []:
            warnings.append(f"{path.suffix}: {w}")

    # Read back real geometry for every placed element (checks use this, not the planned boxes).
    bounds: dict[str, Box] = {}
    overflow: list[str] = []
    wanted = {el.id: node_ids[el.id] for el in spec.placed if el.id in node_ids}
    if wanted:
        info = await vc.call("doc_inspect", {"ids": list(wanted.values())})
        by_id = {n["id"]: n for n in info.get("nodes", [])}
        for eid, nid in wanted.items():
            n = by_id.get(nid)
            if not n:
                continue
            b = n["bounds"]
            bounds[eid] = Box(x=b["x"], y=b["y"], w=b["width"], h=b["height"])
            if n.get("overflow"):
                overflow.append(eid)
    return CompileResult(png=png, pdf=pdf, vectorcraft=vcf, node_ids=node_ids, bounds=bounds, overflow=overflow,
                         step_count=len(steps), warnings=warnings + group_warnings)
