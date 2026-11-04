"""The component library, distilled from the catalog posters. The planner picks components + variants + content;
components own all geometry (guardrail: the LLM never places things)."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from PIL import Image

from poster_agent.checks.pixels import hex_rgb, rel_luminance
from poster_agent.spec import AssetKind, Box, ComponentSpec, ElementKind, IssueCategory, PlacedElement
from poster_agent.spec.brand import ROOT
from poster_agent.spec.formats import MM_TO_PT

from .base import Component, Ctx, register


def _gap(ctx: Ctx, factor: float = 1.0) -> float:
    return 0.012 * ctx.canvas.height * factor * ctx.scale


@lru_cache(maxsize=256)
def _aspect(path: str) -> float:
    with Image.open(ROOT / path) as im:
        return im.width / im.height


def _shape(id_: str, role: str, box: Box, z: int = 1, **props) -> PlacedElement:
    return PlacedElement(id=id_, kind=ElementKind.shape, role=role, box=box, z=z, props=props)


def _image(id_: str, role: str, box: Box, asset_id: str, z: int = 10, locked: bool = False, **props) -> PlacedElement:
    return PlacedElement(id=id_, kind=ElementKind.image, role=role, box=box, z=z, asset_id=asset_id, locked=locked, props=props)


@register
class LogoBlock(Component):
    """Brand logo with the brand's exclusion zone (10 mm) padded around it. Locked: nobody moves/resizes it."""
    name = "LogoBlock"

    def build(self, spec, ctx, x, y, w):
        c = spec.content
        logo_name = c.get("logo") or ctx.brand.default_logo
        logo = ctx.brand.logo(logo_name)
        clear = ctx.brand.logo_rules.clear_space_mm * MM_TO_PT
        rules = ctx.brand.logo_rules
        pct = min(max(logo.min_width_pct, c.get("width_pct", 16.0)), max(rules.max_width_pct, logo.min_width_pct))  # brand maximum
        lw = pct / 100.0 * ctx.canvas.width
        lh = lw / logo.aspect
        cap_h = rules.max_height_pct / 100.0 * ctx.canvas.height
        if lh > cap_h and cap_h * logo.aspect >= logo.min_width_pct / 100.0 * ctx.canvas.width:  # tall logo: cap its height too
            lh, lw = cap_h, cap_h * logo.aspect
        align = c.get("align", ctx.align) if (ctx.align == "left" or c.get("force_align")) else "center"  # one axis per poster
        plated = bool(c.get("plate")) and not logo.solid_field  # a logo with its own solid field needs no plate
        inset = clear if plated else 0.0  # the plate pads the logo by `clear`; it must stay inside the safe area
        lx = {"left": x + inset, "center": x + (w - lw) / 2, "right": x + w - lw - inset}[align]
        el = _image(f"{spec.id}.logo", "logo", Box(x=lx, y=y + clear, w=lw, h=lh), ctx.brand_logo_asset(logo_name), locked=True)
        els = [el]
        if plated:  # a flat plate hugging the logo and its exclusion zone (for logos that need a light field on a dark ground)
            col = c["plate"] if isinstance(c["plate"], str) else "surface"
            els.insert(0, _shape(f"{spec.id}.plate", "logo_plate", Box(x=lx - clear, y=y, w=lw + 2 * clear, h=lh + 2 * clear), z=1,
                                 fill=ctx.color(col)))
        return els, lh + 2 * clear


TITLE_STYLES = ("accent_last", "accent_first", "weight_split", "highlight_last")


def split_title(el: PlacedElement, ctx, style: str) -> list[PlacedElement]:
    """One element per line of a wrapped title so the lines can differ in colour or weight. The words and their order are unchanged;
    a one-line title is returned as it is."""
    lines = el.text.split("\n")
    if len(lines) < 2:
        words = el.text.split()
        if len(words) < 2:
            return [el]
        cut = len(words) - 1 if style in ("accent_last", "weight_split", "highlight_last") else 1  # the last word (or the first) stands apart
        head, tail = " ".join(words[:cut]), " ".join(words[cut:])
        if cut == 1 and style == "accent_first":
            head, tail = words[0], " ".join(words[1:])
        trk = float(el.props.get("tracking", 0.0))
        hw = ctx.measurer.width(head + " ", el.font, el.style, el.size, trk)
        tw = ctx.measurer.width(tail, el.font, el.style, el.size, trk)
        total = hw + tw
        x0 = {"left": el.box.x, "center": el.box.cx - total / 2, "right": el.box.right - total}.get(el.align, el.box.x)
        a = el.model_copy(deep=True, update={"id": f"{el.id}.l0", "text": head, "align": "left", "props": {**el.props, "fit": "autoHeight"},
                                             "box": Box(x=x0, y=el.box.y, w=hw, h=el.box.h)})
        b = el.model_copy(deep=True, update={"id": f"{el.id}.l1", "text": tail, "align": "left", "props": {**el.props, "fit": "autoHeight"},
                                             "box": Box(x=x0 + hw, y=el.box.y, w=tw + 2, h=el.box.h)})
        out1 = [a, b]
        target = b if style != "accent_first" or cut != 1 else a
        if style in ("accent_last", "accent_first"):
            target.color = ctx.color("accent")
        elif style == "weight_split":
            light = ctx.font("light") if "light" in ctx.brand.fonts else None
            if light is not None:
                b.font, b.style = light.family, light.style
        elif style == "highlight_last":
            pad = 0.18 * el.size
            out1.insert(0, _shape(f"{el.id}.hl", "highlight_box", Box(x=b.box.x - pad, y=el.box.y + 0.06 * el.box.h, w=tw + 2 * pad,
                                                                    h=0.9 * el.box.h), z=el.z - 1, fill=ctx.color("primary")))
            b.color = ctx.color("surface")
        return out1
    lead = float(el.props.get("leading") or el.size * 1.2)
    out = []
    for i, ln in enumerate(lines):
        last, first = i == len(lines) - 1, i == 0
        part = el.model_copy(deep=True, update={"id": f"{el.id}.l{i}", "text": ln,
                                                "box": Box(x=el.box.x, y=el.box.y + i * lead, w=el.box.w, h=lead),
                                                "props": {**el.props, "fit": "autoHeight"}})
        if (style == "accent_last" and last) or (style == "accent_first" and first):
            part.color = ctx.color("accent")
        elif style == "highlight_last" and last:  # a solid brand bar behind the last line, its text in the colour made for that fill
            pad = 0.18 * el.size
            tw = ctx.measurer.widest([ln], el.font, el.style, el.size, float(el.props.get("tracking", 0.0)))
            bx = {"left": el.box.x - pad, "center": el.box.cx - tw / 2 - pad, "right": el.box.right - tw - pad}.get(el.align, el.box.x - pad)
            out.append(_shape(f"{el.id}.hl", "highlight_box", Box(x=bx, y=el.box.y + i * lead + 0.06 * lead, w=tw + 2 * pad, h=0.9 * lead), z=el.z - 1,
                              fill=ctx.color("primary")))
            part.color = ctx.color("on_primary")
        elif style == "weight_split" and last:
            light = ctx.font("light") if "light" in ctx.brand.fonts else None
            if light is not None:
                part.font, part.style = light.family, light.style
        out.append(part)
    return out


@register
class TitleStack(Component):
    """eyebrow / title / subtitle / tagline. variant 'display' = huge single-word headline (ENGINEERS)."""
    name = "TitleStack"
    variants = ("default", "display")

    def build(self, spec, ctx, x, y, w):
        c, els, cur = spec.content, [], y
        align = c.get("align", ctx.align)
        role = "display" if spec.variant == "display" else "title"

        def add(part, text, role_, **kw):
            nonlocal cur
            if not text:
                return
            el, h = ctx.text(f"{spec.id}.{part}", role_, text, x, cur, w, align=align, source=spec.id, **kw)
            els.append(el)
            cur += h + _gap(ctx, 0.6)

        add("eyebrow", c.get("eyebrow"), "label", color=c.get("eyebrow_color", "accent"), tracking=c.get("eyebrow_tracking"),
            upper=True, max_lines=1)
        if c.get("title") and c.get("title_style") in TITLE_STYLES:
            el, h = ctx.text(f"{spec.id}.title", role, c["title"], x, cur, w, align=align, source=spec.id,
                             color=c.get("title_color", "text"), max_lines=c.get("max_lines", 3), font_role=c.get("title_font"),
                             factor=c.get("title_factor", 1.0))
            els += split_title(el, ctx, c["title_style"])
            cur += h + _gap(ctx, 0.6)
        else:
            add("title", c.get("title"), role, color=c.get("title_color", "text"), max_lines=c.get("max_lines", 3),
                font_role=c.get("title_font"), factor=c.get("title_factor", 1.0))
        add("subtitle", c.get("subtitle"), "subtitle", color=c.get("subtitle_color", "muted"), max_lines=3)
        add("tagline", c.get("tagline"), "body", color=c.get("tagline_color", "muted"), max_lines=3)
        return els, max(cur - y - _gap(ctx, 0.6), 0.0)


@register
class TextBlock(Component):
    name = "TextBlock"

    def build(self, spec, ctx, x, y, w):
        c = spec.content
        text = c.get("text") or "\n".join(c.get("lines", []))
        el, h = ctx.text(f"{spec.id}.text", c.get("role", "body"), text, x, y, w, color=c.get("color", "text"),
                         align=c.get("align", ctx.align), font_role=c.get("font_role"), upper=c.get("upper", False),
                         tracking=c.get("tracking"), max_lines=c.get("max_lines"), factor=c.get("factor", 1.0), source=spec.id)
        return [el], h


@register
class Divider(Component):
    """Hairline across the width, optionally with a centred label ('— DAY —')."""
    name = "Divider"
    variants = ("default", "ornament")

    def build(self, spec, ctx, x, y, w):
        c = spec.content
        color, thick = ctx.color(c.get("color", "muted")), c.get("thickness", 1.5)
        if spec.variant == "ornament":  # a short chain of small diamonds, centred (a quiet motif between info groups)
            n, d = c.get("count", 5), 0.0085 * ctx.canvas.width * ctx.scale
            step = d * 1.9
            x0 = x + (w - (n - 1) * step) / 2
            cy = y + d * 1.2
            return [_shape(f"{spec.id}.d{i}", "divider", Box(x=x0 + i * step - d / 2, y=cy - d / 2, w=d, h=d), fill=None, stroke=color,
                           stroke_width=max(1.0, thick * 0.7), rotation=45) for i in range(n)], d * 2.4
        label = c.get("label")
        if not label:
            return [_shape(f"{spec.id}.line", "divider", Box(x=x, y=y + 4, w=w, h=thick), fill=color)], thick + 8
        lbl, h = ctx.text(f"{spec.id}.label", c.get("role", "title"), label, x, y, w, color=c.get("label_color", "text"),
                          align="center", font_role=c.get("font_role"), tracking=c.get("tracking", 300), source=spec.id, max_lines=1)
        font = ctx.font(c.get("font_role") or "display")
        lw = ctx.measurer.width(lbl.text, font.family, font.style, lbl.size, lbl.props.get("tracking", 0))
        pad, ly = 0.03 * w, y + h * 0.5 - thick / 2
        side = max((w - lw) / 2 - pad, 0.0)
        return [lbl, _shape(f"{spec.id}.l", "divider", Box(x=x, y=ly, w=side, h=thick), fill=color),
                _shape(f"{spec.id}.r", "divider", Box(x=x + w - side, y=ly, w=side, h=thick), fill=color)], h


@register
class InfoBar(Component):
    """Event facts. 'band': full-bleed block at a canvas edge. 'columns': one row split into equal cells."""
    name = "InfoBar"
    variants = ("band", "columns", "footer")

    def _footer(self, spec, ctx, x, y, w):
        """The brand footer, the same on every poster: a primary band with a thin accent rule on top, the lines in ONE fixed small size,
        laid out in up to three columns with hairlines between them (rows of three when there are more), on the poster margins."""
        c = spec.content
        items = [t for t in c["items"] if t]
        H, safe = ctx.canvas.height, ctx.canvas.safe_box
        pad = 0.018 * H * ctx.scale
        probe, _ = ctx.text(f"{spec.id}.probe", "caption", items[0], safe.x, y, safe.w, factor=1.1)  # the one footer size (never shrunk per line)
        fnt, sty, size = probe.font, probe.style, probe.size
        cols = 1
        for n in (min(len(items), 3), min(len(items), 2), 1):  # the most columns in which every line still wraps to at most 2 lines
            if all(len(ls) <= 2 and ctx.measurer.widest(ls, fnt, sty, size) <= safe.w / n * 0.94
                   for ls in (ctx.measurer.wrap(t, fnt, sty, size, safe.w / n * 0.94) for t in items)):
                cols = n
                break
        cw = safe.w / cols
        color = c.get("text_color", "surface")
        els, cur = [], y + pad * 1.3
        for r in range(0, len(items), cols):
            row, row_h, cells = items[r:r + cols], 0.0, []
            for j, it in enumerate(row):
                inset = 0.0 if j == 0 else 0.04 * cw
                el, h = ctx.text(f"{spec.id}.item{r + j}", "caption", it, safe.x + j * cw + inset, cur, cw - inset - (0.02 * cw if j < cols - 1 else 0.0),
                                 color=color, align="left" if (cols > 1 or ctx.align == "left") else "center", size=size, source=spec.id)
                cells.append(el)
                row_h = max(row_h, h)
            for j in range(1, len(row)):
                els.append(_shape(f"{spec.id}.sep{r + j}", "divider", Box(x=safe.x + j * cw - 1, y=cur, w=2, h=row_h), z=5,
                                  fill=ctx.color(c.get("rule_color", "accent")), opacity=0.7))
            els += cells
            cur += row_h + (0.012 * H * ctx.scale if r + cols < len(items) else 0.0)
        total = cur - y + pad
        els.insert(0, _shape(f"{spec.id}.bg", "info_bar", Box(x=x, y=y, w=w, h=total), z=4, fill=ctx.color(c.get("fill", "primary"))))
        els.insert(1, _shape(f"{spec.id}.rule", "divider", Box(x=x, y=y, w=w, h=max(3.0, 0.0022 * H)), z=5, fill=ctx.color("accent")))
        return els, total

    def build(self, spec, ctx, x, y, w):
        if spec.variant == "footer":
            return self._footer(spec, ctx, x, y, w)
        c = spec.content
        items = c["items"]
        pad = 0.03 * ctx.canvas.width * ctx.scale
        fill = c.get("fill", "primary")
        text_color = c.get("text_color", "on_primary")
        fill_prop = ({"gradient": {"kind": "linear", "start": [x, 0], "end": [x + w, 0], "stops": [
            {"offset": 0.0, "color": ctx.color(fill)}, {"offset": 1.0, "color": ctx.color(c["fill_to"])}]}}
            if c.get("fill_to") else ctx.color(fill))
        els: list[PlacedElement] = []
        if spec.variant == "columns":
            n = len(items)
            cw = (w - 2 * pad) / n
            row_h = 0.0
            cells = []
            for i, it in enumerate(items):
                el, h = ctx.text(f"{spec.id}.item{i}", "subtitle", it, x + pad + i * cw, y + pad, cw, color=text_color,
                                 align="center", font_role="display", max_lines=2, source=spec.id)
                cells.append(el)
                row_h = max(row_h, h)
            total = row_h + 2 * pad
            els.append(_shape(f"{spec.id}.bg", "info_bar", Box(x=x, y=y, w=w, h=total), z=1, fill=fill_prop))
            els += cells
            for i in range(1, n):  # thin separators
                els.append(_shape(f"{spec.id}.sep{i}", "divider", Box(x=x + pad + i * cw - 1, y=y + pad * 1.1, w=2, h=row_h * 0.8),
                                  z=2, fill=ctx.color(text_color), opacity=0.6))
            return els, total
        cur = y + pad
        safe = ctx.canvas.safe_box
        tx, tw = (safe.x, safe.w) if x <= 0.5 else (x + pad, w - 2 * pad)  # a band on a canvas edge: text on the poster margin
        for i, it in enumerate(items):
            el, h = ctx.text(f"{spec.id}.item{i}", "heading", it, tx, cur, tw, color=text_color, align=ctx.align,
                             max_lines=2, factor=0.85 if len(items) <= 2 else 0.68, source=spec.id)  # a long line (an address) may wrap
            els.append(el)
            cur += h
        total = cur - y + pad
        els.insert(0, _shape(f"{spec.id}.bg", "info_bar", Box(x=x, y=y, w=w, h=total), z=1, fill=fill_prop))
        return els, total

    def full_bleed(self, spec):
        return spec.anchor != "flow"


def _circle_photo(ctx: Ctx, id_: str, asset: str | None, cx: float, y: float, d: float, ring: bool = True,
                  framing: str | None = None) -> list[PlacedElement]:
    if not asset or asset not in ctx.assets:
        ctx.issue(IssueCategory.photo_treatment, f"{id_}: missing photo asset {asset!r}", [id_], "supply the photo")
        return []
    box = Box(x=cx - d / 2, y=y, w=d, h=d)
    props = {"framing": framing} if framing else {}  # head / head_and_shoulders / half_body: the pipeline reframes a copy of the photo
    els = [_image(id_, "headshot", box, asset, crop="circle", focus=(0.5, 0.3), **props)]
    if ring:
        els.append(_shape(f"{id_}.ring", "headshot_ring", box, z=11, shape="ellipse", fill=None,
                          stroke=ctx.color("primary"), stroke_width=max(2.0, d * 0.02)))
    return els


@register
class ChiefGuestCard(Component):
    name = "ChiefGuestCard"

    def build(self, spec, ctx, x, y, w):
        c, els, cur = spec.content, [], y
        align = c.get("align", ctx.align)
        if c.get("label"):
            el, h = ctx.text(f"{spec.id}.label", "label", c["label"], x, cur, w, color=c.get("label_color", "accent"), align=align, upper=True,
                             tracking=140, max_lines=1, source=spec.id)
            els.append(el)
            cur += h + _gap(ctx, 0.4)
        if c.get("photo"):  # the photo is optional: a card without one is label + name + designation
            d = c.get("photo_d_pct", 19.0) / 100.0 * ctx.canvas.width * ctx.scale
            cx = {"left": x + d / 2, "center": x + w / 2, "right": x + w - d / 2}[align]
            els += _circle_photo(ctx, f"{spec.id}.photo", c.get("photo"), cx, cur, d, framing=c.get("framing", "face"))
            cur += d + _gap(ctx, 1.0)
        el, h = ctx.text(f"{spec.id}.name", "heading", c["name"], x, cur, w, color=c.get("name_color", "primary"), align=align, max_lines=2,
                         factor=1.25, source=spec.id)  # the name is the largest, boldest line of the card
        els.append(el)
        cur += h + _gap(ctx, 0.3)
        for i, ln in enumerate(c.get("lines", [])):
            el, h = ctx.text(f"{spec.id}.line{i}", "caption", ln, x, cur, w, color=c.get("lines_color", "text"), align=align, max_lines=2,
                             source=spec.id)
            els.append(el)
            cur += h
        return els, cur - y


@register
class SpeakerGrid(Component):
    """Circular headshots in aligned rows; the last row is centred (19 speakers, 4 cols -> 4/4/4/4/3)."""
    name = "SpeakerGrid"

    def build(self, spec, ctx, x, y, w):
        c = spec.content
        speakers = c["speakers"]
        cols = max(1, min(c.get("cols", 4), len(speakers)))
        col_w = w / cols
        d = min(0.5 * col_w, 0.105 * ctx.canvas.width) * ctx.scale ** 0.5
        els: list[PlacedElement] = []
        cur = y
        for r in range(0, len(speakers), cols):
            row = speakers[r:r + cols]
            x0 = x + (cols - len(row)) * col_w / 2
            row_h = 0.0
            for i, sp in enumerate(row):
                sid = f"{spec.id}.s{r + i}"
                cx = x0 + (i + 0.5) * col_w
                els += _circle_photo(ctx, f"{sid}.photo", sp.get("asset"), cx, cur, d, ring=False, framing=c.get("framing", "face"))
                ty = cur + d + _gap(ctx, 0.5)
                nm, nh = ctx.text(f"{sid}.name", "micro", sp["name"], x0 + i * col_w, ty, col_w * 0.96, color="primary",
                                  align="center", font_role="display", max_lines=2, source=spec.id)
                els.append(nm)
                h = d + _gap(ctx, 0.5) + nh
                if sp.get("designation"):
                    ds, dh = ctx.text(f"{sid}.role", "micro", sp["designation"], x0 + i * col_w, ty + nh, col_w * 0.96,
                                      color="text", align="center", max_lines=c.get("role_max_lines", 7), source=spec.id)
                    els.append(ds)
                    h += dh
                row_h = max(row_h, h)
            cur += row_h + _gap(ctx, 1.1)
        return els, cur - y - _gap(ctx, 1.1)


@register
class PartnerLogoRow(Component):
    """Supplied partner logos at one height, spaced evenly (client logos arrive as supplied assets)."""
    name = "PartnerLogoRow"

    def build(self, spec, ctx, x, y, w):
        ids = [a for a in spec.content["assets"] if a in ctx.assets]
        if not ids:
            return [], 0.0
        h = spec.content.get("height_pct", 5.5) / 100.0 * ctx.canvas.height * ctx.scale
        widths = [h * _aspect(ctx.assets[a].path) for a in ids]
        total = sum(widths)
        if total > w:  # shrink uniformly to fit the row
            k = w / total
            h, widths = h * k, [v * k for v in widths]
            total = w
        gap = (w - total) / (len(ids) + 1)
        els, cx = [], x + gap
        for i, (a, aw) in enumerate(zip(ids, widths)):
            els.append(_image(f"{spec.id}.p{i}", "partner_logo", Box(x=cx, y=y, w=aw, h=h), a, locked=True))
            cx += aw + gap
        return els, h


@register
class PhotoStrip(Component):
    """Full-bleed band of N photos (cover-cropped), each over an optional tinted column."""
    name = "PhotoStrip"

    def build(self, spec, ctx, x, y, w):
        c = spec.content
        ids = c["assets"]
        h = c.get("height_pct", 24.0) / 100.0 * ctx.canvas.height
        cw = w / len(ids)
        tints = c.get("tints", [])
        els: list[PlacedElement] = []
        for i, a in enumerate(ids):
            cell = Box(x=x + i * cw, y=y, w=cw, h=h)
            if i < len(tints):
                els.append(_shape(f"{spec.id}.tint{i}", "photo_tint", cell, z=2, fill={"gradient": {
                    "kind": "linear", "start": [cell.cx, y], "end": [cell.cx, y + h], "stops": [
                        {"offset": 0.0, "color": "#ffffff", "opacity": 0.0}, {"offset": 1.0, "color": tints[i]}]}}))
            els.append(_image(f"{spec.id}.p{i}", "photo", cell, a, z=3, crop=c.get("crop", "cover"),
                              focus=tuple(c.get("focus", (0.5, 0.5))),
                              **({"treatment": c["treatment"], "duotone": list(c.get("duotone", ["primary", "surface"]))}
                                 if c.get("treatment") else {})))
        return els, h


@register
class Panel(Component):
    """Filled container laying its children out as an inner stack (e.g. the navy hero panel)."""
    name = "Panel"

    def build(self, spec, ctx, x, y, w):
        from .solver import stack
        pad = spec.content.get("padding_pct", 6.0) / 100.0 * ctx.canvas.width * ctx.scale
        inner, h = stack(spec.children, ctx, Box(x=x + pad, y=y + pad, w=w - 2 * pad, h=0), gap_scale=spec.content.get("gap", 1.0))
        total = h + 2 * pad
        fill = ctx.color(spec.content.get("fill", "primary"))
        if ctx.ground and fill.lower() == ctx.ground.lower():
            # A panel the same colour as the page is invisible as a shape but would hide every decoration behind it: only group.
            return inner, total
        bg = _shape(f"{spec.id}.bg", "panel", Box(x=x, y=y, w=w, h=total), z=1, fill=fill, radius=spec.content.get("radius"))
        return [bg] + inner, total


@register
class HighlightBox(Component):
    """Key phrase in an outlined (or filled) box — the event/programme name. variants: outline | filled."""
    name = "HighlightBox"
    variants = ("outline", "filled")

    def build(self, spec, ctx, x, y, w):
        c = spec.content
        pad = 0.022 * ctx.canvas.width * ctx.scale
        bw = c.get("width_pct", 62.0) / 100.0 * w
        bx = x + (w - bw) / 2
        filled = spec.variant == "filled"
        text_color = c.get("text_color", "on_primary" if filled else "text")
        el, h = ctx.text(f"{spec.id}.text", c.get("role", "heading"), c["text"], bx + pad, y + pad, bw - 2 * pad, color=text_color,
                         align="center", font_role=c.get("font_role", "display"), max_lines=c.get("max_lines", 2), source=spec.id)
        total = h + 2 * pad
        edge = ctx.color(c.get("edge", "accent" if filled else "muted"))
        box = _shape(f"{spec.id}.box", "highlight_box", Box(x=bx, y=y, w=bw, h=total), z=2,
                     fill=ctx.color(c.get("fill", "primary")) if filled else None,
                     **({} if filled else {"stroke": edge, "stroke_width": max(2.0, 0.003 * ctx.canvas.width)}))
        return [box, el], total


@register
class LogoTitleRow(Component):
    """Logo on the left, a thin rule, then a left-aligned eyebrow/title column (conference-style header)."""
    name = "LogoTitleRow"

    def build(self, spec, ctx, x, y, w):
        c = spec.content
        logo_name = c.get("logo") or ctx.brand.default_logo
        logo = ctx.brand.logo(logo_name)
        lw = c.get("left_pct", 26.0) / 100.0 * w
        lh = lw / logo.aspect
        els = [_image(f"{spec.id}.logo", "logo", Box(x=x, y=y, w=lw, h=lh), ctx.brand_logo_asset(logo_name), locked=True)]
        gap = 0.03 * w
        tx, tw, cur = x + lw + 2 * gap, w - lw - 2 * gap, y
        els.append(_shape(f"{spec.id}.rule", "divider", Box(x=x + lw + gap - 1, y=y, w=2, h=lh), fill=ctx.color("muted")))
        if c.get("eyebrow"):
            el, h = ctx.text(f"{spec.id}.eyebrow", "caption", c["eyebrow"], tx, cur, tw, color=c.get("eyebrow_color", "primary"),
                             font_role="display", max_lines=2, source=spec.id)
            els.append(el)
            cur += h + _gap(ctx, 0.3)
        el, h = ctx.text(f"{spec.id}.title", "title", c["title"], tx, cur, tw, color=c.get("title_color", "text"), max_lines=4,
                         factor=c.get("title_factor", 1.0), source=spec.id)
        els.append(el)
        cur += h
        total = max(lh, cur - y)
        if c.get("caption"):  # e.g. the cluster/department under the logo
            clear = ctx.brand.logo_rules.clear_space_mm * MM_TO_PT  # nothing may sit inside the logo's exclusion zone
            cap, ch = ctx.text(f"{spec.id}.caption", "caption", c["caption"], x, y + lh + clear + 2, lw, color="primary",
                               align="center", font_role="display", max_lines=3, source=spec.id)
            els.append(cap)
            total = max(total, lh + clear + 2 + ch)
        return els, total


@register
class HeroImage(Component):
    """A single large image (usually a generated RGBA cutout) centred in the flow. Keeps its own aspect ratio, and grows into spare
    vertical space (elastic) so a hero is never lost in empty area; `fill: false` keeps it at height_pct."""
    name = "HeroImage"
    elastic = True

    def build(self, spec, ctx, x, y, w):
        c = spec.content
        aid = c["asset"]
        if aid not in ctx.assets:
            ctx.issue(IssueCategory.photo_treatment, f"{spec.id}: missing asset {aid!r}", [spec.id], "supply or generate the image")
            return [], 0.0
        aspect = _aspect(ctx.assets[aid].path)
        h = c.get("height_pct", 34.0) / 100.0 * ctx.canvas.height * ctx.scale ** 0.5 + ctx.extra.get(spec.id, 0.0)
        h = min(h, 0.62 * ctx.canvas.height)
        bw = h * aspect
        wmax = c.get("max_width_pct", 100.0) / 100.0 * w
        if bw > wmax:
            bw, h = wmax, wmax / aspect
        frame = c.get("frame")
        if frame is None:  # a generated cutout has no background to put in a card; a photo does
            frame = ctx.assets[aid].kind != AssetKind.generated_hero
        if frame is False:
            bx = {"left": x, "center": x + (w - bw) / 2, "right": x + w - bw}[c.get("align", ctx.align)]
            return [_image(f"{spec.id}.img", "hero", Box(x=bx, y=y, w=bw, h=h), aid, z=8)], h
        return self._framed(spec, ctx, aid, x, y, w, bw, h, aspect)

    def _framed(self, spec, ctx, aid, x, y, w, bw, h, aspect):
        """The hero sits in a clean tonal card with a hairline edge and two accent corner brackets: it reads as a deliberate
        presentation piece, and the card gives the hero visual weight without ornament."""
        W = ctx.canvas.width
        pad = 0.028 * W * ctx.scale
        if bw > w - 2 * pad:
            bw, h = w - 2 * pad, (w - 2 * pad) / aspect
        cw, ch = bw + 2 * pad, h + 2 * pad
        cx = {"left": x, "center": x + (w - cw) / 2, "right": x + w - cw}[spec.content.get("align", ctx.align)]
        dark = bool(ctx.ground) and rel_luminance(hex_rgb(ctx.ground)) < 0.25
        accent = ctx.color("accent")
        edge = max(2.0, 0.0035 * W)
        arm = 0.07 * W  # bracket arm length
        card = _shape(f"{spec.id}.card", "hero_frame", Box(x=cx, y=y, w=cw, h=ch), z=3,
                      fill=ctx.color("surface" if dark else "primary"), opacity=0.08 if dark else 0.05)
        brackets = []
        for i, (bxx, byy, sx, sy) in enumerate(((cx, y, 1, 1), (cx + cw, y + ch, -1, -1))):
            for j, (rw, rh) in enumerate(((arm, edge), (edge, arm))):
                rx = bxx if sx > 0 else bxx - rw
                ry = byy if sy > 0 else byy - rh
                brackets.append(_shape(f"{spec.id}.br{i}{j}", "hero_frame", Box(x=rx, y=ry, w=rw, h=rh), z=4, fill=accent))
        img = _image(f"{spec.id}.img", "hero", Box(x=cx + pad, y=y + pad, w=bw, h=h), aid, z=8)
        return [card, *brackets, img], ch
