"""Structural components: the ways designers fill a poster without ornament - scale (BigTitle), structure (Columns, TabBanner,
ImageBlock) and image integration (PhotoFade). All geometry is derived from the width/height they are given and the brand."""
from __future__ import annotations

from poster_agent.spec import AssetKind, AssetRef, Box, ElementKind, IssueCategory, PlacedElement
from poster_agent.spec.brand import ROOT

from .base import Component, register
from .components import _aspect, _gap, _image, _shape


def _treat(c: dict) -> dict:
    """Props asking for a brand colour treatment on a supplied photo (applied by assets.treatments before compiling)."""
    return {"treatment": c["treatment"], "duotone": list(c.get("duotone", ["primary", "surface"]))} if c.get("treatment") else {}


def _shift_y(els: list[PlacedElement], dy: float) -> list[PlacedElement]:
    return [e.model_copy(update={"box": e.box.model_copy(update={"y": e.box.y + dy})}) for e in els]


@register
class BigTitle(Component):
    """The headline as graphic type: every line is scaled to span the full width (capped at `cap_pct` of the poster height), so a
    short title fills the poster the way a designer's oversized wordmark does. Optional pill label under it.
    content: lines (list) or title (newline separated), colors (one palette name per line), pill (text), pill_fill, pill_text."""
    name = "BigTitle"

    def build(self, spec, ctx, x, y, w):
        c = spec.content
        lines = c.get("lines") or str(c["title"]).split("\n")
        colors = c.get("colors") or []
        font_role = c.get("font_role", "display_heavy")
        font = ctx.font(font_role)
        cap = c.get("cap_pct", 13.0) / 100.0 * ctx.canvas.height * ctx.scale
        els: list[PlacedElement] = []
        cur = y
        for i, line in enumerate(lines):
            w100 = max(ctx.measurer.width(line, font.family, font.style, 100.0), 1.0)
            size = min(100.0 * 0.96 * w / w100, cap)
            el, h = ctx.text(f"{spec.id}.l{i}", "display", line, x, cur, w, size=size,
                             color=colors[i] if i < len(colors) else c.get("color", "primary"), align=c.get("align", ctx.align),
                             font_role=font_role, max_lines=1, source=spec.id)
            els.append(el)
            cur += h
        total = cur - y + 0.08 * (els[-1].size or 0) if els else 0.0
        if c.get("pill"):
            pad = 0.022 * ctx.canvas.width * ctx.scale
            frole = ctx.font("display")
            psize = ctx.size("heading", 0.95)
            pw = min(1.08 * ctx.measurer.width(c["pill"].upper(), frole.family, frole.style, psize, 100) + 4 * pad, w)
            px = {"left": x, "center": x + (w - pw) / 2, "right": x + w - pw}[c.get("align", ctx.align)]
            pt, ph = ctx.text(f"{spec.id}.pill", "heading", c["pill"], px + 2 * pad, y + total + pad * 0.6, pw - 4 * pad,
                              color=c.get("pill_text", "on_primary"), align="center", size=psize, tracking=100, upper=True,
                              max_lines=1, source=spec.id)
            els.append(_shape(f"{spec.id}.pillbg", "highlight_box", Box(x=px, y=y + total, w=pw, h=ph + pad * 1.2), z=2,
                              fill=ctx.color(c.get("pill_fill", "primary")), radius=(ph + pad * 1.2) / 2))
            els.append(pt)
            total += ph + pad * 1.2 + pad * 0.4
        return els, total


@register
class TabBanner(Component):
    """A solid brand tab hanging from the top edge with one line of text (a call to action / strapline). Use anchor 'top'."""
    name = "TabBanner"

    def build(self, spec, ctx, x, y, w):
        c = spec.content
        pad = 0.018 * ctx.canvas.width * ctx.scale
        top = max(pad * 0.6, ctx.brand.safe_margin_pct / 100.0 * ctx.canvas.height * 0.9 - y)
        tw = c.get("width_pct", 64.0) / 100.0 * w
        el, h = ctx.text(f"{spec.id}.text", "heading", c["text"], x + (w - tw) / 2 + pad, y + top, tw - 2 * pad,
                         color=c.get("text_color", "on_primary"), align="center", font_role="display", max_lines=1, upper=True,
                         source=spec.id)
        total = h + top + pad * 0.6
        bg = _shape(f"{spec.id}.bg", "info_bar", Box(x=x + (w - tw) / 2, y=y, w=tw, h=total), z=1,
                    fill=ctx.color(c.get("fill", "primary")))
        return [bg, el], total


@register
class ImageBlock(Component):
    """A supplied image at its own aspect ratio (QR code, small figure), optionally in a thin bordered white card."""
    name = "ImageBlock"

    def build(self, spec, ctx, x, y, w):
        c = spec.content
        aid = c["asset"]
        if aid not in ctx.assets:
            ctx.issue(IssueCategory.photo_treatment, f"{spec.id}: missing asset {aid!r}", [spec.id], "supply the image")
            return [], 0.0
        aspect = _aspect(ctx.assets[aid].path)
        pad = 0.012 * ctx.canvas.width * ctx.scale if c.get("border", True) else 0.0
        iw = min(c.get("width_pct", 60.0) / 100.0 * w, w) - 2 * pad
        ih = iw / aspect
        cw = iw + 2 * pad
        cx = {"left": x, "center": x + (w - cw) / 2, "right": x + w - cw}[c.get("align", ctx.align)]
        els: list[PlacedElement] = []
        if pad:
            els.append(_shape(f"{spec.id}.card", "image_card", Box(x=cx, y=y, w=cw, h=ih + 2 * pad), z=2, fill="#ffffff",
                              stroke=ctx.color(c.get("edge", "primary")), stroke_width=max(1.5, 0.002 * ctx.canvas.width)))
        els.append(_image(f"{spec.id}.img", "image", Box(x=cx + pad, y=y + pad, w=iw, h=ih), aid, z=8))
        return els, ih + 2 * pad


@register
class Columns(Component):
    """Two side-by-side columns separated by a hairline. Each child sets content.column = 'left' | 'right'; the shorter column is
    centred vertically against the taller one. content: left_pct (default 55), rule (default true)."""
    name = "Columns"

    def build(self, spec, ctx, x, y, w):
        from .solver import stack
        c = spec.content
        gutter = 0.06 * w
        lw = c.get("left_pct", 55.0) / 100.0 * w - gutter / 2
        rw = w - lw - gutter
        left = [k for k in spec.children if k.content.get("column", "left") == "left"]
        right = [k for k in spec.children if k.content.get("column") == "right"]
        lels, lh = stack(left, ctx, Box(x=x, y=y, w=lw, h=0))
        rels, rh = stack(right, ctx, Box(x=x + lw + gutter, y=y, w=rw, h=0))
        total = max(lh, rh)
        k = {"top": 0.0, "center": 0.5, "bottom": 1.0}[c.get("valign", "center")]  # where the shorter column sits
        if rh < total:
            rels = _shift_y(rels, (total - rh) * k)
        elif lh < total:
            lels = _shift_y(lels, (total - lh) * k)
        els = lels + rels
        if c.get("rule", True) and left and right:
            els.append(_shape(f"{spec.id}.rule", "divider", Box(x=x + lw + gutter / 2 - 1, y=y + 0.04 * total, w=2, h=0.92 * total),
                              z=2, fill=ctx.color("muted"), opacity=0.5))
        return els, total


@register
class PhotoFade(Component):
    """A supplied photo bleeding off the bottom (or top) edge, full width, dissolving into the page colour through a vector
    gradient: the photo itself is untouched. Use anchor 'bottom' (or 'top' to fade at the photo's bottom edge)."""
    name = "PhotoFade"

    def build(self, spec, ctx, x, y, w):
        c = spec.content
        h = c.get("height_pct", 38.0) / 100.0 * ctx.canvas.height
        fade = h * c.get("fade_pct", 45.0) / 100.0
        col = ctx.color(c["fade_color"]) if c.get("fade_color") else (ctx.ground or ctx.color("surface"))
        at_top = spec.anchor != "top"  # a photo at the bottom fades at its top edge; a top photo fades at its bottom edge
        start_y, end_y = (y, y + fade) if at_top else (y + h, y + h - fade)
        grad = {"gradient": {"kind": "linear", "start": [x + w / 2, start_y], "end": [x + w / 2, end_y], "stops": [
            {"offset": 0.0, "color": col, "opacity": 1.0}, {"offset": 1.0, "color": col, "opacity": 0.0}]}}
        veil = Box(x=x, y=y if at_top else y + h - fade, w=w, h=fade)
        photo = _image(f"{spec.id}.img", "photo", Box(x=x, y=y, w=w, h=h), c["asset"], z=3, crop="cover",
                       focus=tuple(c.get("focus", (0.5, 0.5))), **_treat(c))
        return [photo, _shape(f"{spec.id}.fade", "photo_tint", veil, z=4, fill=grad)], h


ICON_DIR = ROOT / "brands/icons"


def icon_names() -> list[str]:
    return sorted(p.stem for p in ICON_DIR.glob("*.svg"))


@register
class IconRow(Component):
    """A row of icon + label columns separated by thin vertical rules (e.g. domains of interest). Icons are vector (Lucide set,
    brands/icons); the planner picks an icon NAME from the list, labels are verbatim from the brief. content.items =
    [{icon, label}]. The icon is drawn in `icon_color` (default primary)."""
    name = "IconRow"

    def build(self, spec, ctx, x, y, w):
        c = spec.content
        items = c["items"]
        n = len(items)
        cw = w / n
        isz = min(0.62 * cw, 0.075 * ctx.canvas.width) * ctx.scale ** 0.5
        col = ctx.color(c.get("icon_color", "primary"))
        els: list[PlacedElement] = []
        label_h = 0.0
        for i, it in enumerate(items):
            aid = f"icon_{it['icon']}"
            if aid not in ctx.assets:
                path = (ICON_DIR / f"{it['icon']}.svg")
                if not path.exists():
                    ctx.issue(IssueCategory.photo_treatment, f"{spec.id}: unknown icon {it['icon']!r}", [spec.id], "pick an icon from the list")
                    continue
                ctx.assets[aid] = AssetRef(id=aid, kind=AssetKind.icon, path=path.relative_to(ROOT).as_posix(), provenance="brand")
            cx = x + (i + 0.5) * cw
            els.append(_image(f"{spec.id}.i{i}", "icon", Box(x=cx - isz / 2, y=y, w=isz, h=isz), aid, z=8, recolor=col))
            lbl, lh = ctx.text(f"{spec.id}.l{i}", "caption", it["label"], x + i * cw + 0.06 * cw, y + isz + _gap(ctx, 0.5),
                               cw * 0.88, color=c.get("label_color", "text"), align="center", font_role="display", max_lines=3,
                               source=spec.id, upper=True)
            els.append(lbl)
            label_h = max(label_h, lh)
        total = isz + _gap(ctx, 0.5) + label_h
        for i in range(1, n):
            els.append(_shape(f"{spec.id}.sep{i}", "divider", Box(x=x + i * cw - 1, y=y + 0.1 * total, w=2, h=0.8 * total), z=2,
                              fill=ctx.color("muted"), opacity=0.5))
        return els, total


@register
class BigWord(Component):
    """One oversized word or number taken verbatim from the brief (a year, '30', 'JOIN') as the poster's graphic, with an optional
    caption beside it (caption_pos 'right', default) or below it. content: text, caption, color, word_pct (width of the word)."""
    name = "BigWord"

    def build(self, spec, ctx, x, y, w):
        c = spec.content
        right = c.get("caption_pos", "right") == "right" and c.get("caption")
        ww = w * (c.get("word_pct", 58.0) / 100.0 if right else 1.0)
        font_role = c.get("font_role", "display_heavy")
        font = ctx.font(font_role)
        w100 = max(ctx.measurer.width(c["text"], font.family, font.style, 100.0), 1.0)
        size = min(100.0 * 0.96 * ww / w100, 0.24 * ctx.canvas.height * ctx.scale)
        word, wh = ctx.text(f"{spec.id}.word", "display", c["text"], x, y, ww, size=size, color=c.get("color", "primary"),
                            align="left" if right else c.get("align", ctx.align), font_role=font_role, max_lines=1, source=spec.id)
        els = [word]
        if not c.get("caption"):
            return els, wh
        if right:
            cap, ch = ctx.text(f"{spec.id}.caption", "heading", c["caption"], x + ww + 0.03 * w, y, w - ww - 0.03 * w,
                               color=c.get("caption_color", "text"), align="left", font_role="display", max_lines=4, source=spec.id,
                               upper=True)
            cap = cap.model_copy(update={"box": cap.box.model_copy(update={"y": y + (wh - ch) / 2})})
            return els + [cap], wh
        cap, ch = ctx.text(f"{spec.id}.caption", "heading", c["caption"], x, y + wh, w, color=c.get("caption_color", "text"),
                           align=c.get("align", ctx.align), font_role="display", max_lines=2, source=spec.id, upper=True)
        return els + [cap], wh + ch


__all__ = ["BigTitle", "TabBanner", "ImageBlock", "Columns", "PhotoFade", "IconRow", "BigWord", "ElementKind"]
