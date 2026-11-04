"""Components of the house skeleton (header strip -> headline -> kicker -> fact line -> content -> bottom anchor), taken from how the
institution's designers build their posters: facts grouped in compact structures, logos in one row with hairlines, and the bottom of
the poster always occupied by an anchor (a cutout or photo standing on the bottom edge, or a band) that takes whatever height is left."""
from __future__ import annotations

from poster_agent.spec import AssetKind, Box

from .base import Component, register
from .components import _aspect, _gap, _image, _shape


def _rule(id_: str, ctx, x: float, y: float, h: float, color: str = "accent") -> object:
    """A thin vertical hairline separating items in a row."""
    return _shape(id_, "divider", Box(x=x - 1, y=y, w=2, h=h), z=2, fill=ctx.color(color), opacity=0.8)


@register
class FactLine(Component):
    """The date | time | venue line: the brief's own words side by side, separated by thin vertical rules (never by typed characters).
    Falls back to one item per line when the row does not fit the width. content.items = list of verbatim text."""
    name = "FactLine"

    def build(self, spec, ctx, x, y, w):
        c = spec.content
        items = [t for t in c.get("items", []) if t]
        if not items:
            return [], 0.0
        align = c.get("align", ctx.align)
        color, factor = c.get("color", "primary"), c.get("factor", 0.8)
        probe, ph = ctx.text(f"{spec.id}.probe", "heading", items[0], x, y, w, factor=factor)
        size, font, style, trk = probe.size, probe.font, probe.style, float(probe.props.get("tracking", 0.0))
        gap = 1.6 * size
        widths = [ctx.measurer.width(t.upper() if c.get("upper", True) else t, font, style, size, trk) for t in items]
        row_w = sum(widths) + gap * (len(items) - 1)
        els = []
        if row_w <= w * 0.97:  # one row with hairlines between the items
            cx = {"left": x, "center": x + (w - row_w) / 2, "right": x + w - row_w}[align]
            for i, (t, tw) in enumerate(zip(items, widths)):
                el, h = ctx.text(f"{spec.id}.i{i}", "heading", t, cx, y, tw * 1.04 + 2, color=color, align="left", size=size,
                                 upper=c.get("upper", True), max_lines=1, source=spec.id)
                els.append(el)
                if i:
                    els.append(_rule(f"{spec.id}.sep{i}", ctx, cx - gap / 2, y + 0.12 * h, 0.76 * h, c.get("rule_color", "accent")))
                cx += tw + gap
            return els, ph
        cur = y  # too wide: one item per line, same size, no separators
        for i, t in enumerate(items):
            el, h = ctx.text(f"{spec.id}.i{i}", "heading", t, x, cur, w, color=color, align=align, size=size, upper=c.get("upper", True),
                             max_lines=2, source=spec.id)
            els.append(el)
            cur += h
        return els, cur - y


@register
class FactStrip(Component):
    """Labelled facts in columns: a small spaced-capitals label (a structural word such as DATE, VENUE, FEE) above each value from the
    brief, thin rules between columns. content.items = [{label, value}], cols = columns per row (default: all in one row, up to 4)."""
    name = "FactStrip"

    def build(self, spec, ctx, x, y, w):
        c = spec.content
        items = [it for it in c.get("items", []) if it.get("value")]
        if not items:
            return [], 0.0
        cols = max(1, min(c.get("cols", len(items)), 4, len(items)))
        cw = w / cols
        pad = 0.04 * cw
        els, cur = [], y
        for r in range(0, len(items), cols):
            row, row_h = items[r:r + cols], 0.0
            cells = []
            for j, it in enumerate(row):
                cx = x + j * cw + (pad if j else 0.0)
                inner = cw - (2 * pad if j else pad)
                lab, lh = ctx.text(f"{spec.id}.l{r + j}", "micro", it.get("label", ""), cx, cur, inner, color=c.get("label_color", "accent"),
                                   align="left", upper=True, tracking=120, max_lines=1, source=spec.id) if it.get("label") else (None, 0.0)
                val, vh = ctx.text(f"{spec.id}.v{r + j}", "caption", it["value"], cx, cur + lh, inner, color=c.get("value_color", "text"),
                                   align="left", max_lines=3, source=spec.id)
                cells += ([lab] if lab else []) + [val]
                row_h = max(row_h, lh + vh)
            for j in range(1, len(row)):
                els.append(_rule(f"{spec.id}.sep{r + j}", ctx, x + j * cw, cur, row_h, c.get("rule_color", "neutral")))
            els += cells
            cur += row_h + (_gap(ctx, 0.5) if r + cols < len(items) else 0.0)
        return els, cur - y


@register
class LogoStrip(Component):
    """The institution logo and partner logos in one row at one height, separated by hairlines (ours first, as the brand rule says).
    content.logo = brand logo name (default), partners = supplied partner-logo asset ids, height_pct (default 6)."""
    name = "LogoStrip"

    def build(self, spec, ctx, x, y, w):
        c = spec.content
        ids = [ctx.brand_logo_asset(c.get("logo"))] + [a for a in c.get("partners", []) if a in ctx.assets]
        h = min(c.get("height_pct", 6.0), ctx.brand.logo_rules.max_height_pct) / 100.0 * ctx.canvas.height
        widths = [h * _aspect(ctx.assets[a].path) for a in ids]
        gap = 0.06 * ctx.canvas.width
        total = sum(widths) + gap * (len(ids) - 1)
        if total > w:  # shrink uniformly to fit the row
            k = w / total
            h, widths, gap, total = h * k, [v * k for v in widths], gap * k, w
        align = c.get("align", ctx.align)
        cx = {"left": x, "center": x + (w - total) / 2, "right": x + w - total}[align]
        els = []
        for i, (a, aw) in enumerate(zip(ids, widths)):
            if i:
                els.append(_rule(f"{spec.id}.sep{i}", ctx, cx - gap / 2, y + 0.1 * h, 0.8 * h, "neutral"))
            els.append(_image(f"{spec.id}.{'logo' if i == 0 else f'p{i}'}", "logo" if i == 0 else "partner_logo", Box(x=cx, y=y, w=aw, h=h),
                              a, locked=True))
            cx += aw + gap
        return els, h


@register
class BottomCutout(Component):
    """The bottom anchor: a cutout (a person from a supplied photo, or a generated object) standing on the bottom edge of the poster, as
    designers place a speaker or subject. anchor 'bottom'. It takes the height left over below the content (between height_pct and
    max_pct of the poster), so the bottom of the poster is never empty. content: asset, align left|center|right, height_pct (min, 28),
    max_pct (38; 60 when beside), max_width_pct (70), beside (true: it stands beside a text column instead of below the content; see the split skeleton)."""
    name = "BottomCutout"
    band_elastic = True

    def build(self, spec, ctx, x, y, w):
        c = spec.content
        aid = c["asset"]
        if aid not in ctx.assets:
            return [], 0.0
        H = ctx.canvas.height
        cap = c.get("max_pct", 60.0 if c.get("beside") else 38.0)  # beside a text column it may stand taller, as a speaker does
        h = min(c.get("height_pct", 28.0) / 100.0 * H * ctx.scale ** 1.5 + ctx.extra.get(spec.id, 0.0), cap / 100.0 * H)  # shrinks first when content is dense
        aspect = _aspect(ctx.assets[aid].path)
        iw = h * aspect
        wmax = (c.get("max_width_pct", 70.0) / 100.0 * ctx.canvas.width) if not c.get("beside") else 0.42 * ctx.canvas.safe_box.w  # beside: only the column right of the text
        if iw > wmax:
            iw, h = wmax, wmax / aspect
        align = c.get("align", "right" if ctx.align == "left" else "center")
        safe = ctx.canvas.safe_box
        ix = {"left": safe.x * 0.5, "center": (ctx.canvas.width - iw) / 2, "right": ctx.canvas.width - iw - safe.x * 0.5}[align]
        props = {"cutout": True} if ctx.assets[aid].kind in (AssetKind.headshot, AssetKind.photo) else {}  # supplied photo: matted by the pipeline
        if c.get("framing") and props:
            props["framing"] = c["framing"]
        els = []
        if props and c.get("stage", True):  # a supplied person stands in front of a deliberate setting, never on a bare ground
            els += self._stage(spec, ctx, ix, y, iw, h)
        return els + [_image(f"{spec.id}.img", "hero", Box(x=ix, y=y, w=iw, h=h), aid, z=6, **props)], h

    def _stage(self, spec, ctx, ix, y, iw, h):
        """An arch of brand colour behind the person (lighter at the head, brand colour toward the floor) and a soft light behind the head.
        The arch runs down behind the footer band, so the person seems to stand in it."""
        W, safe = ctx.canvas.width, ctx.canvas.safe_box
        aw = min(iw * 1.12, W - 2 * safe.x)
        ax = min(max(ix + iw / 2 - aw / 2, safe.x), W - safe.x - aw)
        top = y - 0.07 * h
        prim = ctx.color("primary").lstrip("#")
        acc = ctx.color("accent").lstrip("#")
        light = "#" + "".join(f"{int(int(prim[i:i + 2], 16) * 0.6 + int(acc[i:i + 2], 16) * 0.4):02x}" for i in (0, 2, 4))  # brand blue lifted toward the accent
        arch = _shape(f"{spec.id}.stage", "stage", Box(x=ax, y=top, w=aw, h=ctx.canvas.height - top), z=2, radius=aw / 2,
                      fill={"gradient": {"kind": "linear", "start": [ax, top], "end": [ax, y + h], "stops": [
                          {"offset": 0.0, "color": light, "opacity": 1.0}, {"offset": 1.0, "color": "#" + prim, "opacity": 1.0}]}})
        d = 0.5 * iw
        glow = _shape(f"{spec.id}.glow", "stage", Box(x=ix + iw / 2 - d / 2, y=y + 0.02 * h, w=d, h=d), z=3, shape="ellipse",
                      fill={"gradient": {"kind": "linear", "start": [0, y], "end": [0, y + d], "stops": [
                          {"offset": 0.0, "color": "#ffffff", "opacity": 0.0}, {"offset": 0.45, "color": "#ffffff", "opacity": 0.32},
                          {"offset": 1.0, "color": "#ffffff", "opacity": 0.0}]}})
        return [arch, glow]
