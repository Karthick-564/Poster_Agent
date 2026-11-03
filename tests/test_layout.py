import asyncio

import pytest

from poster_agent.compile.compiler import compile_spec
from poster_agent.compile.vc_client import ROOT, VectorCraftSession, cli_path
from poster_agent.fixtures import engineers_day, industry_conclave, kciiri, kciiri_components
from poster_agent.layout.fonts import TextMeasurer
from poster_agent.layout.solver import apply_layout, solve_layout
from poster_agent.spec import ComponentSpec, ElementKind, IssueCategory, PosterSpec, CanvasSpec, find_unsourced_text
from poster_agent.spec.brand import load_brand
from poster_agent.spec.formats import MM_TO_PT

needs_vc = pytest.mark.skipif(not cli_path().exists(), reason="vectorcraft-cli not installed")
KCT = load_brand("kct")
FIXTURES = {"kciiri_components": kciiri_components, "industry_conclave": industry_conclave, "engineers_day": engineers_day}


def test_layout_is_deterministic():
    a = solve_layout(kciiri_components.build_spec(), KCT)
    b = solve_layout(kciiri_components.build_spec(), KCT)
    assert [e.model_dump() for e in a.placed] == [e.model_dump() for e in b.placed]


def test_components_carry_no_coordinates():
    def walk(c: ComponentSpec):
        assert not {"x", "y", "box", "width", "height"} & set(c.content), f"{c.id} content has geometry"
        for ch in c.children:
            walk(ch)
    for mod in FIXTURES.values():
        for c in mod.build_spec().components:
            walk(c)


def test_speaker_grid_rows_and_centered_last_row():
    spec = industry_conclave.build_spec()
    apply_layout(spec, KCT)
    photos = [e for e in spec.placed if e.role == "headshot"]
    assert len(photos) == 19
    rows: dict[float, list] = {}
    for p in photos:
        rows.setdefault(round(p.box.y, 1), []).append(p)
    assert sorted(len(r) for r in rows.values()) == [3, 4, 4, 4, 4]
    last = rows[max(rows)]
    safe = spec.canvas.safe_box
    assert sum(p.box.cx for p in last) / 3 == pytest.approx(safe.cx, abs=1.0)  # centred as a group


def test_logo_is_locked_with_exclusion_zone():
    spec = engineers_day.build_spec()
    res = apply_layout(spec, KCT)
    logo = next(e for e in res.placed if e.role == "logo")
    assert logo.locked
    panel = next(e for e in res.placed if e.role == "panel")
    assert logo.box.y - panel.box.y >= KCT.logo_rules.clear_space_mm * MM_TO_PT  # 10 mm above the logo


def test_all_text_traces_to_content_verbatim():
    spec = kciiri_components.build_spec()
    apply_layout(spec, KCT)
    assert find_unsourced_text(spec.placed, kciiri.allowed_text()) == []


def test_overstuffed_poster_reports_fit_issue_not_silent_overflow():
    c = ComponentSpec(id="big", component="TextBlock", content={"text": "word " * 900, "role": "body"})
    spec = PosterSpec(brief_id="x", brand="kct", canvas=CanvasSpec.from_format("a4"), components=[c])
    res = apply_layout(spec, KCT)
    assert not res.fits
    assert any(i.category == IssueCategory.fit and i.source == "solver" for i in spec.issues)


def test_banded_components_pin_to_edges():
    spec = kciiri_components.build_spec()
    apply_layout(spec, KCT)
    band = next(e for e in spec.placed if e.id == "when_where.bg")
    assert band.box.x == 0 and band.box.w == spec.canvas.width and band.box.bottom == pytest.approx(spec.canvas.height)


@needs_vc
def test_measurer_matches_vectorcraft_within_3pct():
    cases = [("ANNUAL DAY 2026", "Montserrat", "ExtraBold", 38, 0), ("Dr. A. Kumar", "Barlow Condensed", "Bold", 34, 0),
             ("CHIEF GUEST", "Barlow Condensed", "Regular", 22, 100), ("Typography sample text", "Montserrat", "Regular", 24, 0)]
    m = TextMeasurer()

    async def run():
        out = []
        async with VectorCraftSession() as vc:
            await vc.command("file.new", width=2000, height=800)
            for text, fam, sty, size, trk in cases:
                r = await vc.command("text.create", text=text, x=20, y=100, size=size, font=fam, style=sty, justify="left")
                if trk:
                    await vc.command("text.setStyle", ids=[r["id"]], tracking=trk)
                info = await vc.call("doc_inspect", {"ids": [r["id"]]})
                out.append(info["nodes"][0]["bounds"]["width"])
        return out

    for (text, fam, sty, size, trk), vcw in zip(cases, asyncio.run(run())):
        assert abs(m.width(text, fam, sty, size, trk) - vcw) / vcw < 0.03, text


@needs_vc
@pytest.mark.parametrize("name", list(FIXTURES))
def test_fixture_posters_render_clean(name):
    spec = FIXTURES[name].build_spec()
    res = apply_layout(spec, KCT)
    assert [i for i in res.issues if i.severity == "error"] == []

    async def run():
        async with VectorCraftSession() as vc:
            return await compile_spec(spec, vc, ROOT / f"runs/test_{name}", name=name)

    out = asyncio.run(run())
    assert out.overflow == [] and out.warnings == []
    canvas = spec.canvas.full_box
    for eid, b in out.bounds.items():
        assert canvas.contains(b, tol=1.0), f"{eid} escapes the canvas: {b}"
