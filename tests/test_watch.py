import asyncio
import shutil
import socket

import numpy as np
import pytest
from PIL import Image

from poster_agent.compile import compiler as comp
from poster_agent.compile.compiler import build_steps, build_units, compile_spec, display_name, watch_groups
from poster_agent.compile.vc_client import (ROOT, VectorCraftError, VectorCraftSession, WatchConfig, app_args, app_path, cli_path,
                                            ensure_app, port_open, session_args)
from poster_agent.spec import Decoration
from poster_agent.enrich import default_preset, is_dark_ground
from poster_agent.finalize import (apply_document, finalize, hand_edit_issues, resolve_target, top_level_nodes)
from poster_agent.fixtures import kciiri_components
from poster_agent.layout.solver import apply_layout
from poster_agent.spec import Box, IssueCategory
from poster_agent.spec.brand import load_brand

needs_vc = pytest.mark.skipif(not cli_path().exists(), reason="vectorcraft-cli not installed")
KCT = load_brand("kct")


def decorated_spec():
    spec = kciiri_components.build_spec()
    apply_layout(spec, KCT)
    spec.decorations = default_preset(spec, KCT, is_dark_ground(spec, KCT)) + [
        Decoration(id="auto_dots", kind="dot_grid", region="center", color="accent", opacity=0.4)]  # a many-part ornament, opted in
    apply_layout(spec, KCT)
    return spec


# ---- how a session reaches VectorCraft (pure) ------------------------------------------------------------------------------

def test_headless_session_is_confined_and_watch_session_attaches_to_the_app():
    roots = [ROOT]
    head = session_args(roots, None)
    assert head[:2] == ["mcp", "--headless"] and head.count("--automation-read-root") == 1 and str(ROOT) in head
    watch = WatchConfig(port=7123)
    assert session_args(roots, watch) == ["mcp", "--connect", "127.0.0.1:7123"]  # the app confines itself; the CLI writes nothing
    assert "--automation-read-root" not in session_args(roots, watch)
    app = app_args(roots, watch)
    assert app[:2] == ["--control", "7123"] and "--automation-write-root" in app and str(ROOT) in app
    assert app_path().name.startswith("vectorcraft") and "cli" not in app_path().name


def test_ensure_app_never_launches_unless_asked_and_reuses_a_running_app(monkeypatch):
    import poster_agent.compile.vc_client as vcc

    def no_real_app(*a, **k):
        raise AssertionError("a test must never launch the real VectorCraft app")

    monkeypatch.setattr(vcc.subprocess, "Popen", no_real_app)
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    assert not port_open(port)
    with pytest.raises(VectorCraftError, match="no VectorCraft app is listening"):
        ensure_app(WatchConfig(port=port, launch=False), [ROOT])
    srv = socket.socket()
    srv.bind(("127.0.0.1", port))
    srv.listen(16)  # room for several probes (a full backlog looks like "nothing listening")
    try:
        assert port_open(port)
        assert ensure_app(WatchConfig(port=port, launch=True), [ROOT]) is None  # already running: reuse it, start nothing
    finally:
        srv.close()


# ---- what gets drawn, named (pure) --------------------------------------------------------------------------------





# ---- watched execution gives the identical poster -----------------------------------------------------------------------------





# ---- finalize: map a hand-edited document back onto the spec (pure) ---------------------------------------------------------------

def nodes_for(spec):
    out = []
    for i, e in enumerate(spec.placed):
        if e.props.get("decorative"):
            continue
        n = {"id": i, "name": e.id, "kind": "Type" if e.kind.value == "text" else "Image",
             "bounds": {"x": e.box.x, "y": e.box.y, "width": e.box.w, "height": e.box.h}, "visible": True, "locked": e.locked}
        if e.kind.value == "text":
            n["text"], n["fill"] = e.text, e.color
        out.append(n)
    return out


def test_unedited_document_has_no_changes():
    spec = kciiri_components.build_spec()
    apply_layout(spec, KCT)
    spec.mandatory = []
    diff = apply_document(spec, nodes_for(spec))
    assert diff.changes == [] and diff.overflow == [] and len(diff.bounds) == len([e for e in spec.placed if not e.props.get("decorative")])


def test_hand_edits_are_detected_and_the_spec_follows_the_document():
    spec = kciiri_components.build_spec()
    apply_layout(spec, KCT)
    nodes = {n["name"]: n for n in nodes_for(spec)}
    nodes["guest.name"]["bounds"]["y"] += 120
    nodes["org.title"]["fill"] = "#ffffff"
    nodes["event.text"]["text"] = "ANNUAL DAY 2027"
    nodes["logo.logo"]["visible"] = False
    nodes["theme.text"]["bounds"]["width"] -= 80
    nodes["guest.name"]["locked"] = False
    del nodes["memorial.text"]
    extra = {"id": 999, "name": "Layer note", "kind": "Type", "text": "HELLO", "fill": "#112233",
             "bounds": {"x": 10, "y": 10, "width": 100, "height": 30}, "visible": True}
    diff = apply_document(spec, list(nodes.values()) + [extra])
    kinds = {(c.kind, c.id) for c in diff.changes}
    assert {("moved", "guest.name"), ("recoloured", "org.title"), ("text_edited", "event.text"), ("hidden", "logo.logo"),
            ("resized", "theme.text"), ("deleted", "memorial.text")} <= kinds and any(c.kind == "added" for c in diff.changes)
    ids = {e.id for e in diff.spec.placed}
    assert "logo.logo" not in ids and "memorial.text" not in ids and any(i.startswith("added.") for i in ids)
    assert next(e for e in diff.spec.placed if e.id == "org.title").color == "#ffffff"
    assert next(e for e in diff.spec.placed if e.id == "event.text").text == "ANNUAL DAY 2027"
    assert diff.bounds["guest.name"].y == pytest.approx(next(e for e in spec.placed if e.id == "guest.name").box.y + 120)






# ---- finalize end to end (real VectorCraft, headless: it is the file that matters, not the window) --------------------------------

async def _edit_document(vcf, edits):
    """Simulate a person editing the saved document in VectorCraft."""
    async with VectorCraftSession() as vc:
        await vc.command("file.open", path=str(vcf))
        nodes = {n["name"]: n["id"] for n in top_level_nodes(await vc.call("doc_inspect", {}))}
        await edits(vc, nodes)
        await vc.command("file.save", path=str(vcf))


@needs_vc
def test_finalize_reports_edits_and_catches_a_hand_edit_that_breaks_the_brand(proj_tmp):
    spec = decorated_spec()
    run = proj_tmp / "run"

    async def make():
        async with VectorCraftSession() as vc:
            res = await compile_spec(spec, vc, run, name="poster")
        (run / "spec.json").write_text(spec.model_dump_json(indent=1), encoding="utf-8")
        return res

    asyncio.run(make())
    clean = asyncio.run(finalize(run, use_ocr=False))  # nothing edited yet
    assert clean.changes == [] and clean.ok, [(i.category.value, i.evidence) for i in clean.report.issues + clean.extra_issues]

    async def edits(vc, nodes):
        await vc.command("object.move", ids=[nodes["guest.name"]], dx=0, dy=60)
        await vc.command("text.setStyle", ids=[nodes["org.title"]], fill="#ffffff")  # white title on a light page
        await vc.command("object.setProps", ids=[nodes["logo.logo"]], visible=False)   # remove the brand logo
        await vc.command("text.create", text="HELLO", x=700, y=40, size=30, font="Montserrat", style="Bold", color="#1e2842")

    asyncio.run(_edit_document(run / "poster.vectorcraft", edits))
    res = asyncio.run(finalize(run, use_ocr=False))
    kinds = {(c.kind, c.id) for c in res.changes}
    assert {("moved", "guest.name"), ("recoloured", "org.title"), ("hidden", "logo.logo")} <= kinds
    assert any(c.kind == "added" and "HELLO" in c.detail for c in res.changes)
    assert not res.ok
    assert any(i.category == IssueCategory.low_contrast and "org.title" in i.element_ids for i in res.report.issues)
    assert any(i.category == IssueCategory.logo_violation for i in res.report.issues + res.extra_issues)  # the logo was removed
    assert (res.out_dir / "final_report.json").exists() and res.png.exists() and res.pdf.exists()
    assert Image.open(res.png).size == (2160, 2700)


def test_a_poster_without_any_logo_fails_the_logo_check(tmp_path):
    from poster_agent.checks import run_checks
    from poster_agent.compile.compiler import CompileResult
    from poster_agent.spec import CanvasSpec, PlacedElement, PosterSpec
    img = Image.new("RGB", (1000, 1000), "white")
    img.save(tmp_path / "p.png")
    for n in ("p.pdf", "p.vectorcraft"):
        (tmp_path / n).write_bytes(b"x")
    spec = PosterSpec(brief_id="t", brand="kct", canvas=CanvasSpec(format="square", width=1000, height=1000, export_scale=1.0), placed=[])
    compiled = CompileResult(png=tmp_path / "p.png", pdf=tmp_path / "p.pdf", vectorcraft=tmp_path / "p.vectorcraft")
    rep = run_checks(spec, KCT, compiled, use_ocr=False, use_emptiness=False)
    assert any(i.category == IssueCategory.logo_violation and "no institution logo" in i.evidence for i in rep.issues)
