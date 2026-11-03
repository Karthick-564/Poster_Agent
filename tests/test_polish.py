"""Polish agent: allow-list validation (pure) and an end-to-end polish of a real render with a scripted LLM (no network)."""
import asyncio

import pytest

from poster_agent.compile.compiler import compile_spec
from poster_agent.compile.vc_client import VectorCraftSession, cli_path
from poster_agent.fixtures import kciiri_components
from poster_agent.layout.solver import apply_layout
from poster_agent.llm import ScriptedLLM
from poster_agent.llm.base import LLMError
from poster_agent.polish import ALLOWED_TOOLS, polish, validate_ops
from poster_agent.spec.brand import load_brand

KCT = load_brand("kct")
needs_vc = pytest.mark.skipif(not cli_path().exists(), reason="vectorcraft-cli not installed")


def built_spec():
    spec = kciiri_components.build_spec()
    apply_layout(spec, KCT)
    return spec


def a_text(spec, role="title"):
    return next(e for e in spec.placed if e.kind.value == "text" and e.role == role and not e.props.get("decorative"))


def test_only_allow_listed_ops_on_unprotected_elements_with_bounded_magnitudes_survive():
    spec = built_spec()
    t = a_text(spec)
    logo = next(e for e in spec.placed if e.role == "logo")
    limit = 0.04 * min(spec.canvas.width, spec.canvas.height)
    good, bad = validate_ops([
        {"tool": "object.move", "id": t.id, "dx": 0, "dy": -4},                      # fine (text boxes span the safe width: nudge vertically)
        {"tool": "object.move", "id": t.id, "dx": limit * 2, "dy": 0},               # too far
        {"tool": "object.move", "id": logo.id, "dx": 2, "dy": 0},                    # brand logo is protected
        {"tool": "text.resize", "id": t.id, "size_pct": -8},                         # fine
        {"tool": "text.resize", "id": t.id, "size_pct": 60},                         # too big a change
        {"tool": "text.recolour", "id": t.id, "color": "primary"},                   # palette name: fine
        {"tool": "text.recolour", "id": t.id, "color": "#ff00ff"},                   # not a palette name
        {"tool": "text.setText", "id": t.id, "text": "something new"},               # copy changes are not allowed
        {"tool": "object.move", "id": "nope", "dx": 1, "dy": 1},                     # unknown id
    ], spec, KCT)
    assert [(o["tool"], o["id"]) for o in good] == [("object.move", t.id), ("text.resize", t.id), ("text.recolour", t.id)]
    assert len(bad) == 6
    assert any("allow-list" in b for b in bad) and any("protected" in b for b in bad) and any("exceeds" in b for b in bad)
    assert set(ALLOWED_TOOLS) == {"object.move", "text.resize", "text.recolour", "object.opacity"}


def test_a_text_move_that_would_leave_the_margins_is_refused():
    spec = built_spec()
    t = a_text(spec)
    spec.placed = [e.model_copy(update={"box": e.box.model_copy(update={"x": spec.canvas.safe_box.x + 1})}) if e.id == t.id else e
                   for e in spec.placed]
    good, bad = validate_ops([{"tool": "object.move", "id": t.id, "dx": -10, "dy": 0}], spec, KCT)
    assert not good and "safe margins" in bad[0]


def _render_run(proj_tmp):
    spec = built_spec()
    run = proj_tmp / "run"
    run.mkdir()

    async def go():
        async with VectorCraftSession() as vc:
            return await compile_spec(spec, vc, run, name="poster")
    asyncio.run(go())
    (run / "spec.json").write_text(spec.model_dump_json(), encoding="utf-8")
    return spec, run


@needs_vc
def test_polish_applies_valid_ops_to_a_copy_and_never_touches_the_original(proj_tmp):
    spec, run = _render_run(proj_tmp)
    t = a_text(spec)
    original = (run / "poster.vectorcraft").read_bytes()
    llm = ScriptedLLM([{"rationale": "nudge the title up", "ops": [
        {"tool": "object.move", "id": t.id, "dx": 0, "dy": -6},
        {"tool": "object.move", "id": next(e.id for e in spec.placed if e.role == "logo"), "dx": 5, "dy": 0}]}])
    res = asyncio.run(polish(run, llm, use_ocr=False))
    assert len(res.ops_applied) == 1 and any("protected" in r for r in res.rejected)
    assert (run / "poster.vectorcraft").read_bytes() == original  # the poster itself is never overwritten
    assert res.vectorcraft and res.vectorcraft.parent == run / "polish" and res.png.exists()
    assert res.after_errors is not None and res.before_errors is not None
    assert (res.accepted and "accepted" in res.reason) or "rejected" in res.reason
    assert (run / "polish" / "polish_report.json").exists()
    assert llm.calls[0]["images"] and "elements" in llm.calls[0]["user"]  # the LLM saw the render and the element list


@needs_vc
def test_polish_with_no_valid_ops_or_no_model_leaves_everything_alone(proj_tmp):
    spec, run = _render_run(proj_tmp)
    res = asyncio.run(polish(run, ScriptedLLM([{"ops": [], "rationale": "already balanced"}]), use_ocr=False))
    assert not res.accepted and res.vectorcraft is None and "left exactly as it is" in res.reason

    class Down:
        name = "down"

        def generate_json(self, *a, **k):
            raise LLMError("quota")
    res2 = asyncio.run(polish(run, Down(), use_ocr=False))
    assert not res2.accepted and "unavailable" in res2.reason
