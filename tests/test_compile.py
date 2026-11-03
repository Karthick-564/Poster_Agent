import asyncio
import re

import pytest
from PIL import Image

from poster_agent.compile.compiler import CompileError, build_steps, build_units, compile_spec
from poster_agent.compile.vc_client import ROOT, VectorCraftSession, cli_path
from poster_agent.fixtures.kciiri import allowed_text, build_spec
from poster_agent.spec import Box, ElementKind, PlacedElement, ScrimSpec

needs_vc = pytest.mark.skipif(not cli_path().exists(), reason="vectorcraft-cli not installed")


def test_steps_are_pure_and_refs_are_consistent():
    spec = build_spec()
    steps, main = build_steps(build_units(spec))
    assert steps == build_steps(build_units(spec))[0]  # deterministic
    for i, s in enumerate(steps):  # every "$N" ref points strictly at an earlier step
        for n in re.findall(r"\$(\d+)\.", str(s["params"])):
            assert int(n) < i, f"step {i} ({s['id']}) refs step {n}"
    assert set(main) >= {"__bg__", "logo", "org", "headshot", "footer_band"}
    # circle crop = place image, then ellipse created AFTER (must be topmost), then clip
    ids = [s["id"] for s in steps]
    k = ids.index("object.clippingMask.make")
    assert ids[k - 2:k + 1] == ["file.place", "shape.ellipse", "object.clippingMask.make"]


def test_elements_drawn_in_z_order():
    spec = build_spec()
    keys = [u.key for u in build_units(spec)]
    zs = {e.id: e.z for e in spec.placed}
    placed_keys = [k for k in keys if k in zs]
    assert [zs[k] for k in placed_keys] == sorted(zs[k] for k in placed_keys)


def test_scrim_is_densest_at_its_edge():
    from poster_agent.compile.compiler import _scrim_gradient
    rect, grad = _scrim_gradient(ScrimSpec(edge="bottom", opacity=0.7, extent=0.4), 1000, 2000)
    g = grad["gradient"]
    assert g["start"][1] > g["end"][1]  # starts below (dense edge), fades upward
    assert g["stops"][0]["opacity"] == 0.7 and g["stops"][1]["opacity"] == 0.0
    assert rect["y"] + rect["height"] > 2000  # overshoots the canvas edge


def test_compile_rejects_invented_text():
    spec = build_spec()
    spec.placed.append(PlacedElement(id="spam", kind=ElementKind.text, role="x", box=Box(x=0, y=0, w=100, h=20),
                                     text="Register now for free lunch", size=12))

    async def run(tmp):
        async with VectorCraftSession() as vc:
            await compile_spec(spec, vc, tmp, allowed_text=allowed_text())

    with pytest.raises(CompileError, match="not traceable"):
        # fails before any VectorCraft call is made, so no CLI is needed
        asyncio.run(_check_without_cli(spec))


async def _check_without_cli(spec):
    class Dummy:
        async def command(self, *a, **k):
            raise AssertionError("compile must reject before drawing")
    await compile_spec(spec, Dummy(), ROOT / "runs/_unused", allowed_text=allowed_text())


@needs_vc
def test_golden_kciiri_compiles_cleanly(tmp_path):
    spec = build_spec()

    async def run():
        async with VectorCraftSession() as vc:
            return await compile_spec(spec, vc, ROOT / "runs/test_kciiri", allowed_text=allowed_text(), name="kciiri")

    res = asyncio.run(run())
    assert res.overflow == [] and res.warnings == []
    with Image.open(res.png) as im:
        assert im.size == (2160, 2700)  # 1080x1350 pt at 2x
    canvas = spec.canvas.full_box
    for eid, b in res.bounds.items():  # real bounds from VectorCraft, not the planned boxes
        assert canvas.contains(b, tol=1.0), f"{eid} escapes the canvas: {b}"
    assert res.pdf.exists() and res.vectorcraft.exists()
