"""Few-shot style references for the planner, derived from hand-built fixtures (real poster structures).

The fixtures are built from real institution posters and are not part of the public repository: when they are absent the planner
simply runs without few-shot examples."""
from __future__ import annotations

import importlib

from poster_agent.spec import PosterSpec

FIXTURES = {"engineers_day": "engineers_day", "industry_conclave": "industry_conclave", "kciiri": "kciiri_components"}


def _plan_dict(spec: PosterSpec) -> dict:
    comps = [c.model_dump(exclude_defaults=True, exclude={"asset_ids"}) for c in spec.components]
    for c in comps:
        if c["component"] == "SpeakerGrid":  # keep the prompt short: 3 of the speakers show the shape
            c["content"] = {**c["content"], "speakers": c["content"]["speakers"][:3]}
    bg = spec.background
    background = ({"kind": "solid", "color": bg.color} if bg.kind == "solid"
                  else {"kind": "gradient", "angle": bg.angle,
                        "gradient": [{"offset": s.offset, "color": s.color} for s in bg.gradient]})
    return {"format": spec.canvas.format, "background": background, "components": comps}


def example_plans(names: list[str]) -> list[dict]:
    out = []
    for n in names:
        try:
            mod = importlib.import_module(f"poster_agent.fixtures.{FIXTURES[n]}")
        except ImportError:  # public checkout without the local fixtures
            continue
        out.append(_plan_dict(mod.build_spec()))
    return out
