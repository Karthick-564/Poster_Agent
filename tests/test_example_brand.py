"""The shipped EXAMPLE brand lets a fresh clone run real layout code without any institution's assets."""
from poster_agent.layout.solver import solve_layout
from poster_agent.spec import CanvasSpec, ComponentSpec, ElementKind, PosterSpec
from poster_agent.spec.brand import load_brand

BRAND = load_brand("_example")


def test_example_brand_loads_with_logos_and_solves_a_layout():
    assert BRAND.logos["emblem"].solid_field and BRAND.default_logo == "emblem"
    comps = [ComponentSpec(id="h", component="HeaderRow", content={"logo": "emblem", "lines": ["1 JAN 2027", "HALL 1"]}),
             ComponentSpec(id="t", component="TitleStack", content={"title": "OPEN DAY", "subtitle": "All welcome"}, emphasis="hero")]
    spec = PosterSpec(brief_id="t", brand="_example", canvas=CanvasSpec.from_format("story"), components=comps,
                      palette=dict(BRAND.palette), assets={})
    res = solve_layout(spec, BRAND)
    texts = [e for e in res.placed if e.kind == ElementKind.text]
    assert texts and not [i for i in res.issues if i.severity == "error" and i.category.value in {"margin_violation", "text_overlap"}]
