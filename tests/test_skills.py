"""Design skills library: loading, genre/format selection, budget, and its use in the planner prompt (no network, no brand data needed)."""
from pathlib import Path

from poster_agent.ingest import Brief
from poster_agent.plan.prompt import GENRE_RECIPES, RECIPES, SYSTEM, build_user_prompt, composition_recipe
from poster_agent.skills import SKILLS_DIR, genre_of, load_skills, parse_skill, select


def brief(title, **kw):
    return Brief(brand="kct", event_title=title, **kw)


def test_every_shipped_skill_parses_and_is_small_enough_to_use():
    skills = load_skills()
    assert {"poster-design-craft", "color-harmony-and-palette", "poster-critique", "print-preflight", "brief-extraction",
            "composition-and-grid", "typography-pairing", "genre-playbooks", "art-direction-flux", "creative-concepts"} <= set(skills)
    for s in skills.values():
        assert s.description and s.sections and s.use, s.name
        assert len(s.digest) <= 520, f"{s.name}: digest too long for a prompt ({len(s.digest)} chars)"


def test_genre_comes_from_the_briefs_own_words():
    assert genre_of(brief("GENERATIVE AI WORKSHOP")) == "workshop"
    assert genre_of(brief("CAMPUS PLACEMENT DRIVE 2027")) == "placement"  # more specific than 'drive'
    assert genre_of(brief("RECRUITMENT DRIVE")) == "recruitment"
    assert genre_of(brief("30 YEAR REUNION CELEBRATIONS")) == "commemoration"
    assert genre_of(brief("DESIGNING A GREEN CAMPUS", tagline="A talk on sustainable campuses")) == "talk"
    assert genre_of(brief("AARAMBH 2027", subtitle="Cultural Committee")) == "cultural"
    assert genre_of(brief("Something unusual")) == "event"


def test_selection_puts_the_genre_playbook_first_stays_in_budget_and_filters_by_stage_and_format():
    text, used = select(brief("GENERATIVE AI WORKSHOP"), budget_chars=3200)
    assert used[0] == "genre-playbooks: Workshop or training" and len(text) <= 3200
    assert "creative-concepts (digest)" in used and "poster-design-craft (digest)" in used
    assert not any(u.startswith("poster-critique") or u.startswith("brief-extraction") for u in used)  # other stages' skills stay out
    assert not any(u.startswith("print-preflight") for u in used)  # print rules only for a3/a4
    assert any(u.startswith("print-preflight") for u in select(brief("X"), fmt="a3")[1])
    tiny, tiny_used = select(brief("GENERATIVE AI WORKSHOP"), budget_chars=600)
    assert len(tiny) <= 600 and tiny_used  # a small budget keeps the most relevant parts only
    assert any(u.startswith("poster-critique") for u in select(brief("X"), purpose="critic")[1])








# ---- the planner prompt --------------------------------------------------------------------------------------------------

def test_planner_prompt_carries_the_skills_recipe_default_format_and_stays_compact():
    from poster_agent.spec.brand import load_brand
    try:
        brand = load_brand("kct")
    except FileNotFoundError:
        return  # public checkout without a brand: covered by the pure tests above
    b = brief("GENERATIVE AI WORKSHOP", extra_lines=["DAY 1", "DAY 2", "CERTIFICATES"])
    user = build_user_prompt(b, brand, {})
    assert "DESIGN SKILLS" in user and "DEFAULT FORMAT" in user and "Workshop or training" in user and "ICONS (names" in user
    assert len(SYSTEM) + len(user) < 21000
    assert "ICONS (names" not in build_user_prompt(brief("PLAIN EVENT"), brand, {})  # icons only when there is list content
    assert '"SpeakerGrid"' not in user and '"BigTitle"' in user  # components the inventory cannot feed are not described


def test_recipes_follow_the_genre_and_differ_between_events_of_the_same_genre():
    assert set(r for names in GENRE_RECIPES.values() for r in names) <= set(RECIPES)
    assert composition_recipe(brief("GENERATIVE AI WORKSHOP"))[0] in GENRE_RECIPES["workshop"]
    assert composition_recipe(brief("ANY WORKSHOP", layout_style="split"))[0] == "split"  # forced by the brief
    names = {composition_recipe(brief(f"{w} WORKSHOP"))[0] for w in ["AI", "ROBOTICS", "DESIGN", "FINANCE", "WRITING", "DATA"]}
    assert len(names) >= 2 and names <= set(GENRE_RECIPES["workshop"])  # several of the genre's archetypes get used
