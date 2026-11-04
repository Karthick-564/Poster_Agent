from types import SimpleNamespace as NS

from poster_agent.plan.prompt import RECIPES
from poster_agent.variants import pick_recipes, rank


def brief(title):
    return NS(event_title=title, style_notes="", extra_lines=[], theme_lines=[], tagline=None, subtitle=None, organizer=None, cta=None,
              contact_lines=[], venue=None, date=None, time=None, layout_style=None)


def run(ok=True, passed=True, errors=0, score=None, empty=0.0):
    rec = NS(errors=errors, score=score, empty_fraction=empty)
    return NS(ok=ok, passed=passed, best_round=0 if ok else None, rounds=[rec] if ok else [])


def test_variants_use_distinct_recipes_and_start_with_the_genres_own():
    r = pick_recipes(brief("GENERATIVE AI WORKSHOP"), 3)
    assert len(r) == 3 and len(set(r)) == 3 and set(r) <= set(RECIPES)


def test_ranking_prefers_produced_then_passed_then_fewer_errors_then_less_dead_space():
    a, b, c, d, e = run(ok=False), run(passed=False, errors=2), run(passed=True, errors=0, empty=0.2), run(passed=True, errors=0, empty=0.1), run(passed=True, errors=1)
    assert rank([a, b, c, d, e]) == [d, c, e, b, a]
