"""Several different designs of one brief, ranked.

`pick_recipes` chooses distinct composition recipes (the genre's own first, then the rest); each becomes a full run with that recipe forced
(brief.layout_style), so the variants differ in STRUCTURE, not just colour. `rank` orders the finished runs by what was measured: a poster
that was produced beats none, one that passed its checks beats one that did not, then fewer errors, a better critic score, less dead space."""
from __future__ import annotations

from poster_agent.plan.prompt import GENRE_RECIPES, RECIPES


def pick_recipes(brief, n: int) -> list[str]:
    from poster_agent.skills import genre_of
    first = list(GENRE_RECIPES.get(genre_of(brief)) or [])
    rest = [r for r in RECIPES if r not in first]
    out: list[str] = []
    for r in first + rest:
        if r not in out and len(out) < n:
            out.append(r)
    return out


def pick_looks(brief, n: int) -> list[str]:
    """n distinct looks, the brief's own first: variants differ in ground, headline treatment and imagery role, not just structure."""
    from poster_agent.plan.looks import LOOKS, look_for
    first = look_for(brief).name
    return ([first] + [k for k in LOOKS if k != first])[:n]


def rank_key(res) -> tuple:
    """Lower is better. `res` is a pipeline RunResult."""
    if not res.ok or res.best_round is None or not res.rounds:
        return (1, 1, 0, 0, 0.0)
    rec = res.rounds[res.best_round]
    return (0, 0 if res.passed else 1, rec.errors, -(rec.score or 0), round(rec.empty_fraction, 3))


def rank(results: list) -> list:
    return sorted(results, key=rank_key)
