from poster_agent.assets.select_person import parse_box
from poster_agent.ingest.brief import Brief
from poster_agent.plan.validate import art_is_relevant


def test_parse_box_scales_the_0_1000_grid_and_survives_fences_and_truncation():
    assert parse_box('[{"bbox_2d": [500, 100, 1000, 900], "label": "man"}]', (2000, 1000)) == (1000, 100, 2000, 900)
    assert parse_box('```json\n[{"bbox_2d": [0, 0, 500, 500]}]\n```', (100, 100)) == (0, 0, 50, 50)
    assert parse_box('[{"bbox_2d": [500, 100, 900, 800], "lab', (1000, 1000)) == (500, 100, 900, 800)
    assert parse_box("[]", (100, 100)) is None


def test_art_must_share_a_subject_word_with_the_brief():
    b = Brief(brand="kct", event_title="ROBOTICS DAY 2026", style_notes="an industrial robot arm over a navy atmosphere")
    assert art_is_relevant("an industrial robot arm on a workshop floor", b)
    assert not art_is_relevant("flowing blue light streaks, deep navy gradient", b)
