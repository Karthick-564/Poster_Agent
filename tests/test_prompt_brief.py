import run_poster
from poster_agent.ingest import load_brief


class Fake:
    def generate_json(self, system, user, **kw):
        return {"event_title": "ROBOTICS DAY", "date": "5 MARCH 2027", "venue": "Hall 1", "style_notes": "bold"}


def test_prompt_becomes_a_brief_folder_and_never_overwrites(tmp_path, monkeypatch):
    monkeypatch.setattr(run_poster, "ROOT", tmp_path)
    monkeypatch.setattr(run_poster, "default_llm", lambda: Fake())
    text = "ROBOTICS DAY on 5 MARCH 2027 at Hall 1, make it bold"
    a = run_poster.brief_from_prompt(text, "kct", "story", None)
    b = run_poster.brief_from_prompt(text, "kct", "story", None)
    assert a.name == "robotics_day" and b.name == "robotics_day_2"
    br = load_brief(a / "brief.yaml")
    assert br.event_title == "ROBOTICS DAY" and br.format == "story" and br.brand == "kct"
