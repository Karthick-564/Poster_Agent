"""outputs/: only the final image of a finished real run, indexed with what was actually measured."""
import json

from PIL import Image

from poster_agent.outputs import publish


def fake_run(root, name, passed, reviewed, ok=True):
    run = root / "runs" / name
    run.mkdir(parents=True)
    Image.new("RGB", (10, 10), "white").save(run / f"{name}.png")
    (run / "report.json").write_text(json.dumps({"ok": ok, "passed": passed, "reviewed": reviewed, "best_round": 2,
                                                  "outputs": {"png": str(run / f"{name}.png")}}), encoding="utf-8")
    return run


def test_publish_copies_only_the_final_png_and_records_the_honest_status(tmp_path):
    out = tmp_path / "outputs"
    publish(fake_run(tmp_path, "a", True, True), out)
    publish(fake_run(tmp_path, "b", True, False), out)
    publish(fake_run(tmp_path, "c", False, False), out)
    assert sorted(p.name for p in out.iterdir()) == ["a.png", "b.png", "c.png", "index.json"]
    status = {e["name"]: e["status"] for e in json.loads((out / "index.json").read_text(encoding="utf-8"))}
    assert status["a"] == "passed checks and vision review" and "no vision review" in status["b"] and status["c"].startswith("not passed")


def test_a_failed_run_publishes_nothing_and_republishing_replaces_the_entry(tmp_path):
    out = tmp_path / "outputs"
    assert publish(fake_run(tmp_path, "bad", False, False, ok=False), out) is None and not out.exists()
    run = fake_run(tmp_path, "x", False, False)
    publish(run, out)
    rep = json.loads((run / "report.json").read_text(encoding="utf-8"))
    rep["passed"] = True
    (run / "report.json").write_text(json.dumps(rep), encoding="utf-8")
    publish(run, out)
    index = json.loads((out / "index.json").read_text(encoding="utf-8"))
    assert [e["name"] for e in index] == ["x"] and "passed checks only" in index[0]["status"]
