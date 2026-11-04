"""Local vision model client + glance-test critic. No network: the endpoint is faked with httpx.MockTransport / scripted replies."""
import json

import httpx
import pytest
from PIL import Image

from poster_agent.critic import critique
from poster_agent.critic.probes import glance_test, prefer, thumbnail
from poster_agent.ingest import Brief
from poster_agent.llm import LLMError
from poster_agent.llm.openai_vlm import OpenAICompatVLM, coerce_reply, unwrap_server_placeholder

PLACEHOLDER = {"hierarchy": 3, "balance": 3, "readability": 3, "brand_fit": 4, "polish": 3,
               "three_second_test": {"event": "Detected Title", "date": "Detected Date", "venue": "Detected Venue", "passed": True},
               "fixes": ["x"], "raw_critique": "Generative AI Workshop"}
ERROR_PLACEHOLDER = {"hierarchy": 3, "balance": 3, "readability": 3, "brand_fit": 3, "polish": 3,
                     "three_second_test": {"passed": False}, "fixes": ["Review processed with warning: CUDA out of memory"]}


def png(tmp_path, name="p.png", size=(400, 600)):
    p = tmp_path / name
    Image.new("RGB", size, "white").save(p)
    return p


class FakeVLM:
    """Stands in for the local model: replies from a queue of plain strings through the same ask_text interface."""
    name = "fake:vlm"
    probe_critic = True

    def __init__(self, replies):
        self.replies, self.calls = list(replies), []

    def ask_text(self, system, user, *, images=None, temperature=0.0):
        self.calls.append({"user": user, "images": images})
        r = self.replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


# ---- the server's canned placeholders must never be read as a verdict -------------------------------------------------------

def test_server_placeholders_are_unwrapped_or_rejected_never_treated_as_an_answer():
    assert unwrap_server_placeholder(PLACEHOLDER) == "Generative AI Workshop"  # the model's real words are inside
    assert unwrap_server_placeholder(ERROR_PLACEHOLDER) == ""  # an internal error: no answer at all
    assert unwrap_server_placeholder({"answer": "yes"}) is None  # a genuine reply is left alone
    assert coerce_reply(json.dumps({"answer": "no", "reason": "plain"})) == {"answer": "no", "reason": "plain"}
    assert coerce_reply("Yes, the title is clear")["answer"] == "yes"  # small models skip JSON
    assert coerce_reply(json.dumps({**PLACEHOLDER, "raw_critique": "No. It is empty."}))["answer"] == "no"
    with pytest.raises(LLMError):
        coerce_reply(json.dumps(ERROR_PLACEHOLDER))
    with pytest.raises(LLMError):
        coerce_reply("a rambling paragraph with no verdict")


def test_client_sends_base64_image_thinking_off_and_unwraps_the_reply(tmp_path, monkeypatch):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        seen["url"] = str(request.url)
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(PLACEHOLDER)}}]})
    real_post = httpx.post
    monkeypatch.setattr(httpx, "post", lambda url, **kw: httpx.Client(transport=httpx.MockTransport(handler)).post(url, **kw))
    vlm = OpenAICompatVLM(base_url="http://gpu:8080/v1", model="LFM3-VL-2B")
    assert vlm.ask_text("sys", "read the title", images=[png(tmp_path)]) == "Generative AI Workshop"
    body = seen["body"]
    assert seen["url"] == "http://gpu:8080/v1/chat/completions" and body["model"] == "LFM3-VL-2B" and body["thinking"] is False
    parts = body["messages"][1]["content"]
    assert parts[0]["image_url"]["url"].startswith("data:image/png;base64,") and parts[-1] == {"type": "text", "text": "read the title"}
    monkeypatch.setattr(httpx, "post", real_post)


def test_client_without_an_endpoint_says_how_to_configure_it(monkeypatch):
    monkeypatch.setattr("poster_agent.llm.openai_vlm._env", lambda name: "")
    vlm = OpenAICompatVLM()
    assert not vlm.available()
    with pytest.raises(LLMError, match="VLM_URL"):
        vlm.chat("s", "u")


# ---- the glance test --------------------------------------------------------------------------------------------------------



def test_glance_test_compares_what_the_model_read_with_the_brief(tmp_path):
    vlm = FakeVLM(["Generative AI Workshop", "12 & 13 December 2026"])
    res = glance_test(vlm, png(tmp_path), "GENERATIVE AI WORKSHOP", "12 & 13 DECEMBER 2026", tmp_path)
    assert [r.passed for r in res] == [True, True]
    assert vlm.calls[0]["images"][0].name == "glance_300.png"  # it was shown the 300 px thumbnail
    bad = glance_test(FakeVLM(["Some Other Event", "none"]), png(tmp_path), "GENERATIVE AI WORKSHOP", "12 DECEMBER 2026", tmp_path)
    assert [r.passed for r in bad] == [False, False] and bad[0].read == "Some Other Event"
    down = glance_test(FakeVLM([LLMError("down"), LLMError("down")]), png(tmp_path), "X EVENT", "1 JAN", tmp_path)
    assert [r.passed for r in down] == [None, None]  # an unusable reply is "unknown", not a pass or a fail


def test_critique_reports_the_glance_test_honestly(tmp_path):
    brief = Brief(brand="kct", event_title="GENERATIVE AI WORKSHOP", date="12 & 13 DECEMBER 2026")
    ok = critique(png(tmp_path), None, None, brief, None, FakeVLM(["Generative AI Workshop", "12 and 13 December 2026"]), tmp_path)
    assert ok.available and ok.method == "glance" and ok.score == 10 and ok.passed and not ok.issues
    half = critique(png(tmp_path), None, None, brief, None, FakeVLM(["Generative AI Workshop", "none"]), tmp_path)
    assert half.score == 5 and [i.severity for i in half.issues] == ["warning"] and not half.passed
    title_fail = critique(png(tmp_path), None, None, brief, None, FakeVLM(["Welcome", "12 & 13 December 2026"]), tmp_path)
    assert title_fail.issues[0].severity == "error" and title_fail.issues[0].category.value == "hierarchy"
    assert "the vision model read the event title as 'Welcome'" in title_fail.issues[0].evidence
    unavailable = critique(png(tmp_path), None, None, brief, None, FakeVLM([LLMError("x"), LLMError("x")]), tmp_path)
    assert not unavailable.available and unavailable.score is None  # no reading -> no verdict and no made-up score


# ---- pairwise preference ---------------------------------------------------------------------------------------------------

def test_pairwise_preference_survives_the_order_swap_or_is_a_tie(tmp_path):
    a, b = png(tmp_path, "a.png"), png(tmp_path, "b.png")
    assert prefer(FakeVLM(["A", "B"]), a, b, tmp_path) == "a"  # picks a's slot in both orders
    assert prefer(FakeVLM(["B", "A"]), a, b, tmp_path) == "b"
    assert prefer(FakeVLM(["A", "A"]), a, b, tmp_path) == "tie"  # position bias: always the left one
    assert prefer(FakeVLM(["A", LLMError("x")]), a, b, tmp_path) == "tie"
    assert prefer(FakeVLM(["The first one looks better.", "B"]), a, b, tmp_path) == "tie"  # not a letter: no guess


def test_large_images_are_downscaled_before_they_are_sent(tmp_path):
    from PIL import Image
    from poster_agent.llm.openai_vlm import _data_url
    big, small = tmp_path / "big.png", tmp_path / "small.png"
    Image.new("RGB", (2160, 3840), "white").save(big)
    Image.new("RGB", (300, 400), "white").save(small)
    assert _data_url(big).startswith("data:image/jpeg;base64,") and len(_data_url(big)) < 30000  # blank, so tiny after JPEG
    assert _data_url(small).startswith("data:image/png;base64,")
