"""Live check against the real local vision model (LFM3-VL-2B on the GPU PC). Skipped when the endpoint does not answer.
Confirms the property the glance test relies on: a big title is read from a 300 px thumbnail, a tiny one is not."""
import pytest
from PIL import Image, ImageDraw, ImageFont

from poster_agent.critic.probes import glance_test, prefer
from poster_agent.llm.openai_vlm import OpenAICompatVLM
from poster_agent.spec.brand import ROOT

vlm = OpenAICompatVLM()
pytestmark = [pytest.mark.slow, pytest.mark.skipif(not vlm.available(), reason="local VLM endpoint not reachable")]
FONT = str(ROOT / "brands/fonts/Montserrat-Bold.ttf")


def poster(path, title_px, hero=False):
    im = Image.new("RGB", (1080, 1920), (30, 40, 66))
    d = ImageDraw.Draw(im)
    d.text((60, 700), "ROBOTICS DAY 2027", font=ImageFont.truetype(FONT, title_px), fill=(255, 255, 255))
    d.text((60, 1500), "14 AUGUST 2027", font=ImageFont.truetype(FONT, 90 if hero else 12), fill=(133, 206, 189))
    if hero:
        d.rectangle((60, 200, 1020, 620), fill=(133, 206, 189))
        d.text((60, 1100), "KCT SEMINAR HALL", font=ImageFont.truetype(FONT, 70), fill=(255, 255, 255))
    im.save(path)
    return path


def test_a_big_title_is_read_at_thumbnail_size_and_a_tiny_one_is_not(proj_tmp):
    big = glance_test(vlm, poster(proj_tmp / "big.png", 130, hero=True), "ROBOTICS DAY 2027", "14 AUGUST 2027", proj_tmp / "w1")
    tiny = glance_test(vlm, poster(proj_tmp / "tiny.png", 14), "ROBOTICS DAY 2027", "14 AUGUST 2027", proj_tmp / "w2")
    assert big[0].passed is True and big[1].passed is True, [r.to_dict() for r in big]
    assert tiny[0].passed is not True and tiny[1].passed is not True, [r.to_dict() for r in tiny]  # unreadable = not a pass


def test_the_model_prefers_the_clearly_better_poster_in_both_orders(proj_tmp):
    good, bad = poster(proj_tmp / "good.png", 130, hero=True), poster(proj_tmp / "bad.png", 14)
    assert prefer(vlm, good, bad, proj_tmp / "p1") in ("a", "tie")
    assert prefer(vlm, bad, good, proj_tmp / "p2") in ("b", "tie")  # never prefers the bad one
