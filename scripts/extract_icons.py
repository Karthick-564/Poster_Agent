"""Write a curated set of Lucide icons (MIT licence) as plain SVG files into brands/icons/ so the poster agent can place them as vectors.
The source is the `lucide-py` package (pip install --target data/work/icons_pkg lucide-py). Usage: python scripts/extract_icons.py"""
from __future__ import annotations

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data/work/icons_pkg/lucide/icons"
OUT = ROOT / "brands/icons"

NAMES = """cog settings wrench cpu zap battery-charging car bot code terminal circuit-board atom flask-conical microscope graduation-cap
book-open users user megaphone handshake briefcase banknote calendar clock map-pin phone mail globe trophy award lightbulb rocket shield
leaf heart-pulse palette camera film music mic newspaper file-text clipboard-list line-chart bar-chart pen-tool layers database wifi radio
plane truck factory building hammer ruler drafting-compass scale languages message-circle star target trending-up droplets sun wind flame
thermometer brain dna stethoscope pill dumbbell medal gamepad-2 monitor smartphone satellite sprout store landmark scroll ticket gift""".split()


def element(tag: str, attrs: dict) -> str:
    a = " ".join(f'{k.replace("_", "-")}="{v}"' for k, v in attrs.items())
    return f"<{tag} {a}/>"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    done, missing = [], []
    for name in NAMES:
        f = SRC / (name.replace("-", "_") + ".py")
        if not f.exists():
            missing.append(name)
            continue
        m = re.search(r"data = (\{.*\})\n", f.read_text(encoding="utf-8"))
        data = ast.literal_eval(m.group(1))
        body = "".join(element(tag, attrs) for item in data["items"] for tag, attrs in item.items())
        svg = ('<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#000000" '
               f'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">{body}</svg>')
        (OUT / f"{name}.svg").write_text(svg, encoding="utf-8")
        done.append(name)
    (OUT / "LICENSE.txt").write_text("Icons from Lucide (https://lucide.dev), ISC/MIT licensed. Free to use with attribution to the "
                                     "Lucide contributors.\n", encoding="utf-8")
    print(f"wrote {len(done)} icons to {OUT}; missing: {missing}")


if __name__ == "__main__":
    main()
