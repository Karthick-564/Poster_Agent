"""A small local page for the agent: paste the requirement, pick brand and format, get the poster (PNG / PDF / .vectorcraft links).

    python -m poster_agent.webapp [--port 8765]

Standard library only, bound to 127.0.0.1. Each request runs `run_poster.py --prompt ...` as a subprocess (so the page shares nothing with
the pipeline but its files) and shows the log, the checks summary and the poster when it finishes."""
from __future__ import annotations

import argparse
import html
import json
import re
import subprocess
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from poster_agent.spec.brand import BRANDS_DIR, ROOT

JOBS: dict[str, dict] = {}
FORMATS = ["story", "fourfive", "square"]


def brands() -> list[str]:
    return sorted(p.parent.name for p in BRANDS_DIR.glob("*/brand.yaml") if not p.parent.name.startswith("_")) or ["kct"]


def page(body: str, refresh: bool = False) -> str:
    meta = '<meta http-equiv="refresh" content="4">' if refresh else ""
    return (f"<!doctype html><meta charset=utf-8><title>Poster Agent</title>{meta}<style>body{{font:15px system-ui;max-width:880px;margin:2rem auto;"
            "padding:0 1rem}}textarea{{width:100%;height:11rem}}img{{max-width:100%;border:1px solid #ccc}}pre{{background:#f4f4f4;padding:.6rem;"
            "overflow:auto;max-height:18rem}}label{{display:block;margin:.6rem 0}}button{{padding:.5rem 1.2rem}}</style>" + body)


def form() -> str:
    opts = lambda xs: "".join(f"<option>{html.escape(x)}</option>" for x in xs)
    return page(f"""<h1>Poster Agent</h1><form method=post action=/make>
<label>Event requirement (dates, venue, speakers, contact lines, style wishes; the exact words you want printed)
<textarea name=prompt required></textarea></label>
<label>Brand <select name=brand>{opts(brands())}</select></label>
<label>Format <select name=format>{opts(FORMATS)}</select></label>
<label><input type=checkbox name=nogen> No generated art (faster, no GPU)</label>
<button>Make the poster</button></form>""")


def run_job(job_id: str, prompt: str, brand: str, fmt: str, no_gen: bool) -> None:
    job = JOBS[job_id]
    cmd = [sys.executable, str(ROOT / "run_poster.py"), "--prompt", prompt, "--brand", brand, "--format", fmt, "--rounds", "2"]
    if no_gen:
        cmd.append("--no-gen")
    log = ROOT / "data" / "work" / f"web_{job_id}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "w", encoding="utf-8") as f:
        subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, env={**__import__("os").environ, "PYTHONPATH": str(ROOT)})
    text = log.read_text(encoding="utf-8", errors="ignore")
    m = re.search(r"brief written -> .*briefs[\\/]([^\\/\r\n]+)[\\/]brief\.yaml", text)
    job.update(done=True, log=text[-4000:], name=m.group(1) if m else None)


def result(job_id: str) -> str:
    job = JOBS.get(job_id)
    if not job:
        return page("<p>Unknown job.</p><p><a href=/>Back</a></p>")
    if not job.get("done"):
        return page(f"<h1>Working...</h1><p>Started {int(time.time() - job['t0'])} s ago. Planning, art, drawing and checking take a few "
                    "minutes.</p>", refresh=True)
    name = job.get("name")
    rp = ROOT / "runs" / str(name) / "report.json" if name else None
    if not rp or not rp.exists():
        return page(f"<h1>No poster</h1><pre>{html.escape(job['log'])}</pre><p><a href=/>Back</a></p>")
    rep = json.loads(rp.read_text(encoding="utf-8"))
    verdict = ("passed all checks" + ("" if rep.get("reviewed") else " (no vision review)")) if rep.get("passed") else "did NOT pass its checks"
    failed = [c["name"] for c in (rep.get("checks") or {}).get("results", []) if not c["passed"]]
    files = "".join(f'<li><a href="/file/{name}/{n}">{n}</a></li>' for n in (f"{name}.png", f"{name}.pdf", f"{name}.vectorcraft")
                    if (ROOT / "runs" / name / n).exists())
    return page(f"<h1>{html.escape(name)}</h1><p>The poster {verdict}." + (f" Failed checks: {', '.join(failed)}." if failed else "")
                + f'</p><img src="/file/{name}/{name}.png"><ul>{files}</ul><p><a href=/>Make another</a></p>')


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, body: bytes, ctype: str = "text/html; charset=utf-8") -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        u = urlparse(self.path)
        if u.path == "/":
            return self._send(200, form().encode())
        if u.path.startswith("/job/"):
            return self._send(200, result(u.path[5:]).encode())
        if u.path.startswith("/file/"):
            parts = u.path[6:].split("/")
            f = (ROOT / "runs" / parts[0] / parts[1]).resolve() if len(parts) == 2 else None
            if f and f.is_file() and (ROOT / "runs").resolve() in f.parents and f.suffix in (".png", ".pdf", ".vectorcraft"):
                ctype = {".png": "image/png", ".pdf": "application/pdf"}.get(f.suffix, "application/octet-stream")
                return self._send(200, f.read_bytes(), ctype)
        self._send(404, b"not found", "text/plain")

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/make":
            return self._send(404, b"not found", "text/plain")
        q = parse_qs(self.rfile.read(int(self.headers.get("Content-Length", 0))).decode("utf-8"))
        prompt = (q.get("prompt") or [""])[0].strip()
        brand, fmt = (q.get("brand") or ["kct"])[0], (q.get("format") or ["story"])[0]
        if not prompt or brand not in brands() or fmt not in FORMATS:
            return self._send(400, b"bad request", "text/plain")
        job_id = uuid.uuid4().hex[:8]
        JOBS[job_id] = {"t0": time.time(), "done": False}
        threading.Thread(target=run_job, args=(job_id, prompt, brand, fmt, "nogen" in q), daemon=True).start()
        self.send_response(303)
        self.send_header("Location", f"/job/{job_id}")
        self.end_headers()

    def log_message(self, *args) -> None:  # quiet
        pass


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    httpd = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Poster Agent page: http://127.0.0.1:{args.port}/  (Ctrl+C to stop)")
    httpd.serve_forever()


if __name__ == "__main__":
    main()
