"""AssetGenerator: Ideogram backgrounds and Qwen hero cutouts, with caching and persisted seeds (reproducible runs).

Logos, headshots and photos are never produced here (guardrail 4) — only atmospheric backgrounds and object heroes.

Every image is screened the moment it comes back from the GPU box: generated art must contain no text. That catches model
refusal banners ("Image blocked by safety filter") and baked-in lettering before they are matted, laid out or cached. A rejected
image is never cached; the next seed is tried, and the rejection reasons are recorded on the GenInfo.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image

from rapidfuzz import fuzz

from poster_agent.checks.ocr import alnum, read_lines, verifiable
from poster_agent.spec import AssetKind, AssetRef, CanvasSpec
from poster_agent.spec.brand import ROOT

from .comfy import ComfyClient
from .matting import alpha_coverage, cutout
from .prompts import (ArtDirection, CalmLayout, caption_text, flux_prompt, ideogram_prompt, qwen_subject_prompt,
                      validate_ideogram_caption)
from .workflows import QWEN_BG_NEGATIVE, flux_workflow, ideogram_workflow, qwen_workflow, round16

TARGET_MEGAPIXELS = 1.05  # Ideogram renders ~1 MP; the compiler scales it to the canvas
ASPECT_SIZES = {"1:1": (1024, 1024), "4:5": (896, 1120), "3:4": (864, 1152), "16:9": (1280, 720), "9:16": (720, 1280)}
ATTEMPTS = 3


def derived_seed(*parts: str) -> int:
    """Deterministic 31-bit seed from the request: same request -> same seed -> same cache entry (replans reuse
    assets); callers pass an explicit seed (or a seed_offset) to get a different draw."""
    return int(hashlib.sha1("|".join(parts).encode()).hexdigest()[:8], 16) % (2 ** 31)


class AssetError(RuntimeError):
    pass


@dataclass
class GenInfo:
    kind: str
    prompt: str
    seed: int
    width: int
    height: int
    model: str
    seconds: float
    cached: bool
    path: str
    rejected: list[str] = field(default_factory=list)  # attempts thrown away (text in the art, blank, bad matte) and why


def bg_size(canvas: CanvasSpec) -> tuple[int, int]:
    s = (TARGET_MEGAPIXELS * 1e6 / (canvas.width * canvas.height)) ** 0.5
    return round16(canvas.width * s), round16(canvas.height * s)


def cover_crop(img: Image.Image, aspect: float) -> Image.Image:
    """Centre-crop to the target aspect ratio (width/height)."""
    w, h = img.size
    if w / h > aspect:
        nw = round(h * aspect)
        return img.crop(((w - nw) // 2, 0, (w - nw) // 2 + nw, h))
    nh = round(w / aspect)
    return img.crop((0, (h - nh) // 2, w, (h - nh) // 2 + nh))


def looks_blank(img: Image.Image) -> bool:
    lum = np.array(img.convert("L"), dtype=np.float32)
    return float(lum.std()) < 2.0


TEXT_CONF = 0.8  # generic lettering must be read confidently: OCR hallucinates short strings on fine patterns at ~0.5
REFUSAL_PHRASE = "imageblockedbysafetyfilter"


def _is_refusal(text: str) -> bool:
    a = alnum(text)
    return len(a) >= 8 and fuzz.partial_ratio(REFUSAL_PHRASE, a) >= 62  # catches garbled banners ('Image clocknd by sgineer')


SEAM_COVERAGE = 0.9  # a straight, full-width tonal step this consistent is an artefact in an abstract background, not a feature


def seam_coverage(img: Image.Image) -> tuple[float, int, int]:
    """(share of the image width/height over which one row/column has a same-signed tonal step > 5 levels, its index, axis 0=row)."""
    g = np.asarray(img.convert("L").resize((256, 256)), dtype=float)
    best = (0.0, 0, 0)
    for ax in (0, 1):
        s = np.diff(g, axis=ax)
        for i in range(s.shape[ax]):
            line = s[i] if ax == 0 else s[:, i]
            for sign in (1, -1):
                cov = float(np.mean(sign * line > 5))
                if cov > best[0]:
                    best = (cov, i, ax)
    return best


FOREIGN_CONF = 0.5  # lettering in any other script (diffusion models invent CJK-like glyphs) is rejected at a lower bar than Latin text


def _foreign_lettering(text: str) -> bool:
    """Two or more non-ASCII letters: glyphs of another script. Generated art has no text in any script, so these are never legitimate."""
    return sum(not c.isascii() for c in alnum(text)) >= 2


def art_problem(img: Image.Image, backdrop: bool = False) -> str | None:
    """Why this generated image must not be used, or None. Art may not contain text: refusal banners (even garbled) and
    confidently-read lettering both count; low-confidence OCR noise on busy patterns does not. A `backdrop` (full-poster
    ground) must also have no hard straight seam across it."""
    if looks_blank(img):
        return "image is blank"
    if backdrop:
        cov, idx, ax = seam_coverage(img)
        if cov >= SEAM_COVERAGE:
            return f"a hard straight {'horizontal' if ax == 0 else 'vertical'} seam crosses the background at {idx / 256:.0%} ({cov:.0%} of its length)"
    hits = []
    for ln in read_lines(img):
        if not verifiable(ln.text):
            if ln.conf >= FOREIGN_CONF and _foreign_lettering(ln.text):
                hits.append(ln)
            continue
        if (ln.conf >= 0.4 and _is_refusal(ln.text)) or (ln.conf >= TEXT_CONF and len(alnum(ln.text)) >= 4):
            hits.append(ln)
    if hits:
        return "text found in the generated image: " + "; ".join(repr(ln.text) for ln in hits[:3])
    return None


class AssetGenerator:
    def __init__(self, comfy: ComfyClient, out_dir: Path, cache_dir: Path | None = None, background_backend: str | None = None):
        import os
        from dotenv import dotenv_values
        self.comfy = comfy
        # which model paints backgrounds: flux (default) | qwen | ideogram; BACKGROUND_BACKEND in the environment or .env switches it
        self.background_backend = (background_backend or os.environ.get("BACKGROUND_BACKEND")
                                   or dotenv_values(ROOT / ".env").get("BACKGROUND_BACKEND") or "flux")
        self.out_dir = Path(out_dir)
        self.cache = Path(cache_dir or ROOT / "data/cache/gen")
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.cache.mkdir(parents=True, exist_ok=True)
        self._last_model: str | None = None

    # ---- internals -------------------------------------------------------
    def _screen(self, data: bytes, backdrop: bool = False) -> str | None:
        return art_problem(Image.open(BytesIO(data)).convert("RGB"), backdrop)

    def _render(self, model: str, workflow: dict, save_node: str, backdrop: bool = False) -> tuple[bytes, bool, str | None]:
        """(image bytes, from cache, problem). Only screened-clean images are cached (a .ok sidecar); a rejected request is
        remembered (.bad) so the same seed is never re-run just to get the same bad image."""
        key = hashlib.sha1(json.dumps(workflow, sort_keys=True).encode()).hexdigest()
        f, ok, bad = self.cache / f"{key}.png", self.cache / f"{key}.ok", self.cache / f"{key}.bad"
        if bad.exists():
            return b"", True, bad.read_text(encoding="utf-8")
        if f.exists():
            data = f.read_bytes()
            if backdrop and seam_coverage(Image.open(BytesIO(data)).convert("RGB"))[0] >= SEAM_COVERAGE:  # cached before seam screening
                f.unlink()
                ok.unlink(missing_ok=True)
                bad.write_text("a hard straight seam crosses the cached background", encoding="utf-8")
                return b"", True, bad.read_text(encoding="utf-8")
            if ok.exists():
                return data, True, None
            if self._screen(data, backdrop) is None:  # an entry from before screening existed: verify once, then trust
                ok.write_text("1")
                return data, True, None
            f.unlink()  # a bad image must never be served from the cache
            bad.write_text("text found in a previously cached image", encoding="utf-8")
            return b"", True, "text found in a previously cached image"
        if self._last_model and self._last_model != model:
            self.comfy.free_vram()  # 24 GB holds one big model at a time
        data = self.comfy.run(workflow, save_node)
        self._last_model = model
        problem = self._screen(data, backdrop)
        if problem is None:
            f.write_bytes(data)
            ok.write_text("1")
        else:
            bad.write_text(problem, encoding="utf-8")
        return data, False, problem

    def _rel(self, path: Path) -> str:
        try:
            return path.resolve().relative_to(ROOT).as_posix()
        except ValueError:
            return path.resolve().as_posix()

    # ---- public ----------------------------------------------------------
    def background(self, asset_id: str, art: ArtDirection, canvas: CanvasSpec, calm: CalmLayout | str, *, seed: int | None = None,
                   preset: str = "default", seed_offset: int = 0, attempts: int = ATTEMPTS,
                   backend: str | None = None) -> tuple[AssetRef, GenInfo]:
        """A text-free atmospheric background. backend 'flux' (default: reliable, ~12 s), 'qwen' (~45 s, keeps requested areas calm) or 'ideogram' (strict JSON caption,
        prone to false 'blocked by safety filter' on abstract scenes in this setup)."""
        backend = backend or self.background_backend
        w, h = bg_size(canvas)
        if backend == "flux":
            prompt_text = flux_prompt(art, calm)
            model_label, model_key = "flux3_klein", "flux3"
            record = prompt_text
            salt = prompt_text

            def make(s: int):
                return flux_workflow(prompt_text, w, h, s, prefix=f"bg_{asset_id}")
        elif backend == "qwen":  # listens to "leave this area calm" and draws no lettering (measured: scripts/model_compare.py)
            prompt_text = flux_prompt(art, calm)
            model_label, model_key = "qwen_image_2.1", "qwen21"
            record = prompt_text
            salt = prompt_text

            def make(s: int):
                return qwen_workflow(prompt_text, w, h, s, negative=QWEN_BG_NEGATIVE, prefix=f"bg_{asset_id}")
        elif backend == "ideogram":
            prompt = ideogram_prompt(art, calm)
            off_schema = validate_ideogram_caption(prompt)
            if off_schema:  # an off-schema caption risks a false 'Image blocked by safety filter': never send one
                raise AssetError(f"{asset_id}: caption does not follow the Ideogram schema: {'; '.join(off_schema)}")
            model_label, model_key = "ideogram4", "ideogram4"
            record = caption_text(prompt)
            salt = json.dumps(prompt, sort_keys=True)

            def make(s: int):
                return ideogram_workflow(prompt, w, h, s, preset, prefix=f"bg_{asset_id}")
        else:
            raise ValueError(f"unknown background backend {backend!r}")

        base = seed if seed is not None else derived_seed(asset_id, salt, f"{w}x{h}") + seed_offset
        rejected: list[str] = []
        for k in range(attempts):
            s = base + k
            wf, node = make(s)
            t = time.time()
            data, cached, problem = self._render(model_key, wf, node, backdrop=True)
            if problem:
                rejected.append(f"seed {s}: {problem}")
                continue
            img = cover_crop(Image.open(BytesIO(data)).convert("RGB"), canvas.width / canvas.height)
            path = self.out_dir / f"{asset_id}.png"
            img.save(path)
            info = GenInfo("background", record, s, img.width, img.height, model_label, time.time() - t, cached,
                           self._rel(path), rejected)
            return AssetRef(id=asset_id, kind=AssetKind.generated_background, path=info.path, provenance="generated", seed=s,
                            prompt=info.prompt), info
        raise AssetError(f"{asset_id}: no usable background after {attempts} attempts ({'; '.join(rejected)})")

    def hero_cutout(self, asset_id: str, subject: str, *, aspect: str = "1:1", seed: int | None = None,
                    attempts: int = ATTEMPTS, seed_offset: int = 0) -> tuple[AssetRef, GenInfo]:
        w, h = ASPECT_SIZES.get(aspect, ASPECT_SIZES["1:1"])
        prompt = qwen_subject_prompt(subject)
        base_seed = seed if seed is not None else derived_seed(asset_id, prompt, aspect) + seed_offset
        rejected: list[str] = []
        for k in range(attempts):  # a refusal banner, baked-in text or a failed matte gets a new seed
            s = base_seed + k
            wf, node = qwen_workflow(prompt, w, h, s, prefix=f"hero_{asset_id}")
            t = time.time()
            data, cached, problem = self._render("qwen21", wf, node)
            if problem:
                rejected.append(f"seed {s}: {problem}")
                continue
            try:
                rgba = cutout(data)
            except ValueError as e:
                rejected.append(f"seed {s}: {e}")
                continue
            cov = alpha_coverage(rgba)
            if not 0.08 < cov < 0.97:
                rejected.append(f"seed {s}: matte coverage {cov:.2f} out of range")
                continue
            path = self.out_dir / f"{asset_id}.png"
            rgba.save(path)
            info = GenInfo("hero_cutout", prompt, s, rgba.width, rgba.height, "qwen_image_2.1", time.time() - t, cached,
                           self._rel(path), rejected)
            return AssetRef(id=asset_id, kind=AssetKind.generated_hero, path=info.path, provenance="generated", seed=s,
                            prompt=prompt), info
        raise AssetError(f"{asset_id}: no usable cutout after {attempts} attempts ({'; '.join(rejected)})")
