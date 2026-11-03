# Poster_Agent

An agent that turns an event brief, a brand profile and supplied assets into a brand-correct poster: PNG, PDF and an editable
`.vectorcraft` file.

```
brief -> plan (LLM picks components + content; validated) -> layout (deterministic solver) -> assets (ComfyUI: Flux backgrounds,
Qwen cutouts) -> compile (native VectorCraft MCP, as a client) -> checks (OCR, contrast, logo rules, geometry, dead space)
-> vision critic + router + bounded refine loop -> optional polish pass
```

## Principles
1. The VectorCraft MCP server is VectorCraft's own (`vectorcraft-cli mcp`); this repo is only an MCP **client** (a test enforces it).
2. `PosterSpec` is the single source of truth; stages never silently override each other.
3. Poster text is verbatim from the brief or brand profile. The LLM never writes poster copy.
4. Logos and photos are never generated or altered by diffusion (crop / mask / brand colour treatment only).
5. No per-brief hacks: behaviour comes from brand profiles, component definitions and the plan.
6. Reports state only what a check actually measured.

## Setup
- Python 3.12, `pip install -e .[dev]`.
- Copy `.env.example` to `.env` and fill in your own keys / URLs (ComfyUI, Gemini, Groq, VectorCraft CLI path).
- VectorCraft (the CLI and desktop app) is **not** included; install it separately and point `VECTORCRAFT_CLI` at it.
- A ComfyUI server with an image model (Flux / Qwen-Image) is optional: without it use `--no-gen`.

## Brands
Institution assets are deliberately **not** in this repository. A brand lives in `brands/<key>/brand.yaml` plus
`brands/<key>/logos/`; the schema is `BrandProfile` in `poster_agent/spec/brand.py`. Create your own and the
tests that need a brand (they are skipped when none is present) will run against it.

## Run
```
python run_poster.py briefs/<name> [--no-gen] [--no-critic] [--rounds N] [--polish]
python run_poster.py briefs/<name> --watch        # draw live in the VectorCraft window; edit by hand, then:
python run_poster.py --finalize runs/<name>       # re-export and re-check what you ended up with
python -m pytest -q                               # fast suite;  python -m pytest -q -m slow  runs the VectorCraft render tests
```
The final image of every real run is also copied to `outputs/<name>.png` (indexed in `outputs/index.json` with its measured status); `outputs/` is git-ignored.
A brief is `briefs/<name>/brief.yaml` plus an optional `assets/` folder of `kind__label.ext` files.

## Third-party
Icons in `brands/icons/` are from [Lucide](https://lucide.dev) (ISC/MIT). Fonts in `brands/fonts/` are Montserrat and Barlow (SIL OFL).
