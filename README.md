# Poster_Agent

An agent that turns an event brief, a brand profile and supplied assets into a brand-correct poster: PNG, PDF and an editable
`.vectorcraft` file.

```
brief -> plan (LLM writes a concept, then picks components + content; validated against its own concept)
      -> layout (deterministic solver; text is placed against the art once the art exists)
      -> assets (ComfyUI: background art, object cutouts; supplied photos are only cropped / matted)
      -> compile (native VectorCraft MCP, as a client) -> checks (OCR, contrast, logo rules, geometry, dead space)
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
- Python 3.12: `python -m venv .venv`, then `.venv\Scripts\python -m pip install -r requirements.txt` (or `pip install -e .[dev]`).
- Copy `.env.example` to `.env` and fill in your own keys / URLs (ComfyUI, Gemini, Groq, VectorCraft CLI path).
- VectorCraft (the CLI and desktop app) is **not** included; install it separately and point `VECTORCRAFT_CLI` at it.
- A ComfyUI server with the image models is optional: without it use `--no-gen`.

## Brands
Institution assets are deliberately **not** in this repository. A brand lives in `brands/<key>/brand.yaml` plus
`brands/<key>/logos/`; the schema is `BrandProfile` in `poster_agent/spec/brand.py`. Create your own and the
tests that need a brand (they are skipped when none is present) will run against it. Logo size limits
(`logo_rules.max_width_pct`) are brand settings, not code.

## Run
```
python run_poster.py briefs/<name> [--no-gen] [--no-critic] [--rounds N] [--polish]
python run_poster.py briefs/<name> --watch        # draw live in the VectorCraft window; edit by hand, then:
python run_poster.py --finalize runs/<name>       # re-export and re-check what you ended up with
python -m pytest -q                               # fast suite;  python -m pytest -q -m slow  runs the VectorCraft render tests
```
The final image of every real run is also copied to `outputs/<name>.png` (indexed in `outputs/index.json` with its measured status); `outputs/` is git-ignored.
A brief is `briefs/<name>/brief.yaml` plus an optional `assets/` folder of `kind__label.ext` files. `runs/<name>/` holds the plan of the
kept poster (`plan.json`), the spec, `report.json` (every seed, prompt and check) and the files.

## How a poster is planned
- **Concept first.** The planner writes a short concept before any layout: the one idea, the focal point (which must be the hero
  component), the reading order, the design devices it uses and the main contrast. The plan is rejected if it does not do what the
  concept says, or uses a device the concept does not name.
- **Design devices** the planner can choose: a two-tone ground (`background.panel`, a block from a named component to the bottom edge, which
  must differ clearly in tone from the ground above it), a `PullQuote` component, a keyline frame / border, big type, a hero image, art as the ground.
- **Art direction.** For generated art the planner writes an art director's brief (focal subject and where it sits, composition and depth,
  graphic devices, colour roles); the system turns it into the image prompt and adds where the text will sit so those areas stay calm.
- **Looks** (`poster_agent/plan/looks.py`) set only the ground and the type style. The header, footer, text colours and headline treatment are the planner's decisions.
- Text is placed *after* the art exists: a placement search scores candidate layouts by the fine detail under the text and the overlap
  with the picture's subject. Several background pictures are tried and the one the text fits best is kept; the area under the text
  is softened locally (never the whole picture) only if it is still busy.

## Image models
The background model is set with `BACKGROUND_BACKEND` in `.env`: `qwen` (default here), `flux` or `ideogram`. Object cutouts use Qwen-Image
and a local matte. Every generated image is OCR-screened on arrival; any lettering, including invented non-Latin glyphs, is rejected.

`scripts/model_compare.py` measured the three models on the same layout (3 briefs x 2 seeds, a small test):

| | real text requested: lines correct | calm areas requested |
|---|---|---|
| FLUX Klein | 16 / 18 | filled with invented lettering in 6 of 6 images |
| Ideogram 4 | 15 / 18 | invented lettering in 5 of 6; sometimes blocked by its safety filter |
| Qwen-Image 2.1 | 10 / 18 | no lettering in 6 of 6; areas about 5x calmer than the rest |

So Qwen-Image paints the backgrounds and our own pipeline sets the real text. The text-matching scorer is approximate.

## Scripts
| script | what it does |
|---|---|
| `scripts/evaluate.py` | runs the fictional evaluation briefs (`eval/briefs/`, built by `make_eval_briefs.py`) and reports what was measured |
| `scripts/stage1_eval.py` | stage-1 scoreboard: free-text request -> structured, verbatim brief (recall, misfiled, invented) |
| `scripts/reference_poster.py` | asks an image model to design a whole poster (a design reference only; its text is never used) |
| `scripts/ideogram_box_test.py` | Ideogram captions with a box per text element (and an empty-areas variant) |
| `scripts/model_compare.py` | the table above: text accuracy and calm-area behaviour per model |
| `scripts/design_stats.py` | headline size, text column and type tiers of any poster image, to compare designs with real posters |
| `scripts/poster_dna.py`, `structure_demo.py` | measurement and structure demos |

## Known limits
- Logo files supplied as opaque coloured rectangles stay rectangles; transparent or vector logos are needed for a logo that sits directly on the ground.
- Plans vary from run to run; the vision critic is lenient (it has scored posters with visible defects 10/10) and the dead-space check can
  flag smooth dark art as empty. Treat a pass as "no failing measured check", not as design quality.
- Generated art never contains people or text; people appear only from supplied photos.
- Picking one person out of a supplied photo by description (`poster_agent/assets/select_person.py`) uses the local vision model and is not yet
  validated on group photos.

## Third-party
Icons in `brands/icons/` are from [Lucide](https://lucide.dev) (ISC/MIT). Fonts in `brands/fonts/` are Montserrat and Barlow (SIL OFL).
