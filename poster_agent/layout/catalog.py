"""Machine-readable description of every component: what the planner may ask for and what we validate against.

`text_keys` lists the content keys that print text (verbatim-checked); `asset_keys` the keys that reference inventory assets;
`color_keys` / `font_keys` the keys holding palette names / font roles. A test keeps this in sync with layout/components.py.
"""
from __future__ import annotations

from typing import Any

CATALOG: dict[str, dict[str, Any]] = {
    "LogoBlock": {
        "desc": "The institution's logo, locked, with its brand exclusion zone. Every poster needs exactly one institution logo "
                "(LogoBlock or LogoTitleRow).",
        "variants": ["default"],
        "content": {"logo": "brand logo name (optional; see BRAND.logos)", "align": "left|center|right",
                    "width_pct": "number, % of canvas width (optional; never below the logo's minimum)", "band": "colour: full-width band behind the logo",
                    "plate": "true or a colour name: a flat plate that hugs the logo. ONLY for logo artwork that has no field of its "
                             "own (BRAND.logos[...].solid_field false) on a dark/busy ground; logos with solid_field true "
                             "(e.g. a navy square) need NO plate: never add one"},
        "required": [], "text_keys": [], "asset_keys": [], "color_keys": ["plate"], "font_keys": [],
    },
    "LogoTitleRow": {
        "desc": "Header row: institution logo on the left, a rule, then a left-aligned eyebrow + title column. Counts as the logo.",
        "variants": ["default"],
        "content": {"logo": "brand logo name", "left_pct": "number, logo column width %", "eyebrow": "text", "title": "text",
                    "caption": "text under the logo", "eyebrow_color": "colour", "title_color": "colour", "title_factor": "number"},
        "required": ["title"], "text_keys": ["eyebrow", "title", "caption"], "asset_keys": [],
        "color_keys": ["eyebrow_color", "title_color"], "font_keys": [],
    },
    "TitleStack": {
        "desc": "Eyebrow / title / subtitle / tagline, centred (or left). variant 'display' = one huge headline word or phrase.",
        "variants": ["default", "display"],
        "content": {"eyebrow": "text", "title": "text", "subtitle": "text", "tagline": "text", "align": "left|center|right",
                    "eyebrow_color": "colour", "title_color": "colour", "subtitle_color": "colour", "tagline_color": "colour",
                    "eyebrow_tracking": "number letter-spacing", "max_lines": "int", "title_factor": "number size multiplier",
                    "title_font": "font role",
                    "title_style": "none|accent_last|accent_first|weight_split|highlight_last: colour, weight or a solid bar behind the last title line"},
        "required": ["title"], "text_keys": ["eyebrow", "title", "subtitle", "tagline"], "asset_keys": [],
        "color_keys": ["eyebrow_color", "title_color", "subtitle_color", "tagline_color"], "font_keys": ["title_font"],
    },
    "PullQuote": {
        "desc": "A large quotation with its attribution under a short accent rule: for tributes, quotes, mission statements. Use the "
                "brief's own words; variant 'large' sets the quote bigger.",
        "variants": ["default", "large"],
        "content": {"quote": "text (verbatim, marks included if the brief has them)", "attribution": "text (verbatim)",
                    "align": "left|center|right", "quote_color": "colour", "attribution_color": "colour", "rule": "bool accent rule above",
                    "rule_color": "colour", "font_role": "font role", "max_lines": "int"},
        "required": ["quote"], "text_keys": ["quote", "attribution"], "asset_keys": [],
        "color_keys": ["quote_color", "attribution_color", "rule_color"], "font_keys": ["font_role"],
    },
    "TextBlock": {
        "desc": "A block of text. role: title|heading|subtitle|body|caption|label|micro.",
        "variants": ["default"],
        "content": {"text": "text (or lines)", "lines": "list of text", "role": "type role", "color": "colour",
                    "align": "left|center|right", "font_role": "font role", "upper": "bool", "tracking": "number",
                    "max_lines": "int", "factor": "number size multiplier"},
        "required": [], "text_keys": ["text", "lines"], "asset_keys": [], "color_keys": ["color"], "font_keys": ["font_role"],
    },
    "Divider": {
        "desc": "Hairline across the width, optionally with a centred label. variant 'ornament' = a small centred chain of diamonds "
                "(a quiet motif between info groups).",
        "variants": ["default", "ornament"],
        "content": {"label": "text (optional)", "color": "colour", "label_color": "colour", "thickness": "number",
                    "role": "type role", "font_role": "font role", "tracking": "number"},
        "required": [], "text_keys": ["label"], "asset_keys": [], "color_keys": ["color", "label_color"], "font_keys": ["font_role"],
    },
    "InfoBar": {
        "desc": "Event facts. variant 'band' = full-width block, one line per item (use anchor top|bottom to pin it to an edge). "
                "variant 'columns' = one row split into equal cells (e.g. time | date | venue).",
        "variants": ["band", "columns", "footer"],
        "content": {"items": "list of text", "fill": "colour", "fill_to": "colour (gradient end, optional)", "text_color": "colour"},
        "required": ["items"], "text_keys": ["items"], "asset_keys": [], "color_keys": ["fill", "fill_to", "text_color"], "font_keys": [],
    },
    "HighlightBox": {
        "desc": "A key phrase (event/programme name) in an outlined or filled box.",
        "variants": ["outline", "filled"],
        "content": {"text": "text", "width_pct": "number", "role": "type role", "font_role": "font role", "max_lines": "int",
                    "text_color": "colour", "edge": "colour", "fill": "colour"},
        "required": ["text"], "text_keys": ["text"], "asset_keys": [],
        "color_keys": ["text_color", "edge", "fill"], "font_keys": ["font_role"],
    },
    "ChiefGuestCard": {
        "desc": "A featured guest: optional label (CHIEF GUEST), the supplied photo cropped to just the FACE in a circle, the name (largest "
                "line), designation lines. This is how a supplied guest photo is shown. Without a photo it still gives the name a hierarchy.",
        "variants": ["default"],
        "content": {"label": "text e.g. CHIEF GUEST", "photo": "asset id", "name": "text", "lines": "list of text",
                    "photo_d_pct": "number photo diameter % of canvas width", "align": "left|center|right",
                    "name_color": "colour", "lines_color": "colour", "label_color": "colour",
                    "framing": "face (default: just the face in the circle)|head|head_and_shoulders|half_body: the supplied photo is cropped, never altered"},
        "required": ["name"], "text_keys": ["label", "name", "lines"], "asset_keys": ["photo"], "color_keys": [], "font_keys": [],
    },
    "SpeakerGrid": {
        "desc": "Circular photos with name and designation in aligned rows; last row centred. content.speakers = "
                "[{name, designation, asset}].",
        "variants": ["default"],
        "content": {"speakers": "list of {name, designation, asset}", "cols": "int columns (default 4)",
                    "role_max_lines": "int", "framing": "head|head_and_shoulders|half_body: reframe the supplied photos by cropping"},
        "required": ["speakers"], "text_keys": ["speakers[].name", "speakers[].designation"], "asset_keys": ["speakers[].asset"],
        "color_keys": [], "font_keys": [],
    },
    "PartnerLogoRow": {
        "desc": "Supplied partner/sponsor logos in one evenly spaced row.",
        "variants": ["default"],
        "content": {"assets": "list of asset ids", "height_pct": "number % of canvas height"},
        "required": ["assets"], "text_keys": [], "asset_keys": ["assets"], "color_keys": [], "font_keys": [],
    },
    "PhotoStrip": {
        "desc": "Full-bleed band of event photos side by side (use anchor bottom).",
        "variants": ["default"],
        "content": {"assets": "list of asset ids", "height_pct": "number", "tints": "list of hex colours", "crop": "cover",
                    "focus": "[x, y] 0..1", "treatment": "duotone (optional): unifies mixed event photos in brand colours",
                    "duotone": "[dark colour, light colour] palette names (default [primary, surface])"},
        "required": ["assets"], "text_keys": [], "asset_keys": ["assets"], "color_keys": [], "font_keys": [],
    },
    "HeroImage": {
        "desc": "One large image kept at its own aspect ratio, centred: typically a generated hero cutout declared in "
                "generated_assets (or a supplied photo).",
        "variants": ["default"],
        "content": {"asset": "asset id", "height_pct": "number % of canvas height (a minimum: the hero also grows into spare space)",
                    "max_width_pct": "number", "align": "left|center|right", "fill": "bool, default true: grow into spare height",
                    "frame": "bool: draw the image in a card (default: yes for photos, no for generated cutouts)"},
        "required": ["asset"], "text_keys": [], "asset_keys": ["asset"], "color_keys": [], "font_keys": [],
    },
    "BigTitle": {
        "desc": "The headline as graphic type: each line is scaled to span the full width (so a short title fills the poster). "
                "Optional rounded pill label under it. Use it as the main title when the poster has little text.",
        "variants": ["default"],
        "content": {"lines": "list of text, one per line (or title with newlines)", "colors": "list of colour names, one per line",
                    "pill": "text for a pill label under the title", "pill_fill": "colour", "pill_text": "colour",
                    "cap_pct": "number, max line height % of poster height (default 13)", "align": "left|center|right",
                    "font_roles": "list of font roles per line, e.g. [light, display_heavy] for a two-weight title"},
        "required": [], "text_keys": ["lines", "title", "pill"], "asset_keys": [],
        "color_keys": ["colors", "pill_fill", "pill_text", "color"], "font_keys": ["font_role"],
    },
    "FactLine": {
        "desc": "The date | time | venue line: the brief's own words side by side with thin vertical rules between them (one item "
                "per line when they do not fit). Use it for the key facts right under the headline.",
        "variants": ["default"],
        "content": {"items": "list of verbatim text (date, time, venue)", "color": "colour", "rule_color": "colour",
                    "align": "left|center|right", "upper": "bool (default true)", "factor": "number size factor (default 0.8)"},
        "required": ["items"], "text_keys": ["items"], "asset_keys": [], "color_keys": ["color", "rule_color"], "font_keys": [],
    },
    "FactStrip": {
        "desc": "Labelled facts in columns: a small spaced-capitals label (DATE, TIME, VENUE, FEE, MODE, DURATION, ELIGIBILITY, "
                "CONTACT, REGISTER, DEADLINE) above each verbatim value, thin rules between columns. For 3-8 short facts.",
        "variants": ["default"],
        "content": {"items": "list of {label, value}", "cols": "int columns per row (max 4)", "label_color": "colour",
                    "value_color": "colour", "rule_color": "colour"},
        "required": ["items"], "text_keys": ["items[].label", "items[].value"], "asset_keys": [],
        "color_keys": ["label_color", "value_color", "rule_color"], "font_keys": [],
    },
    "LogoStrip": {
        "desc": "The header strip when partner logos exist: our logo first, then partner logos, one height, thin rules between.",
        "variants": ["default"],
        "content": {"logo": "brand logo name", "partners": "list of partner-logo asset ids", "height_pct": "number (default 6)",
                    "align": "left|center|right", "band": "colour: full-width band behind the strip"},
        "required": [], "text_keys": [], "asset_keys": ["partners"], "color_keys": [], "font_keys": [],
    },
    "BottomCutout": {
        "desc": "The bottom anchor: a cutout (a generated object from generated_assets; a supplied person photo only when the brief asks "
                "for a standing cutout, its background is removed, the photo itself is not altered) standing on the bottom "
                "edge; anchor 'bottom'. It grows to fill the height the content leaves free, so the poster never has an empty lower "
                "part. beside true = it stands next to the text column (split skeleton) instead of below the content.",
        "variants": ["default"],
        "content": {"asset": "asset id", "align": "left|center|right", "height_pct": "number min height % (default 28)",
                    "max_pct": "number max height % (default 38)", "max_width_pct": "number (default 70; 42 beside)", "beside": "bool", "stage": "bool, default true: a brand-colour arch and soft light behind a supplied person",
                    "framing": "head|head_and_shoulders|half_body: reframe a supplied person photo by cropping first"},
        "required": ["asset"], "text_keys": [], "asset_keys": ["asset"], "color_keys": [], "font_keys": [],
    },
    "QRBlock": {
        "desc": "A vector QR code generated from `url` (the URL must appear in the brief) with an optional verbatim caption under it. Use "
                "it when the brief gives a registration / website link and the poster has room for a code.",
        "variants": ["default"],
        "content": {"url": "text from the brief", "size_pct": "number % of canvas width (default 20)", "color": "colour (dark modules)",
                    "align": "left|center|right", "caption": "verbatim text under the code", "caption_color": "colour"},
        "required": ["url"], "text_keys": ["url", "caption"], "asset_keys": [], "color_keys": ["color", "caption_color"], "font_keys": [],
    },
    "TabBanner": {
        "desc": "A solid brand tab hanging from the top edge with one line (strapline / call to action). anchor 'top'.",
        "variants": ["default"],
        "content": {"text": "text", "fill": "colour", "text_color": "colour", "width_pct": "number"},
        "required": ["text"], "text_keys": ["text"], "asset_keys": [], "color_keys": ["fill", "text_color"], "font_keys": [],
    },
    "ImageBlock": {
        "desc": "A supplied image at its own aspect (QR code, small figure), optionally in a thin bordered card.",
        "variants": ["default"],
        "content": {"asset": "asset id", "width_pct": "number % of the available width", "border": "bool, default true",
                    "align": "left|center|right", "edge": "colour"},
        "required": ["asset"], "text_keys": [], "asset_keys": ["asset"], "color_keys": ["edge"], "font_keys": [],
    },
    "Columns": {
        "desc": "Two side-by-side columns with a hairline between them (e.g. a list on the left, QR + call to action on the right). "
                "Its `children` each set content.column to 'left' or 'right'. Fills the middle of the poster with structure.",
        "variants": ["default"],
        "content": {"left_pct": "number, left column width % (default 55)", "rule": "bool, default true",
                    "valign": "top|center|bottom: where the shorter column sits (default center)"},
        "required": [], "text_keys": [], "asset_keys": [], "color_keys": [], "font_keys": [],
    },
    "PhotoFade": {
        "desc": "A supplied photo bleeding off the bottom edge, full width, dissolving into the page colour through a gradient "
                "(the photo is not altered). Use anchor 'bottom'. Gives a photo-led poster instead of empty space.",
        "variants": ["default"],
        "content": {"asset": "asset id", "height_pct": "number % of poster height (default 38)",
                    "fade_pct": "number % of the photo height that fades (default 45)", "fade_color": "colour (default: page colour)",
                    "max_pct": "number max height % when it grows to fill the free height (default 60)",
                    "focus": "[x, y] 0..1", "treatment": "duotone (optional): brand-colour treatment of a SUPPLIED photo",
                    "duotone": "[dark colour, light colour] palette names (default [primary, surface])"},
        "required": ["asset"], "text_keys": [], "asset_keys": ["asset"], "color_keys": ["fade_color"], "font_keys": [],
    },
    "IconRow": {
        "desc": "A row of vector icon + label columns separated by thin rules (e.g. DOMAINS OF INTEREST). Pick each icon NAME from "
                "ICONS; labels must be verbatim brief text. Gives a structured band that fills the middle of the poster.",
        "variants": ["default"],
        "content": {"items": "list of {label: text, icon: optional icon name hint}; the system picks the icon that matches the label", "icon_color": "colour", "label_color": "colour"},
        "required": ["items"], "text_keys": ["items[].label"], "asset_keys": [], "color_keys": ["icon_color", "label_color"],
        "font_keys": [],
    },
    "BigWord": {
        "desc": "One oversized word or number taken verbatim from the brief (a year, '30', 'JOIN') as the poster's graphic, with an "
                "optional caption beside or below it.",
        "variants": ["default"],
        "content": {"text": "text (short)", "caption": "text", "caption_pos": "right|below", "color": "colour",
                    "caption_color": "colour", "word_pct": "number, word width % when the caption is beside it"},
        "required": ["text"], "text_keys": ["text", "caption"], "asset_keys": [], "color_keys": ["color", "caption_color"],
        "font_keys": [],
    },
    "HeaderRow": {
        "desc": "The institution logo on one side and 1-4 short lines (date, time, venue) aligned on the other: a flexible header "
                "instead of the centred logo-then-title stack. Counts as the logo.",
        "variants": ["default"],
        "content": {"side": "left|right (logo side, default left)", "lines": "list of verbatim text lines", "logo": "brand logo name",
                    "logo_pct": "number, logo width % of canvas (default 18)", "color": "colour", "role": "type role",
                    "band": "colour: a full-width band behind the header strip (set by the look)"},
        "required": [], "text_keys": ["lines"], "asset_keys": [], "color_keys": ["color"], "font_keys": [],
    },
    "Panel": {
        "desc": "Filled container that lays its `children` out as an inner stack (e.g. a navy hero panel with logo + title).",
        "variants": ["default"],
        "content": {"fill": "colour", "padding_pct": "number", "gap": "number", "radius": "number"},
        "required": [], "text_keys": [], "asset_keys": [], "color_keys": ["fill"], "font_keys": [],
    },
}

TYPE_ROLES = ["display", "title", "heading", "subtitle", "body", "caption", "label", "micro"]
EMPHASIS = ["hero", "primary", "secondary", "tertiary"]
ANCHORS = ["flow", "top", "bottom"]
