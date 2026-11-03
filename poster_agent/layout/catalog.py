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
                    "width_pct": "number, % of canvas width (optional; never below the logo's minimum)",
                    "plate": "true or a colour name: a flat plate that hugs the logo. REQUIRED when the poster background is dark "
                             "(brand logos need a flat light field); use this instead of wrapping the logo in a full-width Panel"},
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
                    "title_font": "font role"},
        "required": ["title"], "text_keys": ["eyebrow", "title", "subtitle", "tagline"], "asset_keys": [],
        "color_keys": ["eyebrow_color", "title_color", "subtitle_color", "tagline_color"], "font_keys": ["title_font"],
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
        "variants": ["band", "columns"],
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
        "desc": "A featured guest: optional label, circular photo, name, designation lines.",
        "variants": ["default"],
        "content": {"label": "text e.g. CHIEF GUEST", "photo": "asset id", "name": "text", "lines": "list of text",
                    "photo_d_pct": "number photo diameter % of canvas width"},
        "required": ["name"], "text_keys": ["label", "name", "lines"], "asset_keys": ["photo"], "color_keys": [], "font_keys": [],
    },
    "SpeakerGrid": {
        "desc": "Circular photos with name and designation in aligned rows; last row centred. content.speakers = "
                "[{name, designation, asset}].",
        "variants": ["default"],
        "content": {"speakers": "list of {name, designation, asset}", "cols": "int columns (default 4)",
                    "role_max_lines": "int"},
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
                    "max_width_pct": "number", "align": "left|center|right", "fill": "bool, default true: grow into spare height"},
        "required": ["asset"], "text_keys": [], "asset_keys": ["asset"], "color_keys": [], "font_keys": [],
    },
    "BigTitle": {
        "desc": "The headline as graphic type: each line is scaled to span the full width (so a short title fills the poster). "
                "Optional rounded pill label under it. Use it as the main title when the poster has little text.",
        "variants": ["default"],
        "content": {"lines": "list of text, one per line (or title with newlines)", "colors": "list of colour names, one per line",
                    "pill": "text for a pill label under the title", "pill_fill": "colour", "pill_text": "colour",
                    "cap_pct": "number, max line height % of poster height (default 13)", "align": "left|center|right"},
        "required": [], "text_keys": ["lines", "title", "pill"], "asset_keys": [],
        "color_keys": ["colors", "pill_fill", "pill_text", "color"], "font_keys": ["font_role"],
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
                    "focus": "[x, y] 0..1", "treatment": "duotone (optional): brand-colour treatment of a SUPPLIED photo",
                    "duotone": "[dark colour, light colour] palette names (default [primary, surface])"},
        "required": ["asset"], "text_keys": [], "asset_keys": ["asset"], "color_keys": ["fade_color"], "font_keys": [],
    },
    "IconRow": {
        "desc": "A row of vector icon + label columns separated by thin rules (e.g. DOMAINS OF INTEREST). Pick each icon NAME from "
                "ICONS; labels must be verbatim brief text. Gives a structured band that fills the middle of the poster.",
        "variants": ["default"],
        "content": {"items": "list of {icon: icon name, label: text}", "icon_color": "colour", "label_color": "colour"},
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
